# ファイルパス: train.py
# matsushibadenki/snn3/train.py

import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
# --- ▼ 修正: [annotation-unchecked] noteを解消するため、型ヒントを追加。 ▼ ---
from torch.utils.data import DataLoader, random_split, DistributedSampler, Dataset, Sampler
from dependency_injector.wiring import inject, Provide
from typing import Optional, Tuple, List, Dict, Any, Callable, cast, Union, TYPE_CHECKING
from transformers import PreTrainedTokenizerBase
from omegaconf import DictConfig, OmegaConf # DictConfig, OmegaConf をインポート
from torch.optim import Optimizer # Optimizerをインポート
from torch.optim.lr_scheduler import LRScheduler # LRSchedulerをインポート
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork # AstrocyteNetworkをインポート
# --- ▲ 修正 ▲ ---

from app.containers import TrainingContainer
from snn_research.data.datasets import get_dataset_class, DistillationDataset, DataFormat, SNNBaseDataset
from snn_research.training.trainers import BreakthroughTrainer, ParticleFilterTrainer
from snn_research.training.bio_trainer import BioRLTrainer
# --- ▼ 修正 (SpQuant量子化をインポート) ▼ ---
from snn_research.training.quantization import apply_qat, convert_to_quantized_model, apply_spquant_quantization
# --- ▲ 修正 ▲ ---
# --- ▼ 修正 (SBCと時空間プルーニングをインポート) ▼ ---
from snn_research.training.pruning import apply_sbc_pruning, apply_spatio_temporal_pruning
# --- ▲ 修正 ▲ ---
from scripts.data_preparation import prepare_wikitext_data
from snn_research.core.snn_core import SNNCore
# --- ▼ 修正 (v15): collate_fn を app/utils からインポート ▼ ---
from app.utils import get_auto_device, collate_fn
# --- ▲ 修正 (v15) ▲ ---
# ◾️◾️◾️ 追加: logging ◾️◾️◾️
import logging
logger = logging.getLogger(__name__)

# DIコンテナのセットアップ
container = TrainingContainer()


# --- ▼ 修正 (v14): collate_fn のインデントを修正 (クラスの外に出す) ▼ ---
def collate_fn(tokenizer: PreTrainedTokenizerBase, is_distillation: bool) -> Callable[[List[Any]], Any]:
    """
    データローダー用の Collate 関数。
    """
    def collate(batch: List[Any]) -> Any:
        padding_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        inputs: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        logits: List[torch.Tensor] = [] # Only used if is_distillation

        # Handle different batch item types (dict from HF, tuple from SNNBaseDataset)
        for item in batch:
            if isinstance(item, dict):
                # Ensure keys exist and are tensors or tensor-like
                inp = item.get('input_ids')
                tgt = item.get('labels') # Assuming 'labels' key
                if inp is None or tgt is None: continue # Skip invalid items
                inputs.append(torch.tensor(inp) if not isinstance(inp, torch.Tensor) else inp)
                targets.append(torch.tensor(tgt) if not isinstance(tgt, torch.Tensor) else tgt)
                if is_distillation:
                    lg = item.get('teacher_logits')
                    if lg is not None: logits.append(torch.tensor(lg) if not isinstance(lg, torch.Tensor) else lg)
                    else: logits.append(torch.empty(0)) # Placeholder if missing

            elif isinstance(item, tuple) and len(item) >= 2:
                # Ensure elements are tensors or tensor-like
                inp = item[0]
                tgt = item[1]
                if not isinstance(inp, (torch.Tensor, list, tuple)) or not isinstance(tgt, (torch.Tensor, list, tuple)): continue
                inputs.append(torch.tensor(inp) if not isinstance(inp, torch.Tensor) else inp)
                targets.append(torch.tensor(tgt) if not isinstance(tgt, torch.Tensor) else tgt)
                if is_distillation:
                    if len(item) >= 3:
                         lg = item[2]
                         if lg is not None: logits.append(torch.tensor(lg) if not isinstance(lg, torch.Tensor) else lg)
                         else: logits.append(torch.empty(0))
                    else: logits.append(torch.empty(0))
            else:
                print(f"Warning: Skipping unsupported batch item type: {type(item)}")
                continue # Skip unsupported item types

        if not inputs or not targets: # If batch becomes empty after filtering
            # Return empty structures that match expected types
            if is_distillation:
                return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0, 0), dtype=torch.float32)
            else:
                # 辞書形式を返す (標準のcollate_fnが期待する形式)
                return {
                    "input_ids": torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_val),
                    "attention_mask": torch.nn.utils.rnn.pad_sequence([torch.ones_like(i) for i in inputs], batch_first=True, padding_value=0),
                    "labels": torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
                }

        # --- 標準 (非蒸留) の collate ロジック (辞書を返す) ---
        if not is_distillation:
            padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_val)
            padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
            attention_mask = torch.ones_like(padded_inputs)
            attention_mask[padded_inputs == padding_val] = 0
            return {
                "input_ids": padded_inputs,
                "attention_mask": attention_mask,
                "labels": padded_targets
            }
        
        # --- 蒸留 (タプルを返す) ---
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_val)
        padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
        padded_logits = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=0.0)
        attention_mask = torch.ones_like(padded_inputs)
        attention_mask[padded_inputs == padding_val] = 0
        
        # ターゲットとロジットのシーケンス長を合わせる
        seq_len = padded_inputs.shape[1]
        if padded_targets.shape[1] < seq_len:
            pad = torch.full((padded_targets.shape[0], seq_len - padded_targets.shape[1]), -100, dtype=padded_targets.dtype, device=padded_targets.device)
            padded_targets = torch.cat([padded_targets, pad], dim=1)
        if padded_logits.shape[1] < seq_len:
            pad = torch.full((padded_logits.shape[0], seq_len - padded_logits.shape[1], padded_logits.shape[2]), 0.0, dtype=padded_logits.dtype, device=padded_logits.device)
            padded_logits = torch.cat([padded_logits, pad], dim=1)
            
        return padded_inputs, attention_mask, padded_targets, padded_logits
# --- ▲ 修正 (v14) ▲ ---


# --- ▼ 修正 (v12): @inject を削除し、config: DictConfig を明示的に受け取る ▼ ---
def train(
    args: argparse.Namespace,
    config: DictConfig, # type: ignore[has-type]
    tokenizer: PreTrainedTokenizerBase, # type: ignore[has-type]
) -> None:
# --- ▲ 修正 (v12) ▲ ---
    """学習プロセスを実行するメイン関数"""
    is_distributed = args.distributed
    rank = int(os.environ.get("LOCAL_RANK", -1))
    device = f'cuda:{rank}' if is_distributed and torch.cuda.is_available() else get_auto_device()

    # configがDictConfigであることを確認
    paradigm = config.training.paradigm

    print(f"🚀 学習パラダイム '{paradigm}' で学習を開始します...")

    trainer: Union[BreakthroughTrainer, BioRLTrainer, ParticleFilterTrainer]

    if paradigm.startswith("bio-"):
        # --- 生物学的学習パラダイムの実行 ---
        if paradigm == "bio-causal-sparse":
            print("🧬 適応的因果スパース化を有効にした強化学習を開始します。")
            container.config.training.biologically_plausible.adaptive_causal_sparsification.enabled.from_value(True)
            trainer = container.bio_rl_trainer()
            cast(BioRLTrainer, trainer).train(num_episodes=config.training.epochs)
        elif paradigm == "bio-particle-filter":
            print("🌪️ パーティクルフィルタによる確率的学習を開始します (CPUベース)。")
            container.config.training.biologically_plausible.particle_filter.enabled.from_value(True)
            trainer = container.particle_filter_trainer()
            dummy_data = torch.rand(1, 10, device=device)
            dummy_targets = torch.rand(1, 2, device=device)
            for epoch in range(config.training.epochs):
                loss = cast(ParticleFilterTrainer, trainer).train_step(dummy_data, dummy_targets)
                print(f"Epoch {epoch+1}/{config.training.epochs}: Particle Filter Loss = {loss:.4f}")
        elif paradigm == "bio-probabilistic-hebbian":
            print("🧬 確率的ヘブ学習を開始します...")
            prob_trainer: BioRLTrainer = container.probabilistic_trainer()
            prob_trainer.train(num_episodes=config.training.epochs)
        else:
            raise ValueError(f"不明な生物学的学習パラダイム: {paradigm}")

    elif paradigm in ["gradient_based", "self_supervised", "physics_informed", "probabilistic_ensemble"]:
        # --- 勾配ベース学習パラダイムの実行 ---
        if is_distributed and paradigm != "gradient_based":
            raise NotImplementedError(f"{paradigm} learning does not support DDP yet.")

        is_distillation = paradigm == "gradient_based" and config.training.gradient_based.type == "distillation"

        # --- ▼ 修正 (v14): データパスの決定ロジックを修正 ▼ ---
        # 1. args.data_path (CLI引数) を最優先
        data_path: str
        if args.data_path:
            data_path = args.data_path
            logger.info(f"Using data_path from command line: {data_path}")
        else:
            # 2. config.data.path (設定ファイル)
            data_path_config = OmegaConf.select(config, "data.path", default=None)
            if not isinstance(data_path_config, str):
                data_path = "data/default_data.jsonl" # 最終フォールバック
                logger.warning(f"config.data.path was not a string, using fallback: {data_path}")
            else:
                data_path = data_path_config
            
            # 3. wikitext が存在する場合の上書き (ただし args.data_path が指定されていない場合のみ)
            wikitext_path = "data/wikitext-103_train.jsonl"
            if os.path.exists(wikitext_path):
                data_path = wikitext_path
                logger.info(f"Found wikitext, using data_path: {data_path}")
            else:
                 logger.info(f"Using data_path from config: {data_path}")
        # --- ▲ 修正 (v14) ▲ ---

        DatasetClass = get_dataset_class(DataFormat(config.data.format))
        dataset: SNNBaseDataset
        max_seq_len = OmegaConf.select(config, "model.time_steps", default=128) # Use OmegaConf.select

        if is_distillation:
            data_dir = os.path.dirname(data_path) if os.path.isfile(data_path) else data_path
            distill_jsonl_path = os.path.join(data_dir, "distillation_data.jsonl")
            if not os.path.exists(distill_jsonl_path):
                 raise FileNotFoundError(f"Distillation data not found at {distill_jsonl_path}. Run prepare_distillation_data.py first.")
            dataset = DistillationDataset(file_path=distill_jsonl_path, data_dir=data_dir, tokenizer=tokenizer, max_seq_len=max_seq_len)
        else:
            if not os.path.exists(data_path):
                 if data_path == "data/wikitext-103_train.jsonl": # wikitext が期待されていた場合
                      print(f"Data file '{data_path}' not found. Attempting to prepare WikiText data...")
                      prepared_path = prepare_wikitext_data()
                      if prepared_path != data_path:
                           print(f"Warning: Prepared data path '{prepared_path}' differs from expected '{data_path}'. Using prepared path.")
                           data_path = prepared_path
                      if not os.path.exists(data_path):
                           raise FileNotFoundError(f"Data file not found even after preparation: {data_path}")
                 else:
                      raise FileNotFoundError(f"Data file not found: {data_path}")
            dataset = DatasetClass(file_path=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)

        # Ensure split ratio is valid before splitting
        split_ratio = OmegaConf.select(config, "data.split_ratio", default=0.1)
        if not (0 < split_ratio < 1):
             print(f"Warning: Invalid split_ratio {split_ratio}. Using 0.1.")
             split_ratio = 0.1

        train_size = int((1.0 - split_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        
        # --- ▼ 修正 (v13): インデントエラーを修正 ▼ ---
        # Handle cases where split results in zero size
        if train_size <= 0 or val_size <= 0:
             print(f"Warning: Dataset size {len(dataset)} is too small for split ratio {split_ratio}. Adjusting split.")
             # Example adjustment: ensure at least one sample in validation
             val_size = max(1, int(len(dataset) * 0.05)) # Min 1 sample or 5%
             train_size = len(dataset) - val_size
             if train_size <= 0: 
                 raise ValueError("Dataset too small to split.")
        # --- ▲ 修正 (v13) ▲ ---

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # --- ▼ 修正 ▼ ---
        train_sampler: Optional[Sampler[int]] = DistributedSampler(train_dataset) if is_distributed else None # Sampler[int] に修正
        # --- ▲ 修正 ▲ ---
        
        # --- ▼ 修正 (v14): collate_fn が None にならないように修正 ▼ ---
        collate_fn_instance = collate_fn(tokenizer, is_distillation)
        if collate_fn_instance is None:
            raise RuntimeError("collate_fn factory returned None. Check train.py for errors.")
        train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn_instance, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn_instance, num_workers=0)
        # --- ▲ 修正 (v14) ▲ ---

        snn_model: nn.Module = container.snn_model(backend=args.backend)

        # --- ▼ 修正: SNN5改善レポート 4.2 (SpQuant) を QAT より先に適用 ▼ ---
        # SNN固有の量子化 (SpQuant) を先に試みる
        if OmegaConf.select(config, "training.quantization.spquant.enabled", default=False):
             logger.info("Applying SpQuant-SNN (Membrane Quantization)...")
             snn_model = apply_spquant_quantization(snn_model.to('cpu')) # SpQuantはinplace変更を想定
        
        # PyTorch標準のQAT (SpQuantと併用は通常しないが、設定上は可能)
        elif config.training.quantization.enabled:
            logger.info("Applying PyTorch QAT preparation...")
            snn_model = apply_qat(snn_model.to('cpu'))
        # --- ▲ 修正 ▲ ---
            
        snn_model.to(device)

        if is_distributed:
            snn_model = DDP(snn_model, device_ids=[rank], find_unused_parameters=True)

        # --- ▼ 修正: astrocyte の型を Optional[AstrocyteNetwork] に ▼ ---
        astrocyte: Optional[AstrocyteNetwork] = container.astrocyte_network(snn_model=snn_model) if args.use_astrocyte else None
        # --- ▲ 修正 ▲ ---

        trainer_provider: Callable[..., BreakthroughTrainer]
        optimizer: Optimizer # Use imported Optimizer
        scheduler: Optional[LRScheduler] # Use imported LRScheduler

        if paradigm == "gradient_based":
            optimizer = container.optimizer(params=snn_model.parameters())
            scheduler = container.scheduler(optimizer=optimizer) if config.training.gradient_based.use_scheduler else None
            trainer_provider = container.distillation_trainer if is_distillation else container.standard_trainer
        elif paradigm == "self_supervised":
            optimizer = container.optimizer(params=snn_model.parameters()) # Assuming same optimizer provider
            scheduler = container.scheduler(optimizer=optimizer) if config.training.self_supervised.use_scheduler else None
            trainer_provider = container.self_supervised_trainer
        elif paradigm == "physics_informed":
            optimizer = container.pi_optimizer(params=snn_model.parameters())
            scheduler = container.pi_scheduler(optimizer=optimizer) if config.training.physics_informed.use_scheduler else None
            trainer_provider = container.physics_informed_trainer
        else: # probabilistic_ensemble
            optimizer = container.optimizer(params=snn_model.parameters()) # Assuming same optimizer provider
            scheduler = container.scheduler(optimizer=optimizer) if config.training.probabilistic_ensemble.use_scheduler else None
            trainer_provider = container.probabilistic_ensemble_trainer

        # --- ▼ 修正: trainer_kwargs の型を明示し、astrocyteの型エラーを解消 ▼ ---
        trainer_kwargs: Dict[str, Any] = {
            "model": snn_model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "device": device,
            "rank": rank
            # "astrocyte_network" will be added conditionally below
        }
        if args.use_astrocyte and astrocyte is not None and paradigm in ["gradient_based", "self_supervised", "physics_informed", "probabilistic_ensemble"]:
             trainer_kwargs["astrocyte_network"] = astrocyte # Type matches Optional[AstrocyteNetwork]
        # --- ▲ 修正 ▲ ---


        trainer = trainer_provider(**trainer_kwargs)

        if args.load_ewc_data:
            trainer.load_ewc_data(args.load_ewc_data)

        start_epoch = trainer.load_checkpoint(args.resume_path) if args.resume_path else 0
        for epoch in range(start_epoch, config.training.epochs):
            if train_sampler and isinstance(train_sampler, DistributedSampler): train_sampler.set_epoch(epoch) # isinstanceで型ガード
            trainer.train_epoch(train_loader, epoch)
            if rank in [-1, 0] and (epoch % config.training.eval_interval == 0 or epoch == config.training.epochs - 1):
                val_metrics = trainer.evaluate(val_loader, epoch)
                if epoch % config.training.log_interval == 0:
                    checkpoint_path = os.path.join(config.training.log_dir, f"checkpoint_epoch_{epoch}.pth")
                    # --- ▼ 修正: config.modelを辞書に変換 ▼ ---
                    model_config_dict = OmegaConf.to_container(config.model, resolve=True) if isinstance(config.model, DictConfig) else config.model
                    if not isinstance(model_config_dict, dict): model_config_dict = {} # Fallback
                    trainer.save_checkpoint(path=checkpoint_path, epoch=epoch, metric_value=val_metrics.get('total', float('inf')), tokenizer_name=config.data.tokenizer_name, config=model_config_dict)
                    # --- ▲ 修正 ▲ ---

        if rank in [-1, 0] and args.task_name and config.training.gradient_based.loss.ewc_weight > 0:
            trainer._compute_ewc_fisher_matrix(train_loader, args.task_name)

        # 最終モデルの処理 (量子化、プルーニング)
        # --- ▼ 修正 (SNN5改善レポート 4.1, 4.2, 4.3 対応): プルーニングと量子化の順序変更・新機能追加 ▼ ---
        # --- ▼ 修正 (mypy [assignment]): `type: ignore` を追加 ▼ ---
        if rank in [-1, 0]:
            final_model_wrapped = trainer.model.module if is_distributed else trainer.model
            
            # SNNCoreラッパーから内部モデルを取得
            final_model: nn.Module
            if isinstance(final_model_wrapped, SNNCore):
                final_model = final_model_wrapped.model # type: ignore[assignment]
            else:
                # DDP や他のラッパーが SNNCore をラップしていない場合
                final_model = final_model_wrapped # type: ignore[assignment]
            
            if isinstance(final_model, nn.Module):
                model_to_process = final_model # 処理対象のモデル
                
                # --- 1a. 時空間プルーニング (SNN5改善レポート 4.1) ---
                if OmegaConf.select(config, "training.pruning.spatio_temporal.enabled", default=False):
                    logger.info("Applying Spatio-Temporal Pruning to the final best model...")
                    
                    st_amount: float = OmegaConf.select(config, "training.pruning.spatio_temporal.spatial_amount", default=0.2)
                    st_kl_thresh: float = OmegaConf.select(config, "training.pruning.spatio_temporal.kl_threshold", default=0.01)
                    
                    # (BaseModelからtime_stepsを取得)
                    snn_time_steps: int = cast(int, getattr(model_to_process, 'time_steps', 16))

                    st_pruned_model = apply_spatio_temporal_pruning(
                        model_to_process,
                        dataloader=val_loader, # スタブとして検証ローダーを渡す
                        time_steps=snn_time_steps,
                        spatial_amount=st_amount,
                        kl_threshold=st_kl_thresh
                    )
                    st_pruned_path = os.path.join(config.training.log_dir, 'pruned_spatio_temporal_best_model.pth')
                    torch.save(st_pruned_model.state_dict(), st_pruned_path)
                    logger.info(f"✅ Spatio-Temporal Pruned model saved to {st_pruned_path}")
                    model_to_process = st_pruned_model # 次のステップのため、処理済みモデルを更新

                # --- 1b. SBC プルーニング (SNN5改善レポート 4.3 順序) ---
                if OmegaConf.select(config, "training.pruning.sbc.enabled", default=False): # 'enabled' -> 'sbc.enabled'
                    pruning_amount: float = OmegaConf.select(config, "training.pruning.sbc.amount", default=0.2)
                    logger.info("Applying SBC Pruning to the final model (post ST-pruning if enabled)...")
                    
                    pruned_model = apply_sbc_pruning(
                        model_to_process, 
                        amount=pruning_amount,
                        dataloader_stub=val_loader, # スタブとして検証ローダーを渡す
                        loss_fn_stub=trainer.criterion # スタブとしてトレーナーの損失関数を渡す
                    )
                    pruned_path = os.path.join(config.training.log_dir, 'pruned_sbc_best_model.pth')
                    torch.save(pruned_model.state_dict(), pruned_path)
                    logger.info(f"✅ SBC Pruned model saved to {pruned_path}")
                    model_to_process = pruned_model # 次のステップのため、処理済みモデルを更新
                
                # --- 2a. SNN固有量子化 (SpQuant) (SNN5改善レポート 4.2) ---
                if OmegaConf.select(config, "training.quantization.spquant.enabled", default=False):
                    logger.info("Applying SpQuant-SNN (Membrane Quantization) to the final model (post-pruning if enabled)...")
                    # (SpQuantは訓練前に行うのがQATだが、ここでは訓練後のモデルに適用するスタブ)
                    spquant_model = apply_spquant_quantization(model_to_process.to('cpu'))
                    spquant_path = os.path.join(config.training.log_dir, 'quantized_spquant_best_model.pth')
                    torch.save(spquant_model.state_dict(), spquant_path)
                    logger.info(f"✅ SpQuant (Stub) model saved to {spquant_path}")
                
                # --- 2b. 標準QAT (SNN5改善レポート 4.3 順序) ---
                elif config.training.quantization.enabled:
                    logger.info("Applying PyTorch QAT conversion to the final model (post-pruning if enabled)...")
                    quantized_model = convert_to_quantized_model(model_to_process.to('cpu'))
                    quantized_path = os.path.join(config.training.log_dir, 'quantized_qat_best_model.pth')
                    torch.save(quantized_model.state_dict(), quantized_path)
                    logger.info(f"✅ QAT Quantized model saved to {quantized_path}")
        # --- ▲ 修正 ▲ ---
        # --- ▲ 修正 ▲ ---
            
    else:
        raise ValueError(f"Unknown training paradigm: '{paradigm}'.")

    print("✅ 学習が完了しました。")


def main() -> None:
    parser = argparse.ArgumentParser(description="SNN 統合学習スクリプト")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="基本設定ファイル")
    parser.add_argument("--model_config", type=str, help="モデルアーキテクチャ設定ファイル")
    parser.add_argument("--data_path", type=str, help="データセットのパス（configを上書き）")
    parser.add_argument("--task_name", type=str, help="EWCのためにタスク名を指定 (例: 'sst2')")
    parser.add_argument("--override_config", type=str, action='append', help="設定を上書き (例: 'training.epochs=5')")
    parser.add_argument("--distributed", action="store_true", help="分散学習を有効にする")
    parser.add_argument("--resume_path", type=str, help="チェックポイントから学習を再開する")
    parser.add_argument("--load_ewc_data", type=str, help="事前計算されたEWCのFisher行列とパラメータのパス")
    parser.add_argument("--use_astrocyte", action="store_true", help="アストロサイトネットワークを有効にする (gradient_based系のみ)")
    parser.add_argument("--paradigm", type=str, help="学習パラダイムを上書き (例: gradient_based, bio-causal-sparse, bio-particle-filter)")
    parser.add_argument("--backend", type=str, default="spikingjelly", choices=["spikingjelly", "snntorch"], help="SNNシミュレーションバックエンドライブラリ")
    args = parser.parse_args()

    # Load base config first
    container.config.from_yaml(args.config)

    # Load model config if provided
    if args.model_config:
         try:
             container.config.from_yaml(args.model_config)
         except FileNotFoundError:
             print(f"Warning: Model config file not found: {args.model_config}. Using base config model settings.")
         except Exception as e:
              print(f"Error loading model config '{args.model_config}': {e}. Using base config model settings.")


    # Explicit overrides from command line
    if args.data_path: container.config.data.path.from_value(args.data_path)
    if args.paradigm: container.config.training.paradigm.from_value(args.paradigm)

    # Apply dotted overrides
    if args.override_config:
        for override in args.override_config:
            try:
                keys, value_str = override.split('=', 1)
                # Try to infer type
                try: value: Any = int(value_str)
                except ValueError:
                    try: value = float(value_str)
                    except ValueError:
                        if value_str.lower() == 'true': value = True
                        elif value_str.lower() == 'false': value = False
                        else: value = value_str # Keep as string

                # Use OmegaConf's update method for dotted keys
                OmegaConf.update(container.config(), keys, value, merge=True)
            except Exception as e:
                print(f"Error applying override '{override}': {e}")
    # --- ▼ 修正 (v_syn): 498行目の余分な '}' を削除 ▼ ---
    # (削除) }
    # --- ▲ 修正 (v_syn) ▲ ---


    if args.distributed:
        if not dist.is_available(): raise RuntimeError("Distributed training requested but not available.")
        if not torch.cuda.is_available(): raise RuntimeError("Distributed training requires CUDA.")
        # Ensure WORLD_SIZE and RANK are set if not using torchrun
        if "WORLD_SIZE" not in os.environ: os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        if "RANK" not in os.environ: os.environ["RANK"] = "0" # Default for single node, adjust if needed
        if "LOCAL_RANK" not in os.environ: os.environ["LOCAL_RANK"] = os.environ["RANK"]
        if "MASTER_ADDR" not in os.environ: os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ: os.environ["MASTER_PORT"] = "29500" # Default port

        dist.init_process_group(backend="nccl")

    # Wire the container AFTER all configurations are loaded
    container.wire(modules=[__name__])

    # --- ▼ 修正 (v12): container.config() (dict) を OmegaConf.create() でラップ ▼ ---
    # Get injected config and tokenizer AFTER wiring
    injected_config_dict: dict = container.config() # DIコンテナは dict を返す
    injected_config: DictConfig = OmegaConf.create(injected_config_dict) # OmegaConfオブジェクトに変換
    
    injected_tokenizer: PreTrainedTokenizerBase = container.tokenizer() # 正しい型で取得
    
    train(args, config=injected_config, tokenizer=injected_tokenizer)
    # --- ▲ 修正 (v12) ▲ ---

    if args.distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
