# ファイルパス: train.py
# matsushibadenki/snn3/train.py
# (更新)
# 新しい統合学習実行スクリプト (完全版)
#
# (v1-v11 修正履歴は省略)
#
# 修正 (v12):
# - 健全性チェック (health-check) での `AttributeError: 'dict' object has no attribute 'training'` エラーを解消。
# - DIコンテナ (@inject) が返す config オブジェクトは DictConfig ではなく標準の dict であるため、
#   @inject を削除し、main() 関数内で container.config() から dict を取得後、
#   OmegaConf.create() で明示的に DictConfig に変換してから train() 関数に渡すように変更。

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
from app.utils import get_auto_device
# ◾️◾️◾️ 追加: logging ◾️◾️◾️
import logging
logger = logging.getLogger(__name__)

# DIコンテナのセットアップ
container = TrainingContainer()

# --- ▼ 修正: collate_fn のインデントを修正 ▼ ---
def collate_fn(tokenizer: PreTrainedTokenizerBase, is_distillation: bool) -> Callable[[List[Any]], Any]:
# ... existing code ...
    def collate(batch: List[Any]) -> Any:
        padding_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        inputs: List[torch.Tensor] = []
# ... existing code ...
    return collate
# --- ▲ 修正 ▲ ---


# --- ▼ 修正 (v12): @inject を削除し、config: DictConfig を明示的に受け取る ▼ ---
def train(
    args: argparse.Namespace,
    config: DictConfig, # type: ignore[has-type]
    tokenizer: PreTrainedTokenizerBase, # type: ignore[has-type]
) -> None:
# --- ▲ 修正 (v12) ▲ ---
    """学習プロセスを実行するメイン関数"""
    is_distributed = args.distributed
# ... existing code ...
    rank = int(os.environ.get("LOCAL_RANK", -1))
    device = f'cuda:{rank}' if is_distributed and torch.cuda.is_available() else get_auto_device()

    # configがDictConfigであることを確認
    paradigm = config.training.paradigm

    print(f"🚀 学習パラダイム '{paradigm}' で学習を開始します...")
# ... existing code ...
    trainer: Union[BreakthroughTrainer, BioRLTrainer, ParticleFilterTrainer]

    if paradigm.startswith("bio-"):
# ... existing code ...
            raise ValueError(f"不明な生物学的学習パラダイム: {paradigm}")

    elif paradigm in ["gradient_based", "self_supervised", "physics_informed", "probabilistic_ensemble"]:
        # --- 勾配ベース学習パラダイムの実行 ---
# ... existing code ...
        if is_distributed and paradigm != "gradient_based":
            raise NotImplementedError(f"{paradigm} learning does not support DDP yet.")

        is_distillation = paradigm == "gradient_based" and config.training.gradient_based.type == "distillation"
# ... existing code ...
        # データセットの準備
        wikitext_path = "data/wikitext-103_train.jsonl"
        data_path: str
# ... existing code ...
        if os.path.exists(wikitext_path):
            data_path = wikitext_path
        else:
# ... existing code ...
             data_path = args.data_path or "data/default_data.jsonl"
             # 修正(v12): config は DictConfig なので .get() や OmegaConf.select を使用
             data_path_config = OmegaConf.select(config, "data.path", default=None)
             if not isinstance(data_path_config, str):
                 data_path = args.data_path or "data/default_data.jsonl"
# ... existing code ...
                 print(f"Warning: config.data.path was not a string, using fallback: {data_path}")
             else:
                 data_path = args.data_path or data_path_config

        DatasetClass = get_dataset_class(DataFormat(config.data.format))
# ... existing code ...
        max_seq_len = OmegaConf.select(config, "model.time_steps", default=128) # Use OmegaConf.select

        if is_distillation:
# ... existing code ...
            dataset = DistillationDataset(file_path=distill_jsonl_path, data_dir=data_dir, tokenizer=tokenizer, max_seq_len=max_seq_len)
        else:
            if not os.path.exists(data_path):
# ... existing code ...
                      raise FileNotFoundError(f"Data file not found: {data_path}")
            dataset = DatasetClass(file_path=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)

        # Ensure split ratio is valid before splitting
        split_ratio = OmegaConf.select(config, "data.split_ratio", default=0.1)
# ... existing code ...
             if train_size <= 0: raise ValueError("Dataset too small to split.")


        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# ... existing code ...
        # --- ▼ 修正 ▼ ---
        train_sampler: Optional[Sampler[int]] = DistributedSampler(train_dataset) if is_distributed else None # Sampler[int] に修正
        # --- ▲ 修正 ▲ ---
# ... existing code ...
        val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn(tokenizer, is_distillation), num_workers=0)

        snn_model: nn.Module = container.snn_model(backend=args.backend)

        # --- ▼ 修正: SNN5改善レポート 4.2 (SpQuant) を QAT より先に適用 ▼ ---
# ... existing code ...
        # PyTorch標準のQAT (SpQuantと併用は通常しないが、設定上は可能)
        elif config.training.quantization.enabled:
            logger.info("Applying PyTorch QAT preparation...")
# ... existing code ...
        # --- ▲ 修正 ▲ ---
            
        snn_model.to(device)

        if is_distributed:
# ... existing code ...
            snn_model = DDP(snn_model, device_ids=[rank], find_unused_parameters=True)

        # --- ▼ 修正: astrocyte の型を Optional[AstrocyteNetwork] に ▼ ---
        astrocyte: Optional[AstrocyteNetwork] = container.astrocyte_network(snn_model=snn_model) if args.use_astrocyte else None
# ... existing code ...
        # --- ▲ 修正 ▲ ---

        trainer_provider: Callable[..., BreakthroughTrainer]
        optimizer: Optimizer # Use imported Optimizer
# ... existing code ...
        scheduler: Optional[LRScheduler] # Use imported LRScheduler

        if paradigm == "gradient_based":
            optimizer = container.optimizer(params=snn_model.parameters())
# ... existing code ...
            trainer_provider = container.distillation_trainer if is_distillation else container.standard_trainer
        elif paradigm == "self_supervised":
            optimizer = container.optimizer(params=snn_model.parameters()) # Assuming same optimizer provider
# ... existing code ...
            trainer_provider = container.self_supervised_trainer
        elif paradigm == "physics_informed":
            optimizer = container.pi_optimizer(params=snn_model.parameters())
# ... existing code ...
            trainer_provider = container.physics_informed_trainer
        else: # probabilistic_ensemble
            optimizer = container.optimizer(params=snn_model.parameters()) # Assuming same optimizer provider
# ... existing code ...
            trainer_provider = container.probabilistic_ensemble_trainer

        # --- ▼ 修正: trainer_kwargs の型を明示し、astrocyteの型エラーを解消 ▼ ---
        trainer_kwargs: Dict[str, Any] = {
# ... existing code ...
            "model": snn_model,
            "optimizer": optimizer,
            "scheduler": scheduler,
# ... existing code ...
            "device": device,
            "rank": rank
            # "astrocyte_network" will be added conditionally below
# ... existing code ...
        }
        if args.use_astrocyte and astrocyte is not None and paradigm in ["gradient_based", "self_supervised", "physics_informed", "probabilistic_ensemble"]:
             trainer_kwargs["astrocyte_network"] = astrocyte # Type matches Optional[AstrocyteNetwork]
        # --- ▲ 修正 ▲ ---
# ... existing code ...


        trainer = trainer_provider(**trainer_kwargs)

        if args.load_ewc_data:
# ... existing code ...
            trainer.load_ewc_data(args.load_ewc_data)

        start_epoch = trainer.load_checkpoint(args.resume_path) if args.resume_path else 0
        for epoch in range(start_epoch, config.training.epochs):
# ... existing code ...
            if train_sampler and isinstance(train_sampler, DistributedSampler): train_sampler.set_epoch(epoch) # isinstanceで型ガード
            trainer.train_epoch(train_loader, epoch)
            if rank in [-1, 0] and (epoch % config.training.eval_interval == 0 or epoch == config.training.epochs - 1):
# ... existing code ...
                val_metrics = trainer.evaluate(val_loader, epoch)
                if epoch % config.training.log_interval == 0:
                    checkpoint_path = os.path.join(config.training.log_dir, f"checkpoint_epoch_{epoch}.pth")
                    # --- ▼ 修正: config.modelを辞書に変換 ▼ ---
                    model_config_dict = OmegaConf.to_container(config.model, resolve=True) if isinstance(config.model, DictConfig) else config.model
# ... existing code ...
                    if not isinstance(model_config_dict, dict): model_config_dict = {} # Fallback
                    trainer.save_checkpoint(path=checkpoint_path, epoch=epoch, metric_value=val_metrics.get('total', float('inf')), tokenizer_name=config.data.tokenizer_name, config=model_config_dict)
                    # --- ▲ 修正 ▲ ---
# ... existing code ...

        if rank in [-1, 0] and args.task_name and config.training.gradient_based.loss.ewc_weight > 0:
            trainer._compute_ewc_fisher_matrix(train_loader, args.task_name)

        # 最終モデルの処理 (量子化、プルーニング)
# ... existing code ...
        # --- ▼ 修正 (SNN5改善レポート 4.1, 4.2, 4.3 対応): プルーニングと量子化の順序変更・新機能追加 ▼ ---
        # --- ▼ 修正 (mypy [assignment]): `type: ignore` を追加 ▼ ---
        if rank in [-1, 0]:
# ... existing code ...
            final_model_wrapped = trainer.model.module if is_distributed else trainer.model
            
            # SNNCoreラッパーから内部モデルを取得
# ... existing code ...
            final_model: nn.Module
            if isinstance(final_model_wrapped, SNNCore):
                final_model = final_model_wrapped.model # type: ignore[assignment]
# ... existing code ...
            else:
                # DDP や他のラッパーが SNNCore をラップしていない場合
                final_model = final_model_wrapped # type: ignore[assignment]
            
            if isinstance(final_model, nn.Module):
# ... existing code ...
                model_to_process = final_model # 処理対象のモデル
                
                # --- 1a. 時空間プルーニング (SNN5改善レポート 4.1) ---
                if OmegaConf.select(config, "training.pruning.spatio_temporal.enabled", default=False):
# ... existing code ...
                    
                    st_amount: float = OmegaConf.select(config, "training.pruning.spatio_temporal.spatial_amount", default=0.2)
                    st_kl_thresh: float = OmegaConf.select(config, "training.pruning.spatio_temporal.kl_threshold", default=0.01)
# ... existing code ...
                    
                    # (BaseModelからtime_stepsを取得)
                    snn_time_steps: int = cast(int, getattr(model_to_process, 'time_steps', 16))

                    st_pruned_model = apply_spatio_temporal_pruning(
# ... existing code ...
                        model_to_process,
                        dataloader=val_loader, # スタブとして検証ローダーを渡す
                        time_steps=snn_time_steps,
# ... existing code ...
                        spatial_amount=st_amount,
                        kl_threshold=st_kl_thresh
                    )
# ... existing code ...
                    torch.save(st_pruned_model.state_dict(), st_pruned_path)
                    logger.info(f"✅ Spatio-Temporal Pruned model saved to {st_pruned_path}")
                    model_to_process = st_pruned_model # 次のステップのため、処理済みモデルを更新

                # --- 1b. SBC プルーニング (SNN5改善レポート 4.3 順序) ---
# ... existing code ...
                if OmegaConf.select(config, "training.pruning.sbc.enabled", default=False): # 'enabled' -> 'sbc.enabled'
                    pruning_amount: float = OmegaConf.select(config, "training.pruning.sbc.amount", default=0.2)
                    logger.info("Applying SBC Pruning to the final model (post ST-pruning if enabled)...")
# ... existing code ...
                    
                    pruned_model = apply_sbc_pruning(
                        model_to_process, 
# ... existing code ...
                        amount=pruning_amount,
                        dataloader_stub=val_loader, # スタブとして検証ローダーを渡す
                        loss_fn_stub=trainer.criterion # スタブとしてトレーナーの損失関数を渡す
# ... existing code ...
                    )
                    pruned_path = os.path.join(config.training.log_dir, 'pruned_sbc_best_model.pth')
                    torch.save(pruned_model.state_dict(), pruned_path)
# ... existing code ...
                    logger.info(f"✅ SBC Pruned model saved to {pruned_path}")
                    model_to_process = pruned_model # 次のステップのため、処理済みモデルを更新
                
                # --- 2a. SNN固有量子化 (SpQuant) (SNN5改善レポート 4.2) ---
# ... existing code ...
                if OmegaConf.select(config, "training.quantization.spquant.enabled", default=False):
                    logger.info("Applying SpQuant-SNN (Membrane Quantization) to the final model (post-pruning if enabled)...")
                    # (SpQuantは訓練前に行うのがQATだが、ここでは訓練後のモデルに適用するスタブ)
# ... existing code ...
                    spquant_model = apply_spquant_quantization(model_to_process.to('cpu'))
                    spquant_path = os.path.join(config.training.log_dir, 'quantized_spquant_best_model.pth')
                    torch.save(spquant_model.state_dict(), spquant_path)
# ... existing code ...
                    logger.info(f"✅ SpQuant (Stub) model saved to {spquant_path}")
                
                # --- 2b. 標準QAT (SNN5改善レポート 4.3 順序) ---
                elif config.training.quantization.enabled:
# ... existing code ...
                    logger.info("Applying PyTorch QAT conversion to the final model (post-pruning if enabled)...")
                    quantized_model = convert_to_quantized_model(model_to_process.to('cpu'))
                    quantized_path = os.path.join(config.training.log_dir, 'quantized_qat_best_model.pth')
# ... existing code ...
                    torch.save(quantized_model.state_dict(), quantized_path)
                    logger.info(f"✅ QAT Quantized model saved to {quantized_path}")
        # --- ▲ 修正 ▲ ---
# ... existing code ...
        # --- ▲ 修正 ▲ ---
            
    else:
        raise ValueError(f"Unknown training paradigm: '{paradigm}'.")
# ... existing code ...

    print("✅ 学習が完了しました。")


def main() -> None:
# ... existing code ...
    parser = argparse.ArgumentParser(description="SNN 統合学習スクリプト")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="基本設定ファイル")
    parser.add_argument("--model_config", type=str, help="モデルアーキテクチャ設定ファイル")
# ... existing code ...
    parser.add_argument("--paradigm", type=str, help="学習パラダイムを上書き (例: gradient_based, bio-causal-sparse, bio-particle-filter)")
    parser.add_argument("--backend", type=str, default="spikingjelly", choices=["spikingjelly", "snntorch"], help="SNNシミュレーションバックエンドライブラリ")
    args = parser.parse_args()
# ... existing code ...

    # Load base config first
    container.config.from_yaml(args.config)

    # Load model config if provided
# ... existing code ...
    if args.model_config:
         try:
             container.config.from_yaml(args.model_config)
# ... existing code ...
         except FileNotFoundError:
             print(f"Warning: Model config file not found: {args.model_config}. Using base config model settings.")
         except Exception as e:
# ... existing code ...
              print(f"Error loading model config '{args.model_config}': {e}. Using base config model settings.")


    # Explicit overrides from command line
# ... existing code ...
    if args.data_path: container.config.data.path.from_value(args.data_path)
    if args.paradigm: container.config.training.paradigm.from_value(args.paradigm)

    # Apply dotted overrides
# ... existing code ...
    if args.override_config:
        for override in args.override_config:
            try:
# ... existing code ...
                keys, value_str = override.split('=', 1)
                # Try to infer type
                try: value: Any = int(value_str)
# ... existing code ...
                        elif value_str.lower() == 'false': value = False
                        else: value = value_str # Keep as string

                # Use OmegaConf's update method for dotted keys
                OmegaConf.update(container.config(), keys, value, merge=True)
# ... existing code ...
            except Exception as e:
                print(f"Error applying override '{override}': {e}")


    if args.distributed:
# ... existing code ...
        if "MASTER_ADDR" not in os.environ: os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ: os.environ["MASTER_PORT"] = "29500" # Default port

        dist.init_process_group(backend="nccl")
# ... existing code ...

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
# ... existing code ...
    main()

}
