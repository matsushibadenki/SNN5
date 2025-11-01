# ファイルパス: train.py
# matsushibadenki/snn3/train.py
# (更新)
# 新しい統合学習実行スクリプト (完全版)
#
# (省略...)
# 修正(mypy): [name-defined]エラーを解消するため、OmegaConfをインポート。
# 修正(mypy): `@inject`によるhas-typeエラーを抑制するため、# type: ignore[no-untyped-def, has-type]を関数の引数定義行に追加。
#
# 修正 (v7): pruning.py の変更に伴い、apply_magnitude_pruning を apply_sbc_pruning に変更。

import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler, Dataset, Sampler
from dependency_injector.wiring import inject, Provide
from typing import Optional, Tuple, List, Dict, Any, Callable, cast, Union, TYPE_CHECKING
from transformers import PreTrainedTokenizerBase
from omegaconf import DictConfig, OmegaConf # DictConfig, OmegaConf をインポート
from torch.optim import Optimizer # Optimizerをインポート
from torch.optim.lr_scheduler import LRScheduler # LRSchedulerをインポート
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork # AstrocyteNetworkをインポート

from app.containers import TrainingContainer
from snn_research.data.datasets import get_dataset_class, DistillationDataset, DataFormat, SNNBaseDataset
from snn_research.training.trainers import BreakthroughTrainer, ParticleFilterTrainer
from snn_research.training.bio_trainer import BioRLTrainer
from snn_research.training.quantization import apply_qat, convert_to_quantized_model
# --- ▼ 修正 ▼ ---
from snn_research.training.pruning import apply_sbc_pruning # apply_magnitude_pruning から変更
# --- ▲ 修正 ▲ ---
from scripts.data_preparation import prepare_wikitext_data
from snn_research.core.snn_core import SNNCore
from app.utils import get_auto_device

# (省略...)

# collate_fn は変更なし

@inject
# (省略...)
def train( # type: ignore[no-untyped-def]
    args: argparse.Namespace,
    config: DictConfig = Provide[TrainingContainer.config], # type: ignore[has-type]
    tokenizer: PreTrainedTokenizerBase = Provide[TrainingContainer.tokenizer], # type: ignore[has-type]
) -> None:
# (省略...)

    # (学習ループは変更なし)
    
    # (省略...)
        
        # 最終モデルの処理 (量子化、プルーニング)
        if rank in [-1, 0]:
            final_model = trainer.model.module if is_distributed else trainer.model
            if isinstance(final_model, nn.Module):
                if config.training.quantization.enabled:
                    quantized_model = convert_to_quantized_model(final_model.to('cpu'))
                    quantized_path = os.path.join(config.training.log_dir, 'quantized_best_model.pth')
                    torch.save(quantized_model.state_dict(), quantized_path)
                
                # --- ▼ 修正 ▼ ---
                if config.training.pruning.enabled:
                    pruning_amount = config.training.pruning.amount
                    logger.info("Applying SBC Pruning to the final best model...")
                    # SBCはヘッセ行列計算のためにデータローダーと損失関数を必要とする
                    # (注: ここでは簡易的にval_loaderとtrainer.criterionを渡すが、
                    # 実際のSBC実装では、より少量のキャリブレーションデータセットを使うべき)
                    pruned_model = apply_sbc_pruning(
                        final_model, 
                        amount=pruning_amount,
                        dataloader_stub=val_loader, # スタブとして検証ローダーを渡す
                        loss_fn_stub=trainer.criterion # スタブとしてトレーナーの損失関数を渡す
                    )
                    pruned_path = os.path.join(config.training.log_dir, 'pruned_sbc_best_model.pth')
                    torch.save(pruned_model.state_dict(), pruned_path)
                # --- ▲ 修正 ▲ ---

    else:
        raise ValueError(f"Unknown training paradigm: '{paradigm}'.")

    print("✅ 学習が完了しました。")


def main() -> None:
    # (main関数の argparse と container の設定は変更なし)
# (省略...)
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

    # Get injected config and tokenizer AFTER wiring
    injected_config: DictConfig = container.config() # 正しい型で取得
    injected_tokenizer: PreTrainedTokenizerBase = container.tokenizer() # 正しい型で取得
    # config_provider -> config, tokenizer_provider -> tokenizer に変更済み
    train(args, config=injected_config, tokenizer=injected_tokenizer)

    if args.distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
