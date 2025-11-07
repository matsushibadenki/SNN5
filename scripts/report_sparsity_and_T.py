# ファイルパス: scripts/report_sparsity_and_T.py
#
# Title: SNNモデル スパース性(s)・タイムステップ(T) 診断レポート
#
# Description:
# doc/SNN開発：SNN5プロジェクト改善のための情報収集_追案.md の提案タスク1に基づき、
# 指定されたSNNモデルの2つの重要な効率指標を計測・レポートします。
#   1. T (Time-steps): 推論レイテンシ
#   2. s (Sparsity): 平均スパイク率（エネルギー効率）


import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import logging
from omegaconf import OmegaConf, DictConfig
from typing import Tuple, Optional, cast, Any, Dict, List, Type
from transformers import PreTrainedTokenizerBase

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import TrainingContainer
from snn_research.training.trainers import BreakthroughTrainer
from snn_research.core.snn_core import SNNCore
from snn_research.data.datasets import get_dataset_class, SNNBaseDataset, DataFormat
from snn_research.benchmark.tasks import BenchmarkTask # Type[BenchmarkTask] のためにインポート（ただし未使用）
# --- ▼ 修正 (v5): collate_fn を train.py からインポート ▼ ---
from train import collate_fn as text_collate_fn
# --- ▲ 修正 (v5) ▲ ---


# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

def get_model_time_steps(model: nn.Module) -> int:
    if isinstance(model, SNNCore):
        # SNNCoreラッパーの場合、内部モデルのtime_stepsを参照
        internal_model: nn.Module = model.model
        if hasattr(internal_model, 'time_steps'):
            return cast(int, getattr(internal_model, 'time_steps'))
    elif hasattr(model, 'time_steps'):
        # BaseModelを直接継承している場合
        return cast(int, getattr(model, 'time_steps'))
    
    logger.warning("モデルに 'time_steps' 属性が見つかりません。デフォルト値 16 を使用します。")
    return 16

@torch.no_grad()
def measure_sparsity(
    trainer: BreakthroughTrainer,
    dataloader: DataLoader,
    time_steps: int
) -> Tuple[float, float]:
    """
    データローダーを使用してモデルの平均スパイク率(s)と
    動的推論ステップ(SNN Cutoff)を計測する。
    """
    logger.info("評価データセットを使用して平均スパース性(s)と平均推論ステップを計測中...")
    
    # BreakthroughTrainerの評価メソッドを実行
    # _run_step が 'spike_rate' と 'avg_cutoff_steps' を計算する
    eval_metrics: Dict[str, float] = trainer.evaluate(dataloader, epoch=0)
    
    avg_spike_rate: float = eval_metrics.get('spike_rate', 0.0)
    avg_cutoff_steps: float = eval_metrics.get('avg_cutoff_steps', float(time_steps))
    
    return avg_spike_rate, avg_cutoff_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SNNモデルのスパース性(s)とタイムステップ(T)の診断ツール",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="診断対象のモデルアーキテクチャ設定ファイル。\n例: configs/models/medium.yaml"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/smoke_test_data.jsonl",
        help="評価に使用するデータパス。\n例: data/smoke_test_data.jsonl"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="(オプション) 学習済みモデルの重みファイル (.pth) のパス。"
    )
    args: argparse.Namespace = parser.parse_args()

    # --- 1. DIコンテナと設定のロード ---
    container = TrainingContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml(args.model_config)
    
    # データパスを上書き
    container.config.data.path.from_value(args.data_path)
    
    # --- ▼ 修正 (v4): `train.py` と同様に、Configを明示的にDictConfigとしてロード ▼ ---
    # DIコンテナから dict を取得
    cfg_dict: Dict[str, Any] = container.config()
    # OmegaConfオブジェクトに変換
    cfg: DictConfig = OmegaConf.create(cfg_dict)
    
    # 評価なのでエポック数などは最小に
    # OmegaConfオブジェクトを直接更新
    OmegaConf.update(cfg, "training.epochs", 1, merge=True)
    OmegaConf.update(cfg, "training.batch_size", 4, merge=True)
    # --- ▲ 修正 (v4) ▲ ---
    
    # --- 2. コンポーネントの構築 ---
    device: str = container.device()
    tokenizer: PreTrainedTokenizerBase = container.tokenizer()
    
    # モデルのロード
    model: nn.Module = container.snn_model()
    if args.model_path:
        if Path(args.model_path).exists():
            try:
                checkpoint: Dict[str, Any] = torch.load(args.model_path, map_location=device)
                state_dict: Dict[str, Any] = checkpoint.get('model_state_dict', checkpoint)
                
                # SNNCoreラッパーから内部モデルを取得
                model_to_load: nn.Module = model.model if isinstance(model, SNNCore) else model # type: ignore[attr-defined]
                
                model_to_load.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ 学習済みモデル '{args.model_path}' をロードしました。")
            except Exception as e:
                logger.error(f"❌ モデルのロードに失敗しました: {e}。初期化された重みで続行します。")
        else:
            logger.warning(f"⚠️ モデルファイル '{args.model_path}' が見つかりません。初期化された重みで続行します。")
            
    model.to(device)

    # データローダーの準備
    DatasetClass: Type[SNNBaseDataset] = get_dataset_class(DataFormat(cfg.data.format))
    dataset = DatasetClass(
        file_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=cfg.model.time_steps
    )
    # 簡単な評価データセット（例：最初の10バッチ）
    subset_indices: List[int] = list(range(min(len(dataset), cfg.training.batch_size * 10)))
    if not subset_indices:
        logger.error(f"データセット '{args.data_path}' からデータを読み込めませんでした。")
        return
        
    eval_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    # --- ▼ 修正 (v5): collate_fn_factory を train.py からインポートしたものに修正 ▼ ---
    collate_fn_factory = text_collate_fn
    # --- ▲ 修正 (v5) ▲ ---
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn_factory(tokenizer, is_distillation=False)
    )

    # トレーナー（評価用）の準備
    # ダミーのオプティマイザとスケジューラ
    optimizer: torch.optim.Optimizer = container.optimizer(params=model.parameters())
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = container.scheduler(optimizer=optimizer)
    
    # --- ▼ 修正 (v4): config (DictConfig) をトレーナーに渡す ▼ ---
    trainer: BreakthroughTrainer = container.standard_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=-1,
        enable_visualization=False, # 診断中は可視化不要
    )
    # --- ▲ 修正 (v4) ▲ ---

    # --- 3. 計測の実行 ---
    
    # T (Time-steps) の計測
    time_steps: int = get_model_time_steps(model)
    
    # s (Sparsity) の計測
    avg_sparsity: float
    avg_latency: float
    avg_sparsity, avg_latency = measure_sparsity(trainer, eval_loader, time_steps)
    
    # --- 4. レポートの出力 ---
    logger.info("\n" + "="*30 + " 📊 SNN 効率診断レポート " + "="*30)
    logger.info(f"  モデルコンフィグ: {args.model_config}")
    logger.info(f"  データセット: {args.data_path}")
    logger.info("-" * (64 + 26))
    logger.info(f"  T (最大タイムステップ数): {time_steps}")
    logger.info(f"  s (平均スパース性 / スパイク率): {avg_sparsity:.4f} ({(avg_sparsity*100):.2f} %)")
    logger.info(f"  L (平均推論レイテンシ / Cutoff): {avg_latency:.2f} ステップ")
    logger.info("-" * (64 + 26))
    
    # 戦略的インサイト (doc/SNN開発：SNN5プロジェクト改善のための情報収集_追案.md 1.3 に基づく)
    logger.info("【戦略的インサイト (SNN5改善レポート 1.3に基づく)】")
    is_efficient: bool = True
    if avg_sparsity > 0.07: # 93%スパース性 (7%スパイク率) の閾値
        logger.warning(f"  [!] スパース性 ({avg_sparsity:.2%}) が高いです。SOTAベンチマーク(1.3)では 7% 未満が推奨されます。")
        is_efficient = False
    if time_steps > 16 or avg_latency > 16:
        logger.warning(f"  [!] タイムステップ数 ({time_steps}) またはレイテンシ ({avg_latency:.2f}) が大きいです。SOTAベンチマーク(1.3)では T <= 16 が推奨されます。")
        is_efficient = False
    
    if is_efficient:
        logger.info("  [✅] モデルは高効率（高スパース性・低レイテンシ）の基準を満たしています。")
    else:
        logger.warning("  [⚠️] モデルはエネルギー効率の改善余地があります（高スパイク率または高レイテンシ）。")
        logger.info("      対策案 (doc/ROADMAP.md 4.2-4.4):")
        logger.info("      1. 損失関数にスパース性正則化項を追加する (base_config.yaml の sparsity_reg_weight)。")
        logger.info("      2. 時空間プルーニング (pruning.py) を適用する。")
        logger.info("      3. 膜電位量子化 (quantization.py) を適用する。")

if __name__ == "__main__":
    main()