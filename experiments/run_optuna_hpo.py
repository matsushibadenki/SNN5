# ファイルパス: experiments/run_optuna_hpo.py
# タイトル: P5-4 ハイパーパラメータ最適化 (Optuna) スクリプト (PyTorch準拠)
# 機能説明: 
#   Project SNN4のロードマップ (Phase 5, P5-4) に基づく、
#   'optuna' を使用した SNN モデルのハイパーパラメータ最適化。
#
#   (ダミー実装の解消):
#   - 具象クラス (SequentialSNNNetwork, AbstractTrainer) の
#     型ヒントから [Tensor] を削除。

import sys
import os
import logging
from typing import List, Tuple, Iterable, Dict, Any, Optional, cast

# (mypy) 'snn_research' パッケージをパスに追加
project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- P5-1: PyTorch ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    
    Tensor = torch.Tensor
    TORCH_AVAILABLE = True
    
except ImportError:
    TORCH_AVAILABLE = False
    # (mypy フォールバック)
    Tensor = Any # type: ignore[misc, assignment]
    Dataset = Any # type: ignore[misc, assignment]
    DataLoader = Any # type: ignore[misc, assignment]
    class DummyTensor:
        def shape(self) -> List[int]: return [0]
        def size(self, dim: int) -> int: return 0
        def new_zeros(self, shape: Tuple[int, ...]) -> Any: return self
        def matmul(self, other: Any) -> Any: return self
        def __add__(self, other: Any) -> Any: return self
        def __sub__(self, other: Any) -> Any: return self
        def __mul__(self, other: Any) -> Any: return self
        def __gt__(self, other: Any) -> Any: return self
        def item(self) -> float: return 0.0
        def float(self) -> Any: return self # type: ignore[valid-type]
    torch: Any = Any # type: ignore[misc, assignment, no-redef]
    torch.Tensor = DummyTensor # type: ignore[misc, assignment]


# --- P5-2, P5-3, P5-4: 実験管理・最適化ライブラリ ---
WANDB_AVAILABLE: bool = False
TENSORBOARD_AVAILABLE: bool = False
OPTUNA_AVAILABLE: bool = False

# P5-4: Optuna
try:
    import optuna # type: ignore[import-not-found]
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    Trial = Any # type: ignore[misc, assignment]

# P5-2: WandB
try:
    import wandb  # type: ignore[import-not-found]
    WANDB_AVAILABLE = True
except ImportError:
    pass

# P5-3: TensorBoard
if not WANDB_AVAILABLE and TORCH_AVAILABLE:
    try:
        from torch.utils.tensorboard import SummaryWriter # type: ignore[attr-defined]
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        pass


# --- P1-P4 のコンポーント ---
try:
    from snn_research.config.learning_config import PredictiveCodingConfig
    from snn_research.core.networks.sequential_snn_network import SequentialSNNNetwork
    from snn_research.core.trainer import AbstractTrainer, LoggerProtocol
    from snn_research.core.layers.lif_layer import LIFLayer
    from snn_research.layers.abstract_layer import AbstractLayer
except ImportError as e:
    print(f"Error: Could not import SNN modules. Ensure 'snn_research' is in PYTHONPATH.")
    sys.exit(1)

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger("P5_HPO_Script")


# --- P3-3: ロガー実装 (P5-1 からコピー) ---
class ConsoleLogger(LoggerProtocol):
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        step_val: int = step if step is not None else 0
        if step_val % 5 == 0: 
             logger.info(f"[Epoch {step_val}] Metrics: {data}")

class TensorBoardLogger(LoggerProtocol):
    def __init__(self, log_dir: str) -> None:
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available.")
        self.writer: SummaryWriter = SummaryWriter(log_dir=log_dir) # type: ignore[name-defined]
        logger.info(f"TensorBoardLogger initialized. Logging to: {log_dir}")
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        step = step if step is not None else 0
        scalars_data: Dict[str, Dict[str, float]] = {}
        for key, value in data.items():
            try:
                parts: List[str] = key.split('/', 1)
                if len(parts) == 2:
                    main_tag, sub_tag = parts[1], parts[0]
                    if main_tag not in scalars_data:
                        scalars_data[main_tag] = {}
                    if hasattr(value, 'item'):
                        scalars_data[main_tag][sub_tag] = value.item()
                    else:
                        scalars_data[main_tag][sub_tag] = float(value)
                else:
                    self.writer.add_scalar(key, float(value), step)
            except (ValueError, TypeError): pass
        for main_tag, tag_scalar_dict in scalars_data.items():
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    def close(self) -> None: self.writer.close()

# --- P3-1: ダミーデータセット (P5-1 からコピー) ---
class DummySpikingDataset(Dataset):
    def __init__(self, num_samples: int, time_steps: int, features: int):
        self.num_samples, self.time_steps, self.features = num_samples, time_steps, features
        if TORCH_AVAILABLE:
            self.data: Tensor = (torch.rand(num_samples, time_steps, features) > 0.8).float()
            self.targets: Tensor = (torch.rand(num_samples) * 2).long()
        else:
            self.data, self.targets = [], [] # type: ignore[assignment]
    def __len__(self) -> int: return self.num_samples
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx], self.targets[idx]


# --- P5-4: Optuna Objective 関数 ---

def objective(trial: Trial) -> float:
    """
    P5-4: Optuna の単一トライアルを実行する 'objective' 関数。
    """
    
    global train_loader, eval_loader 
    
    if train_loader is None or eval_loader is None:
        logger.error("DataLoaders are not initialized. Run main_hpo() first.")
        return -1.0 

    # --- 1. ハイパーパラメータのサンプリング (P5-4) ---
    config_dict: Dict[str, Any] = {
        "batch_size": 8, "time_steps": 20, "input_features": 10,
        "lif1_neurons": 32, "lif2_neurons": 2, "epochs": 10,
        
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        ),
        "error_weight": trial.suggest_float(
            "error_weight", 0.1, 1.0
        ),
        "lif_decay": trial.suggest_float(
            "lif_decay", 0.8, 0.99
        ),
        "lif_threshold": trial.suggest_float(
            "lif_threshold", 0.8, 1.5
        ),
    }

    # P1-3: 学習規則の設定
    pc_config = PredictiveCodingConfig(
        learning_rate=config_dict["learning_rate"],
        error_weight=config_dict["error_weight"]
    )

    # --- 3. モデルの構築 (P4, P2) ---
    
    # 修正: 'input_shape'/'output_shape' -> 'input_features'/'neurons'
    # 修正: [torch.Tensor] 型引数を削除
    lif1: LIFLayer = LIFLayer(
        input_features=config_dict["input_features"],
        neurons=config_dict["lif1_neurons"],
        learning_config=pc_config, name="lif1",
        decay=config_dict["lif_decay"], threshold=config_dict["lif_threshold"],
    )

    lif2: LIFLayer = LIFLayer(
        input_features=config_dict["lif1_neurons"],
        neurons=config_dict["lif2_neurons"],
        learning_config=pc_config, name="lif2",
        decay=config_dict["lif_decay"], threshold=config_dict["lif_threshold"],
    )
    
    # 修正 (エラー 39): [Tensor] を削除
    model: SequentialSNNNetwork = SequentialSNNNetwork(
        layers=[lif1, lif2],
        reset_states_on_forward=False 
    )
    
    # 修正: (nn.Parameter の代入は不要。build() が初期化を行う)
    lif1.build()
    lif2.build()

    # --- 4. トレーナーの準備 (P3) ---
    logger_client: LoggerProtocol
    run_name: str = f"trial_{trial.number}"
    
    if WANDB_AVAILABLE:
        logger_client = wandb.init( # type: ignore[attr-defined]
            project="SNN4_P5_HPO",
            config=config_dict, 
            name=run_name,
            reinit=True, 
            group="P5-4_Study"
        )
    elif TENSORBOARD_AVAILABLE:
        logger_client = TensorBoardLogger(log_dir=f"runs/P5-4_Study/{run_name}")
    else:
        logger_client = ConsoleLogger() 
    
    # 修正 (エラー 40): [Tensor] を削除
    trainer: AbstractTrainer = AbstractTrainer(
        model=model, logger_client=logger_client
    )

    # --- 5. 訓練/評価ループの実行 (P5-1) ---
    final_eval_metrics: Dict[str, float] = {}
    
    for epoch in range(config_dict["epochs"]):
        model.reset_states()
        train_metrics: Dict[str, float] = trainer.train_epoch(train_loader)
        
        model.reset_states()
        eval_metrics: Dict[str, float] = trainer.evaluate_epoch(eval_loader)
        
        accuracy: float = eval_metrics.get("accuracy", 0.0)
        trial.report(accuracy, epoch)
        if trial.should_prune():
            if WANDB_AVAILABLE: cast(Any, logger_client).finish()
            elif TENSORBOARD_AVAILABLE: cast(TensorBoardLogger, logger_client).close()
            raise optuna.exceptions.TrialPruned()
            
        final_eval_metrics = eval_metrics

    if WANDB_AVAILABLE: cast(Any, logger_client).finish()
    elif TENSORBOARD_AVAILABLE: cast(TensorBoardLogger, logger_client).close()

    return final_eval_metrics.get("accuracy", 0.0)


# (P5-4) グローバルデータローダー (Optuna デモ用)
train_loader: Optional[DataLoader] = None
eval_loader: Optional[DataLoader] = None

def main_hpo() -> None:
    """
    P5-4: Optuna HPO スタディを実行します。
    """
    global train_loader, eval_loader
    
    if not TORCH_AVAILABLE:
        logger.error("Torch not found. Cannot run HPO.")
        return
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not found. Cannot run HPO. 'pip install optuna'")
        return

    # --- グローバルデータローダーの準備 (P3-1) ---
    train_dataset: DummySpikingDataset = DummySpikingDataset(100, 20, 10)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_dataset: DummySpikingDataset = DummySpikingDataset(20, 20, 10)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

    # --- P5-4: Optuna スタディの実行 ---
    logger.info("Starting Optuna HPO Study (P5-4)...")
    
    study: optuna.Study = optuna.create_study(
        direction="maximize", # 'accuracy' を最大化
        pruner=optuna.pruners.MedianPruner() # 枝刈り
    )
    
    study.optimize(
        objective, 
        n_trials=10 # (ダミー: 10 トライアル実行)
    )

    logger.info("Optuna HPO Study finished.")
    logger.info(f"Best trial (accuracy): {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

if __name__ == "__main__":
    main_hpo()
