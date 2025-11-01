# ファイルパス: experiments/run_dummy_training.py
# タイトル: P5-1, P5-2, P5-3 訓練/評価スクリプト (WandB/TensorBoard統合)
# 機能説明: 
#   Project SNN4のロードマップ (Phase 5) に基づく、
#   実装したコンポーネント (P1-P4) を統合し、訓練と評価のループを実行します。
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

# --- PyTorch (P5-1) ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    
    Tensor = torch.Tensor
    
except ImportError:
    print(
        "PyTorch not found. Please install PyTorch: "
        "'pip install torch torchvision'",
        file=sys.stderr
    )
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


# --- P5-2 & P5-3: 実験管理ロガー ---
WANDB_AVAILABLE: bool = False
TENSORBOARD_AVAILABLE: bool = False

# P5-2: WandB (優先度1)
try:
    import wandb  # type: ignore[import-not-found]
    WANDB_AVAILABLE = True
except ImportError:
    pass

# P5-3: TensorBoard (優先度2)
if not WANDB_AVAILABLE:
    try:
        from torch.utils.tensorboard import SummaryWriter # type: ignore[attr-defined]
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        pass


# --- P1-P4 のコンポーネントをインポート ---
try:
    from snn_research.config.learning_config import PredictiveCodingConfig
    from snn_research.core.networks.sequential_snn_network import SequentialSNNNetwork
    from snn_research.core.trainer import AbstractTrainer, LoggerProtocol
    from snn_research.core.layers.lif_layer import LIFLayer
    from snn_research.layers.abstract_layer import AbstractLayer
    
except ImportError as e:
    print(f"Error: Could not import SNN modules. Ensure 'snn_research' is in PYTHONPATH.")
    print(f"Details: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger: logging.Logger = logging.getLogger("P5_Script")

# --- P3-3: ロガー実装 ---
class ConsoleLogger(LoggerProtocol):
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        step_val: int = step if step is not None else 0
        step_str: str = f"[Epoch {step_val}]"
        log_msgs: List[str] = [f"--- {step_str} Metrics ---"]
        for key, value in data.items():
            try:
                log_msgs.append(f"  {key}: {value:.4f}")
            except (TypeError, ValueError):
                log_msgs.append(f"  {key}: {value}")
        log_msgs.append("------------------------------")
        logger.info("\n".join(log_msgs))

class TensorBoardLogger(LoggerProtocol):
    def __init__(self, log_dir: str = "runs/snn4_p5_dummy") -> None:
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
                    main_tag: str = parts[1]
                    sub_tag: str = parts[0]
                    if main_tag not in scalars_data:
                        scalars_data[main_tag] = {}
                    if hasattr(value, 'item'):
                        scalars_data[main_tag][sub_tag] = value.item()
                    else:
                        scalars_data[main_tag][sub_tag] = float(value)
                else:
                    self.writer.add_scalar(key, float(value), step)
            except (ValueError, TypeError):
                logger.warning(f"[TBLogger] Skipping non-numeric metric: {key}")
        for main_tag, tag_scalar_dict in scalars_data.items():
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    def close(self) -> None: self.writer.close()

# P3-1: ダミーのデータセット (時系列)
class DummySpikingDataset(Dataset):
    def __init__(self, num_samples: int, time_steps: int, features: int):
        self.num_samples: int = num_samples
        self.time_steps: int = time_steps
        self.features: int = features
        if torch != Any:
            self.data: Tensor = (torch.rand(
                num_samples, time_steps, features
            ) > 0.8).float()
            self.targets: Tensor = (torch.rand(num_samples) * 2).long()
        else:
            self.data = [] # type: ignore[assignment]
            self.targets = [] # type: ignore[assignment]
    def __len__(self) -> int: return self.num_samples
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx], self.targets[idx]

def main() -> None:
    if torch == Any:
        logger.error("Torch not found. Cannot run dummy training.")
        return

    # --- 1. ハイパーパラメータ ---
    config_dict: Dict[str, Any] = {
        "batch_size": 8, "time_steps": 20, "input_features": 10,
        "lif1_neurons": 32, "lif2_neurons": 2, "epochs": 3,
        "learning_rate": 0.01, "error_weight": 0.5,
        "lif_decay": 0.95, "lif_threshold": 1.0,
    }

    pc_config = PredictiveCodingConfig(
        learning_rate=config_dict["learning_rate"],
        error_weight=config_dict["error_weight"]
    )

    # --- 2. データローダーの準備 ---
    train_dataset: DummySpikingDataset = DummySpikingDataset(
        num_samples=100, 
        time_steps=config_dict["time_steps"], 
        features=config_dict["input_features"]
    )
    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=config_dict["batch_size"], shuffle=True
    )
    eval_dataset: DummySpikingDataset = DummySpikingDataset(
        num_samples=20, 
        time_steps=config_dict["time_steps"], 
        features=config_dict["input_features"]
    )
    eval_loader: DataLoader = DataLoader(
        eval_dataset, batch_size=config_dict["batch_size"], shuffle=False
    )

    # --- 3. モデルの構築 ---
    logger.info("Building model...")
    
    lif1: LIFLayer = LIFLayer(
        input_features=config_dict["input_features"],
        neurons=config_dict["lif1_neurons"],
        learning_config=pc_config, name="lif1",
        decay=config_dict["lif_decay"],
        threshold=config_dict["lif_threshold"],
    )

    lif2: LIFLayer = LIFLayer(
        input_features=config_dict["lif1_neurons"],
        neurons=config_dict["lif2_neurons"],
        learning_config=pc_config, name="lif2",
        decay=config_dict["lif_decay"],
        threshold=config_dict["lif_threshold"],
    )
    
    # 修正 (エラー 37): [Tensor] を削除
    model: SequentialSNNNetwork = SequentialSNNNetwork(
        layers=[lif1, lif2], 
        reset_states_on_forward=False 
    )
    
    # (LIFLayer が nn.Parameter を初期化するため、
    #  スクリプト側での代入は不要になった)
    lif1.build()
    lif2.build()

    # --- 4. トレーナーの準備 ---
    logger_client: LoggerProtocol
    if WANDB_AVAILABLE:
        logger.info("WandB is available. Initializing wandb run...")
        logger_client = wandb.init( # type: ignore[attr-defined]
            project="SNN4_P5_Dummy",
            config=config_dict,
            name="P5-2_run"
        )
    elif TENSORBOARD_AVAILABLE:
        logger.info("WandB not found. Using TensorBoardLogger.")
        logger_client = TensorBoardLogger()
    else:
        logger.info("WandB and TensorBoard not found. Using ConsoleLogger.")
        logger_client = ConsoleLogger()
    
    # 修正 (エラー 38): [Tensor] を削除
    trainer: AbstractTrainer = AbstractTrainer(
        model=model,
        logger_client=logger_client
    )

    # --- 5. 訓練/評価ループの実行 ---
    logger.info(f"Starting training for {config_dict['epochs']} epochs...")
    
    for epoch in range(config_dict["epochs"]):
        model.reset_states()
        train_metrics: Dict[str, float] = trainer.train_epoch(train_loader)
        model.reset_states()
        eval_metrics: Dict[str, float] = trainer.evaluate_epoch(eval_loader)

    logger.info("Training finished.")
    
    if WANDB_AVAILABLE:
        cast(Any, logger_client).finish()
    elif TENSORBOARD_AVAILABLE:
        cast(TensorBoardLogger, logger_client).close()


if __name__ == "__main__":
    main()
