# ファイルパス: snn_research/core/trainer.py
# タイトル: 抽象学習トレーナー (PyTorch準拠)
# 機能説明: 
#   Project SNN4のロードマップ (Phase 3) に基づき、
#   ネットワークの訓練および評価ループを抽象化・管理するクラス。
#
#   (ダミー実装の解消):
#   - Generic[Tensor] および TypeVar('Tensor') を削除。
#   - PyTorch の 'torch.Tensor' を直接使用します。

import logging
from typing import (
    Dict, Any, Optional, Iterable, List, Tuple, Union,
    Mapping,
    Protocol
)

# --- ダミー実装の解消 (PyTorch のインポート) ---
import torch
from torch import Tensor
import torch.nn as nn

# P2-2 (抽象ネットワーク) をインポート
try:
    from .network import AbstractNetwork
except ImportError:
    # (mypy フォールバック)
    
    # 修正 (エラー 1): ABC をインポート
    from abc import ABC, abstractmethod
    
    class AbstractNetwork(ABC): # type: ignore[no-redef, misc]
        def build_model(self) -> None: pass
        def forward(
            self, i: Tensor, t: Optional[Tensor]
        ) -> Dict[str, Tensor]:
            return {'output': i}
        def update_model(
            self, i: Tensor, t: Optional[Tensor], s: Dict[str, Tensor]
        ) -> Dict[str, Tensor]:
            return {'loss': torch.tensor(0.0)}

# データローダーの型エイリアス (Pytorch/Tensorflow風)
Batch = Tuple[Tensor, Tensor]
DataLoader = Iterable[Batch]

# メトリクスの型エイリアス (P3-2)
MetricValue = Union[Tensor, float, int]
MetricsMap = Mapping[str, MetricValue]
MetricsDict = Dict[str, MetricValue]

# P3-3: ロギングクライアントのインターフェース定義 (Protocol)
class LoggerProtocol(Protocol):
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        ...

# ロガーの設定
logger: logging.Logger = logging.getLogger(__name__)

class AbstractTrainer:
    """
    P3-1, P3-3: ネットワークの訓練・評価ループを管理するトレーナー。
    """

    def __init__(
        self, 
        model: AbstractNetwork,
        logger_client: Optional[LoggerProtocol] = None
    ) -> None:
        self.model: AbstractNetwork = model
        self.logger_client: Optional[LoggerProtocol] = logger_client
        self.current_epoch: int = 0


    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        if logger:
            logger.info(f"Starting training epoch {self.current_epoch}...")
        
        epoch_metrics: List[MetricsMap] = []
        
        for i, batch in enumerate(data_loader):
            inputs: Tensor
            targets: Tensor
            inputs, targets = batch
            
            model_state: Dict[str, Tensor] = self.model.forward(inputs, targets)
            
            batch_metrics: MetricsMap = self.model.update_model(
                inputs, 
                targets, 
                model_state
            )
            
            epoch_metrics.append(batch_metrics)

        aggregated_metrics: Dict[str, float] = self._aggregate_metrics(epoch_metrics)
        if logger:
            logger.info(f"Training epoch finished. Metrics: {aggregated_metrics}")
            
        if self.logger_client:
            log_data: Dict[str, Any] = {
                f"train/{k}": v for k, v in aggregated_metrics.items()
            }
            self.logger_client.log(log_data, step=self.current_epoch)
            
        self.current_epoch += 1
        return aggregated_metrics

    def evaluate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        if logger:
            logger.info(f"Starting evaluation epoch {self.current_epoch}...")
            
        epoch_eval_metrics: List[MetricsMap] = []

        for batch in data_loader:
            inputs: Tensor
            targets: Tensor
            inputs, targets = batch
            
            model_state: Dict[str, Tensor] = self.model.forward(
                inputs, 
                targets=targets
            )
            
            eval_metrics: MetricsMap = self._calculate_eval_metrics(
                model_state, targets
            )
            epoch_eval_metrics.append(eval_metrics)

        aggregated_metrics: Dict[str, float] = self._aggregate_metrics(
            epoch_eval_metrics
        )
        if logger:
            logger.info(f"Evaluation epoch finished. Metrics: {aggregated_metrics}")
            
        if self.logger_client:
            log_data: Dict[str, Any] = {
                f"eval/{k}": v for k, v in aggregated_metrics.items()
            }
            self.logger_client.log(log_data, step=self.current_epoch)
            
        return aggregated_metrics

    def _aggregate_metrics(
        self, 
        metrics_list: List[MetricsMap]
    ) -> Dict[str, float]:
        if not metrics_list:
            return {}

        aggregated: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        
        for batch_metrics in metrics_list:
            for key, value in batch_metrics.items():
                try:
                    float_val: float
                    if isinstance(value, Tensor):
                        float_val = float(value.item())
                    else:
                        float_val = float(value)
                        
                    aggregated[key] = aggregated.get(key, 0.0) + float_val
                    counts[key] = counts.get(key, 0) + 1
                except (ValueError, TypeError):
                    if logger:
                        logger.warning(f"Could not aggregate non-numeric metric '{key}'")

        for key in aggregated:
            if counts[key] > 0:
                aggregated[key] /= counts[key]
        
        return aggregated

    def _calculate_eval_metrics(
        self, 
        model_state: Dict[str, Tensor],
        targets: Tensor
    ) -> Dict[str, float]:
        
        output: Optional[Tensor] = model_state.get('output')
        if output is None:
            return {'accuracy': 0.0}
        
        try:
            predicted: Tensor = output.argmax(dim=1)
            correct: Tensor = (predicted == targets)
            accuracy: float = correct.sum().item() / targets.size(0)
            return {'accuracy': accuracy}
        except Exception as e:
            logger.debug(f"Failed to calculate accuracy: {e}")
            pass
            
        return {'accuracy': 0.0}
