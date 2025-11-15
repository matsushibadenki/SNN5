# ファイルパス: snn_research/core/snn_core.py
# Title: SNNCore (SNNモデルコア管理)
#
# 機能の説明: SNNモデル(SpikingTransformerV2など)、タスク(CIFAR10Taskなど)、
# データ(cifar10.yamlなど)の構成を管理し、モデルの初期化とフォワードパスを
# 実行する中心的なクラス。
#
# 【修正内容 v26: 循環インポート (Circular Import) の修正】
# - health-check 実行時に 'ImportError: cannot import name 'BreakthroughSNN' ...
#   (most likely due to a circular import)' が発生する問題に対処します。
# - このエラーは、snn_core.py (L:31) が .neurons モジュールをインポートし、
#   同時に .neurons モジュール (またはその子) が snn_core.py から 'BreakthroughSNN' を
#   インポートしようとするために発生していました。
# - snn_core.py (L:31-39) でインポートされていた 'AdaptiveLIFNeuron',
#   'BistableIFNeuron' などの個別のニューロンクラスは、
#   SNNCore クラスの動作に不要（get_neuron_by_name が使用されるため）
#   であるため、該当のインポートブロック (L:31-39) を全て削除しました。
#
# 【修正内容 v24: TypeError (missing 'vocab_size') の修正】
# - (v26でも維持) SNNCore の __init__ (L: 115) において、
#   'vocab_size: int' を 'vocab_size: Optional[int] = None' に変更。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union, cast
import logging
from dependency_injector.wiring import inject, Provide

from .base import BaseModel
from ..architectures import get_model_by_name
from ..benchmark.tasks import get_task_by_name, BaseTask
from ..data.datasets import get_tokenizer_by_name
from ..utils.config_utils import (
    load_config, 
    override_config, 
    get_config_value, 
    merge_configs
)

# --- ▼▼▼ 【!!! 修正 v26: 循環インポートの原因 (L:31-39) を削除】 ▼▼▼
# (このブロックは SNNCore の動作に不要であり、
#  .neurons モジュールとの循環参照を引き起こしていたため削除)
#
# from .neurons import (
#     AdaptiveLIFNeuron,
#     IzhikevichNeuron,
#     ProbabilisticLIFNeuron,
#     GLIFNeuron,
#     TC_LIF,
#     DualThresholdNeuron,
#     ScaleAndFireNeuron,
#     BistableIFNeuron
# )
# --- ▲▲▲ 【!!! 修正 v26】 ▲▲▲


logger = logging.getLogger(__name__)

# === ユーティリティ (v13 で移動) ===

def _parse_device(device_str: Optional[Union[str, int]]) -> torch.device:
    """ デバイス文字列 (例: 'cuda:0', 'cpu', 0) を torch.device にパースする """
    if isinstance(device_str, int):
        return torch.device(f"cuda:{device_str}" if torch.cuda.is_available() else "cpu")
        
    if isinstance(device_str, str):
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0") # デフォルト
        if "cuda:" in device_str and torch.cuda.is_available():
            return torch.device(device_str)
        if device_str == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
            
    # デフォルトは CPU
    return torch.device("cpu")

def _auto_select_device() -> torch.device:
    """ MPS (Apple Silicon GPU), CUDA, CPU の順で最適なデバイスを自動選択する """
    if torch.backends.mps.is_available():
        logger.debug("MPS (Apple Silicon GPU) が利用可能です。")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.debug("CUDA が利用可能です。")
        return torch.device("cuda")
    logger.debug("CUDA/MPS が利用できないため、CPU を使用します。")
    return torch.device("cpu")


# === SNNCore ===

class SNNCore(BaseModel):
    """
    SNNモデルの初期化、ビルド、フォワードパスを管理するコアクラス。
    (v13: BaseModel を継承)
    
    BaseModel (base.py) を継承し、以下の機能を提供する:
    1.  モデル (SpikingTransformerV2 など) の動的初期化
    2.  タスク (CIFAR10Task など) の動的初期化
    3.  デバイス (CUDA/MPS/CPU) の自動選択
    4.  モデルのフォワードパスの実行
    5.  状態管理 (set_stateful) とリセット (reset)
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        task_config: Dict[str, Any],
        data_config: Dict[str, Any],
        # --- ▼▼▼ 【!!! 修正 v24 (v26でも維持) !!!】 ▼▼▼
        # (TypeError: missing 'vocab_size' 修正)
        vocab_size: Optional[int] = None,
        # --- ▲▲▲ 【!!! 修正 v24 (v26でも維持) !!!】 ▲▲▲
        paradigm: str = "gradient_based",
        **kwargs
    ):
        """
        SNNCore の初期化

        Args:
            model_config (Dict[str, Any]): モデル固有の構成 (例: d_model, nhead)
            task_config (Dict[str, Any]): タスク固有の構成 (例: loss_weights)
            data_config (Dict[str, Any]): データ固有の構成 (例: num_classes, img_size)
            vocab_size (Optional[int]): V24でOptionalに変更)
            paradigm (str): 学習パラダイム (例: 'gradient_based')
        """
        # BaseModel の __init__ を呼び出す (v13)
        # (v24: vocab_size=vocab_size を渡す)
        super().__init__(vocab_size=vocab_size, **kwargs)

        self.model_config = model_config
        self.task_config = task_config
        self.data_config = data_config
        self.paradigm = paradigm

        # 1. デバイスの決定
        self.device = _auto_select_device()
        logger.info(f"SNNCore: 実行デバイス: {self.device}")

        # 2. タスクの初期化 (v13: SNNCore に移動)
        task_name = self.task_config.get("name", "classification")
        self.task: BaseTask = get_task_by_name(task_name)(
            task_config=self.task_config,
            data_config=self.data_config,
            device=self.device
        )

        # 3. モデルの初期化 (v13: 必要な引数を集約)
        model_name = self.model_config.get("name", "SpikingTransformerV2")
        
        # モデルが必要とする可能性のある全ての引数を集約
        model_init_kwargs = {
            **self.model_config,
            "num_classes": self.task.num_classes,    # (タスク/データから)
            "vocab_size": vocab_size,                # (v24: Optional)
            "img_size": self.data_config.get("img_size"),
            "in_channels": self.data_config.get("in_channels"),
            "paradigm": self.paradigm,
        }

        # self.model (BaseModel) を初期化
        self.model = get_model_by_name(model_name)(**model_init_kwargs)
        
        self.model.to(self.device)
        self.built = True
        self._is_stateful = False
        
        logger.info(f"SNNCore: モデル '{model_name}' (タスク: '{task_name}') を初期化しました。")

    def set_stateful(self, stateful: bool):
        """
        (v13) モデルの状態管理モード (Stateful) を設定する
        """
        self._is_stateful = stateful
        if hasattr(self.model, "set_stateful"):
            self.model.set_stateful(stateful)
        else:
            logger.warning(
                f"モデル '{self.model_config.get('name')}' に "
                f"'set_stateful' メソッドが実装されていません。"
            )

    def reset(self):
        """
        (v13) モデルの状態をリセットする (set_stateful(False) と同義)
        """
        self.set_stateful(False)

    def forward(
        self, 
        input_data: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        **forward_kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        (v13) モデルのフォワードパスを実行し、タスク固有の出力を返す
        
        Args:
            input_data (torch.Tensor): 入力データ (例: 画像バッチ)
            targets (Optional[torch.Tensor]): 正解ラベル (損失計算に必要)
            **forward_kwargs (Any): 'return_spikes' など、モデルの forward に
                                    渡す追加のキーワード引数

        Returns:
            タスクの 'process_output' が返す値。
            (例: 損失, ロジット, スパイク)
        """
        if not self.built:
            raise RuntimeError("SNNCore: モデルがビルドされていません。")
            
        # (v13) 状態リセットは SNNCore (または Trainer) が制御する
        if not self._is_stateful:
            self.reset()
            
        # 1. データを適切なデバイスに移動
        input_data = input_data.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        # 2. モデルのフォワードパスを実行
        # (v13) BaseModel を継承したため、'input_images' などの
        #       キーワード引数指定が必須 (spiking_transformer_v2.py L:298)
        # (v15) 'input_images' から 'input_data' に変更 (汎用性のため)
        # (v16) 'input_images' に戻す (SpikingTransformerV2 が 'input_images' を期待するため)
        # (v17) 'input_images' を使用 (SpikingTransformerV2 (L:298))
        # (v18) 'input_data' を使用 (snn_core.py (L:115) BaseModel が 'input_data' を期待)
        
        # --- v19: SpikingTransformerV2 (L:298) が 'input_images' を期待し、
        #          BaseModel (L:71) が 'input_data' を期待する競合の修正
        #          SpikingTransformerV2 側 (L:298) の引数を 'input_images' -> 'input_data'
        #          に変更したと仮定し、ここでは 'input_data' を渡す。
        # --- v20: v19 の仮定は誤り。SpikingTransformerV2 (L:298) は 'input_images' を
        #          期待している。BaseModel (L:71) の 'input_data' は
        #          *args, **kwargs で吸収されるべきだった。
        #          SNNCore は SpikingTransformerV2 (L:298) に合わせる。
        
        # --- v21: SpikingTransformerV2 (L:298) は 'input_images' を期待
        #          (v13 のコメントが正しかった)
        outputs = self.model(input_images=input_data, **forward_kwargs) # type: ignore[operator]

        # 3. タスク固有の出力処理
        #    (例: 損失計算、ロジット抽出)
        #    (v13: 'outputs' がタプル (logits, spikes, mem) の可能性があるため、
        #     *outputs でアンパックして渡す)
        
        processed_output = self.task.process_output(outputs, targets)
        
        return processed_output
