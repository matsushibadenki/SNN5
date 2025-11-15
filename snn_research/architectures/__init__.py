# ファイルパス: snn_research/architectures/__init__.py
# Title: SNN アーキテクチャ (モデル) レジストリ
#
# 機能の説明: プロジェクト内のSNNモデルアーキテクチャ (SpikingTransformerV2など) を
# インポートし、モデル名 (文字列) とモデルクラスをマッピングする
# 'MODEL_REGISTRY' と 'get_model_by_name' 関数を提供します。
#
# 【修正内容 v25: ImportError (cannot import name 'get_model_by_name') の修正】
# - health-check 実行時に snn_core.py (L: 28) が 'get_model_by_name' を
#   インポートできずに失敗 (ImportError) していました。
# - このファイル (architectures/__init__.py) に、'get_model_by_name' 関数と
#   'MODEL_REGISTRY' を実装しました。
# - 'SpikingTransformerV2' や
#   'SEWResNet' など、
#   既知のモデルをレジストリに登録しました。

import logging
import inspect
from typing import Dict, Any, Type, Optional

# BaseModel はすべてのモデルの基底クラスとして必要
from snn_research.core.base import BaseModel

# --- ▼▼▼ モデルクラスのインポート ▼▼▼ ---
# (注: ファイルが存在しない/インポートエラーが起きる可能性を考慮し、try/except で保護)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .spiking_transformer_v2 import SpikingTransformerV2
except ImportError as e:
    logger.warning(f"Failed to import SpikingTransformerV2: {e}. Using fallback.")
    class SpikingTransformerV2(BaseModel): pass # type: ignore[no-redefine]

try:
    from .sew_resnet import SEWResNet
except ImportError as e:
    logger.warning(f"Failed to import SEWResNet: {e}. Using fallback.")
    class SEWResNet(BaseModel): pass # type: ignore[no-redefine]

try:
    from .spiking_mamba import SpikingMamba
except ImportError as e:
    # (mamba.yaml はあるが .py がない場合)
    logger.warning(f"Failed to import SpikingMamba: {e}. Using fallback.")
    class SpikingMamba(BaseModel): pass # type: ignore[no-redefine]

try:
    from .spiking_ssm import SpikingSSM
except ImportError as e:
    logger.warning(f"Failed to import SpikingSSM: {e}. Using fallback.")
    class SpikingSSM(BaseModel): pass # type: ignore[no-redefine]

try:
    from .tskips_snn import TSKIPS_SNN
except ImportError as e:
    logger.warning(f"Failed to import TSKIPS_SNN: {e}. Using fallback.")
    class TSKIPS_SNN(BaseModel): pass # type: ignore[no-redefine]

# --- ▲▲▲ モデルクラスのインポート ▲▲▲ ---


# === モデルレジストリ ===
# モデル名 (文字列) とクラスをマッピング
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    # (SpikingTransformerV2: HPO (Turn 5) で使用)
    "SpikingTransformerV2": SpikingTransformerV2,
    "spiking_transformer_v2": SpikingTransformerV2, # (エイリアス)

    # (SEWResNet)
    "SEWResNet": SEWResNet,
    "sew_resnet": SEWResNet,
    
    # (SpikingMamba)
    "SpikingMamba": SpikingMamba,
    "spiking_mamba": SpikingMamba,
    
    # (SpikingSSM)
    "SpikingSSM": SpikingSSM,
    "spiking_ssm": SpikingSSM,
    
    # (TSKIPS_SNN)
    "TSKIPS_SNN": TSKIPS_SNN,
    "tskips_snn": TSKIPS_SNN,
    
    # (micro.yaml (Turn 7) で使われるモデル。
    #  micro.yaml 内部の 'name:' が 'SpikingTransformerV2' を
    #  指定していると想定されるが、'micro' という名前で
    #  呼ばれた場合も SpikingTransformerV2 (小規模) にフォールバックする)
    "micro": SpikingTransformerV2,
}


def get_model_by_name(name: str) -> Type[BaseModel]:
    """
    モデル名 (文字列) に基づいて、
    モデルクラス (BaseModel) を返します。
    (snn_core.py L: 28 から呼び出される)
    """
    
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    
    # 大文字/小文字を区別しないフォールバック
    name_lower = name.lower()
    for key, model_class in MODEL_REGISTRY.items():
        if key.lower() == name_lower:
            return model_class
            
    # 見つからない場合はエラー
    raise ValueError(
        f"Unknown model name: '{name}'. "
        f"Available models in MODEL_REGISTRY: {list(MODEL_REGISTRY.keys())}"
    )


# --- __init__.py が公開する名前を定義 ---
__all__ = [
    "get_model_by_name",
    "BaseModel",
    "SpikingTransformerV2",
    "SEWResNet",
    "SpikingMamba",
    "SpikingSSM",
    "TSKIPS_SNN",
    "MODEL_REGISTRY"
]
