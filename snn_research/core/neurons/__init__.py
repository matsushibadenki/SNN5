# ファイルパス: snn_research/core/neurons/__init__.py
# Title: SNNニューロンモデル定義
#
# 【エラー修正】
# 1. ImportError: cannot import name 'AdaptiveLIFNeuron'
#    - snn_core.pyがインポートできるように、必要なクラスを __init__.py に明示的にインポートする。
# 2. TypeError: MemoryModule.__init__() got an unexpected keyword argument 'v_init'
#    - get_neuron_by_name 関数内で、ニューロンクラスが受け入れる引数のみを渡すようフィルタリングを実装 (v12)。

from typing import Optional, Tuple, Any, List, cast, Dict, Type, Union 
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import surrogate, base # type: ignore[import-untyped]
import logging 
import inspect 

# --- ▼▼▼ 【修正 v13: ImportError 解消のため、すべてのニューロンを明示的にインポート】 ▼▼▼ ---
# snn_core.py がインポートするすべてのクラスを、各ニューロンファイルからインポートします。
from .adaptive_lif_neuron import AdaptiveLIFNeuron
from .izhikevich_neuron import IzhikevichNeuron # Assuming file name izhikevich_neuron.py
from .glif_neuron import GLIFNeuron # Assuming file name glif_neuron.py
from .tc_lif import TC_LIF # Assuming file name tc_lif.py
from .dual_threshold_neuron import DualThresholdNeuron # Assuming file name dual_threshold_neuron.py
from .scale_and_fire_neuron import ScaleAndFireNeuron # Assuming file name scale_and_fire_neuron.py
from .probabilistic_lif_neuron import ProbabilisticLIFNeuron 
from .bif_neuron import BistableIFNeuron
# --- ▲▲▲ 【修正 v13】 ▲▲▲ ---

logger = logging.getLogger(__name__)

# --- ▼▼▼ 【修正 v13: ローカルに定義されていたクラス定義を削除 (インポートに置き換え)】 ▼▼▼ ---
# DualThresholdNeuron, ScaleAndFireNeuron などのローカル定義を削除。
# --- ▲▲▲ 【修正 v13】 ▲▲▲ ---

__all__ = [
    "AdaptiveLIFNeuron",
    "IzhikevichNeuron",
    "ProbabilisticLIFNeuron",
    "GLIFNeuron",
    "TC_LIF",
    "DualThresholdNeuron",
    "ScaleAndFireNeuron",
    "BistableIFNeuron"
]

# ニューロンのタイプ名 (文字列) とクラスをマッピング
NEURON_REGISTRY: Dict[str, Type[base.MemoryModule]] = {
    "lif": AdaptiveLIFNeuron,
    "bif": BistableIFNeuron,
    "izhikevich": IzhikevichNeuron,
    "glif": GLIFNeuron,
    "tc_lif": TC_LIF,
    "dual_threshold": DualThresholdNeuron,
    "scale_and_fire": ScaleAndFireNeuron,
    "probabilistic_lif": ProbabilisticLIFNeuron,
}

def get_neuron_by_name(name: str, params: Dict[str, Any]) -> base.MemoryModule:
    """
    ニューロンのタイプ名 (文字列) に基づいて、
    ニューロンクラスのインスタンスを作成して返します。
    (v12: パラメータフィルタリングの強化)
    """
    name_lower = name.lower()
    if name_lower not in NEURON_REGISTRY:
        raise ValueError(
            f"Unknown neuron type: '{name}'. "
            f"Available types: {list(NEURON_REGISTRY.keys())}"
        )
        
    NeuronClass = NEURON_REGISTRY[name_lower]
    
    # 1. 期待される引数を取得
    sig = inspect.signature(NeuronClass.__init__)
    
    # 2. フィルタリングされたパラメータ辞書を準備
    filtered_params = {}
    
    # 3. 'features' が params にない場合の推測ロジックを移動 (フィルタリング前に実行)
    current_params = params.copy()
    if 'features' not in current_params:
        if 'd_model' in current_params:
            current_params['features'] = int(current_params['d_model'])
        elif 'dim_feedforward' in current_params:
            current_params['features'] = int(current_params['dim_feedforward'])

    # 4. 期待される引数 (**kwargs と 'self' を除く) のみを抽出
    accepts_kwargs = False
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD: # **kwargs を受け付けるか
            accepts_kwargs = True
            continue

        # 期待される引数が current_params にあれば追加
        if name in current_params:
            # DualThresholdNeuron の v_init は float にキャストが必要 (互換性維持)
            if name == 'v_init' and isinstance(current_params[name], (int, float, str)):
                try:
                    filtered_params[name] = float(current_params[name])
                    continue
                except (ValueError, TypeError):
                    logger.warning(f"Could not cast '{name}' value to float for {name_lower}.")
                    filtered_params[name] = current_params[name]
            else:
                 filtered_params[name] = current_params[name]
    
    # 5. **kwargs を受け入れる場合、残りのすべての引数を渡す (DualThresholdNeuron など)
    if accepts_kwargs:
         # DualThresholdNeuron は **kwargs を受け付けるため、全ての残りの引数を渡す
         for k, v in current_params.items():
             if k not in filtered_params:
                 filtered_params[k] = v
        
    # 6. AdaptiveLIFNeuron ('lif') の特殊処理:
    #    AdaptiveLIFNeuron (MemoryModule) のコンストラクタは引数を取らないため、
    #    features や v_init などが残っていた場合、ここで強制的に削除する。
    #    (ログが ['self'] のみを期待しているため)
    if name_lower == 'lif':
        # AdaptiveLIFNeuron (MemoryModule) のコンストラクタは引数を取らない
        # ログに登場する残りの引数を確実に削除。
        keys_to_purge = ['v_init', 'features', 'bias_init', 'v_threshold', 'threshold_decay', 'threshold_step', 'bias'] 
        temp_filtered_params = filtered_params.copy()
        for k in keys_to_purge:
             temp_filtered_params.pop(k, None)
        
        # AdaptiveLIFNeuronは引数を取らないため、空の辞書を渡す
        filtered_params = temp_filtered_params 

    try:
        # フィルタリングされたパラメータ辞書を渡してインスタンス化
        # AdaptiveLIFNeuron の場合、{} が渡される
        return NeuronClass(**filtered_params) 
    except TypeError as e:
        logger.error(
            f"Failed to instantiate neuron '{name_lower}' with params: {filtered_params}. "
            f"Error: {e}"
        )
        import inspect
        sig = inspect.signature(NeuronClass.__init__)
        expected_params = list(sig.parameters.keys())
        provided_params = list(filtered_params.keys())
        logger.error(f"'{name_lower}' __init__ expects: {expected_params}")
        logger.error(f"Params provided: {provided_params}")
        
        raise e
