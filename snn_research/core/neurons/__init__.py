# ファイルパス: snn_research/core/neurons/__init__.py
# (更新)
# Title: SNNニューロンモデル定義
#
# (中略)
#
# --- 修正 (mypy) ---
# 1. [union-attr] (L635) : DualThresholdNeuron.forward にて、
#    init_val (float | Tensor) が float の場合に .expand_as を
#    呼べないエラーを修正。isinstance(..., float) で分岐。

from typing import Optional, Tuple, Any, List, cast, Dict, Type, Union # Union をインポート
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import surrogate, base # type: ignore[import-untyped]
import logging 

from .bif_neuron import BistableIFNeuron

logger = logging.getLogger(__name__)

# (AdaptiveLIFNeuron, IzhikevichNeuron, ProbabilisticLIFNeuron, GLIFNeuron, TC_LIF クラス定義は変更なし)
# ... (変更のないクラス定義は省略) ...

class AdaptiveLIFNeuron(base.MemoryModule):
    # ... (変更なし) ...
    pass

class IzhikevichNeuron(base.MemoryModule):
    # ... (変更なし) ...
    pass

class ProbabilisticLIFNeuron(base.MemoryModule):
    # ... (変更なし) ...
    pass

class GLIFNeuron(base.MemoryModule):
    # ... (変更なし) ...
    pass

class TC_LIF(base.MemoryModule):
    # ... (変更なし) ...
    pass


class DualThresholdNeuron(base.MemoryModule):
    """
    Dual Threshold Neuron (エラー補償学習用)。
    SNN5改善レポート (セクション3.1, 引用[6]) に基づく実装。
    量子化エラーと不均一性エラーを削減する。
    v2: Implements PLIF by making tau_mem a learnable parameter.
    """
    log_tau_mem: nn.Parameter
    threshold_high: nn.Parameter # T_h (学習可能なしきい値)
    threshold_low: nn.Parameter  # T_l (デュアルしきい値)
    
    spikes: Tensor
    
    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        threshold_high_init: float = 1.0, # T_h (クリッピング用)
        threshold_low_init: float = 0.5,  # T_l (量子化エラー削減用)
        v_reset: float = 0.0,
        **kwargs: Any, # (v_init 互換性)
    ):
        super().__init__()
        self.features = features
        self.log_tau_mem = nn.Parameter(torch.full((features,), math.log(max(1.1, tau_mem - 1.1))))
        
        # 引用[6]に基づき、2つのしきい値を学習可能パラメータとする
        self.threshold_high = nn.Parameter(torch.full((features,), threshold_high_init))
        self.threshold_low = nn.Parameter(torch.full((features,), threshold_low_init))
        
        self.v_reset = nn.Parameter(torch.full((features,), v_reset))
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        
        # (v_init 互換性)
        self.v_init = float(kwargs.get('v_init', 0.0))

        self.register_buffer("mem", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        super().reset()
        self.mem = None
        self.spikes.zero_()
        self.total_spikes.zero()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """1タイムステップの処理"""
        if not self.stateful:
            self.mem = None

        if self.mem is None or self.mem.shape != x.shape:
            # (v_init 互換性)
            # 引用[6]のT_l/2の代わりにv_initを使用（v_initが設定されていれば）
            
            # --- ▼ mypy [union-attr] 修正 ▼ ---
            init_val_source: Union[float, Tensor]
            if self.v_init != 0.0:
                init_val_source = self.v_init
            else:
                init_val_source = self.threshold_low.detach() / 2.0

            if isinstance(init_val_source, float):
                self.mem = torch.full_like(x, init_val_source)
            else:
                # Tensor a.k.a (self.threshold_low.detach() / 2.0)
                self.mem = init_val_source.expand_as(x)
            # --- ▲ mypy [union-attr] 修正 ▲ ---


        current_tau_mem = torch.exp(self.log_tau_mem) + 1.1
        mem_decay = torch.exp(-1.0 / current_tau_mem)
        
        # 膜電位の更新
        self.mem = self.mem * mem_decay + x
        
        # スパイク生成 (T_h を使用)
        spike_untyped = self.surrogate_function(self.mem - self.threshold_high)
        spike: Tensor = cast(Tensor, spike_untyped)
        
        current_spikes_detached: Tensor = spike.detach()
        
        if current_spikes_detached.ndim > 1:
            self.spikes = current_spikes_detached.mean(dim=0)
        else:
            self.spikes = current_spikes_detached

        with torch.no_grad():
            self.total_spikes += current_spikes_detached.sum() # type: ignore[has-type]
        
        # リセット (デュアルしきい値を使用)
        # 引用[6]の式(7)に基づくリセット
        # S=1 の場合: V[t+1] = V[t] - T_h
        # S=0 の場合: V[t+1] = V[t]
        # ただし、 V[t+1] < T_l の場合は、V[t+1] = V_reset (または T_l/2) にリセット
        
        reset_mem = self.mem - current_spikes_detached * self.threshold_high
        
        # T_l を下回ったニューロンを検出
        below_low_threshold = reset_mem < self.threshold_low
        
        reset_condition = (current_spikes_detached > 0.5) | below_low_threshold
        
        self.mem = torch.where(
            reset_condition,
            self.v_reset.expand_as(self.mem), # V_reset にリセット
            reset_mem # それ以外は減算後の膜電位を維持
        )
        
        return spike, self.mem

class ScaleAndFireNeuron(base.MemoryModule):
    # ... (変更なし) ...
    pass


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

# --- ▼▼▼ 【!!! エラー修正 (log.txt) !!!】 ▼▼▼
# (log.txt の ImportError を解決するため、ファクトリ関数を追加)

# ニューロンのタイプ名 (文字列) とクラスをマッピング
NEURON_REGISTRY: Dict[str, Type[base.MemoryModule]] = {
    # (注: このファイル内のクラス定義に合わせて 'features' 引数を必須とする)
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
    
    Args:
        name (str): ニューロンのタイプ名 (例: "lif")。
        params (Dict[str, Any]): ニューロンのコンストラクタに渡すパラメータ辞書。
        
    Returns:
        base.MemoryModule: インスタンス化されたニューロン。
        
    Raises:
        ValueError: 指定された 'name' がレジストリにない場合。
    """
    name_lower = name.lower()
    if name_lower not in NEURON_REGISTRY:
        raise ValueError(
            f"Unknown neuron type: '{name}'. "
            f"Available types: {list(NEURON_REGISTRY.keys())}"
        )
        
    NeuronClass = NEURON_REGISTRY[name_lower]
    
    # (v_fix_type_error / v_fix_import_error 互換性)
    # spiking_transformer_v2.py は 'd_model' または 'dim_feedforward' を
    # ニューロンの 'features' 引数として期待している可能性がある。
    # 'features' が params にない場合、d_model や d_ff から推測する。
    if 'features' not in params:
        if 'd_model' in params:
            params['features'] = int(params['d_model'])
        elif 'dim_feedforward' in params:
            params['features'] = int(params['dim_feedforward'])
        else:
            # features が見つからず、NeuronClass が features を必要とするかチェック
            import inspect
            sig = inspect.signature(NeuronClass.__init__)
            if 'features' in sig.parameters:
                 logger.warning(
                     f"Neuron type '{name_lower}' requires 'features', but it was not found in params. "
                     f"This might cause an error. Params provided: {list(params.keys())}"
                 )

    
    try:
        # パラメータ辞書を渡してインスタンス化
        return NeuronClass(**params)
    except TypeError as e:
        logger.error(
            f"Failed to instantiate neuron '{name_lower}' with params: {params}. "
            f"Error: {e}"
        )
        # (デバッグ用) 期待される引数と渡された引数のミスマッチの詳細を出力
        import inspect
        sig = inspect.signature(NeuronClass.__init__)
        expected_params = list(sig.parameters.keys())
        provided_params = list(params.keys())
        logger.error(f"'{name_lower}' __init__ expects: {expected_params}")
        logger.error(f"Params provided: {provided_params}")
        
        # 'features' がないのが原因の場合のエラーメッセージ
        if 'features' in expected_params and 'features' not in provided_params:
            logger.error(
                "Critical Error: 'features' (e.g., d_model) was not passed to the neuron constructor."
            )
        raise e

# --- ▲▲▲ 【!!! エラー修正 (log.txt) !!!】 ▲▲▲
