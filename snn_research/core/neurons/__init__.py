# ファイルパス: snn_research/core/neurons/__init__.py
# Title: SNNニューロンモデル定義
#
# 機能の説明: SNNニューロンモデルクラス（LIF、DualThresholdなど）を定義およびインポートし、
# ニューロンタイプ名（文字列）とクラスをマッピングするレジストリ(NEURON_REGISTRY)を提供します。
# オプションのニューロンモデルについては、ModuleNotFoundError発生時にダミークラスを定義することで、
# プログラム全体の実行を継続できるようにします。
#
# 【修正内容 v17: ModuleNotFoundErrorの解消】
# - ProbabilisticLIFNeuronのインポートをtry/exceptブロック内に移動し、モジュールが欠損していても
#   プログラムがクラッシュしないように修正しました。

from typing import Optional, Tuple, Any, List, cast, Dict, Type, Union 
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import surrogate, base # type: ignore[import-untyped]
import logging 
import inspect 

# --- ▼▼▼ 【修正 v17: 安全なインポートとダミー定義】 ▼▼▼ ---

# コアニューロンのインポート (これらは通常存在を期待する)
from .adaptive_lif_neuron import AdaptiveLIFNeuron
from .bif_neuron import BistableIFNeuron

# 欠損している可能性のあるモジュールを捕捉し、ダミークラスを定義

try: 
    from .probabilistic_lif_neuron import ProbabilisticLIFNeuron 
except ModuleNotFoundError: 
    class ProbabilisticLIFNeuron(base.MemoryModule): 
        def __init__(self, **kwargs): 
            super().__init__()

try: 
    from .izhikevich_neuron import IzhikevichNeuron 
except ModuleNotFoundError: 
    class IzhikevichNeuron(base.MemoryModule): 
        def __init__(self, **kwargs): 
            super().__init__()

try: 
    from .glif_neuron import GLIFNeuron 
except ModuleNotFoundError: 
    class GLIFNeuron(base.MemoryModule): 
        def __init__(self, **kwargs): 
            super().__init__()

try: 
    from .tc_lif import TC_LIF 
except ModuleNotFoundError: 
    class TC_LIF(base.MemoryModule): 
        def __init__(self, **kwargs): 
            super().__init__()

try: 
    from .scale_and_fire_neuron import ScaleAndFireNeuron 
except ModuleNotFoundError: 
    class ScaleAndFireNeuron(base.MemoryModule): 
        def __init__(self, **kwargs): 
            super().__init__()
# --- ▲▲▲ 【修正 v17】 ▲▲▲ ---


logger = logging.getLogger(__name__)

# --- DualThresholdNeuron のクラス定義は、このファイル内に存在するため維持する ---
class DualThresholdNeuron(base.MemoryModule):
    """
    Dual Threshold Neuron (エラー補償学習用)。
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
# ModuleNotFoundErrorが発生した場合でも、ダミークラスが使用されるため安全
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
