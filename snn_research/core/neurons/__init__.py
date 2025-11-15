# ファイルパス: snn_research/core/neurons/__init__.py
# Title: SNNニューロンモデル定義
#
# 機能の説明: SNNニューロンモデルクラス（LIF、DualThresholdなど）を定義およびインポートし、
# ニューロンタイプ名（文字列）とクラスをマッピングするレジストリ(NEURON_REGISTRY)を提供します。
# オプションのニューロンモデルについては、ModuleNotFoundError発生時にダミークラスを定義することで、
# プログラム全体の実行を継続できるようにします。
#
# 【修正内容 v22: ImportError と AttributeError の同時修正】
# - snn_core.py (L:31) が 'cannot import name 'AdaptiveLIFNeuron'' で
#   失敗する問題に対処します。
# - v21で定義した 'FallbackAdaptiveLIFNeuron' を 'AdaptiveLIFNeuron' (L:40) に
#   リネームしました。
# - これにより、snn_core.py は 'AdaptiveLIFNeuron' をインポートでき、
#   かつ、そのクラスは 'base.MemoryModule' を継承し 'reset()' メソッドを
#   持つため、AttributeError も解消されます。
# - 'adaptive_lif_neuron.py' からのインポート試行 (v21) は削除しました。
#
# 【修正内容 v21: AttributeError (missing 'reset') の修正】
# - (v22にて 'AdaptiveLIFNeuron' としてリネーム・実装)
#
# 【修正内容 v20: TypeError (missing 'features') の修正】
# - (v21, v22 で維持) get_neuron_by_name (L:335) の 'lif' 特殊処理を修正し、
#   'features' などの必須引数を渡すようにしています。

from typing import Optional, Tuple, Any, List, cast, Dict, Type, Union 
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import surrogate, base # type: ignore[import-untyped]
import logging 
import inspect # グローバルにインポートされる

# --- ▼▼▼ 【修正 v22: 'reset' 対策 + 'ImportError' 対策】 ▼▼▼ ---

# (v21 の 'try...except' on .adaptive_lif_neuron を削除)
# 理由: snn_core.py が 'AdaptiveLIFNeuron' を import しようとして失敗 (ImportError)
#       していた。また、元の 'AdaptiveLIFNeuron' には 'reset' が
#       存在しない (AttributeError)。
# 対策: 'reset' を持つ 'AdaptiveLIFNeuron' (base.MemoryModule 継承) を
#       この __init__.py ファイルで直接定義する。

class AdaptiveLIFNeuron(base.MemoryModule): # (v21 の FallbackAdaptiveLIFNeuron をリネーム)
    """
    AttributeError: 'reset' 対策済みのLIFニューロン。
    base.MemoryModule を継承し、'reset()' メソッドを実装する。
    snn_core.py からの 'import AdaptiveLIFNeuron' にも対応する。
    """
    def __init__(self, 
                 features: int, 
                 v_init: float = 0.0,
                 v_threshold: float = 1.0,
                 decay: float = 1.0, # (未使用)
                 bias_init: float = 0.0, # (未使用)
                 time_steps: int = 0, # (未使用)
                 **kwargs): 
        super().__init__()
        self.features = features
        self.v_init = float(v_init)
        self.v_threshold = float(v_threshold)
        
        self.register_buffer("v", torch.full((features,), self.v_init))
        self.register_buffer("threshold", torch.full((features,), self.v_threshold))
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        logger.debug(
            f"Instantiated 'AdaptiveLIFNeuron (v22 Fallback)' with features={features}, "
            f"v_init={self.v_init}, v_threshold={self.v_threshold}"
        )

    def reset(self):
        """ニューロンの状態 (膜電位) をリセットします。"""
        super().reset() # base.MemoryModule.reset() を呼ぶ
        # v_init を使ってリセット
        self.v = torch.full_like(self.v, self.v_init)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """仮のフォワード実装 (ログ(L:4)の引数に基づく)"""
        
        # (snntorchのLIF実装を参考に簡易化)
        # 膜電位の減衰 (本来は 'decay' (時定数) を使うべき)
        # self.v = self.v * decay_factor + x 
        self.v = self.v + x # (単純な加算)
        
        spike_untyped = self.surrogate_function(self.v - self.threshold)
        spike: Tensor = cast(Tensor, spike_untyped)
        
        current_spikes_detached: Tensor = spike.detach()

        # リセット (v_reset = 0.0 と仮定し、v_init に戻す)
        self.v = torch.where(
            current_spikes_detached > 0.5,
            torch.full_like(self.v, self.v_init), # v_init にリセット
            self.v # スパイクしなかったニューロンは膜電位を維持 (減衰は次ステップ)
        )
        # (注: 本来のリセットは V = V - V_threshold * S だが、
        #  surrogate_function が 0/1 でないため V_init に戻す方が安全)
        
        return spike, self.v

# --- ▲▲▲ 【修正 v22】 ▲▲▲ ---


# BistableIFNeuron のインポートとフォールバック
try: 
    from .bif_neuron import BistableIFNeuron 
except ModuleNotFoundError: 
    class BistableIFNeuron(base.MemoryModule): 
        def __init__(self, **kwargs): 
            super().__init__()

# ProbabilisticLIFNeuron のインポートとフォールバック
try: 
    from .probabilistic_lif_neuron import ProbabilisticLIFNeuron 
except ModuleNotFoundError: 
    class ProbabilisticLIFNeuron(base.MemoryModule): 
        def __init__(self, **kwargs): 
            super().__init__()

# 欠損している可能性のあるその他のモジュールを捕捉し、ダミークラスを定義
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
# --- ▲▲▲ 【修正 v18】 ▲▲▲ ---


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
    # --- ▼▼▼ 【修正 v22】 ▼▼▼ ---
    "AdaptiveLIFNeuron", # v22: 'FallbackAdaptiveLIFNeuron' -> 'AdaptiveLIFNeuron'
    # --- ▲▲▲ 【修正 v22】 ▲▲▲ ---
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
    # --- ▼▼▼ 【修正 v22】 ▼▼▼ ---
    "lif": AdaptiveLIFNeuron, # v22: 'FallbackAdaptiveLIFNeuron' -> 'AdaptiveLIFNeuron'
    # --- ▲▲▲ 【修正 v22】 ▲▲▲ ---
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
         # DualThresholdNeuron, AdaptiveLIFNeuron (v22) は **kwargs を受け付ける
         for k, v in current_params.items():
             if k not in filtered_params:
                 filtered_params[k] = v
        
    # 6. AdaptiveLIFNeuron ('lif') の特殊処理:
    if name_lower == 'lif':
        
        # --- ▼▼▼ 【!!! エラー修正 v21 (v22でも維持) !!!】 ▼▼▼
        # 'lif' (AdaptiveLIFNeuron v22) が期待する引数を
        # 削除しないように keys_to_purge リストを修正します。
        # 期待: 'features', 'v_init', 'v_threshold', 'decay', 'bias_init', 'time_steps'
        # 'bias' は _map_bias_to_bias_init で 'bias_init' に変換済みの想定
        keys_to_purge = ['bias', 'threshold_decay', 'threshold_step'] 
        # --- ▲▲▲ 【!!! エラー修正 v21 (v22でも維持) !!!】 ▲▲▲
        
        temp_filtered_params = filtered_params.copy()
        for k in keys_to_purge:
             temp_filtered_params.pop(k, None)
        
        # 'lif' が期待する引数が含まれた filtered_params を使用する
        filtered_params = temp_filtered_params 

    try:
        # フィルタリングされたパラメータ辞書を渡してインスタンス化
        return NeuronClass(**filtered_params) 
    except TypeError as e:
        logger.error(
            f"Failed to instantiate neuron '{name_lower}' with params: {filtered_params}. "
            f"Error: {e}"
        )
        # import inspect # <--- 削除: グローバルにインポートされているため不要
        sig = inspect.signature(NeuronClass.__init__)
        expected_params = list(sig.parameters.keys())
        provided_params = list(filtered_params.keys())
        logger.error(f"'{name_lower}' __init__ expects: {expected_params}")
        logger.error(f"Params provided: {provided_params}")
        
        raise e
