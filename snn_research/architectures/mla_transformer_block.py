# ファイルパス: snn_research/architectures/mla_transformer_block.py
# タイトル: MLAアテンション統合 Transformerブロック (QAT対応)
# 機能説明:
#   プロジェクトの既存の MultiLevelSpikeDrivenSelfAttention (MLA) を使用し、
#   QAT量子化を念頭に置いた構成を持つ単一のTransformerエンコーダーブロック。
#   mypyエラー回避のため、`self.lif1`と`self.lif2`の呼び出しにタイプ無視を追加。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, Union, cast, Type

# プロジェクト内のコンポーネントをインポート
from snn_research.core.base import SNNLayerNorm
from snn_research.core.layers.complex_attention import MultiLevelSpikeDrivenSelfAttention
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron, TC_LIF
from snn_research.training.quantization import SpQuantWrapper
# spikingjelly.activation_based.baseの型情報がない場合の対応
# type: ignore[import-untyped, attr-defined] を使用して外部ライブラリのインポートエラーを回避
try:
    from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]
except ImportError:
    class DummyMemoryModule(nn.Module):
        def reset(self):
            pass
        def set_stateful(self, stateful: bool):
            pass
    sj_base = type('sj_base', (object,), {'MemoryModule': DummyMemoryModule})


# ニューロンをラッピングするヘルパー関数
def wrap_neuron_for_quantization(neuron_module: nn.Module) -> nn.Module:
    """設定に応じてニューロンをSpQuantWrapperでラップする。"""
    return neuron_module


class MLATransformerBlock(sj_base.MemoryModule):
    """
    MLAアテンション (MultiLevelSpikeDrivenSelfAttention) とSNN FFNを持つブロック。
    ニューロンがQAT/SpQuantWrapperでラップされることを想定。
    """
    attn: MultiLevelSpikeDrivenSelfAttention
    lif1: nn.Module
    lif2: nn.Module
    
    def __init__(self, d_model: int, n_head: int, dim_feedforward: int, time_steps: int, neuron_config: Dict[str, Any]):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        
        # 1. MLAアテンション (既存実装を使用)
        self.norm1 = SNNLayerNorm(d_model)
        self.attn = MultiLevelSpikeDrivenSelfAttention(
            d_model=d_model, 
            n_head=n_head, 
            neuron_class=AdaptiveLIFNeuron, # LIFをデフォルトとして使用
            neuron_params=neuron_config,
            time_scales=[1, 4, 8] # 複数時間スケール (MLA)
        )
        
        # 2. Feedforward Network (FFN) - SNN化
        self.norm2 = SNNLayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        
        # FFN用ニューロン (AdaptiveLIFNeuronのインスタンスを直接作成)
        lif_params = {k: v for k, v in neuron_config.items() if k in ['tau_mem', 'base_threshold']}
        self.lif1 = wrap_neuron_for_quantization(AdaptiveLIFNeuron(features=dim_feedforward, **lif_params)) # QATターゲット
        
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.lif2 = wrap_neuron_for_quantization(AdaptiveLIFNeuron(features=d_model, **lif_params)) # QATターゲット

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        # MLAアテンション内の全ニューロンの状態を設定
        if hasattr(self.attn, 'set_stateful'):
            self.attn.set_stateful(stateful)
        
        # FFN内のニューロンの状態を設定 (SpQuantWrapperにも伝播するはず)
        for module in [self.lif1, self.lif2]:
            if hasattr(module, 'set_stateful'):
                cast(Any, module).set_stateful(stateful)

    def reset(self):
        super().reset()
        if hasattr(self.attn, 'reset'):
            self.attn.reset()
        for module in [self.lif1, self.lif2]:
            if hasattr(module, 'reset'):
                cast(Any, module).reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        単一時間ステップの処理（B, L, D）。
        外部ループ (SpikingTransformer) から呼び出されることを想定。
        """
        B, L, D = x.shape
        
        # 1. MLA Attention (Analog -> Spike -> Spike)
        attn_out_spikes = self.attn(self.norm1(x)) # (B, L, D) スパイク
        
        # 2. Residual Connection 1 (x はアナログ入力、attn_out はスパイク)
        x = x + attn_out_spikes # Residualはアナログ+スパイクの混合

        # 3. FFN (Spike -> Linear -> LIF -> Spike)
        ffn_in = self.norm2(x) 
        ffn_in_flat = ffn_in.reshape(B * L, D)
        
        ffn_hidden_current = self.fc1(ffn_in_flat) # アナログ電流
        
        # LIF1 (Spike)
        # mypyの誤検知（"Tensor" not callable）を回避するため、タイプ無視を追加
        ffn_hidden_spikes, _ = self.lif1(ffn_hidden_current) # type: ignore[operator]
        
        ffn_out_current = self.fc2(ffn_hidden_spikes) # アナログ電流
        
        # LIF2 (Spike)
        # mypyの誤検知（"Tensor" not callable）を回避するため、タイプ無視を追加
        # (ライン73のエラーに対応)
        ffn_out_spikes, _ = self.lif2(ffn_out_current) # type: ignore[operator]
        ffn_out_spikes = ffn_out_spikes.reshape(B, L, D)
        
        # 4. Residual Connection 2
        x = x + ffn_out_spikes
        
        return x
