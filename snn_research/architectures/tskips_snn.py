# ファイルパス: snn_research/architectures/tskips_snn.py
# Title: TSKIPS_SNN (Transformer with Spiking Inverse Kernel Prediction)
#
# 機能の説明: Spiking Inverse Kernel Prediction (SKIP) メカニズムを
# 組み込んだTransformerベースのSNNモデル。
#
# 【修正内容 v27: 循環インポート (Circular Import) の修正】
# - health-check 実行時に 'ImportError: cannot import name 'BreakthroughSNN' ...
#   (most likely due to a circular import)' が発生する問題に対処します。
# - (L: 20) 'from ..core.snn_core import BreakthroughSNN' は、
#   snn_core.py (L:28) -> architectures/__init__.py (L:28) -> tskips_snn.py (L:20)
#   という循環参照を引き起こしていました。
# - (L: 22) 'BreakthroughSNN' は 'SNNCore' ではなく、
#   全てのモデルが継承すべき 'BaseModel' の
#   タイポ（タイプミス）または古い名前であると判断しました。
# - (L: 20, 22) 'BreakthroughSNN' を削除し、'from ..core.base import BaseModel' を
#   インポートして継承するように修正しました。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ..core.layers.complex_attention import ComplexSpikeDrivenAttention
from ..core.neurons import get_neuron_by_name
from ..core.layers.predictive_coding import PredictiveCodingLayer
from ..core.attention import SpikeDrivenSelfAttention

# --- ▼▼▼ 【!!! 修正 v27: 循環インポート修正 !!!】 ▼▼▼
# (from ..core.snn_core import BreakthroughSNN を削除)
from ..core.base import BaseModel # BaseModel をインポート

class TSKIPS_SNN(BaseModel): # 'BreakthroughSNN' -> 'BaseModel' に変更
# --- ▲▲▲ 【!!! 修正 v27】 ▲▲▲
    """
    Transformer with Spiking Inverse Kernel Prediction (TSKIPS_SNN)
    
    (中略)
    """
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        time_steps: int = 16,
        neuron_config: Dict[str, Any] = {},
        sdsa_config: Dict[str, Any] = {},
        pc_config: Dict[str, Any] = {},
        vocab_size: int = 10000, # (v15: BaseModel から渡される)
        num_classes: int = 10,  # (v15: SNNCore から渡される)
        **kwargs
    ):
        # (v15: BaseModel の __init__ を呼び出す)
        super(TSKIPS_SNN, self).__init__(vocab_size=vocab_size, **kwargs)
        
        self.d_model = d_model
        self.nhead = nhead
        self.time_steps = time_steps
        self.neuron_config = neuron_config
        self.sdsa_config = sdsa_config
        self.pc_config = pc_config

        # (v15) vocab_size を使用
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # (SpikingTransformerV2 を参考に
        #  ニューロンの初期化を追加)
        neuron_config_embed = neuron_config.copy()
        neuron_config_embed['features'] = d_model
        self.embed_neuron = get_neuron_by_name(
            neuron_config.get('type', 'lif'), 
            neuron_config_embed
        )

        self.encoder_layers = nn.ModuleList([
            TSKIPS_EncoderLayer(
                d_model, nhead, dim_feedforward, dropout, 
                time_steps, neuron_config, sdsa_config, pc_config
            ) for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TSKIPS_DecoderLayer(
                d_model, nhead, dim_feedforward, dropout, 
                time_steps, neuron_config, sdsa_config, pc_config
            ) for _ in range(num_decoder_layers)
        ])

        # (v15) num_classes を使用
        self.head = nn.Linear(d_model, num_classes)
        
        # (v15) 状態管理
        self._is_stateful = False
        self.built = True

    def set_stateful(self, stateful: bool):
        """ (v15) 状態管理モードを設定 """
        self._is_stateful = stateful
        if not stateful:
            self.reset()
            
        # (v15) SpikingTransformerV2 (L:323) に倣い、
        #       ニューロンのリセット/状態設定を伝播
        if hasattr(self.embed_neuron, 'set_stateful'):
            self.embed_neuron.set_stateful(stateful) # type: ignore[attr-defined]
            
        for layer in self.encoder_layers:
            if hasattr(layer, 'set_stateful'):
                layer.set_stateful(stateful)
        for layer in self.decoder_layers:
            if hasattr(layer, 'set_stateful'):
                layer.set_stateful(stateful)

    def reset(self):
        """ (v15) 状態をリセット """
        if hasattr(self.embed_neuron, 'reset'):
            self.embed_neuron.reset() # type: ignore[attr-defined]
            
        for layer in self.encoder_layers:
            if hasattr(layer, 'reset'):
                layer.reset()
        for layer in self.decoder_layers:
            if hasattr(layer, 'reset'):
                layer.reset()

    def forward(self, input_data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        (v15: BaseModel (L:71) に合わせて引数を 'input_data' に変更)
        
        Args:
            input_data (torch.Tensor): (B, SeqLen) - トークンIDのテンソル
        
        Returns:
            torch.Tensor: (B, NumClasses) - ロジット
        """
        B, SeqLen = input_data.shape
        T = self.time_steps

        # (v15) 状態リセット
        if not self._is_stateful:
            self.reset()

        # 1. 埋め込み
        x = self.embedding(input_data) # (B, SeqLen, D)
        
        # 2. 時間軸でリピート
        x = x.unsqueeze(0).repeat(T, 1, 1, 1) # (T, B, SeqLen, D)
        
        # 3. 埋め込みニューロン (Spiking Embedding)
        # (SpikingTransformerV2 (L:189) に倣う)
        spikes = []
        for t in range(T):
            spike_t, _ = self.embed_neuron(x[t]) # type: ignore[attr-defined]
            spikes.append(spike_t)
        x_spikes = torch.stack(spikes, dim=0) # (T, B, SeqLen, D)

        # 4. エンコーダ
        # (v15: SpikingTransformerV2 (L:331) に倣い、
        #      レイヤーが (T, B, N, C) を処理するように変更)
        memory = x_spikes
        for layer in self.encoder_layers:
            memory = layer(memory) # (T, B, SeqLen, D)
            
        # 5. デコーダ (簡略化のため、エンコーダの出力のみ使用)
        # (v15)
        output = memory
        for layer in self.decoder_layers:
            # (注: 本来デコーダは 'tgt' (ターゲット) も受け取るが、
            #  ここでは 'memory' のみを使用する)
            output = layer(output, memory) # (T, B, SeqLen, D)

        # 6. プーリング
        # (v15: SpikingTransformerV2 (L:348) に倣い、
        #      時間軸とシーケンス軸で平均化)
        x_final = torch.mean(output, dim=(0, 2)) # (B, D)
        
        # 7. 分類ヘッド
        logits = self.head(x_final) # (B, NumClasses)

        # (v15: HPO (Turn 5) のために3つの値を返す)
        avg_spikes = torch.mean(output)
        avg_mem = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # (v15: BaseModel はタプルを返せないため、ロジットのみを返すか、
        #  SNNCore (L:210) 側でタプルを処理する必要がある。
        #  ここでは SNNCore (L:221) がタプルを処理すると仮定し、3つ返す)
        return logits, avg_spikes, avg_mem # type: ignore[return-value]


# (以下、TSKIPS_EncoderLayer と TSKIPS_DecoderLayer の定義が続く)
# (これらのレイヤーは SpikeDrivenSelfAttention を使用する)

class TSKIPS_EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, 
                 time_steps, neuron_config, sdsa_config, pc_config):
        super().__init__()
        self.self_attn = SpikeDrivenSelfAttention(
            dim=d_model, num_heads=nhead, time_steps=time_steps, 
            neuron_config=neuron_config, **sdsa_config
        )
        self.pc_layer = PredictiveCodingLayer(d_model, **pc_config)
        
        # (FFN と Norm)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # (FFNニューロン)
        neuron_config_ffn1 = neuron_config.copy()
        neuron_config_ffn1['features'] = dim_feedforward
        self.ffn_neuron1 = get_neuron_by_name(
            neuron_config.get('type', 'lif'), 
            neuron_config_ffn1
        )
        neuron_config_ffn2 = neuron_config.copy()
        neuron_config_ffn2['features'] = d_model
        self.ffn_neuron2 = get_neuron_by_name(
            neuron_config.get('type', 'lif'), 
            neuron_config_ffn2
        )

    def set_stateful(self, stateful: bool):
        # (v15)
        if hasattr(self.self_attn, 'set_stateful'):
            self.self_attn.set_stateful(stateful)
        if hasattr(self.pc_layer, 'set_stateful'):
            self.pc_layer.set_stateful(stateful)
        if hasattr(self.ffn_neuron1, 'set_stateful'):
            self.ffn_neuron1.set_stateful(stateful) # type: ignore[attr-defined]
        if hasattr(self.ffn_neuron2, 'set_stateful'):
            self.ffn_neuron2.set_stateful(stateful) # type: ignore[attr-defined]

    def reset(self):
        # (v15)
        if hasattr(self.self_attn, 'reset'):
            self.self_attn.reset()
        if hasattr(self.pc_layer, 'reset'):
            self.pc_layer.reset()
        if hasattr(self.ffn_neuron1, 'reset'):
            self.ffn_neuron1.reset() # type: ignore[attr-defined]
        if hasattr(self.ffn_neuron2, 'reset'):
            self.ffn_neuron2.reset() # type: ignore[attr-defined]


    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """ (v15) (T, B, N, C) を処理 """
        T, B, N, C = src.shape
        outputs = []
        
        for t in range(T):
            x_t = src[t] # (B, N, C)
            
            # 1. SDSA
            attn_out = self.self_attn(x_t)
            x_t = self.norm1(x_t + self.dropout(attn_out)) # AddNorm
            
            # 2. PC
            pc_out = self.pc_layer(x_t)
            
            # 3. FFN
            ffn_out = self.linear2(self.dropout(self.ffn_neuron1(self.linear1(pc_out))[0])) # type: ignore[attr-defined]
            ffn_out, _ = self.ffn_neuron2(ffn_out) # type: ignore[attr-defined]
            
            x_t = self.norm2(pc_out + self.dropout(ffn_out)) # AddNorm
            
            outputs.append(x_t)
            
        return torch.stack(outputs, dim=0)


class TSKIPS_DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, 
                 time_steps, neuron_config, sdsa_config, pc_config):
        super().__init__()
        self.self_attn = SpikeDrivenSelfAttention(
            dim=d_model, num_heads=nhead, time_steps=time_steps, 
            neuron_config=neuron_config, **sdsa_config
        )
        self.multihead_attn = SpikeDrivenSelfAttention(
            dim=d_model, num_heads=nhead, time_steps=time_steps, 
            neuron_config=neuron_config, **sdsa_config
        )
        self.pc_layer = PredictiveCodingLayer(d_model, **pc_config)
        
        # (FFN と Norm)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # (FFNニューロン)
        neuron_config_ffn1 = neuron_config.copy()
        neuron_config_ffn1['features'] = dim_feedforward
        self.ffn_neuron1 = get_neuron_by_name(
            neuron_config.get('type', 'lif'), 
            neuron_config_ffn1
        )
        neuron_config_ffn2 = neuron_config.copy()
        neuron_config_ffn2['features'] = d_model
        self.ffn_neuron2 = get_neuron_by_name(
            neuron_config.get('type', 'lif'), 
            neuron_config_ffn2
        )

    def set_stateful(self, stateful: bool):
        # (v15)
        if hasattr(self.self_attn, 'set_stateful'):
            self.self_attn.set_stateful(stateful)
        if hasattr(self.multihead_attn, 'set_stateful'):
            self.multihead_attn.set_stateful(stateful)
        if hasattr(self.pc_layer, 'set_stateful'):
            self.pc_layer.set_stateful(stateful)
        if hasattr(self.ffn_neuron1, 'set_stateful'):
            self.ffn_neuron1.set_stateful(stateful) # type: ignore[attr-defined]
        if hasattr(self.ffn_neuron2, 'set_stateful'):
            self.ffn_neuron2.set_stateful(stateful) # type: ignore[attr-defined]

    def reset(self):
        # (v15)
        if hasattr(self.self_attn, 'reset'):
            self.self_attn.reset()
        if hasattr(self.multihead_attn, 'reset'):
            self.multihead_attn.reset()
        if hasattr(self.pc_layer, 'reset'):
            self.pc_layer.reset()
        if hasattr(self.ffn_neuron1, 'reset'):
            self.ffn_neuron1.reset() # type: ignore[attr-defined]
        if hasattr(self.ffn_neuron2, 'reset'):
            self.ffn_neuron2.reset() # type: ignore[attr-defined]

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """ (v15) (T, B, N, C) を処理 """
        T, B, N, C = tgt.shape
        outputs = []

        for t in range(T):
            tgt_t = tgt[t] # (B, N, C)
            mem_t = memory[t] # (B, N, C)
            
            # 1. Self-Attention (Decoder)
            attn_out = self.self_attn(tgt_t)
            tgt_t = self.norm1(tgt_t + self.dropout(attn_out)) # AddNorm
            
            # 2. Multihead-Attention (Encoder-Decoder)
            # (注: SDSA は K, V を取れないため、mem_t を Q, K, V として使用)
            attn_out_multi = self.multihead_attn(mem_t) 
            tgt_t = self.norm2(tgt_t + self.dropout(attn_out_multi)) # AddNorm
            
            # 3. PC
            pc_out = self.pc_layer(tgt_t)
            
            # 4. FFN
            ffn_out = self.linear2(self.dropout(self.ffn_neuron1(self.linear1(pc_out))[0])) # type: ignore[attr-defined]
            ffn_out, _ = self.ffn_neuron2(ffn_out) # type: ignore[attr-defined]
            
            tgt_t = self.norm3(pc_out + self.dropout(ffn_out)) # AddNorm
            
            outputs.append(tgt_t)
            
        return torch.stack(outputs, dim=0)
