# ファイルパス: matsushibadenki/snn5/SNN5-3acb4dd4029b197f15649dbb0ab217b995d64666/snn_research/architectures/spiking_transformer_v2.py
# Title: Spiking Transformer v2 (SDSA統合版)
# Description: Spike-Driven Self-Attention (SDSA) を組み込んだSpiking Transformerアーキテクチャ。
#
# 【修正 v_fix_bias_key_mapping】:
# - neuron_config 内のキー 'NEURON_BIAS' を AdaptiveLIFNeuron が期待する 'bias_init' に
#   確実にマッピングするようにロジックを強化。
#
# 【修正 v_fix_attribute_error】:
# - 'linear2' が __init__ で定義されているにも関わらず、
#   forward で 'AttributeError' が発生する問題に対処。
#   (クラス属性としての定義と __init__ での初期化を確実に行う)

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Union, cast 
import math
import logging 

# 必要なコアコンポーネントをインポート
from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.attention import SpikeDrivenSelfAttention 
from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ▼ ViT用パッチ埋め込み層 (変更なし) ▼ ---
class PatchEmbedding(nn.Module):
    """ 画像をパッチに分割し、線形射影する (ViTの入力層) """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
# --- ▲ ViT用パッチ埋め込み層 (変更なし) ▲ ---


class SDSAEncoderLayer(sj_base.MemoryModule):
    """
    SDSAを使用したTransformerエンコーダーレイヤー。
    """
    # --- ▼ 【修正 v_fix_attribute_error】: 属性を明示的に型定義 ▼ ---
    input_spike_converter: AdaptiveLIFNeuron
    neuron_ff: AdaptiveLIFNeuron
    neuron_ff2: AdaptiveLIFNeuron
    sdsa: SpikeDrivenSelfAttention
    linear1: nn.Linear
    linear2: nn.Linear # linear2 をクラス属性として明示的に定義
    norm1: SNNLayerNorm
    norm2: SNNLayerNorm
    # --- ▲ 【修正 v_fix_attribute_error】 ▲ ---

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, time_steps: int, neuron_config: dict):
        super().__init__()
        self.sdsa = SpikeDrivenSelfAttention(d_model, nhead, time_steps, neuron_config)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        
        # --- ▼ 【修正 v_fix_bias_key_mapping】: バイアスキーマッピング ▼ ---
        # 1. LIFパラメータをフィルタリング
        lif_params_filtered = {k: v for k, v in neuron_config.items() if k in [
            'tau_mem', 
            'base_threshold', 
            'adaptation_strength', 
            'target_spike_rate', 
            'noise_intensity', 
            'threshold_decay', 
            'threshold_step',
            'evolutionary_leak',
            'gate_input_features',
            'bias_init',      
            'neuron_bias',    
            'NEURON_BIAS',    
        ]}
        
        # 2. キーマッピングの強化
        if 'NEURON_BIAS' in lif_params_filtered:
            lif_params_filtered['bias_init'] = lif_params_filtered.pop('NEURON_BIAS')
        elif 'neuron_bias' in lif_params_filtered: 
            lif_params_filtered['bias_init'] = lif_params_filtered.pop('neuron_bias')
        
        lif_params = lif_params_filtered 
        # --- ▲ 【修正 v_fix_bias_key_mapping】 ▲ ---

        lif_params['threshold_step'] = 0.0

        self.neuron_ff = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=dim_feedforward, **lif_params))

        # --- ▼ 【修正 v_fix_attribute_error】: self.linear2 の初期化 ▼ ---
        # この行が、実行中のコードで欠落しているか、タイプミスしている可能性が
        # 非常に高いです。ここで明示的に定義します。
        self.linear2 = nn.Linear(dim_feedforward, d_model) 
        # --- ▲ 【修正 v_fix_attribute_error】 ▲ ---
        
        self.neuron_ff2 = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=d_model, **lif_params))

        self.norm1 = SNNLayerNorm(d_model)
        self.norm2 = SNNLayerNorm(d_model)

        lif_input_params = lif_params.copy() 
        lif_input_params['base_threshold'] = lif_input_params.get('base_threshold', 0.5) 
        self.input_spike_converter = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=d_model, **lif_input_params))

    def set_stateful(self, stateful: bool):
        self.stateful = stateful
        self.sdsa.set_stateful(stateful)
        self.neuron_ff.set_stateful(stateful)
        self.neuron_ff2.set_stateful(stateful)
        self.input_spike_converter.set_stateful(stateful)

    def reset(self):
        super().reset()
        self.sdsa.reset()
        self.neuron_ff.reset()
        self.neuron_ff2.reset()
        self.input_spike_converter.reset()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        SDSAエンコーダーレイヤーのフォワードパス（スタブ）。
        """
        # 1. SDSAによる自己注意
        attn_output = self.sdsa(src) # (B, N, C) - SDSAは内部でタイムステップを処理

        # 2. Residual Connection 1 + Norm 1 (スパイク + スパイク)
        src_spiked, _ = self.input_spike_converter(src)
        
        x = src_spiked + attn_output 
        x = torch.clamp(x, 0, 1) # スパイクは0か1
        x_norm1 = self.norm1(x) # x_norm1 を定義

        # 3. Feedforward Network
        ff_spikes, _ = self.neuron_ff(self.linear1(x_norm1)) # FFN内部はスパイク
        
        # --- ▼ エラー発生箇所 ▼ ---
        # self.linear2 が __init__ で正しく定義されていれば、ここは動作します。
        ff_output_analog = self.linear2(ff_spikes)
        # --- ▲ エラー発生箇所 ▲ ---
        
        ff_output_spikes, _ = self.neuron_ff2(ff_output_analog) # 出力もスパイク化

        # 4. Residual Connection 2 + Norm 2 (スパイク + スパイク)
        x = x_norm1 + ff_output_spikes 
        x = torch.clamp(x, 0, 1)
        x = self.norm2(x)

        return x

# --- (SpikingTransformerV2 クラスの定義は変更なし) ---
class SpikingTransformerV2(BaseModel):
    """
    SDSA Encoder Layer を使用した Spiking Transformer。
    ViT（画像）とテキストの両方に対応。
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 nhead: int, 
                 num_encoder_layers: int, 
                 dim_feedforward: int, 
                 time_steps: int, 
                 neuron_config: Dict[str, Any],
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 **kwargs: Any):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps 

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embedding.num_patches
        
        self.pos_encoder_text = nn.Parameter(torch.zeros(1, 1024, d_model)) 
        self.pos_encoder_image = nn.Parameter(torch.zeros(1, num_patches, d_model))
        
        self.layers = nn.ModuleList([
            SDSAEncoderLayer(d_model, nhead, dim_feedforward, time_steps, neuron_config)
            for _ in range(num_encoder_layers)
        ])
        self.norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size) 

        self._init_weights()
        print(f"✅ SpikingTransformerV2 (SDSA, ViT compatible) initialized.")
        print(f"   - FFN residual connections are spike-based (Hardware-Friendly).")

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                input_images: Optional[torch.Tensor] = None,
                return_spikes: bool = False, 
                output_hidden_states: bool = False, 
                **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B: int
        N: int 
        x: torch.Tensor
        device: torch.device
        
        SJ_F.reset_net(self) 

        if input_ids is not None:
            B, N = input_ids.shape
            device = input_ids.device
            x = self.token_embedding(input_ids) 
            if N > self.pos_encoder_text.shape[1]:
                 logging.warning(f"Input seq_len ({N}) exceeds max_seq_len ({self.pos_encoder_text.shape[1]})")
                 x = x + self.pos_encoder_text[:, :N, :]
            else:
                 x = x + self.pos_encoder_text[:, :N, :]
        
        elif input_images is not None:
            device = input_images.device
            x = self.patch_embedding(input_images) 
            B, N, C = x.shape
            x = x + self.pos_encoder_image 
        
        else:
            raise ValueError("Either input_ids or input_images must be provided.")

        outputs_over_time = []

        for layer_module in self.layers:
             layer = cast(SDSAEncoderLayer, layer_module)
             layer.set_stateful(True)

        current_x = x 
        for t in range(self.time_steps):
            x_step = current_x 
            
            # (v_hpo_fix_oom): 元の埋め込み `x` を毎ステップ入力
            if t == 0:
                 x_step = x
            else:
                 x_step = current_x # t>0 はスパイク入力を想定

            for layer_module in self.layers:
                layer = cast(SDSAEncoderLayer, layer_module)
                x_step = layer(x_step) 

            outputs_over_time.append(x_step)
            current_x = x_step # スパイクを次のステップの入力とする

        x_final = torch.stack(outputs_over_time).mean(dim=0)

        for layer_module in self.layers:
             layer = cast(SDSAEncoderLayer, layer_module)
             layer.set_stateful(False)

        x_final = self.norm(x_final)

        if output_hidden_states:
             output = x_final
        else:
            if input_images is not None:
                pooled_output = x_final.mean(dim=1) # (B, C)
                output = self.output_projection(pooled_output) # (B, VocabSize)
            else:
                output = self.output_projection(x_final) # (B, N, VocabSize)

        total_spikes = self.get_total_spikes()
        avg_spikes_val = total_spikes / (B * N * self.time_steps) if return_spikes and self.time_steps > 0 else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return output, avg_spikes, mem
