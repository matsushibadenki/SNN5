# ファイルパス: snn_research/architectures/spiking_transformer_v2.py
# Title: Spiking Transformer v2 (SDSA統合版)
# Description: Improvement-Plan.md に基づき、Spike-Driven Self-Attention (SDSA) を
#              組み込んだ新しいSpiking Transformerアーキテクチャのスタブを実装します。
#
# 改善 (v2):
# - SDSAEncoderLayer が spikingjelly.activation_based.base.MemoryModule を継承。
# - set_stateful と reset メソッドを実装し、内部ニューロンの状態管理を
#   学習/推論ループと連携できるように修正。
#
# 修正 (v3):
# - mypy [import-untyped], [name-defined] エラーを修正。
#
# 改善 (v4):
# - doc/ROADMAP.md (セクション5.1, 引用[9]) に基づき、
#   FFNの残差接続を「スパイク+スパイク」に変更し、「非スパイク計算」を排除。
# - doc/SNN開発：SNN5プロジェクト改善のための情報収集.md (セクション5.1, SSSA) に基づき、
#   画像入力 (ViT) のための PatchEmbedding を追加。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Union, cast 
import math
import logging # ロギングを追加

# 必要なコアコンポーネントをインポート
from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.attention import SpikeDrivenSelfAttention # 新しいSDSAモジュール
from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ▼ 改善 (v4): ViT用パッチ埋め込み層 ▼ ---
class PatchEmbedding(nn.Module):
    """ 画像をパッチに分割し、線形射影する (ViTの入力層) """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # (B, C, H, W) -> (B, N_Patches, EmbedDim)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, EmbedDim, H_patch, W_patch)
        x = self.proj(x)
        # (B, EmbedDim, H_patch, W_patch) -> (B, EmbedDim, N_Patches)
        x = x.flatten(2)
        # (B, EmbedDim, N_Patches) -> (B, N_Patches, EmbedDim)
        x = x.transpose(1, 2)
        return x
# --- ▲ 改善 (v4) ▲ ---


class SDSAEncoderLayer(sj_base.MemoryModule):
    """
    SDSAを使用したTransformerエンコーダーレイヤー。
    FFNの残差接続もスパイクベースで行う (ハードウェアフレンドリー版)。
    """
    input_spike_converter: AdaptiveLIFNeuron
    neuron_ff: AdaptiveLIFNeuron
    # --- ▼ 改善 (v4): FFN出力用LIFを追加 ▼ ---
    neuron_ff2: AdaptiveLIFNeuron
    # --- ▲ 改善 (v4) ▲ ---
    sdsa: SpikeDrivenSelfAttention
    linear1: nn.Linear
    linear2: nn.Linear # linear2 をクラス属性として定義

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, time_steps: int, neuron_config: dict):
        super().__init__()
        self.sdsa = SpikeDrivenSelfAttention(d_model, nhead, time_steps, neuron_config)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        
        lif_params = {k: v for k, v in neuron_config.items() if k in ['tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']}
        self.neuron_ff = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=dim_feedforward, **lif_params))

        self.linear2 = nn.Linear(dim_feedforward, d_model) # linear2 を初期化
        # --- ▼ 改善 (v4): FFN出力用LIFを追加 ▼ ---
        # FFNの出力 (linear2 の後) もスパイクさせる
        self.neuron_ff2 = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=d_model, **lif_params))
        # --- ▲ 改善 (v4) ▲ ---

        self.norm1 = SNNLayerNorm(d_model)
        self.norm2 = SNNLayerNorm(d_model)

        lif_input_params = lif_params.copy() 
        lif_input_params['base_threshold'] = lif_input_params.get('base_threshold', 0.5) 
        self.input_spike_converter = cast(AdaptiveLIFNeuron, AdaptiveLIFNeuron(features=d_model, **lif_input_params))

    def set_stateful(self, stateful: bool):
        """
        このレイヤーおよびサブモジュール（SDSA, LIF）のステートフルモードを設定する。
        """
        super().set_stateful(stateful)
        self.sdsa.set_stateful(stateful)
        self.neuron_ff.set_stateful(stateful)
        # --- ▼ 改善 (v4): FFN出力用LIFを追加 ▼ ---
        self.neuron_ff2.set_stateful(stateful)
        # --- ▲ 改善 (v4) ▲ ---
        self.input_spike_converter.set_stateful(stateful)

    def reset(self):
        """
        このレイヤーおよびサブモジュール（SDSA, LIF）の状態をリセットする。
        """
        super().reset()
        self.sdsa.reset()
        self.neuron_ff.reset()
        # --- ▼ 改善 (v4): FFN出力用LIFを追加 ▼ ---
        self.neuron_ff2.reset()
        # --- ▲ 改善 (v4) ▲ ---
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
        # --- ▼ 改善 (v4): FFN残差接続をスパイクベースに変更 ▼ ---
        ff_spikes, _ = self.neuron_ff(self.linear1(x_norm1)) # FFN内部はスパイク
        ff_output_analog = self.linear2(ff_spikes)
        ff_output_spikes, _ = self.neuron_ff2(ff_output_analog) # 出力もスパイク化

        # 4. Residual Connection 2 + Norm 2 (スパイク + スパイク)
        x = x_norm1 + ff_output_spikes # 入力は x ではなく x_norm1 を使用 (Pre-Norm)
        x = torch.clamp(x, 0, 1)
        x = self.norm2(x)
        # --- ▲ 改善 (v4) ▲ ---

        return x

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
                 # --- ▼ 改善 (v4): ViT用パラメータを追加 ▼ ---
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 # --- ▲ 改善 (v4) ▲ ---
                 **kwargs: Any):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps # SDSA層に渡すタイムステップ

        # --- 変更 (v4): 2種類の埋め込み層を定義 ---
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embedding.num_patches
        
        # 位置エンコーディング (テキスト用と画像用)
        self.pos_encoder_text = nn.Parameter(torch.zeros(1, 1024, d_model)) # max_seq_len
        self.pos_encoder_image = nn.Parameter(torch.zeros(1, num_patches, d_model))
        # --- ▲ 変更 (v4) ▲ ---
        
        self.layers = nn.ModuleList([
            SDSAEncoderLayer(d_model, nhead, dim_feedforward, time_steps, neuron_config)
            for _ in range(num_encoder_layers)
        ])
        self.norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size) # 出力はvocab_sizeのまま (画像タスク時は num_classes)

        self._init_weights()
        print(f"✅ SpikingTransformerV2 (SDSA, ViT compatible) initialized.")
        print(f"   - FFN residual connections are spike-based (Hardware-Friendly).")

    # --- ▼ 変更 (v4): 入力キーで分岐する forward ▼ ---
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                input_images: Optional[torch.Tensor] = None,
                return_spikes: bool = False, 
                output_hidden_states: bool = False, 
                **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids (Optional[torch.Tensor]): (B, SeqLen)
            input_images (Optional[torch.Tensor]): (B, C, H, W)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (logits, avg_spikes, mem)
        """
        B: int
        N: int # シーケンス長 (トークン数 または パッチ数)
        x: torch.Tensor
        device: torch.device
        
        SJ_F.reset_net(self) # これで SDSAEncoderLayer の reset が呼ばれる

        if input_ids is not None:
            B, N = input_ids.shape
            device = input_ids.device
            x = self.token_embedding(input_ids) # (B, N, C)
            if N > self.pos_encoder_text.shape[1]:
                 logging.warning(f"Input seq_len ({N}) exceeds max_seq_len ({self.pos_encoder_text.shape[1]})")
                 x = x + self.pos_encoder_text[:, :N, :]
            else:
                 x = x + self.pos_encoder_text[:, :N, :]
        
        elif input_images is not None:
            device = input_images.device
            x = self.patch_embedding(input_images) # (B, N_patches, C)
            B, N, C = x.shape
            x = x + self.pos_encoder_image # (B, N_patches, C)
        
        else:
            raise ValueError("Either input_ids or input_images must be provided.")

        # --- 時間ステップループ (FFN内やSDSAパスのため) ---
        outputs_over_time = []

        # 各レイヤーのLIFニューロン等を Stateful に設定
        for layer_module in self.layers:
             layer = cast(SDSAEncoderLayer, layer_module)
             layer.set_stateful(True)

        current_x = x # ループ内で更新されるテンソル
        for t in range(self.time_steps):
            x_step = current_x # 前のステップの出力を入力とする

            # 各層を適用
            for layer_module in self.layers:
                layer = cast(SDSAEncoderLayer, layer_module)
                x_step = layer(x_step)

            outputs_over_time.append(x_step)
            current_x = x_step # 次のステップの入力のために更新

        # 時間平均を取る
        x_final = torch.stack(outputs_over_time).mean(dim=0)

        # 各レイヤーのLIFニューロン等を Stateless に戻す
        for layer_module in self.layers:
             layer = cast(SDSAEncoderLayer, layer_module)
             layer.set_stateful(False)
        # --- 時間ループ終了 ---

        x_final = self.norm(x_final)

        if output_hidden_states:
             output = x_final
        else:
            # 画像タスクの場合、プーリングが必要 (ViTは通常[CLS]トークンを使う)
            if input_images is not None:
                # [CLS]トークンがない場合、単純に平均プーリング
                pooled_output = x_final.mean(dim=1) # (B, C)
                output = self.output_projection(pooled_output) # (B, VocabSize)
            else:
                # テキストタスク
                output = self.output_projection(x_final) # (B, N, VocabSize)

        total_spikes = self.get_total_spikes()
        avg_spikes_val = total_spikes / (B * N * self.time_steps) if return_spikes and self.time_steps > 0 else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device)

        return output, avg_spikes, mem
    # --- ▲ 変更 (v4) ▲ ---