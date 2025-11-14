# ファイルパス: snn_research/architectures/spiking_transformer_v2.py
# Title: Spiking Transformer v2 (SDSA統合版)
# Description: Spike-Driven Self-Attention (SDSA) を組み込んだSpiking Transformerアーキテクチャ。
#
# (中略)
#
# 【!!! エラー修正 (log.txt) !!!】
# 1. TypeError: expected Tensor as element 0 in argument 0, but got tuple
#    - (L186, L342, L487, L493, L496) ニューロンが返す (spike, v_mem) タプルから、
#      spike [0] のみを取得するよう修正。
# 2. AttributeError: ... object has no attribute 'reset_state'
#    - (L298, L426-L428) メソッド呼び出しを 'reset_state()' から 'reset()' に修正。
#
# 【!!! エラー修正 (log.txt v2) !!!】
# 1. ValueError: too many values to unpack (expected 2)
#    - (L481) self.self_attn (SpikeDrivenSelfAttention) はタプルではなく
#      単一のテンソルを返すと想定されるため、タプル展開を削除。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Union, cast 
import math
import logging 

# 必要なコアコンポーネントをインポート
from snn_research.core.base import BaseModel, SNNLayerNorm
# from snn_research.core.neurons.lif_neuron import LIFNeuron # 古いインポート (使用されていない)
from snn_research.core.neurons import get_neuron_by_name # ニューロンの動的取得
from snn_research.core.attention import SpikeDrivenSelfAttention # SDSAをインポート

# (logging)
logger = logging.getLogger(__name__)

# === ユーティリティ ===

def _extract_v_init(neuron_config: Dict[str, Any]) -> float:
    """ 'v_init' または 'bias' (デバッグ用) を安全に抽出する """
    # v_init があれば最優先
    v_init = neuron_config.get('v_init')
    if v_init is not None:
        try:
            return float(v_init)
        except (ValueError, TypeError):
            pass
            
    # v_init がない場合、デバッグ用の bias キーを探す
    bias = neuron_config.get('bias')
    if bias is not None:
        try:
            # run_distill_hpo.py のデバッグ設定 (例: 2.0)
            bias_float = float(bias) 
            # v_threshold (例: 0.5) 未満の適切な初期電位に変換
            v_threshold = float(neuron_config.get('v_threshold', 0.5))
            # 閾値の99.9%を初期電位とする
            return v_threshold * 0.999 
        except (ValueError, TypeError):
            pass
            
    # デフォルト
    return 0.0


def _map_bias_to_bias_init(neuron_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    'bias' キー (デバッグ用) を 'bias_init' (ニューロン初期化用) にマッピングする。
    'v_init' も抽出し、neuron_config に含める。
    """
    # config をコピーして変更
    config = neuron_config.copy()
    
    # 1. v_init の抽出 (v_init または bias から)
    v_init_val = _extract_v_init(config)
    config['v_init'] = v_init_val # ニューロンが参照できるように設定
    
    # 2. bias_init のマッピング (bias から)
    if 'bias' in config and 'bias_init' not in config:
        try:
            # run_distill_hpo.py の bias (例: 2.0) を bias_init にマッピング
            config['bias_init'] = float(config['bias'])
            logger.debug(f"Mapped 'bias' ({config['bias']}) to 'bias_init' ({config['bias_init']})")
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not map 'bias' to 'bias_init': {e}")
            
    return config


# === Spiking Transformer v2 (Vision) ===

class SpikingVisionEmbedding(nn.Module):
    """
    パッチ埋め込み + 位置埋め込み
    (LIFニューロンによる時間エンコーディングを含む)
    """
    def __init__(self, 
                 img_size: int = 32, 
                 patch_size: int = 4, 
                 in_channels: int = 3, 
                 d_model: int = 256,
                 time_steps: int = 32,
                 neuron_config: Dict[str, Any] = {}):
        super().__init__()
        
        # (v_fix_type_error) HPOからのfloat入力をintにキャスト
        img_size = int(img_size)
        patch_size = int(patch_size)
        in_channels = int(in_channels)
        d_model = int(d_model)
        time_steps = int(time_steps)
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.d_model = d_model
        self.time_steps = time_steps

        # パッチ数 (例: (32/4) * (32/4) = 8*8 = 64)
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.num_patches = num_patches
        
        # パッチ埋め込み (Conv2d)
        self.patch_embed = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 位置埋め込み (CLSトークンなし、パッチのみ)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

        # (v_fix_spike_rate_zero) bias と v_init をマッピング
        neuron_config_mapped = _map_bias_to_bias_init(neuron_config)

        # (前回の修正) 'features' を追加
        neuron_config_mapped['features'] = d_model

        # 時間エンコーディング用ニューロン
        self.neuron = get_neuron_by_name(
            neuron_config_mapped.get('type', 'lif'), 
            neuron_config_mapped
        )
        
        # 初期化
        self._initialize_weights()

    def _initialize_weights(self):
        # 位置埋め込みの初期化 (Truncated Normal)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        # パッチ埋め込みの初期化 (Xavier)
        if isinstance(self.patch_embed, nn.Conv2d):
            nn.init.xavier_uniform_(self.patch_embed.weight)
            if self.patch_embed.bias is not None:
                nn.init.constant_(self.patch_embed.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns: (T, B, N, C) - 時間軸で展開されたテンソル
        """
        B, C, H, W = x.shape
        T = self.time_steps
        
        # 1. パッチ埋め込み: (B, C, H, W) -> (B, D, H', W')
        x = self.patch_embed(x)
        
        # 2. フラット化: (B, D, H', W') -> (B, D, N)
        #    N = H' * W' = num_patches
        x = x.flatten(2)
        
        # 3. 転置: (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)
        
        # 4. 時間軸でリピート: (B, N, D) -> (T, B, N, D)
        #    (各時間ステップで同じ定数電流が入力される)
        x = x.unsqueeze(0).repeat(T, 1, 1, 1)
        
        # 5. 位置埋め込み加算: (T, B, N, D) + (1, 1, N, D)
        #    (pos_embedを (1, N, D) -> (1, 1, N, D) にリシェイプ)
        pos_embed = self.pos_embed.unsqueeze(0)
        x = x + pos_embed
        
        # 6. 時間エンコーディング (LIFニューロン): (T, B, N, D) -> (T, B, N, D)
        #    (functional.reset_net が外部で呼び出される想定)
        spikes = []
        for t in range(T):
            # --- ▼▼▼ 【!!! エラー修正 (TypeError) !!!】 ▼▼▼
            # ニューロンは (spike, v_mem) のタプルを返すため、スパイク[0]のみを取得
            spike_t, _ = self.neuron(x[t]) # (B, N, D)
            # --- ▲▲▲ 【!!! エラー修正 (TypeError) !!!】 ▲▲▲
            spikes.append(spike_t)
            
        # 7. スタック: List[(B, N, D)] -> (T, B, N, D)
        x_spikes = torch.stack(spikes, dim=0)
        
        return x_spikes


class SpikingTransformerV2(BaseModel):
    """
    Spike-Driven Self-Attention (SDSA) を組み込んだ Spiking Transformer (ViTベース)。
    CIFAR-10 (img_size=32, patch_size=4) を想定。
    """
    def __init__(
        self,
        # ViT パラメータ (CIFAR-10 デフォルト)
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        
        # Transformer パラメータ
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        
        # SNN パラメータ
        time_steps: int = 32,
        neuron_config: Dict[str, Any] = {},
        sdsa_config: Dict[str, Any] = {},
        
        # その他
        dropout: float = 0.1,
        
        **kwargs # (snn_core.py から渡される 'vocab_size' などを吸収)
    ):
        # (前回の修正: super() 呼び出しを引数なしに変更済み)
        super(SpikingTransformerV2, self).__init__()
        
        # (v_fix_type_error) HPOからのfloat入力をintにキャスト
        img_size = int(img_size)
        patch_size = int(patch_size)
        in_channels = int(in_channels)
        num_classes = int(num_classes)
        d_model = int(d_model)
        nhead = int(nhead)
        num_encoder_layers = int(num_encoder_layers)
        dim_feedforward = int(dim_feedforward)
        time_steps = int(time_steps)

        self.d_model = d_model
        self.nhead = nhead
        self.time_steps = time_steps
        
        # (v_fix_spike_rate_zero) bias と v_init をマッピング
        neuron_config_mapped = _map_bias_to_bias_init(neuron_config)

        # 1. 埋め込み層
        self.embedding = SpikingVisionEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
            time_steps=time_steps,
            neuron_config=neuron_config_mapped
        )
        
        self.num_patches = self.embedding.num_patches

        # 2. Transformer エンコーダ
        self.layers = nn.ModuleList([
            SDSAEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, # EncoderLayer自体のDropout
                self_attn_dropout=dropout, # SDSAに渡すDropout
                activation_dropout=dropout, # FFNに渡すDropout
                neuron_config=neuron_config_mapped,
                sdsa_config=sdsa_config
            ) for _ in range(num_encoder_layers)
        ])
        
        # (前回の修正) プーリング用ニューロンにも 'features' (d_model) が必要
        neuron_config_mapped_pool = neuron_config_mapped.copy()
        neuron_config_mapped_pool['features'] = d_model

        # 3. プーリング用ニューロン (Mean-Spike-Pooling)
        self.pool_neuron = get_neuron_by_name(
            neuron_config_mapped_pool.get('type', 'lif'), 
            neuron_config_mapped_pool
        )
        
        # 4. 分類ヘッド (Linear)
        self.head = nn.Linear(d_model, num_classes)
        
        # 5. 状態管理
        self._is_stateful = False
        self.built = True # (v_fix_type_error のため早期に True に設定)
        
    def set_stateful(self, stateful: bool):
        self._is_stateful = stateful
        # (ニューロンの状態リセットは forward で T=0 の時に行う)
        
        # 各レイヤーに状態管理モードを伝播
        for layer in self.layers:
            if isinstance(layer, SDSAEncoderLayer):
                layer.set_stateful(stateful)

    def forward(self, input_images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        input_images: (B, C, H, W)
        """
        B = input_images.shape[0]
        
        # 状態リセット
        if not self._is_stateful:
            # --- ▼▼▼ 【!!! エラー修正 (AttributeError) !!!】 ▼▼▼
            self.pool_neuron.reset() # 修正: reset_state -> reset
            # --- ▲▲▲ 【!!! エラー修正 (AttributeError) !!!】 ▲▲▲
            
            # (Embedding と EncoderLayer のニューロンは SpikingVisionEmbedding と
            #  SDSAEncoderLayer.set_stateful でリセットされる)

        # 1. 埋め込み: (B, C, H, W) -> (T, B, N, D)
        # (内部でニューロンがリセットされる)
        x_spikes = self.embedding(input_images)
        
        # 2. エンコーダ: (T, B, N, D) -> (T, B, N, D)
        for layer in self.layers:
            # (SDSAEncoderLayer は T, B, N, D を受け取れないため、時間ステップでループ)
            # (注: 本来のTransformerは (N, B, D) または (B, N, D) を期待するが、
            #  SDSAEncoderLayer が時間軸ループを内包していないため、ここでループする)
            
            outputs_t = []
            for t in range(self.time_steps):
                # (B, N, D)
                out_t = layer(x_spikes[t]) 
                outputs_t.append(out_t)
            
            # (T, B, N, D)
            x_spikes = torch.stack(outputs_t, dim=0)

        # 3. プーリング (Mean-Spike-Pooling over Time)
        # (T, B, N, D) -> (B, N, D) (時間軸で平均化)
        x_pooled = torch.mean(x_spikes, dim=0) 
        
        # 4. プーリング用ニューロン (オプション)
        # (B, N, D) -> (B, N, D)
        
        # --- ▼▼▼ 【!!! エラー修正 (TypeError) !!!】 ▼▼▼
        # プーリング用ニューロンもタプル (spike, v_mem) を返すため、[0] を取得
        x_pooled, _ = self.pool_neuron(x_pooled) 
        # --- ▲▲▲ 【!!! エラー修正 (TypeError) !!!】 ▲▲▲
        
        # 5. プーリング (Mean over Patches)
        # (B, N, D) -> (B, D) (パッチ軸で平均化)
        x_final = torch.mean(x_pooled, dim=1)
        
        # 6. 分類ヘッド
        # (B, D) -> (B, NumClasses)
        output = self.head(x_final)
        
        return output


class SDSAEncoderLayer(nn.Module):
    """
    Spike-Driven Self-Attention (SDSA) を使用する Transformer Encoder レイヤー。
    (注: このレイヤーは時間ステップ T=1 での処理のみを想定)
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 self_attn_dropout: float = 0.0,
                 activation_dropout: float = 0.1,
                 neuron_config: Dict[str, Any] = {},
                 sdsa_config: Dict[str, Any] = {},
                 layer_norm_eps: float = 1e-5,
                 name: str = "SDSAEncoderLayer"):
        super(SDSAEncoderLayer, self).__init__()
        
        # (v_fix_type_error) HPOからのfloat入力をintにキャスト
        d_model = int(d_model)
        nhead = int(nhead)
        dim_feedforward = int(dim_feedforward)
        
        self.name = name
        self.built = False

        # (v_fix_spike_rate_zero) bias と v_init をマッピング
        neuron_config_mapped = _map_bias_to_bias_init(neuron_config)

        # 1. Spike-Driven Self-Attention (SDSA)
        # (SDSA内部のニューロンは d_model を features として必要とする)
        sdsa_neuron_config = neuron_config_mapped.copy()
        sdsa_neuron_config['features'] = d_model
        
        # --- ▼▼▼ 【!!! エラー修正 (v3) !!!】 ▼▼▼
        # 'attention.py' の SpikeDrivenSelfAttention.__init__ の
        # シグネチャ (dim, num_heads, time_steps, neuron_config) に
        # 正確に合わせます。
        # 'dropout' と 'sdsa_config' は SDSA 自体には渡しません。
        self.self_attn = SpikeDrivenSelfAttention(
            dim=d_model,
            num_heads=nhead,
            time_steps=1, # このレイヤーは T=1 で動作
            neuron_config=sdsa_neuron_config
            # 'add_noise_if_silent' と 'noise_prob' は
            # sdsa_config から取得するか、デフォルト値を使います。
            # sdsa_config の内容が不明なため、ここではデフォルトに任せます。
        )
        # --- ▲▲▲ 【!!! エラー修正 (v3) !!!】 ▲▲▲
        
        # 2. Feedforward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 3. Normalization
        # (SNNLayerNorm は時間軸 T を持つことを期待するが、
        #  このレイヤーは T=1 で動作するため、標準の LayerNorm を使用)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # 4. Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout) # FFN の最後

        # 5. SNN ニューロン
        
        # (前回の修正) Add&Norm (残差接続) 後のニューロン
        neuron_config_addnorm1 = neuron_config_mapped.copy()
        neuron_config_addnorm1['features'] = d_model
        self.neuron = get_neuron_by_name(
            neuron_config_addnorm1.get('type', 'lif'), 
            neuron_config_addnorm1
        )
        
        # (前回の修正) FFN (linear1) 後のニューロン
        neuron_config_ffn1 = neuron_config_mapped.copy()
        neuron_config_ffn1['features'] = dim_feedforward
        self.ffn_neuron1 = get_neuron_by_name(
            neuron_config_ffn1.get('type', 'lif'), 
            neuron_config_ffn1
        )
        
        # (前回の修正) FFN (linear2) 後のニューロン
        neuron_config_ffn2 = neuron_config_mapped.copy()
        neuron_config_ffn2['features'] = d_model
        self.ffn_neuron2 = get_neuron_by_name(
            neuron_config_ffn2.get('type', 'lif'), 
            neuron_config_ffn2
        )

        self._is_stateful = False
        self.built = True

    def set_stateful(self, stateful: bool):
        """
        ニューロンの状態保持モード (stateful) を切り替えます。
        """
        self._is_stateful = stateful
        # ニューロンの状態をリセット
        if not stateful:
            # --- ▼▼▼ 【!!! エラー修正 (AttributeError) !!!】 ▼▼▼
            self.neuron.reset() # 修正: reset_state -> reset
            self.ffn_neuron1.reset() # 修正: reset_state -> reset
            self.ffn_neuron2.reset() # 修正: reset_state -> reset
            # --- ▲▲▲ 【!!! エラー修正 (AttributeError) !!!】 ▲▲▲
        
        # SNNLayerNorm の状態も切り替え
        if isinstance(self.norm1, SNNLayerNorm):
            self.norm1.set_stateful(stateful)
        if isinstance(self.norm2, SNNLayerNorm):
            self.norm2.set_stateful(stateful)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        LIFニューロンとSDSAを使用したフォワードパス。
        入力 `src` は (B, N, C) のテンソル。
        """
        if not self.built:
            raise RuntimeError(f"Layer {self.name} has not been built.")

        # 1. SDSA (Spike-Driven Self-Attention)
        # --- ▼▼▼ 【!!! エラー修正 (ValueError) !!!】 ▼▼▼
        # self.self_attn はタプルではなく単一のテンソルを返すため、タプル展開を削除
        x_step = self.self_attn(src) # (B, N, C)
        # --- ▲▲▲ 【!!! エラー修正 (ValueError) !!!】 ▲▲▲
        
        # 2. Add & Norm (残差接続 1)
        src = src + self.dropout1(x_step)
        
        # 3. 発火 (LIF)
        # --- ▼▼▼ 【!!! エラー修正 (TypeError) !!!】 ▼▼▼
        src, _ = self.neuron(src) 
        # --- ▲▲▲ 【!!! エラー修正 (TypeError) !!!】 ▲▲▲
        
        # 4. Norm 1
        src = self.norm1(src)

        # 5. Feedforward (FFN)
        x_step = self.linear1(src)
        # --- ▼▼▼ 【!!! エラー修正 (TypeError) !!!】 ▼▼▼
        x_step, _ = self.ffn_neuron1(x_step) # FFN内ニューロン1
        x_step = self.dropout2(x_step)
        x_step = self.linear2(x_step)
        x_step, _ = self.ffn_neuron2(x_step) # FFN内ニューロン2
        # --- ▲▲▲ 【!!! エラー修正 (TypeError) !!!】 ▲▲▲
        
        # 6. Add & Norm (残差接続 2)
        src = src + self.dropout3(x_step)
        
        # 7. Norm 2
        src = self.norm2(src)
        
        return src
