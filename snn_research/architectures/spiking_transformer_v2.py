# ファイルパス: snn_research/architectures/spiking_transformer_v2.py
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
#
# 【修正 v_fix_spike_rate_zero】:
# - `run_distill_hpo.py` から渡される `neuron_config` 内の `bias` キーを
#   `bias_init` にマッピングするロジックを追加。
# - 【v_init 修正】: `v_init` (初期膜電位) をニューロンに渡すロジックを追加。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Union, cast 
import math
import logging 

# 必要なコアコンポーネントをインポート
from snn_research.core.base import BaseModel, SNNLayerNorm
# from snn_research.core.neurons.lif_neuron import LIFNeuron # 古いインポート (使用されていない)
from snn_research.core.neurons.adaptive_lif_neuron import AdaptiveLIFNeuron # これが使用されている
from snn_research.core.attention import SpikingSelfAttention, SpikeDrivenSelfAttention

# (Pdb)
# import pdb

# (Pdb)
# from ..core.layers.abstract_snn_layer import LayerOutput
LayerOutput = Dict[str, torch.Tensor]


# ロガーの設定 (v_fix_bias_key_mapping)
logger: logging.Logger = logging.getLogger(__name__)


class SpikingTransformerV2(BaseModel):
    """
    Spike-Driven Self-Attention (SDSA) を組み込んだSpiking Transformerアーキテクチャ。
    ViTアーキテクチャ（パッチ埋め込み）と互換性がある。
    """
    def __init__(
        self,
        d_model: int,
        n_head: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        time_steps: int,
        neuron_config: Dict[str, Any],
        sdsa_config: Dict[str, Any],
        # ViT互換のためのパラメータ
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        # (v_hpo_fix_bias_key_mapping): bias_init を直接受け取るように変更
        bias_init: float = 0.0,
        **kwargs: Any
    ) -> None:
        
        # (v_hpo_fix_bias_key_mapping):
        # HPO (run_distill_hpo.py) から 'bias' キーで渡される場合に対応
        # kwargs から 'bias' を取得し、'bias_init' にマッピング
        hpo_bias = kwargs.get('bias', 0.0)
        
        # (v_fix_spike_rate_zero):
        # HPOから渡される 'neuron_bias' (小文字) にも対応
        if hpo_bias == 0.0:
            hpo_bias = kwargs.get('neuron_bias', 0.0)

        # ログで渡されたバイアスを確認
        if hpo_bias != 0.0:
            logger.info(f"[SpikingTransformerV2] 🧠 Overriding bias_init with HPO value: {hpo_bias}")
            # 'bias_init' を上書き
            bias_init = hpo_bias
        
        # neuron_config に 'bias_init' を設定
        # (v_fix_spike_rate_zero): 既存のキーを上書きしないように修正
        if 'bias_init' not in neuron_config:
            neuron_config['bias_init'] = bias_init
        
        # (v_hpo_fix_bias_key_mapping): 
        # 'NEURON_BIAS' が存在する場合、'bias_init' より優先する
        if 'NEURON_BIAS' in neuron_config:
            neuron_config['bias_init'] = neuron_config['NEURON_BIAS']
        
        # (v_fix_spike_rate_zero):
        # 'bias' が存在する場合、'bias_init' より優先する
        if 'bias' in neuron_config:
            neuron_config['bias_init'] = neuron_config['bias']

        # デバッグログで最終的なバイアスを確認
        logger.info(f"[SpikingTransformerV2] 🧠 Final bias_init for layers: {neuron_config['bias_init']}")
        
        
        super().__init__(time_steps=time_steps, **kwargs)

        self.d_model = d_model
        self.n_head = n_head
        self.time_steps = time_steps

        # --- ViT パッチ埋め込み ---
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * (patch_size ** 2)
        
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置エンベディング
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))
        # -------------------------

        self.layers = nn.ModuleList([
            SDSAEncoderLayer(
                d_model=d_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                time_steps=time_steps,
                neuron_config=neuron_config,
                sdsa_config=sdsa_config,
                name=f"SDSAEncoderLayer_{i}",
                # (v_hpo_fix_bias_key_mapping): 修正済みの bias_init を渡す
                bias_init=neuron_config['bias_init'] 
            ) for i in range(num_layers)
        ])

        self.norm = SNNLayerNorm(d_model, time_steps=time_steps)
        
        # 出力プロジェクション (分類ヘッド)
        self.output_projection = nn.Linear(d_model, num_classes)

        self.built = True


    def forward(
        self,
        x: torch.Tensor,
        input_images: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False
    ) -> Union[torch.Tensor, LayerOutput]:
        
        if not self.built:
            raise RuntimeError(f"Layer {self.name} has not been built.")

        # ViT互換フォワード
        # (B, C, H, W) -> (B, N, D)
        x_patched = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x_patched + self.pos_embed

        outputs_over_time: List[torch.Tensor] = []
        
        # 状態をリセット (推論/学習時に必要)
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

        # 状態をリセット (ステートレスに戻す)
        for layer_module in self.layers:
             layer = cast(SDSAEncoderLayer, layer_module)
             layer.set_stateful(False)

        x_final = self.norm(x_final)

        if output_hidden_states:
             # (Pdb)
             # output = x_final
             output: LayerOutput = {
                'last_hidden_state': x_final,
                'all_hidden_states': torch.stack(outputs_over_time)
             }
        else:
            # 分類タスクの場合 (input_images が None でない)
            if input_images is not None:
                # (B, N, C) -> (B, C) プーリング
                pooled_output = x_final.mean(dim=1) 
                output = self.output_projection(pooled_output) # (B, NumClasses)
            else:
                # Transformerの標準的な出力 (B, N, C) -> (B, N, VocabSize)
                output = self.output_projection(x_final) 

        # (Pdb)
        # return output
        # メトリクスのために辞書形式で返す
        
        # スパイク数を収集
        total_spikes = self.get_total_spikes()
        avg_spike_rate = total_spikes / (self.get_total_neurons() * self.time_steps)

        # (Pdb)
        # スパース性を計算 (オプション)
        # sparsity_loss = self.calculate_sparsity_loss()

        return {
            'output': output, # 'output' キーにテンソルを格納
            'activity': avg_spike_rate, # 'activity' キーにスパイク率を格納
            'total_spikes': total_spikes,
            # 'sparsity_loss': sparsity_loss
        }


class SDSAEncoderLayer(nn.Module):
    """
    Spike-Driven Self-Attention (SDSA) を組み込んだTransformerエンコーダレイヤー。
    """
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
        time_steps: int,
        neuron_config: Dict[str, Any],
        sdsa_config: Dict[str, Any],
        name: str = "SDSAEncoderLayer",
        # (v_hpo_fix_bias_key_mapping): bias_init を直接受け取るように変更
        bias_init: float = 0.0 
    ) -> None:
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.n_head = n_head
        self.time_steps = time_steps
        self._is_stateful = True # デフォルトはステートフル (推論/学習時)

        # --- v_init 修正: ここで v_init を取得 ---
        # ログ (log5.txt) の V_INIT (Forced): 0.499 を反映させる
        v_init = neuron_config.get('v_init', 0.0)
        # ----------------------------------------
        
        # (v_hpo_fix_bias_key_mapping):
        # neuron_config から 'NEURON_BIAS' または 'bias_init' を取得
        # run_distill_hpo.py から 'bias' キーで渡される場合にも対応
        bias = neuron_config.get('NEURON_BIAS', 
               neuron_config.get('bias_init', 
               neuron_config.get('bias', 0.0)))
        
        # (v_fix_spike_rate_zero):
        # HPOから渡される 'neuron_bias' (小文字) にも対応
        if bias == 0.0:
            bias = neuron_config.get('neuron_bias', 0.0)

        # デバッグログで渡されたバイアスとv_initを確認
        if bias != 0.0 or v_init != 0.0:
            logger.info(f"[{self.name}] 🧠 Overriding neuron params: bias_init={bias}, v_init={v_init}")
        
        # v_threshold は spiking_transformer.yaml から正しく渡されている (0.5)
        v_threshold_s = neuron_config.get('v_threshold', 1.0)
        decay_s = neuron_config.get('decay', 0.95)
        
        # (v_hpo_fix_bias_key_mapping): 'bias' を 'bias_init' として渡す
        neuron_params = {
            'threshold': v_threshold_s,
            'decay': decay_s,
            'bias_init': bias, # (v_fix_spike_rate_zero) 修正済みの bias を渡す
            'v_init': v_init,  # --- v_init 修正: パラメータとして渡す ---
            **neuron_config
        }

        self.self_attn = SpikeDrivenSelfAttention(
            d_model, n_head, dropout=dropout, **sdsa_config
        )

        # (v_fix_attribute_error): linear2を先に定義
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = SNNLayerNorm(d_model, time_steps=time_steps)
        self.norm2 = SNNLayerNorm(d_model, time_steps=time_steps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        
        # --- v_init 修正: 3箇所の AdaptiveLIFNeuron に neuron_params (v_init 含む) を渡す ---
        self.neuron = AdaptiveLIFNeuron(
            features=d_model,
            **neuron_params
        )
        self.ffn_neuron1 = AdaptiveLIFNeuron(
            features=dim_feedforward,
            **neuron_params
        )
        self.ffn_neuron2 = AdaptiveLIFNeuron(
            features=d_model,
            **neuron_params
        )
        
        self.built = True

    def set_stateful(self, stateful: bool) -> None:
        """
        ネットワークの状態管理 (ステートフル/ステートレス) を切り替えます。
        """
        self._is_stateful = stateful
        # ニューロンの状態をリセット
        if not stateful:
            self.neuron.reset_state()
            self.ffn_neuron1.reset_state()
            self.ffn_neuron2.reset_state()
        
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
        # (B, N, C) -> (B, N, C)
        # SDSAは内部でニューロン (LIF) を持ち、スパイクを出力する
        x_step, _ = self.self_attn(src) 
        
        # 2. Add & Norm (残差接続 1)
        # x_step はスパイク (0 or 1)、src は前の層のスパイク (または埋め込み)
        src = src + self.dropout1(x_step)
        
        # 3. 発火 (LIF)
        # (v_hpo_fix_residual): ここで非スパイクの残差接続 `src` を
        # スパイクに変換 (または膜電位を更新) する
        # AdaptiveLIFNeuron は (B, N, C) の電流を受け取り、(B, N, C) のスパイクを返す
        src = self.neuron(src) 
        
        # 4. Norm 1
        src = self.norm1(src)

        # 5. Feedforward (FFN)
        # (B, N, C) -> (B, N, C*4)
        x_step = self.linear1(src)
        x_step = self.dropout2(x_step)
        # 6. 発火 (LIF)
        x_step = self.ffn_neuron1(x_step) 

        # (B, N, C*4) -> (B, N, C)
        x_step = self.linear2(x_step)
        x_step = self.dropout3(x_step)
        # 7. 発火 (LIF)
        x_step = self.ffn_neuron2(x_step)

        # 8. Add & Norm (残差接続 2)
        src = src + x_step
        
        # 9. Norm 2
        src = self.norm2(src)

        return src
