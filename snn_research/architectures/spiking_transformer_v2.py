# ファイルパス: snn_research/architectures/spiking_transformer_v2.py
# Title: Spiking Transformer v2 (SDSA統合版)
# Description: Spike-Driven Self-Attention (SDSA) を組み込んだSpiking Transformerアーキテクチャ。
#
# (中略)
#
# 【修正 v_fix_spike_rate_zero】:
# - `run_distill_hpo.py` から渡される `neuron_config` 内の `bias` キーを
#   `bias_init` にマッピングするロジックを追加。
# - 【v_init 修正】: `v_init` (初期膜電位) をニューロンに渡すロジックを追加。
#
# 【修正 v_fix_import_error】:
# - 存在しない 'SpikingSelfAttention' のインポートを削除 (log6.txt)
#
# 【修正 v_fix_type_error (log9.txt)】:
# - HPO (dependency_injector) 経由で int 型引数が float (例: 256.0) として
#   渡されることが原因で TypeError が発生するため、
#   __init__ の冒頭で全ての整数引数を int() で明示的にキャストする。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Union, cast 
import math
import logging 

# 必要なコアコンポーネントをインポート
from snn_research.core.base import BaseModel, SNNLayerNorm
# from snn_research.core.neurons.lif_neuron import LIFNeuron # 古いインポート (使用されていない)
from snn_research.core.neurons.adaptive_lif_neuron import AdaptiveLIFNeuron # これが使用されている

# --- 修正 v_fix_import_error ---
from snn_research.core.attention import SpikeDrivenSelfAttention 
# ---------------------------------

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
        
        # --- 修正 v_fix_type_error (log9.txt) ---
        # HPO (dependency_injector) が float を渡すため、全て int にキャスト
        _d_model = int(d_model)
        _n_head = int(n_head)
        _num_layers = int(num_layers)
        _dim_feedforward = int(dim_feedforward)
        _time_steps = int(time_steps)
        _img_size = int(img_size)
        _patch_size = int(patch_size)
        _in_channels = int(in_channels)
        _num_classes = int(num_classes)
        # ----------------------------------------
        
        # (v_hpo_fix_bias_key_mapping):
        # (中略: bias_init のロジックは変更なし)
        hpo_bias = kwargs.get('bias', 0.0)
        if hpo_bias == 0.0:
            hpo_bias = kwargs.get('neuron_bias', 0.0)
        if hpo_bias != 0.0:
            logger.info(f"[SpikingTransformerV2] 🧠 Overriding bias_init with HPO value: {hpo_bias}")
            bias_init = hpo_bias
        if 'bias_init' not in neuron_config:
            neuron_config['bias_init'] = bias_init
        if 'NEURON_BIAS' in neuron_config:
            neuron_config['bias_init'] = neuron_config['NEURON_BIAS']
        if 'bias' in neuron_config:
            neuron_config['bias_init'] = neuron_config['bias']
        logger.info(f"[SpikingTransformerV2] 🧠 Final bias_init for layers: {neuron_config['bias_init']}")
        
        
        super().__init__(time_steps=_time_steps, **kwargs) # _time_steps を使用

        self.d_model = _d_model # _d_model を使用
        self.n_head = _n_head # _n_head を使用
        self.time_steps = _time_steps # _time_steps を使用

        # --- ViT パッチ埋め込み ---
        self.patch_size = _patch_size # _patch_size を使用

        # --- 修正 v_fix_type_error (log9.txt) ---
        # キャスト済みのローカル変数を使用
        num_patches = (_img_size // _patch_size) ** 2
        patch_dim = _in_channels * (_patch_size ** 2)
        
        self.patch_embed = nn.Conv2d(
            _in_channels, _d_model, 
            kernel_size=_patch_size, stride=_patch_size
        )
        
        # 位置エンベディング (エラー発生箇所)
        # num_patches も int() でキャストして万全を期す
        self.pos_embed = nn.Parameter(torch.randn(1, int(num_patches), _d_model))
        # -------------------------

        self.layers = nn.ModuleList([
            SDSAEncoderLayer(
                d_model=_d_model,
                n_head=_n_head,
                dim_feedforward=_dim_feedforward,
                dropout=dropout,
                time_steps=_time_steps,
                neuron_config=neuron_config,
                sdsa_config=sdsa_config,
                name=f"SDSAEncoderLayer_{i}",
                bias_init=neuron_config['bias_init'] 
            ) for i in range(_num_layers) # _num_layers を使用
        ])

        self.norm = SNNLayerNorm(_d_model, time_steps=_time_steps)
        
        # 出力プロジェクション (分類ヘッド)
        self.output_projection = nn.Linear(_d_model, _num_classes) # _num_classes を使用

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

        # スパイク数を収集
        total_spikes = self.get_total_spikes()
        avg_spike_rate = total_spikes / (self.get_total_neurons() * self.time_steps)

        return {
            'output': output, # 'output' キーにテンソルを格納
            'activity': avg_spike_rate, # 'activity' キーにスパイク率を格納
            'total_spikes': total_spikes,
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
        bias_init: float = 0.0 
    ) -> None:
        super().__init__()
        self.name = name
        
        # --- 修正 v_fix_type_error (log9.txt) ---
        # このレイヤーに渡される引数もキャスト
        _d_model = int(d_model)
        _n_head = int(n_head)
        _dim_feedforward = int(dim_feedforward)
        _time_steps = int(time_steps)
        # ----------------------------------------
        
        self.d_model = _d_model
        self.n_head = _n_head
        self.time_steps = _time_steps
        self._is_stateful = True # デフォルトはステートフル (推論/学習時)

        # --- v_init 修正: (変更なし) ---
        v_init = neuron_config.get('v_init', 0.0)
        
        # (v_hpo_fix_bias_key_mapping): (変更なし)
        bias = neuron_config.get('NEURON_BIAS', 
               neuron_config.get('bias_init', 
               neuron_config.get('bias', 0.0)))
        if bias == 0.0:
            bias = neuron_config.get('neuron_bias', 0.0)
        if bias != 0.0 or v_init != 0.0:
            logger.info(f"[{self.name}] 🧠 Overriding neuron params: bias_init={bias}, v_init={v_init}")
        
        v_threshold_s = neuron_config.get('v_threshold', 1.0)
        decay_s = neuron_config.get('decay', 0.95)
        
        neuron_params = {
            'threshold': v_threshold_s,
            'decay': decay_s,
            'bias_init': bias, 
            'v_init': v_init,  
            **neuron_config
        }

        self.self_attn = SpikeDrivenSelfAttention(
            _d_model, _n_head, dropout=dropout, **sdsa_config
        )

        # (v_fix_attribute_error): (変更なし)
        self.linear2 = nn.Linear(_dim_feedforward, _d_model)

        self.norm1 = SNNLayerNorm(_d_model, time_steps=_time_steps)
        self.norm2 = SNNLayerNorm(_d_model, time_steps=_time_steps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(_d_model, _dim_feedforward)
        
        # --- v_init 修正: (変更なし) ---
        self.neuron = AdaptiveLIFNeuron(
            features=_d_model,
            **neuron_params
        )
        self.ffn_neuron1 = AdaptiveLIFNeuron(
            features=_dim_feedforward,
            **neuron_params
        )
        self.ffn_neuron2 = AdaptiveLIFNeuron(
            features=_d_model,
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
        x_step, _ = self.self_attn(src) 
        
        # 2. Add & Norm (残差接続 1)
        src = src + self.dropout1(x_step)
        
        # 3. 発火 (LIF)
        src = self.neuron(src) 
        
        # 4. Norm 1
        src = self.norm1(src)

        # 5. Feedforward (FFN)
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
