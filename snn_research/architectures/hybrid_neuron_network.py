# ファイルパス: snn_research/architectures/hybrid_neuron_network.py
# (実装完了版)
# Title: Hybrid Neuron Spiking CNN
# Description: Improvement-Plan.md に基づき、
#              LIFニューロンとBIFニューロンを混在させたSpiking CNNを実装。
#
# 改善 (v2):
# - ダミー実装だった `forward` メソッドを、`stateful` モードと
#   `time_steps` ループを正しく処理する完全な実装に置き換える。
# - `functional.reset_net(self)` が BIF ニューロンもリセットできるようにする
#   (bif_neuron.py側の MemoryModule 継承が前提)。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Type, cast

# 必要なニューロンクラスをインポート
from snn_research.core.neurons import AdaptiveLIFNeuron as LIFNeuron
from snn_research.core.neurons.bif_neuron import BistableIFNeuron as BIFNeuron
from snn_research.core.base import BaseModel # BaseModelをインポート

# spikingjellyのfunctionalをリセットに利用
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]

class HybridSpikingCNN(BaseModel):
    """
    LIFニューロンとBIFニューロンを混在させたSpiking CNN。
    """
    fc1: nn.Linear
    fc_neuron: nn.Module
    fc2: nn.Linear

    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 10,
                 use_bif_layers: List[int] = [1],
                 time_steps: int = 16,
                 lif_params: Optional[Dict[str, Any]] = None,
                 bif_params: Optional[Dict[str, Any]] = None,
                 # --- ▼ 追加: 入力サイズを仮定 ▼ ---
                 input_size: int = 224
                 # --- ▲ 追加 ▲ ---
                 ):
        super().__init__()
        self.time_steps = time_steps
        self.use_bif_layers = use_bif_layers

        if lif_params is None:
            lif_params = {'tau_mem': 10.0, 'base_threshold': 1.0}
        if bif_params is None:
            bif_params = {'v_threshold_high': 1.0, 'v_reset': 0.6, 'bistable_strength': 0.25, 'unstable_equilibrium_offset': 0.5}

        self.features = nn.ModuleList()
        self.neuron_types: List[str] = [] # 各層のニューロンタイプを記録

        current_channels = input_channels
        current_size = input_size
        
        # --- レイヤー定義 ---
        # Layer 0 (Conv + Neuron + Pool)
        out_channels_0 = 64
        self.features.append(nn.Conv2d(current_channels, out_channels_0, kernel_size=3, padding=1))
        if 0 in use_bif_layers:
            self.features.append(BIFNeuron(features=out_channels_0, **bif_params))
            self.neuron_types.append("BIF")
        else:
            self.features.append(LIFNeuron(features=out_channels_0, **lif_params))
            self.neuron_types.append("LIF")
        self.features.append(nn.AvgPool2d(2))
        current_channels = out_channels_0
        current_size //= 2

        # Layer 1 (Conv + Neuron + Pool)
        out_channels_1 = 128
        self.features.append(nn.Conv2d(current_channels, out_channels_1, kernel_size=3, padding=1))
        if 1 in use_bif_layers:
            self.features.append(BIFNeuron(features=out_channels_1, **bif_params))
            self.neuron_types.append("BIF")
        else:
            self.features.append(LIFNeuron(features=out_channels_1, **lif_params))
            self.neuron_types.append("LIF")
        self.features.append(nn.AvgPool2d(2))
        current_channels = out_channels_1
        current_size //= 2

        # Layer 2 (Conv + Neuron + Pool)
        out_channels_2 = 256
        self.features.append(nn.Conv2d(current_channels, out_channels_2, kernel_size=3, padding=1))
        if 2 in use_bif_layers:
            self.features.append(BIFNeuron(features=out_channels_2, **bif_params))
            self.neuron_types.append("BIF")
        else:
            self.features.append(LIFNeuron(features=out_channels_2, **lif_params))
            self.neuron_types.append("LIF")
        self.features.append(nn.AvgPool2d(2))
        current_channels = out_channels_2
        current_size //= 2

        # Classifier
        flattened_size = current_channels * current_size * current_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 512)
        # 最終層手前は安定なLIFを使用
        self.fc_neuron = LIFNeuron(features=512, **lif_params)
        self.neuron_types.append("LIF_FC")
        self.fc2 = nn.Linear(512, num_classes)

        self._init_weights()
        print(f"✅ HybridSpikingCNN initialized (Input: {input_size}x{input_size}). BIF layers at indices: {use_bif_layers}")
        print(f"   Neuron sequence: {self.neuron_types}")

    def forward(self, input_images: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ハイブリッドCNNのフォワードパス（実装完了版）。
        Statefulな時間ループを実行する。
        """
        B, C, H, W = input_images.shape
        device = input_images.device

        # --- 状態リセット ---
        # BIFNeuron が MemoryModule を継承したため、reset_net が両方をリセットする
        SJ_F.reset_net(self)
        
        # --- 全ニューロンを Stateful モードに設定 ---
        for module in self.modules():
            if isinstance(module, (LIFNeuron, BIFNeuron)):
                cast(Any, module).set_stateful(True)

        output_voltages_sum = torch.zeros(B, self.fc2.out_features, device=device)
        
        # --- 時間ステップループ ---
        for t in range(self.time_steps):
            # 毎ステップ同じ画像を入力 (レートコーディングの簡易版)
            x: torch.Tensor = input_images
            neuron_layer_idx: int = 0

            # --- CNN特徴抽出レイヤー ---
            for i, layer in enumerate(self.features):
                if isinstance(layer, (LIFNeuron, BIFNeuron)):
                    # ニューロン層の処理
                    # (B, C, H, W) -> (B*H*W, C) にreshapeしてニューロンに入力
                    B_c, C_c, H_c, W_c = x.shape
                    x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C_c)
                    
                    spikes, _ = layer(x_reshaped) # type: ignore[operator]
                    
                    # 出力形状を元に戻す
                    x = spikes.view(B_c, H_c, W_c, C_c).permute(0, 3, 1, 2)
                    neuron_layer_idx += 1
                else:
                    # Conv, Pool 層
                    x = layer(x) # type: ignore[operator]

            # --- 分類層 ---
            x_flat = self.flatten(x)
            
            x_fc1_current = self.fc1(x_flat)
            x_fc1_spikes, _ = self.fc_neuron(x_fc1_current) # type: ignore[operator]
            
            # 最終層はロジット（アナログ値）
            final_output_t = self.fc2(x_fc1_spikes)

            # 時間全体の出力を累積
            output_voltages_sum += final_output_t

        # --- ループ終了後: Stateful を解除 ---
        for module in self.modules():
            if isinstance(module, (LIFNeuron, BIFNeuron)):
                cast(Any, module).set_stateful(False)

        # 時間平均を取って最終的なロジットとする
        final_logits = output_voltages_sum / self.time_steps

        # スパイク数と膜電位 (簡易的に返す)
        avg_spikes_val = self.get_total_spikes() / (B * self.time_steps) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        mem = torch.tensor(0.0, device=device) # 最終膜電位はここでは意味を持たない

        return final_logits, avg_spikes, mem
