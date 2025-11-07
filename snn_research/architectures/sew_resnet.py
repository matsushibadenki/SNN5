# ファイルパス: snn_research/architectures/sew_resnet.py
# (新規作成)
#
# Title: SEW (Spike-Element-Wise) ResNet
#
# Description:
# doc/SNN開発：基本設計思想.md (セクション2.1, 引用[9, 26]) に基づき、
# SNNの深層学習を可能にするための残差学習（Residual Learning）を実装する。
#
# このアーキテクチャは、標準的なANNのResNetにおける恒等写像（Identity Mapping）を、
# スパイクニューロンのダイナミクスに適応させたものです。
#
# 注意点 (設計思想.md 引用[9, 27]):
# このSEW-ResNetの残差接続（加算ゲート）は、SNNの膜電位（アナログ値）または
# スパイク（バイナリ値）と、恒等写像のスパイク（バイナリ値）を加算します。
# これが「非スパイク計算（浮動小数点加算）」と見なされ、一部の純粋な
# イベント駆動型ニューロモーフィックハードウェアとの互換性がない可能性が
# 指摘されています。
#
# mypy --strict 準拠。
#
# 修正 (v2): mypy [name-defined] エラーを解消するため、loggingをインポート。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Type, Optional, List, cast
# --- ▼ 修正 ▼ ---
import logging
# --- ▲ 修正 ▲ ---

# SNNのコアコンポーネントをインポート
from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from spikingjelly.activation_based import functional as SJ_F # type: ignore

# --- ▼ 修正 ▼ ---
# ロガー設定
logger = logging.getLogger(__name__)
# --- ▲ 修正 ▲ ---

class SEWResidualBlock(nn.Module):
    """
    SEW (Spike-Element-Wise) 残差ブロック。
    (Conv -> Norm -> LIF -> Conv -> Norm) + Identity (LIF)
    """
    conv1: nn.Conv2d
    norm1: nn.Module # SNNLayerNorm or BatchNorm
    lif1: nn.Module
    conv2: nn.Conv2d
    norm2: nn.Module
    lif_shortcut: nn.Module
    lif_out: nn.Module
    downsample: Optional[nn.Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any]
    ) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # Note: SNNLayerNormは(..., C)を期待するが、ここでは (B, C, H, W) の時間ステップごとを想定
        # SpikingJellyの慣例に従いBatchNormを使用する（ただしBNは非SNN的）
        # ここでは互換性のため BatchNorm を使用
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.lif1 = neuron_class(features=out_channels, **neuron_params)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        # --- 残差接続 (Shortcut) ---
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # チャンネル数またはサイズが異なる場合、ショートカットも変換
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        # スパイクベースの恒等写像のためのLIF
        self.lif_shortcut = neuron_class(features=out_channels, **neuron_params)
        
        # 最終的な加算後のLIF
        self.lif_out = neuron_class(features=out_channels, **neuron_params)

    def forward(self, x_spike: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_spike (torch.Tensor): 入力スパイク (B, T, C_in, H, W)
        
        Returns:
            torch.Tensor: ブロックの出力スパイク (B, T, C_out, H', W')
        """
        B, T, C_in, H, W = x_spike.shape
        
        # --- 1. ショートカットパス (Identity) ---
        shortcut_spikes: List[torch.Tensor] = []
        
        # ショートカット用のLIFをStatefulに設定
        SJ_F.reset_net(self.lif_shortcut)
        neuron_sc: nn.Module = cast(nn.Module, self.lif_shortcut)
        if hasattr(neuron_sc, 'set_stateful'):
            getattr(neuron_sc, 'set_stateful')(True)

        identity: torch.Tensor = x_spike # (B, T, C_in, H, W)
        
        # ダウンサンプル処理 (時間軸でループ)
        identity_downsampled_t_list: List[torch.Tensor] = []
        if self.downsample:
            identity_flat_time: torch.Tensor = identity.reshape(B * T, C_in, H, W)
            identity_downsampled_flat: torch.Tensor = self.downsample(identity_flat_time)
            _, C_out, H_out, W_out = identity_downsampled_flat.shape
            identity_downsampled: torch.Tensor = identity_downsampled_flat.reshape(B, T, C_out, H_out, W_out)
        else:
            identity_downsampled = identity
            _, C_out, H_out, W_out = identity_downsampled.shape

        # ショートカットパスのLIF (時間軸ループ)
        for t_idx in range(T):
            x_sc_t: torch.Tensor = identity_downsampled[:, t_idx, ...] # (B, C_out, H_out, W_out)
            # (B, C, H, W) -> (B*H*W, C)
            x_sc_t_flat: torch.Tensor = x_sc_t.permute(0, 2, 3, 1).reshape(-1, C_out)
            spike_sc_t_flat, _ = self.lif_shortcut(x_sc_t_flat)
            spike_sc_t: torch.Tensor = spike_sc_t_flat.reshape(B, H_out, W_out, C_out).permute(0, 3, 1, 2)
            shortcut_spikes.append(spike_sc_t)
            
        if hasattr(neuron_sc, 'set_stateful'):
            getattr(neuron_sc, 'set_stateful')(False)

        # --- 2. メインパス (Conv) ---
        main_path_spikes: List[torch.Tensor] = []
        
        # メインパスのLIFをStatefulに設定
        SJ_F.reset_net(self.lif1)
        SJ_F.reset_net(self.lif_out)
        neuron1: nn.Module = cast(nn.Module, self.lif1)
        neuron_out: nn.Module = cast(nn.Module, self.lif_out)
        if hasattr(neuron1, 'set_stateful'): getattr(neuron1, 'set_stateful')(True)
        if hasattr(neuron_out, 'set_stateful'): getattr(neuron_out, 'set_stateful')(True)

        for t_idx in range(T):
            x_t: torch.Tensor = x_spike[:, t_idx, ...] # (B, C_in, H, W)
            
            # Conv1 -> Norm1 -> LIF1
            y_t: torch.Tensor = self.conv1(x_t)
            y_t = self.norm1(y_t)
            # (B, C_out, H', W') -> (B*H'*W', C_out)
            B_t, C_t, H_t, W_t = y_t.shape
            y_t_flat: torch.Tensor = y_t.permute(0, 2, 3, 1).reshape(-1, C_t)
            spike1_t_flat, _ = self.lif1(y_t_flat)
            spike1_t: torch.Tensor = spike1_t_flat.reshape(B_t, H_t, W_t, C_t).permute(0, 3, 1, 2)

            # Conv2 -> Norm2
            y_t = self.conv2(spike1_t)
            y_t = self.norm2(y_t) # (B, C_out, H', W')

            # --- 3. 加算 (Add Gate) ---
            # (設計思想.md 引用[9, 27] で指摘される非スパイク計算（加算）)
            # メインパスの電流(y_t)とショートカットパスのスパイク(shortcut_spikes[t_idx])を加算
            residual_input: torch.Tensor = y_t + shortcut_spikes[t_idx]
            
            # 最終的なLIF
            # (B, C_out, H', W') -> (B*H'*W', C_out)
            res_in_flat: torch.Tensor = residual_input.permute(0, 2, 3, 1).reshape(-1, C_out)
            spike_out_t_flat, _ = self.lif_out(res_in_flat)
            spike_out_t: torch.Tensor = spike_out_t_flat.reshape(B_t, H_t, W_t, C_t).permute(0, 3, 1, 2)
            
            main_path_spikes.append(spike_out_t)

        if hasattr(neuron1, 'set_stateful'): getattr(neuron1, 'set_stateful')(False)
        if hasattr(neuron_out, 'set_stateful'): getattr(neuron_out, 'set_stateful')(False)

        # (T, B, C_out, H', W') -> (B, T, C_out, H', W')
        return torch.stack(main_path_spikes, dim=1)


class SEWResNet(BaseModel):
    """
    SEW (Spike-Element-Wise) ResNet アーキテクチャ (スタブ実装)。
    SpikingCNN (snn_core.py) をベースに、残差ブロックで置き換える。
    """
    def __init__(
        self,
        num_classes: int = 10, # SpikingCNNとの互換性のため vocab_size を num_classes に
        time_steps: int = 16,
        neuron_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any # vocab_size を kwargs で受け取る
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        
        if neuron_config is None:
            neuron_config = {'type': 'lif', 'tau_mem': 10.0, 'base_threshold': 1.0}
            
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        neuron_class: Type[nn.Module]
        if neuron_type_str == 'lif':
            neuron_class = AdaptiveLIFNeuron
            neuron_params = {k: v for k, v in neuron_params.items() if k in ['tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']}
        elif neuron_type_str == 'izhikevich':
            neuron_class = IzhikevichNeuron
            neuron_params = {k: v for k, v in neuron_params.items() if k in ['features', 'a', 'b', 'c', 'd', 'dt']}
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type_str}")

        self.in_channels = 64
        
        # --- 1. 入力層 (SpikingCNNと同様) ---
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(self.in_channels)
        self.lif1 = neuron_class(features=self.in_channels, **neuron_params)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- 2. 残差ブロック層 ---
        # (ResNet-18/34風のスタブ)
        self.layer1 = self._make_layer(self.in_channels, 64, 2, 1, neuron_class, neuron_params)
        self.layer2 = self._make_layer(64, 128, 2, 2, neuron_class, neuron_params)
        self.layer3 = self._make_layer(128, 256, 2, 2, neuron_class, neuron_params)
        self.layer4 = self._make_layer(256, 512, 2, 2, neuron_class, neuron_params)
        
        # --- 3. 分類層 ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()
        logger.info("✅ SEWResNet (Stub) initialized.")

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any]
    ) -> nn.Sequential:
        
        layers: List[nn.Module] = []
        # 最初のブロックでストライド（ダウンサンプリング）を適用
        layers.append(SEWResidualBlock(in_channels, out_channels, stride, neuron_class, neuron_params))
        # 残りのブロック
        for _ in range(1, num_blocks):
            layers.append(SEWResidualBlock(out_channels, out_channels, 1, neuron_class, neuron_params))
        return nn.Sequential(*layers)

    def forward(
        self, 
        input_images: torch.Tensor, 
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, C, H, W = input_images.shape
        device: torch.device = input_images.device
        
        # (B, C, H, W) -> (B, T, C, H, W)
        # レートコーディング (簡易版)
        x_spikes_t: torch.Tensor = input_images.unsqueeze(1).repeat(1, self.time_steps, 1, 1, 1)
        
        # 状態リセット
        SJ_F.reset_net(self)
        
        # --- 1. 入力層 (時間軸ループ) ---
        spikes_history: List[torch.Tensor] = []
        
        # 入力層のLIFをStatefulに設定
        neuron1: nn.Module = cast(nn.Module, self.lif1)
        if hasattr(neuron1, 'set_stateful'):
            getattr(neuron1, 'set_stateful')(True)

        for t in range(self.time_steps):
            x_t: torch.Tensor = x_spikes_t[:, t, ...] # (B, C, H, W)
            y_t: torch.Tensor = self.conv1(x_t)
            y_t = self.norm1(y_t)
            # (B, C, H, W) -> (B*H*W, C)
            B_t, C_t, H_t, W_t = y_t.shape
            y_t_flat: torch.Tensor = y_t.permute(0, 2, 3, 1).reshape(-1, C_t)
            spike_t_flat, _ = self.lif1(y_t_flat)
            spike_t: torch.Tensor = spike_t_flat.reshape(B_t, H_t, W_t, C_t).permute(0, 3, 1, 2)
            spikes_history.append(spike_t)
            
        if hasattr(neuron1, 'set_stateful'):
            getattr(neuron1, 'set_stateful')(False)

        x_spikes: torch.Tensor = torch.stack(spikes_history, dim=1) # (B, T, C, H, W)
        
        # MaxPool (時間軸にも適用)
        # (B, T, C, H, W) -> (B*T, C, H, W)
        x_flat_time: torch.Tensor = x_spikes.reshape(B * self.time_steps, self.in_channels, H, W)
        pooled_flat: torch.Tensor = self.pool1(x_flat_time)
        _, C_pool, H_pool, W_pool = pooled_flat.shape
        x_spikes = pooled_flat.reshape(B, self.time_steps, C_pool, H_pool, W_pool)

        # --- 2. 残差ブロック層 ---
        # (SEWResidualBlockは内部で時間ループを処理)
        x_spikes = self.layer1(x_spikes)
        x_spikes = self.layer2(x_spikes)
        x_spikes = self.layer3(x_spikes)
        x_spikes = self.layer4(x_spikes)

        # --- 3. 分類層 ---
        # (B, T, C, H, W) -> (B, C, H, W) (時間平均)
        x_analog: torch.Tensor = x_spikes.mean(dim=1) 
        
        x_analog = self.avgpool(x_analog) # (B, C, 1, 1)
        x_analog = torch.flatten(x_analog, 1) # (B, C)
        logits: torch.Tensor = self.fc(x_analog) # (B, num_classes)
        
        # --- 互換性のため (logits, avg_spikes, mem) を返す ---
        avg_spikes_val: float = self.get_total_spikes() / (B * self.time_steps) if return_spikes and self.time_steps > 0 else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        mem: torch.Tensor = torch.tensor(0.0, device=device)

        return logits, avg_spikes, mem