# ファイルパス: snn_research/architectures/sew_resnet.py
# Title: SEW (Spiking Element-Wise) ResNet
#
# 機能の説明: Spiking Element-Wise (SEW) ブロックを使用した
# ResNetアーキテクチャの実装。
#
# 【修正内容 v30.1: 循環インポート (Circular Import) の修正】
# - health-check 実行時に 'ImportError: ... (most likely due to a circular import)'
#   が発生する問題に対処します。
# - (L: 28) 'from snn_research.core.snn_core import SNNCore' が、
#   snn_core.py (L:28) -> architectures/__init__.py (L:19) -> sew_resnet.py (L:28)
#   という循環参照を引き起こしていました。
# - (L: 251) 'SNNCore' を継承するのは誤りであり、全てのモデルが継承すべき
#   'BaseModel' に修正しました。
# - (L: 28) 'SNNCore' のインポートを削除し、'from ..core.base import BaseModel' を
#   インポートするように変更しました。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Type, Union, cast

from spikingjelly.activation_based import layer, functional, surrogate, base # type: ignore[import-untyped]

from ..core.neurons import get_neuron_by_name

# --- ▼▼▼ 【!!! 修正 v30.1: 循環インポート修正 !!!】 ▼▼▼
# (from snn_research.core.snn_core import SNNCore を削除)
from ..core.base import BaseModel # BaseModel をインポート
# --- ▲▲▲ 【!!! 修正 v30.1】 ▲▲▲


# (中略: SEWBlock, SEWResNetBlock の定義)

class SEWBlock(nn.Module):
    """
    SEW (Spiking Element-Wise) Block
    (中略)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 neuron_config: Dict[str, Any],
                 time_steps: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_steps = time_steps

        # (v15) neuron_config から 'features' を設定
        neuron_config_conv1 = neuron_config.copy()
        neuron_config_conv1['features'] = in_channels
        neuron_config_conv2 = neuron_config.copy()
        neuron_config_conv2['features'] = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.neuron1 = get_neuron_by_name(
            neuron_config_conv1.get('type', 'lif'),
            neuron_config_conv1
        )

        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.neuron2 = get_neuron_by_name(
            neuron_config_conv2.get('type', 'lif'),
            neuron_config_conv2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x は (T, B, C, H, W) を想定 """
        T, B, C, H, W = x.shape

        # (v15) 時間ループ
        outputs = []
        for t in range(T):
            x_t = x[t] # (B, C, H, W)

            # (v15) SpikingTransformerV2 (L:528) に倣い、
            #       ニューロンは (spike, mem) のタプルを返すと仮定
            
            # Path 1
            out1, _ = self.neuron1(self.bn1(x_t)) # type: ignore[attr-defined]
            out1 = self.conv1(out1)

            # Path 2
            out2, _ = self.neuron2(self.bn2(x_t)) # type: ignore[attr-defined]
            out2 = self.conv2(out2)

            # Element-wise addition
            out_t = out1 + out2
            outputs.append(out_t)

        return torch.stack(outputs, dim=0)

    def set_stateful(self, stateful: bool):
        # (v15)
        if hasattr(self.neuron1, 'set_stateful'):
            self.neuron1.set_stateful(stateful) # type: ignore[attr-defined]
        if hasattr(self.neuron2, 'set_stateful'):
            self.neuron2.set_stateful(stateful) # type: ignore[attr-defined]

    def reset(self):
        # (v15)
        if hasattr(self.neuron1, 'reset'):
            self.neuron1.reset() # type: ignore[attr-defined]
        if hasattr(self.neuron2, 'reset'):
            self.neuron2.reset() # type: ignore[attr-defined]


class SEWResNetBlock(nn.Module):
    """
    SEW ResNet Block (Shortcutあり)
    (中略)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 neuron_config: Dict[str, Any],
                 time_steps: int):
        super().__init__()
        self.sew_block = SEWBlock(in_channels, out_channels,
                                  kernel_size, stride, padding,
                                  neuron_config, time_steps)
        
        # Shortcut (Downsample)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x は (T, B, C, H, W) """
        
        # (v15) 時間ループ
        T, B, C, H, W = x.shape
        
        # (v15) Shortcut は時間依存なし (T, B, C, H, W) -> (T, B, C', H', W')
        #       (B*T, C, H, W) にリシェイプして適用
        x_reshaped = x.view(T * B, C, H, W)
        shortcut_out_reshaped = self.shortcut(x_reshaped)
        shortcut_out = shortcut_out_reshaped.view(T, B,
                                                  shortcut_out_reshaped.shape[1],
                                                  shortcut_out_reshaped.shape[2],
                                                  shortcut_out_reshaped.shape[3])

        # (v15) SEWBlock は (T, B, C, H, W) を受け取る
        sew_out = self.sew_block(x)

        # (v15) Add
        out = sew_out + shortcut_out
        return out

    def set_stateful(self, stateful: bool):
        # (v15)
        if hasattr(self.sew_block, 'set_stateful'):
            self.sew_block.set_stateful(stateful)

    def reset(self):
        # (v15)
        if hasattr(self.sew_block, 'reset'):
            self.sew_block.reset()


# --- ▼▼▼ 【!!! 修正 v30.1: 循環インポート修正 !!!】 ▼▼▼
class SEWResNet(BaseModel): # 'SNNCore' -> 'BaseModel' に変更
# --- ▲▲▲ 【!!! 修正 v30.1】 ▲▲▲
    """
    SEW (Spiking Element-Wise) ResNet
    (中略)
    """
    def __init__(
        self,
        block: Type[SEWResNetBlock],
        layers: List[int],
        num_classes: int = 10,
        in_channels: int = 3,
        time_steps: int = 16,
        neuron_config: Dict[str, Any] = {},
        **kwargs # (v15: BaseModel から vocab_size を吸収)
    ):
        # (v15: BaseModel の __init__ を呼び出す)
        super(SEWResNet, self).__init__(**kwargs)
        
        self.in_channels = 64
        self.time_steps = time_steps
        self.neuron_config = neuron_config

        # (v15) neuron_config から 'features' を設定
        neuron_config_conv1 = neuron_config.copy()
        neuron_config_conv1['features'] = self.in_channels
        neuron_config_pool = neuron_config.copy()
        neuron_config_pool['features'] = 512 * block.expansion # type: ignore[attr-defined]

        self.conv1 = nn.Conv2d(in_channels, self.in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.neuron1 = get_neuron_by_name(
            neuron_config_conv1.get('type', 'lif'),
            neuron_config_conv1
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], 1)
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.pool_neuron = get_neuron_by_name(
            neuron_config_pool.get('type', 'lif'),
            neuron_config_pool
        )
        self.fc = nn.Linear(512 * block.expansion, num_classes) # type: ignore[attr-defined]

        # (v15) 状態管理
        self._is_stateful = False
        self.built = True

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels,
                            kernel_size=3, stride=stride, padding=1,
                            neuron_config=self.neuron_config,
                            time_steps=self.time_steps))
        self.in_channels = out_channels * block.expansion # type: ignore[attr-defined]
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels,
                                kernel_size=3, stride=1, padding=1,
                                neuron_config=self.neuron_config,
                                time_steps=self.time_steps))
        return nn.Sequential(*layers)

    def set_stateful(self, stateful: bool):
        """ (v15) 状態管理モードを設定 """
        self._is_stateful = stateful
        if not stateful:
            self.reset()

        # (v15) ニューロンと全レイヤーに伝播
        if hasattr(self.neuron1, 'set_stateful'):
            self.neuron1.set_stateful(stateful) # type: ignore[attr-defined]
        if hasattr(self.pool_neuron, 'set_stateful'):
            self.pool_neuron.set_stateful(stateful) # type: ignore[attr-defined]
            
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if hasattr(layer, 'set_stateful'):
                layer.set_stateful(stateful) # type: ignore[attr-defined]
            else:
                # (v15) nn.Sequential の中のブロックに伝播
                for block in layer:
                    if hasattr(block, 'set_stateful'):
                        block.set_stateful(stateful)

    def reset(self):
        """ (v15) 状態をリセット """
        if hasattr(self.neuron1, 'reset'):
            self.neuron1.reset() # type: ignore[attr-defined]
        if hasattr(self.pool_neuron, 'reset'):
            self.pool_neuron.reset() # type: ignore[attr-defined]
            
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if hasattr(layer, 'reset'):
                layer.reset() # type: ignore[attr-defined]
            else:
                for block in layer:
                    if hasattr(block, 'reset'):
                        block.reset()

    def forward(self, input_data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        (v15: BaseModel (L:71) に合わせて引数を 'input_data' に変更)
        
        Args:
            input_data (torch.Tensor): (B, C, H, W)
        
        Returns:
            torch.Tensor: (B, NumClasses) - ロジット
        """
        T = self.time_steps
        B, C, H, W = input_data.shape

        # (v15) 状態リセット
        if not self._is_stateful:
            self.reset()
            
        # (v15) (B, C, H, W) -> (T, B, C, H, W)
        x = input_data.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        
        # (v15) 時間ループ
        outputs_conv1 = []
        for t in range(T):
            x_t = x[t] # (B, C, H, W)
            
            # (v15) Conv1 -> BN1 -> Neuron1
            out_t = self.conv1(x_t)
            out_t = self.bn1(out_t)
            out_t, _ = self.neuron1(out_t) # type: ignore[attr-defined]
            
            # (v15) MaxPool (時間ごと)
            out_t = self.maxpool(out_t)
            outputs_conv1.append(out_t)
            
        x = torch.stack(outputs_conv1, dim=0) # (T, B, C', H', W')

        # (v15) SEWResNetBlock は (T, B, C, H, W) を受け取る
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (T, B, 512*exp, H'', W'')

        # (v15) プーリング (時間軸で平均化)
        x = torch.mean(x, dim=0) # (B, 512*exp, H'', W'')
        
        x = self.avgpool(x) # (B, 512*exp, 1, 1)
        
        # (v15) プーリングニューロン (AvgPool の後)
        #       (注: SpikingTransformerV2 (L:342) では
        #        時間ループ内で適用しているが、ここでは時間平均後に適用)
        x, _ = self.pool_neuron(x) # type: ignore[attr-defined]
        
        x = torch.flatten(x, 1) # (B, 512*exp)
        
        logits = self.fc(x) # (B, NumClasses)

        # (v15: HPO (Turn 5) のために3つの値を返す)
        avg_spikes = torch.mean(x)
        avg_mem = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # (v15: SNNCore (L:221) がタプルを処理すると仮定)
        return logits, avg_spikes, avg_mem # type: ignore[return-value]
