# ファイルパス: snn_research/architectures/spiking_ssm.py
# Title: Spiking State Space Model (SSM)
#
# 機能の説明: 状態空間モデル (SSM) のアーキテクチャをSNNに
# 適用したモデル (Spiking Mamba のコア要素)。
#
# 【修正内容 v28: 循環インポート (Circular Import) の修正】
# - health-check 実行時に 'ImportError: ... (most likely due to a circular import)'
#   が発生する問題に対処します。
# - (L: 20) 'from snn_research.core.snn_core import SNNCore' が、
#   snn_core.py (L:28) -> architectures/__init__.py (L:25) -> spiking_ssm.py (L:20)
#   という循環参照を引き起こしていました。
# - (L: 22) 'SNNCore' を継承するのは誤りであり、全てのモデルが継承すべき
#   'BaseModel' に修正しました。
# - (L: 20) 'SNNCore' のインポートを削除し、'from ..core.base import BaseModel' を
#   インポートするように変更しました。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from snn_research.core.mamba_core import SNN_SSM # Mamba (SSM) のコア
from snn_research.core.neurons import get_neuron_by_name

# --- ▼▼▼ 【!!! 修正 v28: 循環インポート修正 !!!】 ▼▼▼
# (from snn_research.core.snn_core import SNNCore を削除)
from ..core.base import BaseModel # BaseModel をインポート

class SpikingSSM(BaseModel): # 'SNNCore' -> 'BaseModel' に変更
# --- ▲▲▲ 【!!! 修正 v28】 ▲▲▲
    """
    Spiking State Space Model (SSM)
    
    (中略)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int,
        time_steps: int,
        neuron_config: Dict[str, Any],
        vocab_size: int = 10000, # (v15: BaseModel から)
        num_classes: int = 10,  # (v15: SNNCore から)
        **kwargs
    ):
        # (v15: BaseModel の __init__ を呼び出す)
        super(SpikingSSM, self).__init__(vocab_size=vocab_size, **kwargs)
        
        self.d_model = d_model
        self.d_state = d_state
        self.time_steps = time_steps
        self.neuron_config = neuron_config

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

        self.ssm_layer = SNN_SSM(
            d_model=d_model,
            d_state=d_state,
            time_steps=time_steps,
            neuron_config=neuron_config
        )

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
        if hasattr(self.ssm_layer, 'set_stateful'):
            self.ssm_layer.set_stateful(stateful)

    def reset(self):
        """ (v15) 状態をリセット """
        if hasattr(self.embed_neuron, 'reset'):
            self.embed_neuron.reset() # type: ignore[attr-defined]
        if hasattr(self.ssm_layer, 'reset'):
            self.ssm_layer.reset()

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

        # 4. SSM レイヤー
        # (v15: (T, B, N, C) を処理するように変更)
        output_spikes = self.ssm_layer(x_spikes) # (T, B, SeqLen, D)
        
        # 5. プーリング
        # (v15: SpikingTransformerV2 (L:348) に倣い、
        #      時間軸とシーケンス軸で平均化)
        x_final = torch.mean(output_spikes, dim=(0, 2)) # (B, D)
        
        # 6. 分類ヘッド
        logits = self.head(x_final) # (B, NumClasses)

        # (v15: HPO (Turn 5) のために3つの値を返す)
        avg_spikes = torch.mean(output_spikes)
        avg_mem = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # (v15: SNNCore (L:221) がタプルを処理すると仮定)
        return logits, avg_spikes, avg_mem # type: ignore[return-value]
