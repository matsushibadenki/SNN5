# ファイルパス: snn_research/io/spike_encoder.py
# (改修)
#
# Title: Spike Encoder (TTFS実装版)
#
# Description:
# - 人工脳アーキテクチャの「符号化層」を担うコンポーネント。
# - SensoryReceptorから受け取った内部表現を、SNNが処理可能な
#   スパイクパターンに変換（符号化）する。
# - 【技術指令】指令2「レートコーディング依存の削除」に基づき、
#   高効率な Time-to-First-Spike (TTFS) 符号化をデフォルトとして実装する。
#
# 改善 (v2):
# - doc/SNN開発：基本設計思想.md (セクション4.1, 引用[70]) に基づき、
#   学習可能な(微分可能な)TTFSエンコーダ `DifferentiableTTFSEncoder` を追加。
#
# 修正 (v3): mypy [name-defined] エラーを解消するため、surrogate をインポート。

import torch
import torch.nn as nn 
from typing import Dict, Any, Optional
import math 
# --- ▼ 修正 ▼ ---
from spikingjelly.activation_based import surrogate # type: ignore
# --- ▲ 修正 ▲ ---

class SpikeEncoder:
    """
    感覚情報をスパイクパターンに符号化するモジュール。
    """
    def __init__(self, num_neurons: int, max_rate: int = 100):
        """
        Args:
            num_neurons (int): 符号化に使用するニューロンの数。
            max_rate (int): 最大発火率 (Hz)。
        """
        self.num_neurons = num_neurons
        self.max_rate = max_rate
        print("⚡️ スパイクエンコーダーモジュールが初期化されました。(デフォルト: TTFS)")

    def encode(
        self,
        sensory_info: Dict[str, Any],
        duration: int = 50, # ◾️ タイムステップ（期間）
        encoding_type: str = "ttfs" # ◾️ デフォルトをTTFSに変更
    ) -> torch.Tensor:
        """
        感覚情報をスパイクパターンに変換する。

        Args:
            sensory_info (Dict[str, Any]): SensoryReceptorからの出力。
            duration (int): スパイクを生成する期間 (タイムステップ数)。
            encoding_type (str): "ttfs" (デフォルト) または "rate"。

        Returns:
            torch.Tensor: 生成されたスパイクパターン (time_steps, num_neurons)。
        """
        content: Any = sensory_info.get('content')
        
        if sensory_info['type'] == 'text' and isinstance(content, str):
            if encoding_type == "ttfs":
                return self._ttfs_encode_text(content, duration)
            elif encoding_type == "rate":
                return self._rate_encode_text(content, duration)
        
        elif sensory_info['type'] == 'numeric' and isinstance(content, (int, float)):
             # 0から1の範囲に正規化されていると仮定
             normalized_value: float = max(0.0, min(1.0, float(content)))
             if encoding_type == "ttfs":
                 return self._ttfs_encode_value(normalized_value, duration)
        
        print(f"⚠️ サポートされていないエンコードタイプ ({sensory_info['type']}, {encoding_type}) です。")
        return torch.zeros((duration, self.num_neurons))

    def _ttfs_encode_value(self, value: float, duration: int) -> torch.Tensor:
        """
        単一の正規化された値 [0, 1] をTTFSでエンコードする。
        値が強いほど（1に近いほど）、早く発火する。
        """
        spikes = torch.zeros((duration, self.num_neurons))
        if value <= 0.0:
            return spikes # 0以下の入力は発火しない

        fire_time: int = math.floor((1.0 - value) * (duration - 1))
        
        if 0 <= fire_time < duration:
            spikes[fire_time, 0] = 1.0
            
        print(f"📈 数値 {value:.2f} をTTFS符号化 (T={fire_time}) しました。")
        return spikes

    def _ttfs_encode_text(self, text: str, duration: int) -> torch.Tensor:
        """
        テキストをTime-to-First-Spike (TTFS) でエンコードする。
        テキストの順序が時間にマッピングされる。
        """
        time_steps = duration
        spikes = torch.zeros((time_steps, self.num_neurons))
        
        for char_index, char in enumerate(text):
            if char_index >= time_steps:
                break 
            
            neuron_id = ord(char) % self.num_neurons
            spikes[char_index, neuron_id] = 1.0

        print(f"📉 テキストを {time_steps}x{self.num_neurons} のスパイクパターンにTTFS符号化しました。")
        return spikes

    def _rate_encode_text(self, text: str, duration: int) -> torch.Tensor:
        """
        テキストをレート符号化する。(指令により非推奨だが機能としては残す)
        """
        time_steps = duration
        spikes = torch.zeros((time_steps, self.num_neurons))

        for char_index, char in enumerate(text):
            neuron_id = ord(char) % self.num_neurons
            fire_prob: float = (self.max_rate * (duration / 1000.0)) / time_steps
            if fire_prob <= 0:
                continue
            
            poisson_spikes: torch.Tensor = torch.poisson(torch.full((time_steps,), fire_prob))
            spikes[:, neuron_id] += poisson_spikes

        spikes = torch.clamp(spikes, 0, 1)
        
        print(f"📈 (非推奨) テキストを {time_steps}x{self.num_neurons} のスパイクパターンにレート符号化しました。")
        return spikes


# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始 (DifferentiableTTFSEncoder)◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
class DifferentiableTTFSEncoder(nn.Module):
    """
    doc/SNN開発：基本設計思想.md (セクション4.1, 引用[70]) に基づく、
    学習可能な（微分可能な）TTFSエンコーダ。

    入力アナログ値 `x` (0~1) を `(1 - x) * T_max` のように発火時間にマッピングする
    プロセスにおいて、そのマッピングの鋭敏さ（`sensitivity`）を学習可能にする。
    """
    def __init__(self, num_neurons: int, duration: int, initial_sensitivity: float = 10.0):
        """
        Args:
            num_neurons (int): エンコード対象のニューロン数。
            duration (int): スパイクを生成する期間 (タイムステップ数)。
            initial_sensitivity (float): 発火タイミングの鋭敏さ（学習可能）。
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.duration = duration
        
        # 学習可能なパラメータ (各ニューロンが独自のマッピング感度を持つ)
        self.sensitivity = nn.Parameter(torch.full((num_neurons,), initial_sensitivity))
        self.time_steps_tensor = nn.Parameter(torch.arange(0, duration, dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False) # (1, 1, T)

    def forward(self, x_analog: torch.Tensor) -> torch.Tensor:
        """
        アナログ入力 (B, N) をTTFSスパイク (B, N, T) に微分可能に変換する。

        Args:
            x_analog (torch.Tensor): アナログ入力 (Batch, num_neurons)。値は [0, 1] に正規化されている想定。

        Returns:
            torch.Tensor: 生成されたスパイクパターン (Batch, num_neurons, time_steps)。
        """
        B, N = x_analog.shape
        if N != self.num_neurons:
            raise ValueError(f"Input dimension ({N}) does not match num_neurons ({self.num_neurons})")

        # 1. 目標発火時間を計算
        # x=1.0 -> target_time=0
        # x=0.0 -> target_time=(duration-1)
        target_fire_time = (1.0 - x_analog) * (self.duration - 1) # (B, N)
        
        # (B, N) -> (B, N, T)
        target_fire_time_expanded = target_fire_time.unsqueeze(-1)
        
        # (1, 1, T)
        time_steps = self.time_steps_tensor

        # 2. 微分可能なスパイク生成 (代理勾配のアイデアを利用)
        # 時間ステップ `t` と `target_fire_time` の差を計算
        # `t` が `target_fire_time` を超えた瞬間に発火 (値が負になる)
        distance = time_steps - target_fire_time_expanded # (B, N, T)
        
        # 鋭敏さ（学習可能）を適用
        # sensitivity が大きいほど、target_fire_time での発火が鋭くなる
        # Sigmoidの代理勾配 (surrogate.fast_sigmoidなど) を使う
        # ここでは単純な Sigmoid を使う
        spike_probs = torch.sigmoid(-distance * self.sensitivity.view(1, -1, 1))
        
        # 3. 確率的スパイクではなく、「最初の」スパイクのみを選択
        # 累積確率が0.5を超えた最初の時点を見つける (微分可能な近似)
        # (簡易実装: 確率をそのままスパイクの「強度」として扱う)
        # SNNの訓練では、この確率的な値が代理勾配として機能する
        
        # ここでは最も単純な実装として、確率をそのまま「ソフトなスパイク」として返す
        # 実際のSNN (LIF) はバイナリ入力を期待するため、これはあくまで
        # 「学習可能なエンコーダ層」としての実装例
        
        # TTFSの性質（最初のスパイクのみ）を厳密にするため、
        # 累積確率を計算し、その差分を取る
        cumulative_probs = torch.cumsum(spike_probs, dim=-1)
        # 最初のスパイクのみを1に、残りを0にする (微分可能ではない)
        # spikes = (cumulative_probs > 0.5) & (torch.roll(cumulative_probs, 1, -1) <= 0.5)
        # spikes = spikes.float()
        
        # 微分可能な近似として、soft-winner-take-all (例: Gumbel-Softmax) が必要だが、
        # ここでは surrogate.fast_sigmoid のような代理勾配関数で代用する
        # (代理勾配関数は通常、(x - threshold) を入力に取る)
        
        # surrogate.fast_sigmoid を使う例 (代理勾配)
        # これにより、forwardでは 0/1 に近い値 (Heaviside)、backwardでは勾配が流れる
        
        # --- ▼ 修正 ▼ ---
        # surrogate が未定義だったため修正
        spikes = surrogate.fast_sigmoid(self.duration - 1 - distance * self.sensitivity.view(1, -1, 1)) # 仮の実装
        # --- ▲ 修正 ▲ ---

        return spikes.permute(0, 2, 1) # (B, T, N) に形状を合わせる
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
