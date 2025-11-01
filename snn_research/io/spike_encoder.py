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

import torch
from typing import Dict, Any, Optional
import math # ◾️◾️◾️ 追加 ◾️◾️◾️

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
        
        # ◾️◾️◾️ 追加: 数値入力のエンコード ◾️◾️◾️
        elif sensory_info['type'] == 'numeric' and isinstance(content, (int, float)):
             # 0から1の範囲に正規化されていると仮定
             normalized_value: float = max(0.0, min(1.0, float(content)))
             if encoding_type == "ttfs":
                 return self._ttfs_encode_value(normalized_value, duration)
        # ◾️◾️◾️ ここまで ◾️◾️◾️

        # 不明なタイプやレートコーディング指定のない場合は空のスパイク列を返す
        print(f"⚠️ サポートされていないエンコードタイプ ({sensory_info['type']}, {encoding_type}) です。")
        return torch.zeros((duration, self.num_neurons))

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始 (TTFS Value)◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def _ttfs_encode_value(self, value: float, duration: int) -> torch.Tensor:
        """
        単一の正規化された値 [0, 1] をTTFSでエンコードする。
        値が強いほど（1に近いほど）、早く発火する。
        """
        spikes = torch.zeros((duration, self.num_neurons))
        if value <= 0.0:
            return spikes # 0以下の入力は発火しない

        # 値をタイムステップにマッピング (非線形マッピングも可)
        # value=1.0 -> fire_time=0
        # value=0.0 -> fire_time=duration-1 (または発火しない)
        fire_time: int = math.floor((1.0 - value) * (duration - 1))
        
        # 簡易的に、最初のニューロンに発火を割り当てる
        if 0 <= fire_time < duration:
            spikes[fire_time, 0] = 1.0
            
        print(f"📈 数値 {value:.2f} をTTFS符号化 (T={fire_time}) しました。")
        return spikes
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始 (TTFS Text)◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def _ttfs_encode_text(self, text: str, duration: int) -> torch.Tensor:
        """
        テキストをTime-to-First-Spike (TTFS) でエンコードする。
        テキストの順序が時間にマッピングされる。
        """
        time_steps = duration
        spikes = torch.zeros((time_steps, self.num_neurons))
        
        # テキストの各文字をタイムステップに割り当てる
        for char_index, char in enumerate(text):
            if char_index >= time_steps:
                break # 期間を超える文字は無視
            
            # 文字のASCII値をニューロンIDとして使用
            neuron_id = ord(char) % self.num_neurons
            
            # char_index のタイミングで発火
            spikes[char_index, neuron_id] = 1.0

        print(f"📉 テキストを {time_steps}x{self.num_neurons} のスパイクパターンにTTFS符号化しました。")
        return spikes
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def _rate_encode_text(self, text: str, duration: int) -> torch.Tensor:
        """
        テキストをレート符号化する。(指令により非推奨だが機能としては残す)
        """
        time_steps = duration
        spikes = torch.zeros((time_steps, self.num_neurons))

        for char_index, char in enumerate(text):
            # 文字のASCII値をニューロンIDとして使用
            neuron_id = ord(char) % self.num_neurons
            
            # 発火率を計算
            fire_prob: float = (self.max_rate * (duration / 1000.0)) / time_steps
            if fire_prob <= 0:
                continue
            
            # ポアソン分布に従うスパイクを生成
            poisson_spikes: torch.Tensor = torch.poisson(torch.full((time_steps,), fire_prob))
            spikes[:, neuron_id] += poisson_spikes

        # スパイクは0か1なので、1より大きい値は1にクリップ
        spikes = torch.clamp(spikes, 0, 1)
        
        print(f"📈 (非推奨) テキストを {time_steps}x{self.num_neurons} のスパイクパターンにレート符号化しました。")
        return spikes