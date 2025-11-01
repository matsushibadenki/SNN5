# ファイルパス: snn_research/io/spike_decoder.py
# (新規作成)
#
# Title: Spike Decoder (スパイクデコーダー)
#
# Description:
# - 人工脳アーキテクチャの「復号化層」を担うコンポーネント。
# - SNN内部で処理されたスパイクパターンを、人間が理解可能な
#   抽象的な情報（例: テキスト）に変換（復号化）する。
# - 設計書に基づき、スパイクカウント法を実装する。

import torch
from typing import Dict, Any

class SpikeDecoder:
    """
    スパイクパターンを抽象的な情報に復号化するモジュール。
    """
    def __init__(self, num_neurons: int):
        """
        Args:
            num_neurons (int): 符号化に使用されたニューロンの数。
        """
        self.num_neurons = num_neurons
        print("⚡️ スパイクデコーダーモジュールが初期化されました。")

    def decode(self, spike_pattern: torch.Tensor) -> str:
        """
        スパイクパターンをスパイクカウント法を用いてテキストに復号化する。

        Args:
            spike_pattern (torch.Tensor):
                復号化するスパイクパターン (time_steps, num_neurons)。

        Returns:
            str: 復号化されたテキスト。
        """
        if spike_pattern.shape[1] != self.num_neurons:
            raise ValueError(f"入力スパイクパターンのニューロン数 ({spike_pattern.shape[1]}) が"
                             f"デコーダーのニューロン数 ({self.num_neurons}) と一致しません。")

        # スパイクカウント法: 各ニューロンの発火総数を計算
        spike_counts = torch.sum(spike_pattern, dim=0)

        # 最も発火したニューロンのインデックスを取得
        most_active_neuron_ids = torch.argsort(spike_counts, descending=True)

        decoded_text = ""
        for neuron_id in most_active_neuron_ids:
            count = spike_counts[neuron_id].item()
            # 一定回数以上発火したニューロンのみを文字として解釈
            if count > 0:
                # ニューロンIDを文字のASCII値と見なして変換
                try:
                    decoded_text += chr(neuron_id.item())
                except ValueError:
                    # ASCII範囲外のIDは無視
                    pass
        
        print(f"📉 スパイクパターンをテキストに復号化しました。 -> '{decoded_text}'")
        return decoded_text