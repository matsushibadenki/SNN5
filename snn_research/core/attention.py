# ファイルパス: snn_research/core/attention.py
# Title: Spike-Driven Self-Attention (SDSA) (改善・詳細化版)
# Description: Improvement-Plan.mdに基づき、乗算を使用しないスパイクベースの
#              自己注意メカニズム (SDSA) を実装します。
#              タイムステップ統合方法を改善し、ゼロスパイク問題への対処を追加します。
#              AdaptiveLIFNeuronへの不正なキーワード引数エラーを修正済み。

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
import random # randomをインポート

# SDSAで使用するスパイクニューロン (例: LIF)
from .neurons import AdaptiveLIFNeuron as LIFNeuron
# 代理勾配関数
from spikingjelly.activation_based import surrogate # type: ignore
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpikeDrivenSelfAttention(nn.Module):
    """
    Spike-Driven Self-Attention (SDSA) の改善・詳細化版実装。
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 time_steps: int,
                 neuron_config: dict,
                 add_noise_if_silent: bool = True, # ゼロスパイク時にノイズを加えるか (デフォルトTrue)
                 noise_prob: float = 0.01 # ノイズを加える確率
                ):
        """
        Args:
            dim (int): モデルの次元数。
            num_heads (int): アテンションヘッド数。
            time_steps (int): スパイク生成のためのタイムステップ数。
            neuron_config (dict): スパイクニューロンの設定。
            add_noise_if_silent (bool): 全スパイクが0の場合にノイズを加えるか。
            noise_prob (float): ノイズとしてスパイクを発生させる確率。
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.time_steps = time_steps
        self.add_noise_if_silent = add_noise_if_silent
        self.noise_prob = noise_prob

        # 線形変換層 (入力 -> Q, K, V)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # スパイク生成ニューロン
        # AdaptiveLIFNeuronに渡せるパラメータのみフィルタリング
        lif_params = {k: v for k, v in neuron_config.items() if k in ['tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']}
        # SDSA専用の閾値パラメータがあれば優先して使用
        lif_params['base_threshold'] = neuron_config.get("sdsa_threshold", lif_params.get('base_threshold', 1.0))

        # surrogate_function は AdaptiveLIFNeuron 内部で定義されるため、引数から削除
        self.lif_q = LIFNeuron(features=dim, **lif_params)
        self.lif_k = LIFNeuron(features=dim, **lif_params)
        self.lif_v = LIFNeuron(features=dim, **lif_params) # Vもスパイク化

        # 出力層
        self.to_out = nn.Linear(dim, dim)

        logging.info("✅ SpikeDrivenSelfAttention (Improved Implementation) initialized.")
        logging.info(f"   - Time steps: {self.time_steps}")
        logging.info(f"   - Add noise if silent: {self.add_noise_if_silent}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SDSAのフォワードパスを実行します。

        Args:
            x (torch.Tensor): 入力テンソル (Batch, Num_Tokens, Dim)。

        Returns:
            torch.Tensor: アテンション適用後の出力テンソル (Batch, Num_Tokens, Dim)。
        """
        B, N, C = x.shape
        device = x.device

        # 線形変換
        q_lin = self.to_q(x) # (B, N, C)
        k_lin = self.to_k(x)
        v_lin = self.to_v(x)

        # タイムステップループでスパイクを生成
        s_q_list, s_k_list, s_v_list = [], [], []

        # ニューロンの状態をリセット
        self.reset()

        # ニューロンをstatefulモードに設定
        # ⚠️ 注意: 実際の使用時には、Transformer層の外部でstateful設定を行う方が適切
        self.lif_q.set_stateful(True)
        self.lif_k.set_stateful(True)
        self.lif_v.set_stateful(True)

        for t in range(self.time_steps):
            # (B, N, C) -> (B*N, C) にreshapeしてニューロンに入力
            s_q_t, _ = self.lif_q(q_lin.reshape(B * N, C)) # (B*N, C) binary spikes
            s_k_t, _ = self.lif_k(k_lin.reshape(B * N, C))
            s_v_t, _ = self.lif_v(v_lin.reshape(B * N, C)) # Vもスパイク化

            # (B*N, C) -> (B, N, C) に戻してリストに追加
            s_q_list.append(s_q_t.reshape(B, N, C))
            s_k_list.append(s_k_t.reshape(B, N, C))
            s_v_list.append(s_v_t.reshape(B, N, C))

        # ニューロンをstatelessモードに戻す
        self.lif_q.set_stateful(False)
        self.lif_k.set_stateful(False)
        self.lif_v.set_stateful(False)

        # タイムステップの情報を集約 (合計して最大1にクリップ)
        # 複数回スパイクしても1回のスパイクとして扱うことで、レートコーディングの影響を低減
        s_q_agg = torch.stack(s_q_list).sum(dim=0).clamp(max=1.0) # (B, N, C)
        s_k_agg = torch.stack(s_k_list).sum(dim=0).clamp(max=1.0)
        s_v_agg = torch.stack(s_v_list).sum(dim=0).clamp(max=1.0)

        # --- ゼロスパイク問題への対処 (オプション) ---
        if self.add_noise_if_silent:
            # バッチ内で全くスパイクしなかったサンプルを特定 (QとKの両方がゼロの場合にノイズ注入)
            q_silent_samples = torch.all(s_q_agg == 0, dim=(-1,-2)) # (B,) boolean
            k_silent_samples = torch.all(s_k_agg == 0, dim=(-1,-2))
            silent_mask = q_silent_samples & k_silent_samples

            if silent_mask.any():
                num_silent = silent_mask.sum().item()
                logging.debug(f"Injecting noise into Q and K for {num_silent} silent samples.")
                # ノイズ（ランダムなスパイク）を生成
                noise_q = torch.bernoulli(torch.full_like(s_q_agg[silent_mask], self.noise_prob))
                noise_k = torch.bernoulli(torch.full_like(s_k_agg[silent_mask], self.noise_prob))
                # 元のスパイク(ゼロ)とノイズの最大値を取る（OR演算に相当）
                s_q_agg[silent_mask] = torch.max(s_q_agg[silent_mask], noise_q)
                s_k_agg[silent_mask] = torch.max(s_k_agg[silent_mask], noise_k)
                # Vにも同様にノイズを加えるか検討 (ここでは加えない)
        # --- ここまでゼロスパイク対処 ---

        # ヘッドに分割
        s_q = s_q_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)
        s_k = s_k_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)
        s_v = s_v_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)

        # --- SDSA コア計算 (A = K⊙V, Out = A×⃝Q) ---
        # ⊙ は要素ごとの積 (バイナリなのでAND演算)
        # ×⃝ も要素ごとの積 (バイナリなのでAND演算)
        # 論文によっては A = K^T @ V のような行列積を使う場合もあるが、ここでは要素積
        # ⚠️ 注意: この単純な要素積ではトークン間の相互作用が限定的になる可能性がある
        a = s_k * s_v # (B, H, N, Dh)
        attention_out = a * s_q # (B, H, N, Dh)
        # --- ここまで ---

        # ヘッドを結合
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(B, N, C)

        # 出力変換
        out = self.to_out(attention_out)

        return out

    def reset(self):
        """ニューロンの状態をリセット"""
        if hasattr(self.lif_q, 'reset'):
            self.lif_q.reset()
        if hasattr(self.lif_k, 'reset'):
            self.lif_k.reset()
        if hasattr(self.lif_v, 'reset'):
            self.lif_v.reset()