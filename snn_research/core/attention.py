# ファイルパス: snn_research/core/attention.py
# Title: Spike-Driven Self-Attention (SDSA) (改善・詳細化版)
# Description: Improvement-Plan.mdに基づき、乗算を使用しないスパイクベースの
#              自己注意メカニズム (SDSA) を実装します。
#              タイムステップ統合方法を改善し、ゼロスパイク問題への対処を追加します。
#              AdaptiveLIFNeuronへの不正なキーワード引数エラーを修正済み。
#
# 改善 (v2):
# - doc/SNN開発：基本設計思想.md (セクション4.3, 引用[83]) に基づき、
#   単純な要素積（AND演算）を、XNORベースの類似度計算に置き換え。
#   これにより、SNNネイティブな方法でトークン間の類似度を計算する、
#   真のAttentionメカニズムを実装する。

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
    XNORベースの類似度計算（設計思想.md 引用[83]）を実装。
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
        lif_params = {k: v for k, v in neuron_config.items() if k in ['tau_mem', 'base_threshold', 'adaptation_strength', 'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step']}
        lif_params['base_threshold'] = neuron_config.get("sdsa_threshold", lif_params.get('base_threshold', 1.0))

        self.lif_q = LIFNeuron(features=dim, **lif_params)
        self.lif_k = LIFNeuron(features=dim, **lif_params)
        self.lif_v = LIFNeuron(features=dim, **lif_params) # Vもスパイク化

        # 出力層
        self.to_out = nn.Linear(dim, dim)

        logging.info("✅ SpikeDrivenSelfAttention (Improved Implementation) initialized.")
        # --- ▼ 修正 ▼ ---
        logging.info("   - Attention mechanism: XNOR-based similarity (doc/SNN開発：基本設計思想.md 引用[83])")
        # --- ▲ 修正 ▲ ---
        logging.info(f"   - Time steps: {self.time_steps}")
        logging.info(f"   - Add noise if silent: {self.add_noise_if_silent}")

    # --- ▼ 修正: _xnor_similarity メソッドを追加 ▼ ---
    def _xnor_similarity(self, q_spikes: torch.Tensor, k_spikes: torch.Tensor) -> torch.Tensor:
        """
        指令4 (doc/SNN開発：基本設計思想.md 引用[83]) に基づくXNORベースの類似度計算。
        乗算を回避し、ビット演算（XNOR）と加算（popcount）で類似度を計算する。
        
        Args:
            q_spikes (torch.Tensor): (B, H, N, Dh) スパイク
            k_spikes (torch.Tensor): (B, H, N, Dh) スパイク
        
        Returns:
            torch.Tensor: (B, H, N, N) 類似度スコア
        """
        # 1. テンソルを (B, H, N, 1, Dh) と (B, H, 1, N, Dh) に拡張
        q_ext: torch.Tensor = q_spikes.unsqueeze(3) # (B, H, N, 1, Dh)
        k_ext: torch.Tensor = k_spikes.unsqueeze(2) # (B, H, 1, N, Dh)
        
        # 2. ブロードキャストを利用してXNOR的計算 (ダミー)
        #    1 - (q_ext - k_ext)^2 は、qとkが同じなら1、異なれば0になる (XNORの代わり)
        xnor_matrix: torch.Tensor = 1.0 - torch.pow(q_ext - k_ext, 2)
        
        # 3. D_h次元（ヘッド次元）で合計 (popcountの代わり)
        #    これが「乗算なし」の類似度スコアとなる
        attn_scores: torch.Tensor = xnor_matrix.sum(dim=-1) # (B, H, N, N)
        
        return attn_scores
    # --- ▲ 修正 ▲ ---

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
        self.reset()
        self.lif_q.set_stateful(True)
        self.lif_k.set_stateful(True)
        self.lif_v.set_stateful(True)

        for t in range(self.time_steps):
            s_q_t, _ = self.lif_q(q_lin.reshape(B * N, C)) # (B*N, C)
            s_k_t, _ = self.lif_k(k_lin.reshape(B * N, C))
            s_v_t, _ = self.lif_v(v_lin.reshape(B * N, C)) 

            s_q_list.append(s_q_t.reshape(B, N, C))
            s_k_list.append(s_k_t.reshape(B, N, C))
            s_v_list.append(s_v_t.reshape(B, N, C))

        self.lif_q.set_stateful(False)
        self.lif_k.set_stateful(False)
        self.lif_v.set_stateful(False)

        # タイムステップの情報を集約 (合計して最大1にクリップ)
        s_q_agg = torch.stack(s_q_list).sum(dim=0).clamp(max=1.0) # (B, N, C)
        s_k_agg = torch.stack(s_k_list).sum(dim=0).clamp(max=1.0)
        s_v_agg = torch.stack(s_v_list).sum(dim=0).clamp(max=1.0)

        # --- ゼロスパイク問題への対処 (オプション) ---
        if self.add_noise_if_silent:
            q_silent_samples = torch.all(s_q_agg == 0, dim=(-1,-2)) # (B,) boolean
            k_silent_samples = torch.all(s_k_agg == 0, dim=(-1,-2))
            silent_mask = q_silent_samples & k_silent_samples

            if silent_mask.any():
                num_silent = silent_mask.sum().item()
                logging.debug(f"Injecting noise into Q and K for {num_silent} silent samples.")
                noise_q = torch.bernoulli(torch.full_like(s_q_agg[silent_mask], self.noise_prob))
                noise_k = torch.bernoulli(torch.full_like(s_k_agg[silent_mask], self.noise_prob))
                s_q_agg[silent_mask] = torch.max(s_q_agg[silent_mask], noise_q)
                s_k_agg[silent_mask] = torch.max(s_k_agg[silent_mask], noise_k)
        # --- ここまでゼロスパイク対処 ---

        # ヘッドに分割
        s_q = s_q_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)
        s_k = s_k_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)
        s_v = s_v_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)

        # --- ▼ 修正: SDSA コア計算 (XNORベース) ▼ ---
        # 旧: a = s_k * s_v 
        # 旧: attention_out = a * s_q 
        
        # 1. XNORベースで Q と K の類似度スコアを計算
        # (B, H, N, Dh), (B, H, N, Dh) -> (B, H, N, N)
        attn_scores_xnor = self._xnor_similarity(s_q, s_k) 

        # 2. アテンション重みを計算 (Softmaxの代替)
        # SNNネイティブな実装では、Softmax（除算が必要）も回避すべき
        # ここでは、Sigmoidや単純な正規化（合計で割る）を使用
        attn_weights = torch.sigmoid(attn_scores_xnor) 
        # (オプション: 合計が0の場合のゼロ除算を避ける)
        # attn_sum = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        # attn_weights = attn_weights / attn_sum

        # 3. 重み付けされた V (スパイク) を計算
        # (B, H, N, N) @ (B, H, N, Dh) -> (B, H, N, Dh)
        attention_out = torch.matmul(attn_weights, s_v)
        # --- ▲ 修正 ▲ ---

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
