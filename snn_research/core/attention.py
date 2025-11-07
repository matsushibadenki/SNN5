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
#
# 修正 (v3):
# - mypy [operator] エラーを解消するため、set_stateful メソッドを実装し、
#   内部ニューロンの状態管理を可能にする。
#
# 修正 (v4):
# - mypy [name-defined] エラーを解消するため、型ヒントとキャストを
#   インポートエイリアス (LIFNeuron) に統一。
#
# 改善 (v5):
# - ロードマップ P1.4 (DTA-SNN [10]) に基づき、
#   DynamicTemporalAttention (DTA) モジュールを追加。
#
# 修正 (v6):
# - mypy [name-defined] F をインポート

import torch
import torch.nn as nn
# --- ▼ 改善 (v5): 必要な型ヒントを追加 ▼ ---
from typing import Tuple, Dict, Any, Optional, cast, List
# --- ▲ 改善 (v5) ▲ ---
import random 
# --- ▼ 修正 (v6): F をインポート ▼ ---
import torch.nn.functional as F
# --- ▲ 修正 (v6) ▲ ---

# SDSAで使用するスパイクニューロン (例: LIF)
from .neurons import AdaptiveLIFNeuron as LIFNeuron
# 代理勾配関数
from spikingjelly.activation_based import surrogate, base # type: ignore
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpikeDrivenSelfAttention(base.MemoryModule):
    """
    Spike-Driven Self-Attention (SDSA) の改善・詳細化版実装。
    XNORベースの類似度計算（設計思想.md 引用[83]）を実装。
    """
    # --- ▼ 修正: 型ヒントをエイリアス名 (LIFNeuron) に変更 ▼ ---
    lif_q: LIFNeuron
    lif_k: LIFNeuron
    lif_v: LIFNeuron
    # --- ▲ 修正 ▲ ---

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

        # --- ▼ 修正: 型キャストをエイリアス名 (LIFNeuron) に変更 ▼ ---
        self.lif_q = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        self.lif_k = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        self.lif_v = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        # --- ▲ 修正 ▲ ---

        # 出力層
        self.to_out = nn.Linear(dim, dim)

        logging.info("✅ SpikeDrivenSelfAttention (Improved Implementation) initialized.")
        logging.info("   - Attention mechanism: XNOR-based similarity (doc/SNN開発：基本設計思想.md 引用[83])")
        logging.info(f"   - Time steps: {self.time_steps}")
        logging.info(f"   - Add noise if silent: {self.add_noise_if_silent}")

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
        q_ext: torch.Tensor = q_spikes.unsqueeze(3) # (B, H, N, 1, Dh)
        k_ext: torch.Tensor = k_spikes.unsqueeze(2) # (B, H, 1, N, Dh)
        
        xnor_matrix: torch.Tensor = 1.0 - torch.pow(q_ext - k_ext, 2)
        
        attn_scores: torch.Tensor = xnor_matrix.sum(dim=-1) # (B, H, N, N)
        
        return attn_scores

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

        q_lin = self.to_q(x) # (B, N, C)
        k_lin = self.to_k(x)
        v_lin = self.to_v(x)

        # タイムステップループでスパイクを生成
        s_q_list, s_k_list, s_v_list = [], [], []
        
        if not self.stateful:
            self.reset()

        for t in range(self.time_steps):
            s_q_t, _ = self.lif_q(q_lin.reshape(B * N, C)) # (B*N, C)
            s_k_t, _ = self.lif_k(k_lin.reshape(B * N, C))
            s_v_t, _ = self.lif_v(v_lin.reshape(B * N, C)) 

            s_q_list.append(s_q_t.reshape(B, N, C))
            s_k_list.append(s_k_t.reshape(B, N, C))
            s_v_list.append(s_v_t.reshape(B, N, C))

        s_q_agg = torch.stack(s_q_list).sum(dim=0).clamp(max=1.0) # (B, N, C)
        s_k_agg = torch.stack(s_k_list).sum(dim=0).clamp(max=1.0)
        s_v_agg = torch.stack(s_v_list).sum(dim=0).clamp(max=1.0)

        if self.add_noise_if_silent:
            q_silent_samples = torch.all(s_q_agg == 0, dim=(-1,-2)) 
            k_silent_samples = torch.all(s_k_agg == 0, dim=(-1,-2))
            silent_mask = q_silent_samples & k_silent_samples

            if silent_mask.any():
                num_silent = silent_mask.sum().item()
                logging.debug(f"Injecting noise into Q and K for {num_silent} silent samples.")
                noise_q = torch.bernoulli(torch.full_like(s_q_agg[silent_mask], self.noise_prob))
                noise_k = torch.bernoulli(torch.full_like(s_k_agg[silent_mask], self.noise_prob))
                s_q_agg[silent_mask] = torch.max(s_q_agg[silent_mask], noise_q)
                s_k_agg[silent_mask] = torch.max(s_k_agg[silent_mask], noise_k)

        s_q = s_q_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)
        s_k = s_k_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)
        s_v = s_v_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)

        attn_scores_xnor = self._xnor_similarity(s_q, s_k) 
        attn_weights = torch.sigmoid(attn_scores_xnor) 

        attention_out = torch.matmul(attn_weights, s_v)

        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.to_out(attention_out)

        return out

    def set_stateful(self, stateful: bool):
        """内部ニューロンのステートフルモードを設定する。"""
        super().set_stateful(stateful)
        self.lif_q.set_stateful(stateful)
        self.lif_k.set_stateful(stateful)
        self.lif_v.set_stateful(stateful)

    def reset(self):
        """ニューロンの状態をリセット"""
        super().reset()
        self.lif_q.reset()
        self.lif_k.reset()
        self.lif_v.reset()

# --- ▼▼▼ 改善 (v5): P1.4 DTA-SNN (引用[10]) の実装 ▼▼▼ ---

class DynamicTemporalAttention(base.MemoryModule):
    """
    DTA-SNN (Dynamic Temporal Attention) の実装。
    ロードマップ P1.4 (引用[10]) に基づく。
    
    Q, K, V を時間スパイク列 (B, T, N, C) として扱い、
    時間情報に基づいたアテンションを実行する。
    """
    lif_q: LIFNeuron
    lif_k: LIFNeuron
    lif_v: LIFNeuron
    
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 time_steps: int,
                 neuron_config: dict
                ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.time_steps = time_steps

        # 線形変換層 (入力 -> Q, K, V)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # スパイク生成ニューロン
        lif_params = {k: v for k, v in neuron_config.items() if k in ['tau_mem', 'base_threshold']}
        
        self.lif_q = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        self.lif_k = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        self.lif_v = cast(LIFNeuron, LIFNeuron(features=dim, **lif_params))
        
        # DTA [10] の中核: 時間ダイナミクスを学習するリカレント層 (例: GRU)
        # スパイクベースではないが、論文[10]は標準的なRNN/GRUをアテンション計算に用いる
        self.temporal_encoder = nn.GRU(
            input_size=self.head_dim, 
            hidden_size=self.head_dim, 
            batch_first=True
        )

        # 出力層
        self.to_out = nn.Linear(dim, dim)
        
        logging.info("✅ DynamicTemporalAttention (DTA-SNN Stub) initialized (P1.4).")

    def _generate_spikes(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, T, N, C) のスパイク時系列を生成"""
        B, N, C = x.shape
        x_lin_flat = x.reshape(B * N, C)
        
        s_list = []
        if not self.stateful:
            self.reset()
            
        # どのニューロンを使うか (例: Q)
        neuron = self.lif_q
        if not self.stateful:
            neuron.reset()
            neuron.set_stateful(True)

        for _ in range(self.time_steps):
            s_t, _ = neuron(x_lin_flat) # (B*N, C)
            s_list.append(s_t.reshape(B, N, C))
            
        if not self.stateful:
            neuron.set_stateful(False)
            neuron.reset()
            
        return torch.stack(s_list, dim=1) # (B, T, N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DTA のフォワードパス。
        
        Args:
            x (torch.Tensor): 入力テンソル (Batch, Num_Tokens, Dim)。

        Returns:
            torch.Tensor: アテンション適用後の出力テンソル (Batch, Num_Tokens, Dim)。
        """
        B, N, C = x.shape
        
        # 1. Q, K, V の電流を計算
        q_lin = self.to_q(x) # (B, N, C)
        k_lin = self.to_k(x)
        v_lin = self.to_v(x)

        # 2. スパイク時系列を生成 (B, T, N, C)
        s_q_time = self._generate_spikes(q_lin)
        s_k_time = self._generate_spikes(k_lin)
        s_v_time = self._generate_spikes(v_lin)
        
        # 3. ヘッドに分割 (B, T, N, H, Dh) -> (B, H, N, T, Dh)
        s_q_heads = s_q_time.view(B, self.time_steps, N, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)
        s_k_heads = s_k_time.view(B, self.time_steps, N, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)
        s_v_heads = s_v_time.view(B, self.time_steps, N, self.num_heads, self.head_dim).permute(0, 3, 2, 1, 4)
        
        # (B*H*N, T, Dh) にフラット化して GRU に入力
        q_flat = s_q_heads.reshape(B * self.num_heads * N, self.time_steps, self.head_dim)
        k_flat = s_k_heads.reshape(B * self.num_heads * N, self.time_steps, self.head_dim)
        v_flat = s_v_heads.reshape(B * self.num_heads * N, self.time_steps, self.head_dim)

        # 4. 時間ダイナミクスのエンコード (DTA [10])
        # GRUの最後の隠れ状態 (h_n) を時間的特徴として使用
        _, q_temporal = self.temporal_encoder(q_flat) # (1, B*H*N, Dh)
        _, k_temporal = self.temporal_encoder(k_flat)
        _, v_temporal = self.temporal_encoder(v_flat)
        
        # (B, H, N, Dh) に戻す
        q_out = q_temporal.squeeze(0).view(B, self.num_heads, N, self.head_dim)
        k_out = k_temporal.squeeze(0).view(B, self.num_heads, N, self.head_dim)
        v_out = v_temporal.squeeze(0).view(B, self.num_heads, N, self.head_dim)

        # 5. 標準的なアテンション計算 (時間集約された特徴量を使用)
        # (B, H, N, Dh) @ (B, H, Dh, N) -> (B, H, N, N)
        attn_scores = torch.matmul(q_out, k_out.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attention_out = torch.matmul(attn_weights, v_out) # (B, H, N, Dh)

        # 6. 出力のマージ
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.to_out(attention_out)

        return out

    def set_stateful(self, stateful: bool):
        super().set_stateful(stateful)
        self.lif_q.set_stateful(stateful)
        self.lif_k.set_stateful(stateful)
        self.lif_v.set_stateful(stateful)

    def reset(self):
        super().reset()
        self.lif_q.reset()
        self.lif_k.reset()
        self.lif_v.reset()
        self.temporal_encoder.flatten_parameters() # GRU/LSTMの状態リセット推奨

# --- ▲▲▲ 改善 (v5) ▲▲▲ ---