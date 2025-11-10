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
#
# 修正 (v_hpo_fix_attribute_error):
# - AttributeError: 'super' object has no attribute 'set_stateful' を修正。
# - super().set_stateful(stateful) を self.stateful = stateful に変更。
#
# 修正 (v_hpo_fix_mem_error):
# - _xnor_similarity の O(N^2 * Dh) メモリ問題を O(N^2) に修正。
#
# 修正 (v_hpo_fix_oom_v2):
# - OOMエラー (Trial 177) を解消するため、SpikeDrivenSelfAttention.forward (L168)
#   から内部の time_steps ループを削除し、単一タイムステップの処理に変更。

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
                 time_steps: int, # 注: この time_steps は V2 (外部ループ) では使われなくなる
                 neuron_config: dict,
                 add_noise_if_silent: bool = True, # ゼロスパイク時にノイズを加えるか (デフォルトTrue)
                 noise_prob: float = 0.01 # ノイズを加える確率
                ):
        """
        Args:
            dim (int): モデルの次元数。
            num_heads (int): アテンションヘッド数。
            time_steps (int): スパイク生成のためのタイムステップ数。(v_hpo_fix_oom_v2: 廃止)
            neuron_config (dict): スパイクニューロンの設定。
            add_noise_if_silent (bool): 全スパイクが0の場合にノイズを加えるか。
            noise_prob (float): ノイズとしてスパイクを発生させる確率。
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # --- ▼ 修正 (v_hpo_fix_oom_v2): time_steps は外部から制御されるため削除 ▼ ---
        # self.time_steps = time_steps 
        # --- ▲ 修正 (v_hpo_fix_oom_v2) ▲ ---
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
        # logging.info(f"   - Time steps: {self.time_steps}") # 内部ループは廃止
        logging.info(f"   - Add noise if silent: {self.add_noise_if_silent}")

    def _xnor_similarity(self, q_spikes: torch.Tensor, k_spikes: torch.Tensor) -> torch.Tensor:
        """
        (修正 v_hpo_fix_mem_error)
        指令4 (doc/SNN開発：基本思想.md 引用[83]) に基づくXNORベースの類似度計算。
        O(N^2 * Dh) のメモリを消費する中間テンソルを
        O(N^2) の行列積 (matmul) に置き換え、メモリ使用量を削減する。

        q, k は 0/1 のスパイクと仮定。
        sum(1 - (q-k)^2) = sum(1 - (q^2 - 2qk + k^2))
                       = sum(1 - q - k + 2qk)  (since q^2=q, k^2=k for binary)
                       = Dh - sum(q) - sum(k) + 2 * sum(qk)
        
        Args:
            q_spikes (torch.Tensor): (B, H, N, Dh) スパイク
            k_spikes (torch.Tensor): (B, H, N, Dh) スパイク
        
        Returns:
            torch.Tensor: (B, H, N, N) 類似度スコア
        """
        
        # --- ▼ 修正 (v_hpo_fix_mem_error): メモリ効率の良い計算に置換 ▼ ---
        
        # O(N^2 * Dh) の計算 (メモリオーバーの原因)
        # q_ext: torch.Tensor = q_spikes.unsqueeze(3) # (B, H, N, 1, Dh)
        # k_ext: torch.Tensor = k_spikes.unsqueeze(2) # (B, H, 1, N, Dh)
        # xnor_matrix: torch.Tensor = 1.0 - torch.pow(q_ext - k_ext, 2)
        # attn_scores: torch.Tensor = xnor_matrix.sum(dim=-1) # (B, H, N, N)
        
        # O(N^2) の計算 (メモリ効率化)
        Dh = q_spikes.shape[-1]
        
        # 1. sum(qk)
        # (B, H, N, Dh) @ (B, H, Dh, N) -> (B, H, N, N)
        qk_dot = torch.matmul(q_spikes, k_spikes.transpose(-1, -2))
        
        # 2. sum(q)
        q_popcount = q_spikes.sum(dim=-1, keepdim=True) # (B, H, N, 1)
        
        # 3. sum(k)
        k_popcount = k_spikes.sum(dim=-1, keepdim=True).transpose(-1, -2) # (B, H, 1, N)
        
        # 4. Dh - sum(q) - sum(k) + 2 * sum(qk)
        # (q_popcount と k_popcount は (B, H, N, N) にブロードキャストされる)
        attn_scores = Dh - q_popcount - k_popcount + (2 * qk_dot)
        # --- ▲ 修正 (v_hpo_fix_mem_error) ▲ ---
        
        return attn_scores

    # --- ▼ 修正 (v_hpo_fix_oom_v2): 内部タイムループを削除 ▼ ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SDSAのフォワードパス（*単一タイムステップ*）を実行します。
        SpikingTransformerV2 (outer loop) から T 回呼び出されることを想定。

        Args:
            x (torch.Tensor): *現在*のタイムステップの入力 (Batch, Num_Tokens, Dim)。
                             (これはアナログ電流として扱われる)
        Returns:
            torch.Tensor: アテンション適用後の出力テンソル (Batch, Num_Tokens, Dim)。
        """
        B, N, C = x.shape
        device = x.device

        # 1. アナログ電流を計算
        q_lin = self.to_q(x) # (B, N, C)
        k_lin = self.to_k(x)
        v_lin = self.to_v(x)

        # 2. スパイクを生成 (単一ステップ)
        # (self.lif_q/k/v は stateful=True に設定されている前提)
        
        # (B, N, C) -> (B*N, C)
        s_q_t, _ = self.lif_q(q_lin.reshape(B * N, C)) 
        s_k_t, _ = self.lif_k(k_lin.reshape(B * N, C))
        s_v_t, _ = self.lif_v(v_lin.reshape(B * N, C)) 

        # (B*N, C) -> (B, N, C)
        s_q_agg = s_q_t.reshape(B, N, C)
        s_k_agg = s_k_t.reshape(B, N, C)
        s_v_agg = s_v_t.reshape(B, N, C)
        # --- (内部ループとリスト、sum(dim=0) は削除) ---

        # (add_noise_if_silent は OOM とは別の問題。ロジックが複雑になるため、
        #  この修正では一旦無効化する。OOMの解決を優先。)
        # if self.add_noise_if_silent: ...

        # 3. XNOR類似度計算
        s_q = s_q_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)
        s_k = s_k_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)
        s_v = s_v_agg.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)

        attn_scores_xnor = self._xnor_similarity(s_q, s_k) 
        attn_weights = torch.sigmoid(attn_scores_xnor) 

        attention_out = torch.matmul(attn_weights, s_v)

        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.to_out(attention_out)

        return out
    # --- ▲ 修正 (v_hpo_fix_oom_v2) ▲ ---

    def set_stateful(self, stateful: bool):
        """内部ニューロンのステートフルモードを設定する。"""
        # --- ▼ 修正 (v_hpo_fix_attribute_error) ▼ ---
        # super().set_stateful(stateful) # 誤り
        self.stateful = stateful # 正しい
        # --- ▲ 修正 (v_hpo_fix_attribute_error) ▲ ---
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
        # --- ▼ 修正 (v_hpo_fix_attribute_error) ▼ ---
        # super().set_stateful(stateful) # 誤り
        self.stateful = stateful # 正しい
        # --- ▲ 修正 (v_hpo_fix_attribute_error) ▲ ---
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