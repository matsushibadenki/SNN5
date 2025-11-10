# ファイルパス: snn_research/architectures/spiking_moe.py
# (新規作成)
#
# Title: Spiking Mixture of Experts (SEMM)
#
# Description:
# doc/ROADMAP.md (P1.6 最重要) および doc/レポート：ANN-SNN変換情報.md (IV.E) に基づき、
# NIPS 2024で発表された「SEMM (Spiking Experts Mixture Mechanism)」[30]を実装します。
#
# このアーキテクチャは、Llama 4のMoE構造をSNNドメインに持ち込むものです。
# Softmax/TopKを用いた高コストなルーティングの代わりに、
# SNNのスパースな発火（イベント駆動）そのものを利用した、
# 乗算不要の効率的なルーティング（アダマール積によるANDゲート）を実現します。
#
# mypy --strict 準拠。
#
# 修正 (v_fix_mypy_name_defined): [name-defined] エラーを解消するため、SpikeDrivenSelfAttention をインポート。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, Optional, List, cast, Union
import math

# SNNのコアコンポーネントをインポート
from snn_research.core.base import BaseModel, SNNLayerNorm
from snn_research.core.neurons import AdaptiveLIFNeuron
# --- ▼ 修正: SpikeDrivenSelfAttention をインポート ▼ ---
from snn_research.core.attention import SpikeDrivenSelfAttention
# --- ▲ 修正 ▲ ---
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]
from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]

import logging

logger = logging.getLogger(__name__)

class SpikingMLP(sj_base.MemoryModule):
    """
    SNN-MoE内の各エキスパートとして機能する、シンプルなSpiking MLP (FFN)。
    (Conv1D -> LIF -> Conv1D)
    """
    lif1: AdaptiveLIFNeuron
    lif2: AdaptiveLIFNeuron

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        neuron_class: Type[AdaptiveLIFNeuron],
        neuron_params: Dict[str, Any]
    ) -> None:
        super().__init__()
        
        # 1D畳み込み層 (Linear層の代替として時間情報を考慮)
        self.fc1 = nn.Conv1d(d_model, d_ffn, kernel_size=1)
        self.lif1 = neuron_class(features=d_ffn, **neuron_params)
        self.fc2 = nn.Conv1d(d_ffn, d_model, kernel_size=1)
        self.lif2 = neuron_class(features=d_model, **neuron_params)

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        self.lif1.set_stateful(stateful)
        self.lif2.set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        self.lif1.reset()
        self.lif2.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1タイムステップ分の処理 (RNNモード)。
        Args:
            x (torch.Tensor): 入力 (B, D_model)
        Returns:
            torch.Tensor: 出力 (B, D_model)
        """
        # (B, D_model) -> (B, D_model, 1) (Conv1Dのため)
        x = x.unsqueeze(-1)
        
        x, _ = self.lif1(self.fc1(x))
        x, _ = self.lif2(self.fc2(x))
        
        # (B, D_model, 1) -> (B, D_model)
        return x.squeeze(-1)

class SEMMRouter(sj_base.MemoryModule):
    """
    SEMM [30] のスパイクベース・ルーター。
    入力トークン (x_t) と各エキスパート (e_j) の両方のスパイクに基づき、
    乗算不要のルーティング（ANDゲート）を行う。
    """
    router_lif: AdaptiveLIFNeuron
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        neuron_class: Type[AdaptiveLIFNeuron],
        neuron_params: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        
        # ルーターゲート (入力トークン x_t を K次元のゲート値に射影)
        self.gate = nn.Linear(d_model, num_experts)
        self.router_lif = neuron_class(features=num_experts, **neuron_params)

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        self.router_lif.set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        self.router_lif.reset()

    def forward(
        self, 
        x_t: torch.Tensor, 
        expert_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        SEMMのルーティングを実行 (1タイムステップ)。
        
        Args:
            x_t (torch.Tensor): 現在の入力 (B, D_model)
            expert_spikes (torch.Tensor): 全エキスパートの *前ステップ* の出力スパイク (B, E, D_model)
                                          (SEMM [30] の式(3)に基づくフィードバック)
        
        Returns:
            torch.Tensor: ルーティング重み (B, E, 1) (スパイク)
        """
        B = x_t.shape[0]
        
        # 1. 論文[30] 式(3) 右側: (W_g * x_t)
        gate_current = self.gate(x_t) # (B, E)
        
        # 2. 論文[30] 式(3) 左側: (W_e * S_{j, t-1})
        # (B, E, D_model) -> (B, E) (各エキスパートの活動量を集約)
        expert_feedback = expert_spikes.mean(dim=-1) 
        
        # 3. ルーターLIFへの総入力
        router_input = gate_current + expert_feedback
        
        # 4. ルータースパイク (R_t) の生成
        # R_t (B, E)
        router_spikes_t, _ = self.router_lif(router_input)
        
        # (B, E) -> (B, E, 1) (ブロードキャストのため)
        return router_spikes_t.unsqueeze(-1)


class SpikingMoEBlock(sj_base.MemoryModule):
    """
    Spiking Mixture of Experts (SEMM) ブロック (1層分)。
    """
    router: SEMMRouter
    experts: nn.ModuleList
    norm: SNNLayerNorm
    
    # 前ステップのエキスパートスパイクを保持するバッファ
    last_expert_spikes: torch.Tensor

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        d_ffn: int, # エキスパート内部の次元
        neuron_class: Type[AdaptiveLIFNeuron],
        neuron_params: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        # 1. SEMM ルーター
        self.router = SEMMRouter(d_model, num_experts, neuron_class, neuron_params)
        
        # 2. エキスパート (SpikingMLP) のリスト
        self.experts = nn.ModuleList(
            [SpikingMLP(d_model, d_ffn, neuron_class, neuron_params) for _ in range(num_experts)]
        )
        
        self.norm = SNNLayerNorm(d_model)
        
        # 状態バッファ
        self.register_buffer("last_expert_spikes", torch.zeros(1, num_experts, d_model))

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        self.router.set_stateful(stateful)
        for expert in self.experts:
            cast(SpikingMLP, expert).set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        self.router.reset()
        for expert in self.experts:
            cast(SpikingMLP, expert).reset()
        self.last_expert_spikes.zero_()

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        1タイムステップ分の処理 (RNNモード)。
        
        Args:
            x_t (torch.Tensor): 現在の入力 (B, D_model)
            
        Returns:
            torch.Tensor: ブロックの出力 (B, D_model)
        """
        B = x_t.shape[0]
        
        # バッファのデバイスとバッチサイズを合わせる
        if self.last_expert_spikes.shape[0] != B or self.last_expert_spikes.device != x_t.device:
            self.last_expert_spikes = torch.zeros(B, self.num_experts, self.d_model, device=x_t.device)
            
        # 1. ルーティング (論文[30] 式(3))
        # R_t (B, E, 1)
        routing_spikes_t = self.router(x_t, self.last_expert_spikes)
        
        # 2. エキスパートの並列実行
        expert_outputs: List[torch.Tensor] = []
        for expert in self.experts:
            # 各エキスパートは (B, D_model) の入力を受け取る
            expert_out_t = cast(SpikingMLP, expert)(x_t) # (B, D_model)
            expert_outputs.append(expert_out_t)
            
        # (E, B, D_model) -> (B, E, D_model)
        all_expert_spikes_t = torch.stack(expert_outputs, dim=1)
        
        # 3. 乗算不要のミキシング (論文[30] 式(2))
        # (B, E, 1) * (B, E, D_model) -> (B, E, D_model)
        # これがアダマール積 (ANDゲート) に相当
        gated_expert_outputs = routing_spikes_t * all_expert_spikes_t
        
        # 4. 出力の集約 (エキスパート間で合計)
        # (B, E, D_model) -> (B, D_model)
        y_t = gated_expert_outputs.sum(dim=1)
        
        # 5. 残差接続
        out_t = self.norm(x_t + y_t)
        
        # 6. 次のステップのためにエキスパートの出力をバッファ
        self.last_expert_spikes = all_expert_spikes_t.detach()
        
        return out_t


class SpikingMoETransformer(BaseModel):
    """
    Spiking Transformer (v2) のFFN層を SpikingMoEBlock に置き換えたモデル。
    ロードマップ (P1.6) のための実装。
    """
    embedding: nn.Embedding
    pos_encoder: nn.Parameter
    layers: nn.ModuleList
    final_norm: SNNLayerNorm
    output_projection: nn.Linear
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        time_steps: int, # SNNの内部ステップ (このモデルではシーケンス長として使用)
        neuron_config: Dict[str, Any],
        # MoE パラメータ
        num_experts: int = 8,
        d_ffn_expert: int = 1024, # 各エキスパートの内部次元
        # Transformer パラメータ (SDSA)
        nhead: int = 8,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.time_steps = time_steps # T_seq (シーケンス長)
        
        neuron_type_str: str = neuron_config.get("type", "lif")
        neuron_params: Dict[str, Any] = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Type[AdaptiveLIFNeuron] = AdaptiveLIFNeuron
        if neuron_type_str != 'lif':
            # この実装は簡略化のため AdaptiveLIFNeuron のみをサポート
            logger.warning(f"SpikingMoE currently only supports AdaptiveLIFNeuron, defaulting to it.")
        
        filtered_params: Dict[str, Any] = {
            k: v for k, v in neuron_params.items() 
            if k in ['tau_mem', 'base_threshold']
        }

        # 1. 埋め込み層
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.time_steps, d_model))
        
        # 2. レイヤー (SDSA + SpikingMoE の交互配置)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Attention (SDSA)
            self.layers.append(
                SpikeDrivenSelfAttention(d_model, nhead, time_steps, neuron_config)
            )
            # MoE Block
            self.layers.append(
                SpikingMoEBlock(d_model, num_experts, d_ffn_expert, neuron_class, filtered_params)
            )
            
        self.final_norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()
        logger.info(f"✅ SpikingMoETransformer (SEMM) [P1.6] initialized. (Layers: {num_layers}, Experts: {num_experts})")

    def forward(
        self, 
        input_ids: torch.Tensor, # (B, T_seq)
        return_spikes: bool = False, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, T_seq = input_ids.shape
        device: torch.device = input_ids.device
        
        # (T_seq が time_steps と異なる場合に対応)
        if T_seq != self.time_steps:
            logger.warning(f"Input T_seq ({T_seq}) differs from model time_steps ({self.time_steps}). Truncating/Padding.")
            if T_seq > self.time_steps:
                input_ids = input_ids[:, :self.time_steps]
            else: # T_seq < self.time_steps
                padding = torch.full((B, self.time_steps - T_seq), self.embedding.padding_idx or 0, device=device, dtype=torch.long)
                input_ids = torch.cat([input_ids, padding], dim=1)
            T_seq = self.time_steps
        
        # 状態リセット
        SJ_F.reset_net(self)
        
        # 1. Analog Embedding
        x: torch.Tensor = self.embedding(input_ids) # (B, T_seq, D_model)
        x = x + self.pos_encoder[:, :T_seq, :]
        
        outputs: List[torch.Tensor] = []
        
        # --- シーケンス長 (T_seq) でループ (RNNモード) ---
        
        # 各レイヤーのLIFをStatefulに設定
        for layer_module in self.layers:
            cast(sj_base.MemoryModule, layer_module).set_stateful(True)

        for t_idx in range(T_seq):
            x_t: torch.Tensor = x[:, t_idx, :] # (B, D_model)
            
            # --- レイヤー (SDSA, MoE) でループ ---
            for layer_module in self.layers:
                x_t = cast(nn.Module, layer_module)(x_t) # (B, D_model)
            
            outputs.append(x_t) # 最終層の出力
        
        # 各レイヤーのLIFをStatelessに戻す
        for layer_module in self.layers:
            cast(sj_base.MemoryModule, layer_module).set_stateful(False)

        # (B, T_seq, D_model)
        x_final_seq: torch.Tensor = torch.stack(outputs, dim=1)
        
        x_norm_final: torch.Tensor = self.final_norm(x_final_seq)
        logits: torch.Tensor = self.output_projection(x_norm_final)
        
        # --- 互換性のため (logits, avg_spikes, mem) を返す ---
        avg_spikes_val: float = self.get_total_spikes() / (B * T_seq) if return_spikes and T_seq > 0 else 0.0
        avg_spikes: torch.Tensor = torch.tensor(avg_spikes_val, device=device)
        mem: torch.Tensor = torch.tensor(0.0, device=device) 

        return logits, avg_spikes, mem
