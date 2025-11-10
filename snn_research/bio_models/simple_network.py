# ファイルパス: snn_research/bio_models/simple_network.py
# (修正)
# 修正: learning_rule.update がタプルを返すようになったため、
#       戻り値を正しくアンパックして使用する。
# 修正: CausalTraceCreditAssignmentEnhancedV2 に対応
#
# 改善 (v2):
# - doc/The-flow-of-brain-behavior.md および doc/プロジェクト強化案の調査.md (セクション2.3) に基づき、
#   単一の学習則しか持てなかった制約を解消。
# - シナプス可塑性ルール (synaptic_rule) と 恒常性維持ルール (homeostatic_rule) を
#   別々に受け取り、両方を適用できるように __init__ と update_weights を変更。
#
# 改善 (v3):
# - doc/ROADMAP.md (P8.2) および doc/The-flow-of-brain-behavior.md に基づき、
#   E/I分離（興奮性/抑制性ニューロンの分離）とデールの法則を実装する。
# - __init__:
#   - `layer_sizes: List[int]` を `layer_configs: List[Dict[str, int]]` に変更。
#     各層のEニューロン数 (n_e) と Iニューロン数 (n_i) を指定できるようにする。
#   - `weights` を `weights_ee`, `weights_ei`, `weights_ie`, `weights_ii` の
#     4つの nn.ParameterList に分割。
# - forward:
#   - デールの法則を適用。Eニューロンからの重みは `F.relu()` で正に、
#     Iニューロンからの重みは `-F.relu()` で負に制約する。
#   - E/I集団間の相互作用を計算するようロジックを修正。
# - update_weights:
#   - 学習則が4つの重み行列すべてに適用されるように修正。
#   - デールの法則の制約を学習後にも適用（clamp_weightsメソッド）。
#
# 修正 (v4):
# - mypy [name-defined] エラーを解消するため、cast をインポート。

import torch
import torch.nn as nn
# --- ▼ 修正 (v4): cast をインポート ▼ ---
from typing import Dict, Any, Optional, Tuple, List, cast
import torch.nn.functional as F
# --- ▲ 修正 (v4) ▲ ---

from .lif_neuron import BioLIFNeuron
from snn_research.learning_rules.base_rule import BioLearningRule
# --- ▼ 修正 ▼ ---
# V2 クラスをインポート
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignmentEnhancedV2
# --- ▲ 修正 ▲ ---


class BioSNN(nn.Module):
    """
    (改善 v3) E/I分離とデールの法則を実装した生物学的SNN (P8.2)。
    """
    def __init__(
        self, 
        # --- ▼ 修正 (v3): layer_sizes を layer_configs に変更 ▼ ---
        layer_configs: List[Dict[str, int]], # 例: [{"n_e": 80, "n_i": 20}, {"n_e": 50, "n_i": 10}]
        input_size: int, # 入力層のサイズは別途指定
        # --- ▲ 修正 (v3) ▲ ---
        neuron_params: dict, 
        synaptic_rule: BioLearningRule,
        homeostatic_rule: Optional[BioLearningRule] = None,
        sparsification_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.layer_configs = layer_configs
        self.input_size = input_size
        self.synaptic_rule = synaptic_rule
        self.homeostatic_rule = homeostatic_rule
        self.sparsification_enabled = sparsification_config.get("enabled", False) if sparsification_config else False
        self.contribution_threshold = sparsification_config.get("contribution_threshold", 0.0) if sparsification_config else 0.0
        if self.sparsification_enabled:
            print(f"🧬 適応的因果スパース化が有効です (貢献度閾値: {self.contribution_threshold})")
        if self.homeostatic_rule:
            print(f"⚖️ 恒常性維持ルール ({type(self.homeostatic_rule).__name__}) が有効です。")
        if not self.layer_configs:
             raise ValueError("layer_configs must not be empty.")

        self.layers_e = nn.ModuleList()
        self.layers_i = nn.ModuleList()
        
        # --- ▼ 修正 (v3): E/I分離した重み行列 ▼ ---
        self.weights_ee = nn.ParameterList()
        self.weights_ei = nn.ParameterList() # I -> E (抑制性)
        self.weights_ie = nn.ParameterList() # E -> I (興奮性)
        self.weights_ii = nn.ParameterList() # I -> I (抑制性)
        # --- ▲ 修正 (v3) ▲ ---

        current_input_dim_e = input_size
        current_input_dim_i = 0 # 入力層は抑制性を持たないと仮定

        for config in layer_configs:
            n_e = config["n_e"]
            n_i = config["n_i"]
            
            # ニューロン層の作成
            self.layers_e.append(BioLIFNeuron(n_e, neuron_params))
            if n_i > 0:
                self.layers_i.append(BioLIFNeuron(n_i, neuron_params))
            
            # --- 重み行列の作成 (デール則のため4分割) ---
            # E -> E
            self.weights_ee.append(nn.Parameter(torch.rand(n_e, current_input_dim_e) * 0.5))
            # E -> I
            if n_i > 0:
                self.weights_ie.append(nn.Parameter(torch.rand(n_i, current_input_dim_e) * 0.5))
            
            if current_input_dim_i > 0:
                # I -> E
                self.weights_ei.append(nn.Parameter(torch.rand(n_e, current_input_dim_i) * 0.5))
                # I -> I
                if n_i > 0:
                    self.weights_ii.append(nn.Parameter(torch.rand(n_i, current_input_dim_i) * 0.5))
            
            # 次の層の入力次元を更新
            current_input_dim_e = n_e
            current_input_dim_i = n_i
            
        print(f"✅ E/I分離型 BioSNN (P8.2) が {len(self.layers_e)} 層で構築されました。")


    def _apply_dale_law(self) -> None:
        """デールの法則（重みの符号制約）をインプレースで適用する。"""
        with torch.no_grad():
            for w in self.weights_ee: w.data = F.relu(w.data)
            for w in self.weights_ie: w.data = F.relu(w.data)
            # 抑制性ニューロンからの重みは負の値（の絶対値）として扱う
            for w in self.weights_ei: w.data = F.relu(w.data)
            for w in self.weights_ii: w.data = F.relu(w.data)

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        E/I分離モデルのフォワードパス。
        Args:
            input_spikes (torch.Tensor): 入力スパイク (N_input,)
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: 
                (最終層のEニューロンスパイク, 全層のE/Iスパイク履歴)
        """
        all_spikes_history: List[torch.Tensor] = [input_spikes]
        
        spikes_e_prev = input_spikes
        spikes_i_prev = None # 最初の層には抑制性入力はない

        # デールの法則（重みの符号制約）を適用
        self._apply_dale_law()

        for i in range(len(self.layers_e)):
            layer_e = cast(BioLIFNeuron, self.layers_e[i])
            layer_i: Optional[BioLIFNeuron] = None
            if i < len(self.layers_i):
                layer_i = cast(BioLIFNeuron, self.layers_i[i])
            
            # 1. 興奮性 (E) ニューロンへの電流を計算
            current_e = torch.matmul(self.weights_ee[i], spikes_e_prev) # E -> E (興奮性)
            
            if i > 0 and spikes_i_prev is not None and i-1 < len(self.weights_ei):
                # I -> E (抑制性)
                current_e -= torch.matmul(self.weights_ei[i-1], spikes_i_prev)
            
            spikes_e_t = layer_e(current_e) # (N_e,)
            spikes_i_t: Optional[torch.Tensor] = None
            
            # 2. 抑制性 (I) ニューロンへの電流を計算
            if layer_i is not None:
                current_i = torch.matmul(self.weights_ie[i], spikes_e_prev) # E -> I (興奮性)
                
                if i > 0 and spikes_i_prev is not None and i-1 < len(self.weights_ii):
                    # I -> I (抑制性)
                    current_i -= torch.matmul(self.weights_ii[i-1], spikes_i_prev)
                
                spikes_i_t = layer_i(current_i) # (N_i,)
                
                # 履歴に保存 (EとIを結合)
                all_spikes_history.append(torch.cat([spikes_e_t, spikes_i_t]))
                spikes_i_prev = spikes_i_t
            else:
                # 抑制性ニューロンがない層
                all_spikes_history.append(spikes_e_t)
                spikes_i_prev = None

            # 次のステップの入力
            spikes_e_prev = spikes_e_t

        return spikes_e_prev, all_spikes_history # 最終層の興奮性スパイクと全履歴を返す


    def update_weights(
        self,
        all_layer_spikes: List[torch.Tensor], # E/I結合済みスパイクのリスト
        optional_params: Optional[Dict[str, Any]] = None
    ):
        """(改善 v3) E/I分離モデルの重みを学習則に基づき更新する。"""
        if not self.training:
            return

        backward_credit_e: Optional[torch.Tensor] = None
        backward_credit_i: Optional[torch.Tensor] = None
        current_params = optional_params.copy() if optional_params else {}

        for i in reversed(range(len(self.layers_e))):
            # --- 1. スパイク履歴の分離 ---
            pre_spikes_all = all_layer_spikes[i]
            post_spikes_all = all_layer_spikes[i+1]
            
            n_e_pre: int
            n_i_pre: int = 0
            spikes_i_pre: Optional[torch.Tensor] = None
            
            if i == 0:
                n_e_pre = self.input_size
                spikes_e_pre = pre_spikes_all
            else:
                n_e_pre = self.layer_configs[i-1]["n_e"]
                n_i_pre = self.layer_configs[i-1]["n_i"]
                spikes_e_pre = pre_spikes_all[:n_e_pre]
                if n_i_pre > 0:
                    spikes_i_pre = pre_spikes_all[n_e_pre:]

            n_e_post = self.layer_configs[i]["n_e"]
            n_i_post = self.layer_configs[i]["n_i"]
            spikes_e_post = post_spikes_all[:n_e_post]
            spikes_i_post: Optional[torch.Tensor] = None
            if n_i_post > 0:
                spikes_i_post = post_spikes_all[n_e_post:]
            
            # --- 2. クレジット信号の準備 ---
            current_credit_e = current_params.get("reward", 0.0)
            current_credit_i = current_params.get("reward", 0.0)
            if backward_credit_e is not None:
                current_credit_e += backward_credit_e.mean().item() * 0.1
            if backward_credit_i is not None:
                current_credit_i += backward_credit_i.mean().item() * 0.1
                
            params_e = current_params.copy(); params_e["reward"] = current_credit_e
            params_i = current_params.copy(); params_i["reward"] = current_credit_i

            # --- 3. 学習則の適用 (4つの行列すべて) ---
            
            # --- E -> E ---
            dw_ee, bwd_e_from_ee = self._apply_rules(self.weights_ee[i], spikes_e_pre, spikes_e_post, params_e)
            
            bwd_e_from_ie = torch.zeros_like(spikes_e_pre)
            if spikes_i_post is not None and i < len(self.weights_ie):
                # --- E -> I ---
                dw_ie, bwd_e_from_ie_new = self._apply_rules(self.weights_ie[i], spikes_e_pre, spikes_i_post, params_i)
                self.weights_ie[i].data += dw_ie
                bwd_e_from_ie = bwd_e_from_ie_new

            backward_credit_e_t = bwd_e_from_ee + bwd_e_from_ie # E_pre への総クレジット
            backward_credit_i_t = torch.zeros_like(spikes_e_pre) # デフォルト (I_pre がない場合)
            
            if i > 0 and spikes_i_pre is not None:
                bwd_i_from_ei = torch.zeros_like(spikes_i_pre)
                bwd_i_from_ii = torch.zeros_like(spikes_i_pre)
                
                if i-1 < len(self.weights_ei):
                    # --- I -> E ---
                    dw_ei, bwd_i_from_ei_new = self._apply_rules(self.weights_ei[i-1], spikes_i_pre, spikes_e_post, params_e)
                    self.weights_ei[i-1].data += dw_ei
                    bwd_i_from_ei = bwd_i_from_ei_new
                
                if spikes_i_post is not None and i-1 < len(self.weights_ii):
                    # --- I -> I ---
                    dw_ii, bwd_i_from_ii_new = self._apply_rules(self.weights_ii[i-1], spikes_i_pre, spikes_i_post, params_i)
                    self.weights_ii[i-1].data += dw_ii
                    bwd_i_from_ii = bwd_i_from_ii_new
                
                backward_credit_i_t = bwd_i_from_ei + bwd_i_from_ii # I_pre への総クレジット

            # 重み更新 (E -> E/I)
            self.weights_ee[i].data += dw_ee
            
            # 次のループのためのクレジット信号を更新
            backward_credit_e = backward_credit_e_t
            backward_credit_i = backward_credit_i_t

        # 最終的な重みの制約（デールの法則）を適用
        self.clamp_weights()

    def _apply_rules(
        self, 
        weights: nn.Parameter, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor, 
        params: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(v3) シナプス則と恒常性則の両方を適用するヘルパー"""
        
        dw_synaptic, backward_credit = self.synaptic_rule.update(
            pre_spikes=pre_spikes,
            post_spikes=post_spikes,
            weights=weights,
            optional_params=params
        )
        
        dw_homeostasis = torch.zeros_like(weights.data)
        if self.homeostatic_rule:
            dw_homeo, _ = self.homeostatic_rule.update(
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                weights=weights,
                optional_params=params
            )
            dw_homeostasis = dw_homeo

        dw = dw_synaptic + dw_homeostasis
        
        if backward_credit is None:
            backward_credit = torch.zeros(pre_spikes.shape[0], device=pre_spikes.device) # 形状を pre_spikes に合わせる

        # スパース化 (オプション)
        if self.sparsification_enabled and isinstance(self.synaptic_rule, CausalTraceCreditAssignmentEnhancedV2):
            causal_contribution = self.synaptic_rule.get_causal_contribution()
            if causal_contribution is not None:
                contribution_mask = causal_contribution > self.contribution_threshold
                dw = dw * contribution_mask

        return dw, backward_credit

    def clamp_weights(self) -> None:
        """デールの法則（重みの符号制約）を学習後に強制的に適用する。"""
        with torch.no_grad():
            for w_list in [self.weights_ee, self.weights_ie, self.weights_ei, self.weights_ii]:
                for w in w_list:
                    w.data.clamp_(min=0) # すべての重みを非負に保つ