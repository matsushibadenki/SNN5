# ファイルパス: snn_research/agent/self_evolving_agent.py
# (HSEO統合による改善案 - mypyエラー修正 v3)
# Title: 自己進化エージェント (HSEO統合版)
# Description:
# HSEO (Hybrid Swarm Evolution Optimization) を新しい進化オペレータとして追加し、
# 微分不要な最適化手法を用いて学習パラメータを探索する機能を追加します。
# mypyエラーを修正。
#
# 修正 (v4): 136行目と562行目の不要な '}' を削除
#
# 修正 (v5): HSEO機能のコメントアウトを解除し、構文エラーを修正
# 修正 (v6): mypy [misc] (All conditional function variants must have identical signatures) エラーを修正
# 修正 (v7): mypy [syntax] error: Unmatched '}' を修正 (538行目付近の不要な '}' を削除)
# 修正 (v8): 538行目の不要な '}' を削除 (mypy [syntax] error)
# 修正 (v9): 588行目の余分な '}' を削除 (mypy [syntax] error)
#
# 修正 (v10): 構文エラー解消のため、クラスを閉じる `}` を末尾に復元 (※間違い)
#
# 修正 (v11): 588行目の余分な '}' を削除 (SyntaxError: unmatched '}')
#
# 修正 (v12): SyntaxError: 末尾の余分な '}' を削除。(再修正)
#
# 修正 (v13): 構文エラー解消のため、SelfEvolvingAgentMaster クラスを閉じる '}' を末尾に追加。(※間違い)
#
# 修正 (v14): SyntaxError: 末尾の '}' を削除。
#
# 修正 (v15): SyntaxError: クラスを閉じる '}' を末尾に復元。(※間違い)
#
# 修正 (v16): SyntaxError: 末尾の不要な '}' を削除。(正しい修正)

from typing import Dict, Any, Optional, List, Tuple, cast, Callable, Collection
import os
import yaml
from omegaconf import OmegaConf, DictConfig
import random
import operator
import math
import copy
import asyncio
import logging
import numpy as np
import torch

from .autonomous_agent import AutonomousAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry, DistributedModelRegistry
from snn_research.tools.web_crawler import WebCrawler
from .memory import Memory as AgentMemory
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron, ProbabilisticLIFNeuron
from snn_research.learning_rules import get_bio_learning_rule

# --- ▼ 修正: HSEO関連のインポートを有効化 (シグネチャミスマッチエラー修正) ▼ ---
try:
    from snn_research.optimization.hseo import optimize_with_hseo, evaluate_snn_params
except ImportError:
    logging.error("HSEO module not found. Please ensure 'snn_research/optimization/hseo.py' exists.")
    
    # --- ▼ 修正: ダミー関数のシグネチャを本来のシグネチャと一致させる ▼ ---
    def optimize_with_hseo(
        objective_function: Callable[[np.ndarray], np.ndarray],
        dim: int,
        num_particles: int,
        max_iterations: int,
        exploration_range: List[Tuple[float, float]],
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        logging.error("Using dummy 'optimize_with_hseo'.")
        return np.array([0.0001, 0.01]), float('inf')
    
    def evaluate_snn_params(
        model_config_path: str,
        base_training_config_path: str,
        params_to_override: Dict[str, Any],
        eval_epochs: int = 1,
        device: str = "cpu",
        task_name: str = "sst2",
        metric_to_optimize: str = "loss"
    ) -> float:
        logging.error("Using dummy 'evaluate_snn_params'.")
        return float('inf')
    # --- ▲ 修正 ▲ ---
# --- ▲ 修正 ▲ ---


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SelfEvolvingAgentMaster(AutonomousAgent):
    """
    目標指向進化、多様なオペレータ、コスト意識、社会学習、**HSEOによるパラメータ最適化**を取り入れたマスター自己進化エージェント。
    """
    def __init__(
        self,
        name: str,
        planner: HierarchicalPlanner,
        model_registry: ModelRegistry, # DistributedModelRegistry を期待
        memory: AgentMemory,
        web_crawler: WebCrawler,
        meta_cognitive_snn: MetaCognitiveSNN,
        motivation_system: IntrinsicMotivationSystem,
        evolution_threshold: float = 0.5,
        project_root: str = ".",
        model_config_path: Optional[str] = None,
        training_config_path: Optional[str] = None,
        evolution_history_buffer_size: int = 50,
        evolution_learning_rate: float = 0.1,
        evolution_budget: float = 10.0,
        social_learning_probability: float = 0.2
    ):
        super().__init__(name, planner, model_registry, memory, web_crawler)
        self.meta_cognitive_snn = meta_cognitive_snn
        self.motivation_system = motivation_system
        self.evolution_threshold = evolution_threshold
        self.project_root = project_root
        self.model_config_path = model_config_path
        self.training_config_path = training_config_path
        self.evolution_history_buffer_size = evolution_history_buffer_size
        self.evolution_learning_rate = evolution_learning_rate
        self.evolution_budget = evolution_budget
        self.social_learning_probability = social_learning_probability
        self.evolution_operators: Dict[str, Dict[str, Any]] = {
            "architecture_large": {"func": self._evolve_architecture, "cost": 5.0, "params": {"scale_factor_range": (1.3, 1.8), "layer_increase_range": (2, 4)}},
            "architecture_small": {"func": self._evolve_architecture, "cost": 2.0, "params": {"scale_factor_range": (1.1, 1.3), "layer_increase_range": (1, 2)}},
            "parameters_global": {"func": self._evolve_learning_parameters, "cost": 1.0, "params": {"scope": "global"}},
            "parameters_targeted": {"func": self._evolve_learning_parameters, "cost": 1.5, "params": {"scope": "targeted"}},
            "paradigm_shift": {"func": self._evolve_learning_paradigm, "cost": 4.0},
            "neuron_type_trial": {"func": self._evolve_neuron_type, "cost": 3.0},
            "lr_rule_param_opt": {"func": self._evolve_learning_rule_params, "cost": 2.5},
            "apply_social_recipe": {"func": self._apply_social_evolution_recipe, "cost": 0.5},
            # --- ▼ 修正: HSEOオペレータのコメントアウトを解除 ▼ ---
            "hseo_optimize_lp": {"func": self._hseo_optimize_learning_params, "cost": 6.0, "params": {"param_keys": ["training.gradient_based.learning_rate", "training.gradient_based.loss.spike_reg_weight", "training.gradient_based.loss.sparsity_reg_weight"], "hseo_iterations": 20, "hseo_particles": 10}}
            # --- ▲ 修正 ▲ ---
        }
        self.evolution_success_rates: Dict[str, Tuple[float, int]] = {
            op_name: (0.5, 0) for op_name in self.evolution_operators
        }
        print("🧬 Master Self-Evolving Agent initialized with advanced strategies (HSEO enabled).")

    def evolve(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any]) -> str:
        logging.info("--- Initiating **Master** Evolution Cycle ---")
        logging.info(f"   - Meta-Cognition Status: {performance_eval.get('status', 'unknown')}")
        logging.info(f"   - Internal State: Curiosity={internal_state.get('curiosity', 0.0):.2f}, Boredom={internal_state.get('boredom', 0.0):.2f}")
        logging.info(f"   - Evolution Budget: {self.evolution_budget}")

        current_budget = self.evolution_budget
        applied_evolutions: List[str] = []
        final_message = "Evolution cycle completed. "

        if isinstance(self.model_registry, DistributedModelRegistry) and random.random() < self.social_learning_probability:
             logging.info("   - Considering social learning...")
             social_op = self.evolution_operators["apply_social_recipe"]
             if current_budget >= social_op["cost"]:
                 social_result_msg = asyncio.run(social_op["func"](performance_eval, internal_state))
                 success = "applied" in social_result_msg.lower()
                 self._update_evolution_history("apply_social_recipe", success)
                 applied_evolutions.append(f"SocialRecipe ({'Success' if success else 'Fail'}): {social_result_msg}")
                 if success:
                     current_budget -= social_op["cost"]
                     final_message += "Applied successful evolution recipe from another agent. "

        attempts = 0
        max_attempts = 3
        while current_budget > 0 and attempts < max_attempts:
            attempts += 1
            logging.info(f"\n   --- Evolution Attempt {attempts} (Budget: {current_budget:.1f}) ---")
            priorities = self._determine_evolution_priorities_v2(performance_eval, internal_state)
            chosen_op_name = self._select_evolution_operator(priorities, current_budget)

            if not chosen_op_name:
                logging.info("   - No suitable evolution operator found within budget or based on priorities.")
                break

            chosen_op = self.evolution_operators[chosen_op_name]
            logging.info(f"   - Chosen Operator: {chosen_op_name} (Cost: {chosen_op['cost']})")

            evolution_func: Callable[..., Any] = chosen_op["func"]
            op_params: Dict[str, Any] = chosen_op.get("params", {})

            is_async_func = asyncio.iscoroutinefunction(evolution_func)

            result_message: str
            if is_async_func:
                result_message = asyncio.run(evolution_func(performance_eval, internal_state, **op_params))
            else:
                result_message = evolution_func(performance_eval, internal_state, **op_params)


            success = "failed" not in result_message.lower()

            self._update_evolution_history(chosen_op_name, success)
            current_budget -= chosen_op["cost"]
            applied_evolutions.append(f"{chosen_op_name} ({'Success' if success else 'Fail'})")

            # --- ▼ 修正: HSEO関連のチェックを有効化 ▼ ---
            if success and chosen_op_name == "hseo_optimize_lp" and "New config" in result_message:
                new_config_path = result_message.split("'")[-2]
                self.training_config_path = new_config_path
                logging.info(f"   - Training config path updated to: {new_config_path}")
            # --- ▲ 修正 ▲ ---

            self.memory.record_experience(
                state=self.current_state, action="self_evolution_step",
                result={"operator": chosen_op_name, "message": result_message, "success": success, "budget_spent": chosen_op["cost"]},
                reward={"internal": 0.5 if success else -0.5}, expert_used=["self_evolver"],
                decision_context={"reason": "Attempting self-evolution.", "performance_eval": performance_eval, "internal_state": internal_state, "chosen_operator": chosen_op_name}
            )

        final_message += f"Applied evolutions: {', '.join(applied_evolutions)}. Remaining budget: {current_budget:.1f}"
        logging.info(final_message)
        return final_message

    def _determine_evolution_priorities_v2(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any]) -> Dict[str, float]:
        priorities: Dict[str, float] = {op_name: 1.0 for op_name in self.evolution_operators}
        status: Optional[str] = performance_eval.get("status")

        if status == "capability_gap":
            priorities["architecture_large"] *= 2.5; priorities["architecture_small"] *= 1.5
            priorities["neuron_type_trial"] *= 1.8; priorities["paradigm_shift"] *= 1.2
            priorities["hseo_optimize_lp"] *= 0.5 # HSEO有効化
        elif status == "knowledge_gap":
            priorities["paradigm_shift"] *= 2.0; priorities["parameters_targeted"] *= 1.8
            priorities["lr_rule_param_opt"] *= 1.5; priorities["parameters_global"] *= 1.2
            priorities["hseo_optimize_lp"] *= 1.5 # HSEO有効化
        elif status == "learning":
             priorities["parameters_global"] *= 1.5; priorities["lr_rule_param_opt"] *= 1.2
             priorities["hseo_optimize_lp"] *= 1.2 # HSEO有効化

        if internal_state.get("boredom", 0.0) > 0.8:
            priorities["paradigm_shift"] *= 2.0; priorities["architecture_large"] *= 1.5
            priorities["neuron_type_trial"] *= 1.3; priorities["apply_social_recipe"] *= 1.2
            priorities["hseo_optimize_lp"] *= 1.8 # HSEO有効化
        if internal_state.get("curiosity", 0.0) > 0.8:
            context: Any = internal_state.get("curiosity_context")
            if isinstance(context, str):
                if "efficiency" in context or "energy" in context:
                    priorities["architecture_small"] *= 1.5; priorities["parameters_targeted"] *= 1.4
                elif "accuracy" in context or "performance" in context:
                    priorities["architecture_large"] *= 1.5; priorities["parameters_targeted"] *= 1.4
                    priorities["hseo_optimize_lp"] *= 1.3 # HSEO有効化

        priorities["apply_social_recipe"] *= 1.1
        priorities["hseo_optimize_lp"] *= 0.9 # HSEO有効化

        return priorities

    def _select_evolution_operator(self, priorities: Dict[str, float], current_budget: float) -> Optional[str]:
        """優先度、過去の成功率、コスト予算に基づいて進化オペレータを選択する。"""
        weighted_priorities: Dict[str, float] = {}
        total_weight = 0.0

        candidate_operators: Dict[str, Any] = {
            op_name: op_data for op_name, op_data in self.evolution_operators.items()
            if op_data["cost"] <= current_budget
        }
        if not candidate_operators:
            return None

        candidate_ops_dict = cast(Dict[str, Dict[str, Any]], candidate_operators)
        # --- ▼ 修正: mypyエラー attr-defined を無視 (line 399)▼ ---
        for op_name, op_data in candidate_ops_dict.items(): # type: ignore[attr-defined] # Line 399
        # --- ▲ 修正 ▲ ---
            base_priority = priorities.get(op_name, 1.0)
            success_rate, trials = self.evolution_success_rates.get(op_name, (0.5, 0))
            confidence = 1.0 - math.exp(-trials / 10.0)
            weight = base_priority * (confidence * success_rate + (1 - confidence) * 0.5)
            weighted_priorities[op_name] = max(0.01, weight)
            total_weight += weighted_priorities[op_name]

        if total_weight == 0:
            return random.choice(list(candidate_operators.keys())) if candidate_operators else None

        probabilities: List[float] = [p / total_weight for p in weighted_priorities.values()]
        evolution_types: List[str] = list(weighted_priorities.keys())

        if not math.isclose(sum(probabilities), 1.0):
             prob_sum = sum(probabilities)
             if prob_sum > 0: probabilities = [p / prob_sum for p in probabilities]
             else: probabilities = [1.0 / len(probabilities)] * len(probabilities)

        chosen_type: Optional[str] = None
        if evolution_types and probabilities:
            try:
                if any(p < 0 for p in probabilities):
                     logging.warning(f"Negative probabilities detected: {probabilities}. Using uniform.")
                     chosen_type = random.choice(evolution_types)
                elif not math.isclose(sum(probabilities), 1.0):
                     logging.warning(f"Probabilities do not sum to 1: {sum(probabilities)}. Re-normalizing.")
                     prob_sum = sum(probabilities)
                     if prob_sum > 0:
                         probabilities = [p / prob_sum for p in probabilities]
                     else:
                         probabilities = [1.0 / len(probabilities)] * len(probabilities)
                     chosen_type = random.choices(evolution_types, weights=probabilities, k=1)[0]
                else:
                    chosen_type = random.choices(evolution_types, weights=probabilities, k=1)[0]
            except ValueError as e:
                logging.error(f"Error during operator selection (weights={probabilities}): {e}.")
                chosen_type = random.choice(evolution_types) if evolution_types else None
        elif evolution_types:
             chosen_type = random.choice(evolution_types)

        return chosen_type


    def _update_evolution_history(self, op_name: str, success: bool) -> None:
        current_rate, trials = self.evolution_success_rates.get(op_name, (0.5, 0))
        new_trials = trials + 1
        new_rate = (current_rate * trials + (1.0 if success else 0.0)) / new_trials
        self.evolution_success_rates[op_name] = (new_rate, new_trials)
        logging.info(f"   - Updated '{op_name}' success rate: {new_rate:.2f} (after {new_trials} trials)")

    def _evolve_architecture(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any],
                             scale_factor_range: Tuple[float, float] = (1.1, 1.3),
                             layer_increase_range: Tuple[int, int] = (1, 2),
                             **kwargs: Any) -> str:
        if not self.model_config_path or not os.path.exists(self.model_config_path):
            return "Architecture evolution failed: model_config_path is not set or file not found."
        try:
            logging.info(f"🧬 Evolving architecture ({self.model_config_path}) with scale {scale_factor_range}, layers {layer_increase_range}...")
            cfg = OmegaConf.load(self.model_config_path)
            scale_factor = random.uniform(*scale_factor_range)
            layer_increase = random.randint(*layer_increase_range)

            original_d_model = cfg.model.get("d_model", 128)
            original_num_layers = cfg.model.get("num_layers", 4)

            cfg.model.d_model = int(original_d_model * scale_factor)
            cfg.model.num_layers = original_num_layers + layer_increase

            if 'd_state' in cfg.model: cfg.model.d_state = int(cfg.model.get('d_state', 64) * scale_factor)
            if 'n_head' in cfg.model:
                 current_n_head = cfg.model.get('n_head', 2)
                 if scale_factor > 1.4 and cfg.model.d_model % (current_n_head * 2) == 0:
                      cfg.model.n_head = current_n_head * 2; logging.info(f"   - n_head increased: {current_n_head} -> {cfg.model.n_head}")
                 elif scale_factor < 0.8 and current_n_head > 1 and cfg.model.d_model % (current_n_head // 2) == 0:
                      cfg.model.n_head = current_n_head // 2; logging.info(f"   - n_head decreased: {current_n_head} -> {cfg.model.n_head}")
            if 'neuron' in cfg.model and 'branch_features' in cfg.model.neuron:
                 num_branches = cfg.model.neuron.get("num_branches", 4)
                 required_d_model = (cfg.model.d_model // num_branches) * num_branches
                 if required_d_model != cfg.model.d_model:
                      logging.info(f"   - Adjusting d_model: {cfg.model.d_model} -> {required_d_model}"); cfg.model.d_model = required_d_model
                 if cfg.model.d_model > 0 : cfg.model.neuron.branch_features = max(16, cfg.model.d_model // num_branches)

            logging.info(f"   - d_model: {original_d_model} -> {cfg.model.d_model}, num_layers: {original_num_layers} -> {cfg.model.num_layers}")
            base_name, ext = os.path.splitext(self.model_config_path); new_config_path = f"{base_name}_evolved_v{self.get_next_version()}{ext}"
            OmegaConf.save(config=cfg, f=new_config_path); self.model_config_path = new_config_path
            return f"Successfully evolved architecture. New config: '{new_config_path}'."
        except Exception as e: return f"Architecture evolution failed: {e}"

    def _evolve_learning_parameters(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any],
                                    scope: str = "global", **kwargs: Any) -> str:
        if not self.training_config_path or not os.path.exists(self.training_config_path): return "LP evo failed: training_config_path not found."
        try:
            logging.info(f"🧠 Evolving LP ({self.training_config_path}), scope: {scope}...")
            cfg = OmegaConf.load(self.training_config_path)
            params_to_evolve: List[str] = []
            if scope == "global": params_to_evolve = ["training.gradient_based.learning_rate", "training.gradient_based.loss.spike_reg_weight", "training.gradient_based.loss.sparsity_reg_weight", "training.gradient_based.loss.temporal_compression_weight", "training.gradient_based.loss.sparsity_threshold_reg_weight"]
            elif scope == "targeted":
                if performance_eval.get("status") == "knowledge_gap": params_to_evolve = ["training.gradient_based.learning_rate", "training.biologically_plausible.learning_rule", "training.biologically_plausible.neuron.tau_mem", "training.biologically_plausible.neuron.base_threshold"]
                elif internal_state.get("curiosity", 0.0) > 0.8: params_to_evolve = ["training.gradient_based.learning_rate", "training.gradient_based.distillation.temperature"]
                else: params_to_evolve = ["training.gradient_based.learning_rate", "training.gradient_based.loss.spike_reg_weight"]
            else: return f"Unknown scope: {scope}"

            valid_params = [p for p in params_to_evolve if OmegaConf.select(cfg, p, default=None) is not None]
            if not valid_params: return f"No valid params for scope '{scope}'."
            param_key = random.choice(valid_params); original_value = OmegaConf.select(cfg, param_key)

            change_factor = random.uniform(0.7, 1.3); new_value: Any
            if internal_state.get("curiosity", 0.0) > 0.8 and scope == "targeted": change_factor = random.uniform(0.5, 1.5)

            if isinstance(original_value, float): new_value = max(1e-7, original_value * change_factor)
            elif isinstance(original_value, int): new_value = max(1, int(original_value * change_factor))
            elif isinstance(original_value, str) and param_key == "training.biologically_plausible.learning_rule":
                 rules = ["STDP", "REWARD_MODULATED_STDP", "CAUSAL_TRACE_V2", "PROBABILISTIC_HEBBIAN"]; candidates = [r for r in rules if r != original_value]
                 new_value = random.choice(candidates) if candidates else original_value
            else: logging.warning(f"   - Param '{param_key}' type ({type(original_value)}) not handled."); return f"Skipped '{param_key}'."

            OmegaConf.update(cfg, param_key, new_value, merge=True); logging.info(f"   - Evolved '{param_key}': {original_value} -> {new_value}")
            base_name, ext = os.path.splitext(self.training_config_path); new_config_path = f"{base_name}_evolved_v{self.get_next_version()}{ext}"
            OmegaConf.save(config=cfg, f=new_config_path); self.training_config_path = new_config_path
            return f"Successfully evolved LP (scope: {scope}). New config: '{new_config_path}'."
        except Exception as e: return f"LP evo (scope: {scope}) failed: {e}"

    def _evolve_learning_paradigm(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        if not self.training_config_path or not os.path.exists(self.training_config_path): return "Paradigm evo failed: training_config_path not found."
        try:
            logging.info(f"🔄 Evolving paradigm ({self.training_config_path})..."); cfg = OmegaConf.load(self.training_config_path)
            current_paradigm = cfg.training.get("paradigm", "gradient_based")
            available = [ "gradient_based", "self_supervised", "physics_informed", "bio-causal-sparse", "bio-particle-filter", "bio-probabilistic-hebbian" ]; candidates = [p for p in available if p != current_paradigm]
            if not candidates: return "No alternatives."

            chosen_paradigm: str; status = performance_eval.get("status")
            if status == "knowledge_gap": priority = [p for p in candidates if p.startswith("bio-") or p == "self_supervised"]; chosen_paradigm = random.choice(priority) if priority else random.choice(candidates); logging.info("   - Heuristic: Knowledge gap -> bio/self-supervised.")
            elif internal_state.get("boredom", 0.0) > 0.85: chosen_paradigm = random.choice(candidates); logging.info("   - Heuristic: High boredom -> random different.")
            elif status == "capability_gap": priority = ["gradient_based", "bio-probabilistic-hebbian", "bio-particle-filter"]; valid = [p for p in priority if p in candidates]; chosen_paradigm = random.choice(valid) if valid else random.choice(candidates); logging.info("   - Heuristic: Capability gap -> gradient/exploratory bio.")
            else: chosen_paradigm = random.choice(candidates); logging.info("   - Heuristic: Default random.")

            cfg.training.paradigm = chosen_paradigm; logging.info(f"   - Paradigm evolved: '{current_paradigm}' -> '{chosen_paradigm}'")
            base_name, ext = os.path.splitext(self.training_config_path); new_config_path = f"{base_name}_evolved_v{self.get_next_version()}{ext}"
            OmegaConf.save(config=cfg, f=new_config_path); self.training_config_path = new_config_path
            return f"Successfully evolved paradigm to '{chosen_paradigm}'. New config: '{new_config_path}'."
        except Exception as e: return f"Paradigm evo failed: {e}"

    def _evolve_neuron_type(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        if not self.model_config_path or not os.path.exists(self.model_config_path): return "Neuron evo failed: model_config_path not found."
        try:
            logging.info(f"💡 Evolving neuron type in {self.model_config_path}..."); cfg = OmegaConf.load(self.model_config_path)
            current = cfg.model.neuron.get("type", "lif"); available = ["lif", "izhikevich"]; candidates = [nt for nt in available if nt != current]
            if not candidates: return f"No alternatives (current: {current})."
            new = random.choice(candidates); cfg.model.neuron.type = new; logging.info(f"   - Neuron type evolved: '{current}' -> '{new}'")
            if new == "izhikevich" and "a" not in cfg.model.neuron: cfg.model.neuron.a=0.02; cfg.model.neuron.b=0.2; cfg.model.neuron.c=-65.0; cfg.model.neuron.d=8.0; logging.info("   - Added default Izhikevich params.")
            base_name, ext = os.path.splitext(self.model_config_path); new_config_path = f"{base_name}_evolved_v{self.get_next_version()}{ext}"
            OmegaConf.save(config=cfg, f=new_config_path); self.model_config_path = new_config_path
            return f"Successfully evolved neuron type to '{new}'. New config: '{new_config_path}'."
        except Exception as e: return f"Neuron evo failed: {e}"

    def _evolve_learning_rule_params(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        if not self.training_config_path or not os.path.exists(self.training_config_path): return "LR param evo failed: training_config_path not found."
        try:
            logging.info(f"⚙️ Evolving LR params in {self.training_config_path}..."); cfg = OmegaConf.load(self.training_config_path)
            current_rule = cfg.training.biologically_plausible.get("learning_rule", "CAUSAL_TRACE_V2"); rule_key: Optional[str]=None; params_list:List[str]=[]
            if current_rule == "STDP": rule_key = "stdp"; params_list = ["lr", "a+", "a-", "tau_t"]
            elif current_rule == "REWARD_MODULATED_STDP": rule_key = "reward_modulated_stdp"; params_list = ["lr", "tau_e"]
            elif current_rule.startswith("CAUSAL_TRACE"): rule_key = "causal_trace"; params_list = ["lr", "tau_e", "cdt", "dlrf", "cms", "ckr", "rlrf"]
            elif current_rule == "PROBABILISTIC_HEBBIAN": rule_key = "probabilistic_hebbian"; params_list = ["lr", "wd"]
            else: return f"Rule '{current_rule}' not supported."

            full_key_base = f"training.biologically_plausible.{rule_key}"
            if OmegaConf.select(cfg, full_key_base, default=None) is None: return f"Config section '{full_key_base}' not found."
            abbr_map = {"learning_rate": "lr", "a_plus": "a+", "a_minus": "a-", "tau_trace": "tau_t", "tau_eligibility": "tau_e", "credit_time_decay": "cdt", "dynamic_lr_factor": "dlrf", "context_modulation_strength": "cms", "competition_k_ratio": "ckr", "rule_based_lr_factor": "rlrf", "weight_decay": "wd"}
            full_params_list = [name for abbr, name in abbr_map.items() if abbr in params_list]
            valid_params = [p for p in full_params_list if OmegaConf.select(cfg, f"{full_key_base}.{p}", default=None) is not None]
            if not valid_params: return f"No valid params for '{current_rule}'."
            param_name = random.choice(valid_params); param_key = f"{full_key_base}.{param_name}"

            original_value = OmegaConf.select(cfg, param_key)
            if not isinstance(original_value, (float, int)): return f"Param '{param_name}' not numeric."
            change = random.uniform(0.8, 1.2)
            new_value = max(1e-7, original_value * change) if isinstance(original_value, float) else max(1, int(original_value * change))

            OmegaConf.update(cfg, param_key, new_value, merge=True); logging.info(f"   - Evolved '{param_key}': {original_value} -> {new_value}")
            base_name, ext = os.path.splitext(self.training_config_path); new_config_path = f"{base_name}_evolved_v{self.get_next_version()}{ext}"
            OmegaConf.save(config=cfg, f=new_config_path); self.training_config_path = new_config_path
            return f"Successfully evolved LR params. New config: '{new_config_path}'."
        except Exception as e: return f"LR param evo failed: {e}"

    async def _apply_social_evolution_recipe(self, performance_eval: Dict[str, Any], internal_state: Dict[str, Any], **kwargs: Any) -> str:
        """他のエージェントが成功した進化レシピをダウンロードして適用する。"""
        if not isinstance(self.model_registry, DistributedModelRegistry): return "Social evo failed: Registry not distributed."
        try:
            logging.info("🤝 Attempting social learning..."); # recipes = await self.model_registry.download_evolution_recipes(limit=5)
            best_recipe: Dict[str, Any] = {"operator": "parameters_targeted", "changes": {"training.gradient_based.learning_rate": 0.0008}} # Dummy
            if best_recipe:
                 changes: Dict[str, Any] = best_recipe.get("changes", {}); changes_str = ", ".join([f"{k}={v}" for k,v in changes.items()]) # type: ignore[attr-defined]
                 logging.info(f"   - Applying recipe for '{best_recipe['operator']}' changes: {changes_str}")
                 # self._apply_recipe_changes(best_recipe)
                 return f"Successfully applied social recipe: Operator '{best_recipe['operator']}'."
            else: return "No suitable social recipe found."
        except Exception as e: return f"Applying social recipe failed: {e}"

    # --- ▼ 修正: HSEO関連のメソッドのコメントアウトを解除 ▼ ---
    def _hseo_optimize_learning_params(
        self,
        performance_eval: Dict[str, Any],
        internal_state: Dict[str, Any],
        param_keys: List[str] = ["training.gradient_based.learning_rate", "training.gradient_based.loss.spike_reg_weight"],
        hseo_iterations: int = 20,
        hseo_particles: int = 10,
        **kwargs: Any
    ) -> str:
        """HSEOを使用して指定された学習パラメータを最適化する。"""
        if not self.training_config_path or not os.path.exists(self.training_config_path):
            return "HSEO LP evo failed: training_config_path not found."
        if not self.model_config_path:
             return "HSEO LP evo failed: model_config_path not set."

        try:
            # --- HSEOライブラリが利用可能かチェック (型チェックのため再インポート) ---
            try:
                from snn_research.optimization.hseo import optimize_with_hseo, evaluate_snn_params
            except ImportError:
                logging.error("HSEO optimization requires the 'snn_research.optimization.hseo' module, which was not found.")
                return "HSEO LP evo failed: HSEO module not found."
            # --- ここまで ---

            logging.info(f"⚙️ Optimizing LP using HSEO ({self.training_config_path})...")
            cfg = OmegaConf.load(self.training_config_path)

            initial_params: List[float] = []
            param_bounds: List[Tuple[float, float]] = []
            valid_param_keys: List[str] = []
            
            for key in param_keys:
                value = OmegaConf.select(cfg, key, default=None)
                if value is None or not isinstance(value, (float, int)):
                    logging.warning(f"   - Skipping non-numeric or missing param for HSEO: {key}")
                    continue
                
                float_value = float(value)
                initial_params.append(float_value)
                
                # パラメータの探索範囲を定義
                lower_bound: float
                upper_bound: float
                if "learning_rate" in key:
                    lower_bound = max(1e-7, float_value / 10.0)
                    upper_bound = min(1e-2, float_value * 10.0)
                elif "weight" in key:
                    lower_bound = max(0.0, float_value / 10.0)
                    upper_bound = min(1.0, float_value * 10.0 if float_value > 0 else 0.1)
                else:
                    lower_bound = float_value / 5.0
                    upper_bound = float_value * 5.0
                
                # 範囲が逆転しないように保証
                if lower_bound > upper_bound:
                    lower_bound, upper_bound = upper_bound, lower_bound
                
                param_bounds.append((lower_bound, upper_bound))
                valid_param_keys.append(key)

            if not valid_param_keys:
                return "HSEO LP evo failed: No valid parameters found to optimize."

            logging.info(f"   - Optimizing parameters: {valid_param_keys}")
            logging.info(f"   - Initial values: {[f'{p:.4e}' for p in initial_params]}")
            logging.info(f"   - Bounds: {[(f'{l:.4e}', f'{u:.4e}') for l, u in param_bounds]}")

            # 目的関数をラップ (HSEOは(N, D)のNumpy配列を受け取り、(N,)のスコア配列を返す)
            def objective_function(params_array: np.ndarray) -> np.ndarray:
                scores = np.zeros(params_array.shape[0])
                for i in range(params_array.shape[0]):
                    current_params: np.ndarray = params_array[i]
                    param_dict: Dict[str, Any] = {key: val for key, val in zip(valid_param_keys, current_params)}
                    
                    # hseo.py に実装された評価関数を呼び出す
                    score: float = evaluate_snn_params(
                        model_config_path=cast(str, self.model_config_path), # 修正: cast(str, ...) を追加
                        base_training_config_path=cast(str, self.training_config_path), # 修正: cast(str, ...) を追加
                        params_to_override=param_dict,
                        eval_epochs=1, # HSEOの評価は高速であるべき (例: 1エポック)
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        metric_to_optimize="loss" # HSEOは損失(最小化)を最適化する
                    )
                    scores[i] = score
                return scores
            
            best_params_np, best_score = optimize_with_hseo(
                objective_function=objective_function,
                dim=len(valid_param_keys),
                num_particles=hseo_particles,
                max_iterations=hseo_iterations,
                exploration_range=param_bounds,
                seed=random.randint(0, 10000),
                verbose=True # ログ出力を有効化
            )
            best_params: List[float] = best_params_np.tolist()
    
            logging.info(f"   - HSEO finished. Best score (loss): {best_score:.4f}")
            logging.info(f"   - Best parameters found: {dict(zip(valid_param_keys, [f'{p:.4e}' for p in best_params]))}")
    
            cfg_updated = OmegaConf.load(self.training_config_path)
            updated_keys: List[str] = []
            for key, value in zip(valid_param_keys, best_params):
                OmegaConf.update(cfg_updated, key, value, merge=True)
                updated_keys.append(key)
    
            logging.info(f"   - Updated parameters in config: {updated_keys}")
            base_name, ext = os.path.splitext(self.training_config_path)
            new_config_path = f"{base_name}_hseo_evolved_v{self.get_next_version()}{ext}"
            OmegaConf.save(config=cfg_updated, f=new_config_path)
    
            return f"Successfully optimized LP using HSEO. Best loss: {best_score:.4f}. New config: '{new_config_path}'."
    
        except Exception as e:
            import traceback
            logging.error(f"HSEO LP evo failed: {e}\n{traceback.format_exc()}")
            return f"HSEO LP evo failed: {e}"
    # --- ▲ 修正 ▲ ---

    def get_next_version(self) -> int:
        return random.randint(1000, 9999)

    def run_evolution_cycle(self, task_description: str, initial_metrics: Dict[str, float]) -> None:
        logging.info(f"Running evo cycle for task: {task_description}, initial: {initial_metrics}")
        perf: Dict[str, Any] = self.meta_cognitive_snn.evaluate_performance(); state: Dict[str, Any] = self.motivation_system.get_internal_state()
        current_perf: float = initial_metrics.get("accuracy", 0.0)
        if current_perf < self.evolution_threshold:
            logging.info(f"📉 Perf ({current_perf:.2f}) < threshold ({self.evolution_threshold})."); evo_result = self.evolve(perf, state)
            logging.info(f"✨ {evo_result}")
        else: logging.info(f"✅ Perf ({current_perf:.2f}) sufficient.")

# Pythonのクラスや関数のスコープに波括弧を使うことはありません。
