# ファイルパス: app/containers.py
# Title: 依存性注入（DI）コンテナ定義
# Description:
#   プロジェクト全体の依存性注入（DI）コンテナを定義します。
#   TrainingContainer, AgentContainer, AppContainer, BrainContainer が含まれます。
#   HOPT (HPO) 実行時の設定ファイル読み込みタイミングの問題（tokenizer_name is None）
#   に対処するため、tokenizer への即時参照をすべて遅延参照 (.provided または providers.Callable)
#   に変更しています。

import torch
from dependency_injector import containers, providers
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from transformers import AutoTokenizer
import os
# --- ▼ 修正: Union をインポート ▼ ---
from typing import TYPE_CHECKING, Dict, Any, cast, Optional, List, Union
# --- ▲ 修正 ▲ ---
from omegaconf import DictConfig, OmegaConf # DictConfig, OmegaConf をインポート

# --- プロジェクト内モジュールのインポート ---
# (省略...)
from snn_research.core.snn_core import SNNCore
from snn_research.deployment import SNNInferenceEngine
from snn_research.training.losses import ( CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss, PlannerLoss, ProbabilisticEnsembleLoss )
from snn_research.training.trainers import ( BreakthroughTrainer, DistillationTrainer, SelfSupervisedTrainer, PhysicsInformedTrainer, ProbabilisticEnsembleTrainer, ParticleFilterTrainer, PlannerTrainer )
from snn_research.training.bio_trainer import BioRLTrainer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.planner_snn import PlannerSNN
from .services.chat_service import ChatService
from .services.image_classification_service import ImageClassificationService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter
from snn_research.distillation.model_registry import SimpleModelRegistry, DistributedModelRegistry, ModelRegistry
from snn_research.tools.web_crawler import WebCrawler
# --- ▼ 修正: `get_bio_learning_rule` と `BioLearningRule` をインポート ▼ ---
from snn_research.learning_rules import ProbabilisticHebbian, get_bio_learning_rule, BioLearningRule, CausalTraceCreditAssignmentEnhancedV2
# --- ▲ 修正 ▲ ---
from snn_research.core.neurons import ProbabilisticLIFNeuron
from snn_research.bio_models.simple_network import BioSNN
from snn_research.rl_env.grid_world import GridWorldEnv
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.causal_inference_engine import CausalInferenceEngine
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.hybrid_perception_cortex import HybridPerceptionCortex
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.benchmark import TASK_REGISTRY
from .utils import get_auto_device
from snn_research.agent.digital_life_form import DigitalLifeForm
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgentMaster # Master クラスをインポート
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .adapters.snn_langchain_adapter import SNNLangChainAdapter
    from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
    from typing import Optional


# ... (Helper functions remain the same) ...
def _calculate_t_max(epochs: int, warmup_epochs: int) -> int:
    return max(1, epochs - warmup_epochs)

def _create_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int) -> LRScheduler:
    warmup_scheduler = LinearLR(optimizer=optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    main_scheduler_t_max = _calculate_t_max(epochs=epochs, warmup_epochs=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=main_scheduler_t_max)
    return SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

def _load_planner_snn_factory(planner_snn_instance, model_path: str, device: str):
    model = planner_snn_instance
    # --- ▼ 修正 (v7): model_path が None の場合に os.path.exists がエラーになるのを修正 ▼ ---
    if model_path and os.path.exists(model_path):
    # --- ▲ 修正 (v7) ▲ ---
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded PlannerSNN from '{model_path}'.")
        except Exception as e: print(f"⚠️ Failed to load PlannerSNN: {e}.")
    else: print(f"⚠️ PlannerSNN model not found: {model_path}.")
    return model.to(device)


class TrainingContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    task_registry = providers.Object(TASK_REGISTRY)
    device = providers.Factory(get_auto_device)
    
    # --- ▼ 修正 (v_health_check_fix_v8): [no-redef] L112の重複定義を削除 ▼ ---
    
    # --- ▼▼▼ 【!!! HPO修正 (1/7): この行をコメントアウト】 ▼▼▼ ---
    # tokenizer = providers.Factory(AutoTokenizer.from_pretrained, pretrained_model_name_or_path=config.data.tokenizer_name) # L112: 削除
    # --- ▲▲▲ 【!!! HPO修正 (1/7)】 ▲▲▲ ---
    
    @providers.Factory
    def tokenizer(config_provider=config): # L114: こちらを残す
    # --- ▲ 修正 ▲ ---
        """
        DIコンテナから呼び出される際に、最新の設定を読み込んで
        Tokenizerをインスタンス化するファクトリ。
        """
                
        # このファクトリが呼び出された時点（train.py L376でsnn_modelが
        # 解決される際など）で config_provider() を実行し、
        # 最新の設定辞書を取得する。
        config_dict = config_provider() 
        
        # OmegaConf.create で DictConfig に変換し、安全にアクセス
        cfg = OmegaConf.create(config_dict)
        
        # OmegaConf.select を使って安全に値を取得
        # smoke_test_config.yaml の "gpt2" が読み込まれるはず
        tokenizer_name = OmegaConf.select(cfg, "data.tokenizer_name", default=None)
        
        if tokenizer_name is None:
            logger.error("config.data.tokenizer_name is None in tokenizer factory. Defaulting to 'gpt2'.")
            tokenizer_name = "gpt2"
            
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_name)
    
    # SNNCore が config 引数としてプロバイダオブジェクト(Configuration)ではなく、
    # 解決された値(dict)を受け取るように .provided を使用します。
    snn_model = providers.Factory(
        # --- ▼▼▼ 【!!! HPO修正 (2/7): NameError (NNCore -> SNNCore)】 ▼▼▼ ---
        SNNCore,
        # --- ▲▲▲ 【!!! HPO修正 (2/7)】 ▲▲▲ ---
        
        # 修正前 (Stale Config):
        # config=config.model.provided,
        
        # 修正後 (Dynamic Config):
        # 呼び出し時に config.provided (解決済みの辞書) を lambda に渡し、
        # その辞書から 'model' キーを取得して SNNCore に渡す
        config=providers.Callable(
            lambda c: c.get('model', {}), # config['model'] を動的に取得
            c=config.provided              # lambda の 'c' 引数に解決済みの config dict を注入
        ),
        
        # --- ▼▼▼ 【!!! HPO修正 (3/7): この行をコメントアウト】 ▼▼▼ ---
        # vocab_size=tokenizer.provided.vocab_size
        # --- ▲▲▲ 【!!! HPO修正 (3/7)】 ▲▲▲ ---
    )
    # --- ▲ 修正 (v_hpo_fix_4) ▲ ---
    astrocyte_network = providers.Factory(AstrocyteNetwork, snn_model=snn_model)
    meta_cognitive_snn: providers.Provider[MetaCognitiveSNN] = providers.Factory(MetaCognitiveSNN, **(config.training.meta_cognition.to_dict() or {}))
    optimizer = providers.Factory(AdamW, lr=config.training.gradient_based.learning_rate)
    scheduler = providers.Factory(_create_scheduler, optimizer=optimizer, epochs=config.training.epochs, warmup_epochs=config.training.gradient_based.warmup_epochs)
    
    # --- ▼▼▼ 【!!! HPO修正 (4/7): tokenizer -> tokenizer.provided】 ▼▼▼ ---
    standard_trainer = providers.Factory(BreakthroughTrainer, criterion=providers.Factory(CombinedLoss, ce_weight=config.training.gradient_based.loss.ce_weight, spike_reg_weight=config.training.gradient_based.loss.spike_reg_weight, mem_reg_weight=config.training.gradient_based.loss.mem_reg_weight, sparsity_reg_weight=config.training.gradient_based.loss.sparsity_reg_weight, tokenizer=tokenizer.provided, ewc_weight=config.training.gradient_based.loss.ewc_weight), grad_clip_norm=config.training.gradient_based.grad_clip_norm, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    # --- ▲▲▲ 【!!! HPO修正 (4/7)】 ▲▲▲ ---
    
    # --- ▼▼▼ 【!!! HPO修正 (5/7): tokenizer -> tokenizer.provided】 ▼▼▼ ---
    distillation_trainer = providers.Factory(DistillationTrainer, criterion=providers.Factory(DistillationLoss, tokenizer=tokenizer.provided, ce_weight=config.training.gradient_based.distillation.loss.ce_weight, distill_weight=config.training.gradient_based.distillation.loss.distill_weight, spike_reg_weight=config.training.gradient_based.distillation.loss.spike_reg_weight, mem_reg_weight=config.training.gradient_based.distillation.loss.mem_reg_weight, sparsity_reg_weight=config.training.gradient_based.distillation.loss.sparsity_reg_weight, temperature=config.training.gradient_based.distillation.loss.temperature), grad_clip_norm=config.training.gradient_based.grad_clip_norm, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    # --- ▲▲▲ 【!!! HPO修正 (5/7)】 ▲▲▲ ---
    
    pi_optimizer = providers.Factory(AdamW, lr=config.training.physics_informed.learning_rate)
    pi_scheduler = providers.Factory(_create_scheduler, optimizer=pi_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.physics_informed.warmup_epochs)
    
    # --- ▼▼▼ 【!!! HPO修正 (6/7): tokenizer -> tokenizer.provided】 ▼▼▼ ---
    physics_informed_trainer = providers.Factory(PhysicsInformedTrainer, criterion=providers.Factory(PhysicsInformedLoss, ce_weight=config.training.physics_informed.loss.ce_weight, spike_reg_weight=config.training.physics_informed.loss.spike_reg_weight, mem_smoothness_weight=config.training.physics_informed.loss.mem_smoothness_weight, tokenizer=tokenizer.provided), grad_clip_norm=config.training.physics_informed.grad_clip_norm, use_amp=config.training.physics_informed.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    # --- ▲▲▲ 【!!! HPO修正 (6/7)】 ▲▲▲ ---
    
    # --- ▼ 改善 (v6): 生物学的学習ルールの定義 ▼ ---
    
    # 1. シナプス可塑性ルール (例: CausalTrace)
    synaptic_learning_rule = providers.Factory(
        get_bio_learning_rule,
        name=config.training.biologically_plausible.learning_rule, # "CAUSAL_TRACE"
        # --- ▼ 修正 (v_health_check_fix_v8): .provided を使用 ▼ ---
        params=config.training.biologically_plausible.provided 
        # --- ▲ 修正 ▲ ---
    )
    
    # 2. 恒常性維持ルール (例: BCM)
    homeostatic_learning_rule = providers.Factory(
        get_bio_learning_rule,
        name="BCM", 
        # --- ▼ 修正 (v_health_check_fix_v8): .provided を使用 ▼ ---
        params=config.training.biologically_plausible.provided
        # --- ▲ 修正 ▲ ---
    )
    
    # --- ▲ 改善 (v6) ▲ ---

    # --- ▼ 改善 (v6): bio_rl_agent のファクトリを修正 ▼ ---
    bio_rl_agent = providers.Factory(
        ReinforcementLearnerAgent, 
        input_size=4, 
        output_size=4, 
        device=device,
        synaptic_rule=synaptic_learning_rule,     # 注入
        homeostatic_rule=homeostatic_learning_rule # 注入
    )
    # --- ▲ 改善 (v6) ▲ ---

    grid_world_env = providers.Factory(GridWorldEnv, device=device)
    bio_rl_trainer = providers.Factory(BioRLTrainer, agent=bio_rl_agent, env=grid_world_env)
    
    particle_filter_trainer = providers.Factory(
        ParticleFilterTrainer, 
        base_model=providers.Factory(BioSNN, layer_sizes=[10, 5, 2], neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0}, synaptic_rule=providers.Object(None), homeostatic_rule=providers.Object(None)), 
        # --- ▼ 修正 (v_health_check_fix_v9) ▼ ---
        config=config.provided, # config -> config.provided
        # --- ▲ 修正 (v_health_check_fix_v9) ▲ ---
        device=device
    )
    
    # --- ▼▼▼ 【!!! HPO修正 (7/7): tokenizer -> tokenizer.provided】 ▼▼▼ ---
    planner_snn = providers.Factory(PlannerSNN, vocab_size=providers.Callable(len, tokenizer.provided), d_model=config.model.d_model, d_state=config.model.d_state, num_layers=config.model.num_layers, time_steps=config.model.time_steps, n_head=config.model.n_head, num_skills=10, neuron_config=config.model.neuron)
    # --- ▲▲▲ 【!!! HPO修正 (7/7)】 ▲▲▲ ---
    
    planner_optimizer = providers.Factory(AdamW, lr=config.training.planner.learning_rate)
    planner_loss = providers.Factory(PlannerLoss)
    model_registry: providers.Provider[ModelRegistry] = providers.Selector(
        # --- ▼ 修正 (v_health_check_fix_v8): .provided を使用 ▼ ---
        providers.Callable(
            lambda cfg: cfg.get("model_registry", {}).get("provider", "file"), 
            cfg=config.provided # config.provided を lambda の cfg 引数に注入
        ),
        # --- ▲ 修正 ▲ ---
        file=providers.Singleton(SimpleModelRegistry, registry_path=config.model_registry.file.path.or_none()),
        distributed=providers.Singleton(DistributedModelRegistry, registry_path=config.model_registry.file.path.or_none())
    )
    # --- ▼ 修正 (v_health_check_fix_v8): .provided と .get() を使用 ▼ ---
    probabilistic_neuron_params: providers.Provider[Dict[str, Any]] = providers.Factory(
        lambda cfg: cfg.get('training', {}).get('biologically_plausible', {}).get('probabilistic_neuron', {}), 
        cfg=config.provided
    )
    probabilistic_learning_rule: providers.Provider[Optional[BioLearningRule]] = providers.Factory(
        lambda cfg: ProbabilisticHebbian(
            learning_rate=cfg.get('training', {}).get('biologically_plausible', {}).get('probabilistic_hebbian', {}).get('learning_rate', 0.01), 
            weight_decay=cfg.get('training', {}).get('biologically_plausible', {}).get('probabilistic_hebbian', {}).get('weight_decay', 0.0)
        ) if cfg.get('training', {}).get('biologically_plausible', {}).get('probabilistic_hebbian') else None, 
        cfg=config.provided
    )
    probabilistic_model = providers.Factory(
        BioSNN, 
        layer_sizes=[10, 5, 2], 
        neuron_params=probabilistic_neuron_params.provider, # .provider (Factory) を渡す
        synaptic_rule=probabilistic_learning_rule.provider, # .provider (Factory) を渡す
        homeostatic_rule=providers.Object(None), 
        sparsification_config=config.training.biologically_plausible.adaptive_causal_sparsification.provided
    )
    probabilistic_agent = providers.Factory(
        ReinforcementLearnerAgent, 
        input_size=4, output_size=4, device=device, 
        synaptic_rule=probabilistic_learning_rule.provider, # .provider (Factory) を渡す
        homeostatic_rule=providers.Object(None)
    )
    probabilistic_trainer = providers.Factory(BioRLTrainer, agent=probabilistic_agent, env=grid_world_env)
    bio_learning_rule = providers.Factory(
        get_bio_learning_rule, 
        name=config.training.biologically_plausible.learning_rule, 
        params=config.training.biologically_plausible.provided
    )


class AgentContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
