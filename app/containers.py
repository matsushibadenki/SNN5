# ファイルパス: app/containers.py
# (動的ロードUI対応 v5 - TypeError 修正 v2)
# 修正: _registry_path_provider を .get() で安全にアクセスするように変更。
#
# 改善 (v6):
# - doc/The-flow-of-brain-behavior.md との整合性を高めるため、
#   snn_research/agent/reinforcement_learner_agent.py (v2) の変更に対応。
# - `bio_rl_agent` プロバイダが、`synaptic_rule` と `homeostatic_rule` の
#   両方をインスタンス化して `ReinforcementLearnerAgent` に注入するように修正。
#
# 修正 (v7):
# - run_brain_simulation.py での TypeError: stat: path should be string... not NoneType を修正。
# - _load_planner_snn_factory (L83) で model_path が None の場合に os.path.exists を呼び出さないよう修正。

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
    tokenizer = providers.Factory(AutoTokenizer.from_pretrained, pretrained_model_name_or_path=config.data.tokenizer_name)
    
    @providers.Factory
    def tokenizer(config_provider=config):
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
        SNNCore,
        config=config.model.provided, # .provided を追加
        vocab_size=tokenizer.provided.vocab_size
    )
    astrocyte_network = providers.Factory(AstrocyteNetwork, snn_model=snn_model)
    meta_cognitive_snn: providers.Provider[MetaCognitiveSNN] = providers.Factory(MetaCognitiveSNN, **(config.training.meta_cognition.to_dict() or {}))
    optimizer = providers.Factory(AdamW, lr=config.training.gradient_based.learning_rate)
    scheduler = providers.Factory(_create_scheduler, optimizer=optimizer, epochs=config.training.epochs, warmup_epochs=config.training.gradient_based.warmup_epochs)
    standard_trainer = providers.Factory(BreakthroughTrainer, criterion=providers.Factory(CombinedLoss, ce_weight=config.training.gradient_based.loss.ce_weight, spike_reg_weight=config.training.gradient_based.loss.spike_reg_weight, mem_reg_weight=config.training.gradient_based.loss.mem_reg_weight, sparsity_reg_weight=config.training.gradient_based.loss.sparsity_reg_weight, tokenizer=tokenizer, ewc_weight=config.training.gradient_based.loss.ewc_weight), grad_clip_norm=config.training.gradient_based.grad_clip_norm, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    distillation_trainer = providers.Factory(DistillationTrainer, criterion=providers.Factory(DistillationLoss, tokenizer=tokenizer, ce_weight=config.training.gradient_based.distillation.loss.ce_weight, distill_weight=config.training.gradient_based.distillation.loss.distill_weight, spike_reg_weight=config.training.gradient_based.distillation.loss.spike_reg_weight, mem_reg_weight=config.training.gradient_based.distillation.loss.mem_reg_weight, sparsity_reg_weight=config.training.gradient_based.distillation.loss.sparsity_reg_weight, temperature=config.training.gradient_based.distillation.loss.temperature), grad_clip_norm=config.training.gradient_based.grad_clip_norm, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    pi_optimizer = providers.Factory(AdamW, lr=config.training.physics_informed.learning_rate)
    pi_scheduler = providers.Factory(_create_scheduler, optimizer=pi_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.physics_informed.warmup_epochs)
    physics_informed_trainer = providers.Factory(PhysicsInformedTrainer, criterion=providers.Factory(PhysicsInformedLoss, ce_weight=config.training.physics_informed.loss.ce_weight, spike_reg_weight=config.training.physics_informed.loss.spike_reg_weight, mem_smoothness_weight=config.training.physics_informed.loss.mem_smoothness_weight, tokenizer=tokenizer), grad_clip_norm=config.training.physics_informed.grad_clip_norm, use_amp=config.training.physics_informed.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    
    # --- ▼ 改善 (v6): 生物学的学習ルールの定義 ▼ ---
    
    # 1. シナプス可塑性ルール (例: CausalTrace)
    synaptic_learning_rule = providers.Factory(
        get_bio_learning_rule,
        name=config.training.biologically_plausible.learning_rule, # "CAUSAL_TRACE"
        params=config.training.biologically_plausible # パラメータ辞書全体を渡す
    )
    
    # 2. 恒常性維持ルール (例: BCM)
    #    base_config.yaml の "bcm" ブロックを参照する
    homeostatic_learning_rule = providers.Factory(
        get_bio_learning_rule,
        name="BCM", # ハードコード (またはconfigで指定)
        params=config.training.biologically_plausible # BCMパラメータもこの配下にあると仮定
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
    
    # ( ... 残りのプロバイダは変更なし ...)
    particle_filter_trainer = providers.Factory(ParticleFilterTrainer, base_model=providers.Factory(BioSNN, layer_sizes=[10, 5, 2], neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0}, synaptic_rule=providers.Object(None), homeostatic_rule=providers.Object(None)), config=config, device=device) # BioSNNの引数を修正
    planner_snn = providers.Factory(PlannerSNN, vocab_size=providers.Callable(len, tokenizer), d_model=config.model.d_model, d_state=config.model.d_state, num_layers=config.model.num_layers, time_steps=config.model.time_steps, n_head=config.model.n_head, num_skills=10, neuron_config=config.model.neuron)
    planner_optimizer = providers.Factory(AdamW, lr=config.training.planner.learning_rate)
    planner_loss = providers.Factory(PlannerLoss)
    model_registry: providers.Provider[ModelRegistry] = providers.Selector(
        providers.Callable(lambda cfg: cfg.get("model_registry", {}).get("provider", "file"), config.provided),
        file=providers.Singleton(SimpleModelRegistry, registry_path=config.model_registry.file.path.or_none()),
        distributed=providers.Singleton(DistributedModelRegistry, registry_path=config.model_registry.file.path.or_none())
    )
    probabilistic_neuron_params: providers.Provider[Dict[str, Any]] = providers.Factory(lambda cfg: cfg.training.biologically_plausible.probabilistic_neuron.to_dict() if cfg.training.biologically_plausible.probabilistic_neuron() else {}, config.provided)
    probabilistic_learning_rule: providers.Provider[Optional[BioLearningRule]] = providers.Factory(lambda cfg: ProbabilisticHebbian(learning_rate=cfg.training.biologically_plausible.probabilistic_hebbian.learning_rate.as_float(), weight_decay=cfg.training.biologically_plausible.probabilistic_hebbian.weight_decay.as_float()) if cfg.training.biologically_plausible.probabilistic_hebbian() else None, config.provided)
    probabilistic_model = providers.Factory(BioSNN, layer_sizes=[10, 5, 2], neuron_params=probabilistic_neuron_params, synaptic_rule=probabilistic_learning_rule, homeostatic_rule=providers.Object(None), sparsification_config=config.training.biologically_plausible.adaptive_causal_sparsification) # BioSNNの引数を修正
    probabilistic_agent = providers.Factory(ReinforcementLearnerAgent, input_size=4, output_size=4, device=device, synaptic_rule=probabilistic_learning_rule, homeostatic_rule=providers.Object(None)) # Agentの引数を修正
    probabilistic_trainer = providers.Factory(BioRLTrainer, agent=probabilistic_agent, env=grid_world_env)
    bio_learning_rule = providers.Factory( get_bio_learning_rule, name=config.training.biologically_plausible.learning_rule, params=config.training.biologically_plausible )

class AgentContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    training_container: providers.Provider[TrainingContainer] = providers.Container(TrainingContainer, config=config)
    device = providers.Factory(get_auto_device)
    model_registry = providers.Callable(lambda tc: tc.model_registry(), tc=training_container)
    web_crawler = providers.Singleton(WebCrawler)
    rag_system = providers.Factory(RAGSystem, vector_store_path=providers.Callable(lambda log_dir: os.path.join(log_dir, "vector_store") if log_dir else "runs/vector_store", log_dir=config.training.log_dir))
    memory = providers.Factory(Memory, rag_system=rag_system, memory_path=providers.Callable(lambda log_dir: os.path.join(log_dir, "agent_memory.jsonl") if log_dir else "runs/agent_memory.jsonl", log_dir=config.training.log_dir))
    loaded_planner_snn = providers.Singleton( _load_planner_snn_factory, planner_snn_instance=providers.Callable(lambda tc: tc.planner_snn(), tc=training_container), model_path=config.training.planner.model_path.or_none(), device=device )
    hierarchical_planner = providers.Factory( HierarchicalPlanner, model_registry=model_registry, rag_system=rag_system, memory=memory, planner_model=loaded_planner_snn, tokenizer_name=config.data.tokenizer_name, device=device )
    autonomous_agent = providers.Singleton( AutonomousAgent, name="AutonomousAgentBase", planner=hierarchical_planner, model_registry=model_registry, memory=memory, web_crawler=web_crawler )
    self_evolving_agent_master = providers.Singleton( # 名前を変更
        SelfEvolvingAgentMaster, # Master クラスを使用
        name="SelfEvolvingAgentMaster", # 名前を更新
        planner=hierarchical_planner,
        model_registry=model_registry, # DistributedModelRegistry を期待
        memory=memory,
        web_crawler=web_crawler,
        meta_cognitive_snn=providers.Callable( lambda tc: tc.meta_cognitive_snn(), tc=training_container.provider ),
        motivation_system=providers.Singleton(IntrinsicMotivationSystem), # ここで Singleton として定義
        model_config_path=config.model.path.or_none(), # 設定から取得
        training_config_path=providers.Object("configs/base_config.yaml") # 固定パス or 設定から取得
    )


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # --- ▼ 修正: _registry_path_provider を .get() で安全にアクセスするように変更 ▼ ---
    _registry_path_provider = providers.Callable(
        # cfg引数にはconfigプロバイダーの「値」(dictまたはDictConfig)が渡される
        # したがって cfg() ではなく cfg.get() を使う
        # --- ▼▼▼ 修正: cfgがNoneの場合のフォールバックを追加 ▼▼▼ ---
        lambda cfg: (
            cfg.get('model_registry', {}).get('file', {}).get('path', "runs/model_registry.json")
            if cfg is not None and isinstance(cfg, dict) 
            else "runs/model_registry.json" # cfgがNoneの場合のフォールバック
        ),
        # --- ▲▲▲ 修正 ▲▲▲ ---
        cfg=config
    )
    
    model_registry = providers.Singleton(
        SimpleModelRegistry,
        registry_path=_registry_path_provider # configに依存
    )
    # --- ▲ 修正 ▲ ---

    # --- SNN Inference Engines (動的ロードのためFactoryのまま) ---
    snn_inference_engine = providers.Factory(SNNInferenceEngine)

    # --- Services (動的ロードのためFactoryのまま) ---
    chat_service = providers.Factory(
        ChatService,
        snn_engine=snn_inference_engine
        # max_len は ChatService 内部で snn_engine.config から取得
    )
    image_classification_service = providers.Factory(
        ImageClassificationService,
        engine=snn_inference_engine
    )
    
    langchain_adapter: providers.Provider['SNNLangChainAdapter'] = providers.Factory(
        SNNLangChainAdapter,
        snn_engine=snn_inference_engine
    )


class BrainContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    agent_container: providers.Provider[AgentContainer] = providers.Container(AgentContainer, config=config)
    app_container = providers.Container(AppContainer, config=config)
    global_workspace = providers.Singleton( GlobalWorkspace, model_registry=providers.Callable(lambda ac: ac.model_registry(), ac=agent_container) )
    motivation_system = providers.Callable(lambda ac: ac.self_evolving_agent_master().motivation_system, ac=agent_container) # Master Agent から取得
    num_neurons = providers.Factory(lambda: 256) # Define num_neurons
    sensory_receptor = providers.Singleton(SensoryReceptor)
    spike_encoder = providers.Singleton(SpikeEncoder, num_neurons=num_neurons) # Use defined num_neurons
    actuator = providers.Singleton(Actuator, actuator_name="voice_synthesizer")
    perception_cortex = providers.Singleton(HybridPerceptionCortex, workspace=global_workspace, num_neurons=num_neurons, feature_dim=64, som_map_size=(8, 8), stdp_params=config.training.biologically_plausible.stdp.to_dict()) # Use defined num_neurons
    prefrontal_cortex = providers.Singleton(PrefrontalCortex, workspace=global_workspace, motivation_system=motivation_system)
    hippocampus = providers.Singleton(Hippocampus, workspace=global_workspace, capacity=50)
    cortex = providers.Singleton(Cortex)
    amygdala = providers.Singleton(Amygdala, workspace=global_workspace)
    basal_ganglia = providers.Singleton(BasalGanglia, workspace=global_workspace)
    cerebellum = providers.Singleton(Cerebellum)
    motor_cortex = providers.Singleton(MotorCortex, actuators=['voice_synthesizer'])
    causal_inference_engine = providers.Singleton( CausalInferenceEngine, rag_system=providers.Callable(lambda ac: ac.rag_system(), ac=agent_container), workspace=global_workspace )
    artificial_brain = providers.Singleton( ArtificialBrain, global_workspace=global_workspace, motivation_system=motivation_system, sensory_receptor=sensory_receptor, spike_encoder=spike_encoder, actuator=actuator, perception_cortex=perception_cortex, prefrontal_cortex=prefrontal_cortex, hippocampus=hippocampus, cortex=cortex, amygdala=amygdala, basal_ganglia=basal_ganglia, cerebellum=cerebellum, motor_cortex=motor_cortex, causal_inference_engine=causal_inference_engine )
    autonomous_agent = providers.Callable(lambda ac: ac.autonomous_agent(), ac=agent_container)
    # --- ▼ 改善 (v6): bio_rl_agent の取得先を training_container に変更 ▼ ---
    rl_agent = providers.Callable(lambda ac: ac.training_container().bio_rl_agent(), ac=agent_container)
    # --- ▲ 改善 (v6) ▲ ---
    self_evolving_agent = providers.Callable(lambda ac: ac.self_evolving_agent_master(), ac=agent_container) # Master Agent を参照
    digital_life_form = providers.Singleton(
        DigitalLifeForm,
        planner=providers.Callable(lambda ac: ac.hierarchical_planner(), ac=agent_container),
        autonomous_agent=autonomous_agent,
        rl_agent=rl_agent,
        self_evolving_agent=self_evolving_agent, # Master Agent を注入
        motivation_system=motivation_system,
        meta_cognitive_snn=providers.Callable( lambda ac_instance: ac_instance.training_container().meta_cognitive_snn(), ac_instance=agent_container.provider ),
        memory=providers.Callable(lambda ac: ac.memory(), ac=agent_container),
        physics_evaluator=providers.Singleton(PhysicsEvaluator),
        symbol_grounding=providers.Singleton( SymbolGrounding, rag_system=providers.Callable(lambda ac: ac.rag_system(), ac=agent_container) ),
        langchain_adapter=app_container.langchain_adapter,
        global_workspace=global_workspace
    )
