# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# タイトル: 人工脳 統合認知サイクル (内発的動機付け統合)
# 機能説明:
# - 人工脳の全コンポーネントを統合し、知覚から行動までの一連の認知プロセスを実行する。
# - グローバル・ワークスペース理論に基づき、「意識的認知サイクル」を実行。
# - サイクル終了時に経験を評価し、内発的動機システムを更新する。
#
# 改善点(v3):
# - IntrinsicMotivationSystemを統合し、認知サイクルの最後に経験（予測誤差、成功率など）を評価して
#   内発的動機（好奇心、退屈など）を更新するロジックを追加。

from typing import Dict, Any, List
import asyncio
import re
import torch

# IO and encoding
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
# Core cognitive modules
from .hybrid_perception_cortex import HybridPerceptionCortex
from .prefrontal_cortex import PrefrontalCortex
# Memory systems
from .hippocampus import Hippocampus
from .cortex import Cortex
# Value and action selection
from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
# Motor control
from .cerebellum import Cerebellum
from .motor_cortex import MotorCortex
# Central hub
from .global_workspace import GlobalWorkspace
# Causal Engine
from .causal_inference_engine import CausalInferenceEngine
# Motivation System
from .intrinsic_motivation import IntrinsicMotivationSystem


class ArtificialBrain:
    """
    認知アーキテクチャ全体を統合し、「意識的認知サイクル」を制御する人工脳システム。
    """
    def __init__(
        self,
        # Central Hub
        global_workspace: GlobalWorkspace,
        # Motivation System
        motivation_system: IntrinsicMotivationSystem,
        # Input/Output
        sensory_receptor: SensoryReceptor,
        spike_encoder: SpikeEncoder,
        actuator: Actuator,
        # Core Cognitive Flow
        perception_cortex: HybridPerceptionCortex,
        prefrontal_cortex: PrefrontalCortex,
        # Memory
        hippocampus: Hippocampus,
        cortex: Cortex,
        # Value and Action
        amygdala: Amygdala,
        basal_ganglia: BasalGanglia,
        # Motor
        cerebellum: Cerebellum,
        motor_cortex: MotorCortex,
        # Causal Engine
        causal_inference_engine: CausalInferenceEngine
    ):
        print("🚀 人工脳システムの起動を開始...")
        self.workspace = global_workspace
        self.motivation_system = motivation_system
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
        self.perception = perception_cortex
        self.pfc = prefrontal_cortex
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.amygdala = amygdala
        self.basal_ganglia = basal_ganglia
        self.cerebellum = cerebellum
        self.motor = motor_cortex
        self.causal_engine = causal_inference_engine # インスタンスを保持
        
        self.cycle_count = 0
        print("✅ 人工脳システムの全モジュールが正常に起動しました。")

    def run_cognitive_cycle(self, raw_input: Any):
        """
        グローバル・ワークスペース理論に基づいた、意識的認知サイクルを実行する。
        """
        self.cycle_count += 1
        print(f"\n--- 🧠 新しい認知サイクルを開始 ({self.cycle_count}) --- \n入力: '{raw_input}'")
        
        # 1. 並列的な情報処理とGlobalWorkspaceへのアップロード
        # 1a. 感覚入力 -> スパイク変換 -> 知覚
        sensory_info = self.receptor.receive(raw_input)
        spike_pattern = self.encoder.encode(sensory_info, duration=50)
        self.perception.perceive_and_upload(spike_pattern)
        
        # 1b. 情動評価
        if isinstance(raw_input, str):
            self.amygdala.evaluate_and_upload(raw_input)

        # 1c. 記憶との関連性評価
        perception_result = self.workspace.get_information("perception")
        if perception_result:
            self.hippocampus.evaluate_relevance_and_upload(perception_result['features'])

        # 2. 意識の選択：最も顕著な情報を選択し、全体へブロードキャスト
        #    (この中でCausalInferenceEngineのコールバックが自動で呼ばれる)
        self.workspace.conscious_broadcast_cycle()
        conscious_content = self.workspace.conscious_broadcast_content
        
        if conscious_content is None:
            print("🤔 意識に上るほどの情報がなく、行動は選択されませんでした。")
            # 意識に何も上らなくても、経験として評価する
            prediction_error, success_rate, task_similarity, loss = 0.1, 0.0, 0.0, 0.1
        else:
            # 3. トップダウン処理と行動決定（各モジュールのコールバックが自動で実行される）
            #    - PFCが目標を決定 (handle_conscious_broadcast)
            #    - BasalGangliaが行動を選択 (handle_conscious_broadcast)
            
            # 4. 運動実行
            selected_action = self.basal_ganglia.selected_action
            if selected_action:
                motor_commands = self.cerebellum.refine_action_plan(selected_action)
                command_logs = self.motor.execute_commands(motor_commands)
                self.actuator.run_command_sequence(command_logs)

            # 5. 記憶の固定：意識に上った情報をエピソードとして短期記憶に保存
            episode = {'type': 'conscious_experience', 'content': conscious_content, 'source_input': raw_input}
            self.hippocampus.store_episode(episode)

            # --- 6. 経験の評価と内発的動機の更新 ---
            # 6a. prediction_errorの計算
            prediction_error = 0.9 if self.causal_engine.just_inferred else 0.1
            self.causal_engine.reset_inference_flag()

            # 6b. success_rateの計算
            success_rate = 1.0 if self.basal_ganglia.selected_action else 0.0

            # 6c. task_similarityの計算
            task_similarity = 0.0
            recent_episodes = self.hippocampus.retrieve_recent_episodes(1)
            if recent_episodes:
                last_episode_content = recent_episodes[0].get('content')
                if isinstance(last_episode_content, dict) and isinstance(conscious_content, dict):
                     current_features = conscious_content.get('features')
                     last_features = last_episode_content.get('features')
                     if isinstance(current_features, torch.Tensor) and isinstance(last_features, torch.Tensor):
                         task_similarity = torch.nn.functional.cosine_similarity(
                             current_features.flatten(),
                             last_features.flatten(),
                             dim=0
                         ).item()

            # 6d. lossの計算 (ダミー)
            loss = 0.1

        print("📊 経験を評価し、内発的動機を更新しています...")
        self.motivation_system.update_metrics(
            prediction_error=prediction_error,
            success_rate=success_rate,
            task_similarity=task_similarity,
            loss=loss
        )
        print(f"  - Motivation metrics updated: error={prediction_error:.2f}, success={success_rate:.2f}, similarity={task_similarity:.2f}")

        # 7. 記憶の固定化 (5サイクルごと)
        if self.cycle_count % 5 == 0:
            self.consolidate_memories()

        print("--- ✅ 認知サイクル完了 ---")

    def consolidate_memories(self):
        """海馬から大脳皮質へ記憶を固定するプロセス。"""
        print("💾 記憶の固定化プロセスを開始...")
        episodes = self.hippocampus.get_and_clear_episodes_for_consolidation()
        for episode in episodes:
            self.cortex.consolidate_memory(episode)