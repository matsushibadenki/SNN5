# ファイルパス: snn_research/agent/digital_life_form.py
# (mypy修正 v5)

import time
import logging
import torch
import random
import json
import asyncio
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import operator
import os

from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgentMaster # Master クラスをインポート
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner

if TYPE_CHECKING:
    from app.adapters.snn_langchain_adapter import SNNLangChainAdapter
    from snn_research.training.bio_trainer import BioRLTrainer
    from snn_research.rl_env.grid_world import GridWorldEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DigitalLifeForm:
    """
    プランナーと連携し、目標に基づいた計画を実行する、進化したオーケストレーター。
    """
    def __init__(
        self,
        planner: HierarchicalPlanner,
        autonomous_agent: AutonomousAgent,
        rl_agent: ReinforcementLearnerAgent,
        self_evolving_agent: SelfEvolvingAgentMaster, # Master クラスの型ヒントを使用
        motivation_system: IntrinsicMotivationSystem,
        meta_cognitive_snn: MetaCognitiveSNN,
        memory: Memory,
        physics_evaluator: PhysicsEvaluator,
        symbol_grounding: SymbolGrounding,
        langchain_adapter: "SNNLangChainAdapter",
        global_workspace: GlobalWorkspace
    ):
        self.planner = planner
        self.autonomous_agent = autonomous_agent
        self.rl_agent = rl_agent
        self.self_evolving_agent = self_evolving_agent
        self.motivation_system = motivation_system
        self.meta_cognitive_snn = meta_cognitive_snn
        self.memory = memory
        self.physics_evaluator = physics_evaluator
        self.symbol_grounding = symbol_grounding
        self.langchain_adapter = langchain_adapter
        self.workspace = global_workspace
        self.running = False
        self.state: Dict[str, Any] = {"last_action": None, "last_result": None, "last_task": "unknown"}

    # ... (start, stop, life_cycle, life_cycle_step, _formulate_goal, _handle_causal_credit は変更なし) ...
    def start(self): self.running = True; logging.info("DigitalLifeForm activated."); asyncio.create_task(self.life_cycle())
    def stop(self): self.running = False; logging.info("DigitalLifeForm deactivating.")

    async def life_cycle(self):
        while self.running:
            await self.life_cycle_step()
            await asyncio.sleep(10)

    async def life_cycle_step(self):
        logging.info("\n--- 🧠 New Cognitive Cycle ---")
        self._handle_causal_credit()

        internal_state = self.motivation_system.get_internal_state()
        performance_eval = self.meta_cognitive_snn.evaluate_performance()
        goal = self._formulate_goal(internal_state, performance_eval)
        logging.info(f"🎯 New Goal: {goal}")

        plan = await self.planner.create_plan(goal)

        if not plan.task_list:
            logging.warning("Planner could not create a plan. Idling.")
            return

        logging.info(f"📋 Plan Created: {[task.get('task') for task in plan.task_list]}")
        for task in plan.task_list:
            action = task.get('task')
            if not action: continue

            logging.info(f"▶️ Executing task: {action}")
            result, reward, expert_used = await self._execute_action(action, internal_state, performance_eval)

            if isinstance(result, dict): self.symbol_grounding.process_observation(result, context=f"action '{action}'")
            decision_context = {"goal": goal, "plan": [t.get('task') for t in plan.task_list], "performance_eval": performance_eval, "internal_state": internal_state}
            self.memory.record_experience(self.state, action, result, {"external": reward}, expert_used, decision_context)

            prediction_error = random.random() * 0.5; success_rate = 1.0 if reward > 0 else 0.0
            task_similarity = random.random(); loss = random.random() * 0.1
            self.motivation_system.update_metrics(prediction_error, success_rate, task_similarity, loss)

            self.state["last_action"] = action; self.state["last_result"] = result
            logging.info(f"  - Task Result: {str(result)[:100]}, Reward: {reward:.2f}")

            if reward < 0:
                 logging.warning(f"  - Task '{action}' failed. Considering self-evolution...")
                 evolve_result = self.self_evolving_agent.evolve(performance_eval, internal_state)
                 logging.info(f"    - Evolution result: {evolve_result}")
                 break

    def _formulate_goal(self, internal_state: Dict[str, Any], performance_eval: Dict[str, Any]) -> str:
        if internal_state.get("curiosity", 0.0) > 0.8 and internal_state.get("curiosity_context"):
            topic = internal_state.get("curiosity_context"); topic_str = str(topic)[:50] if topic else "unknown"
            return f"Explore unknown concept related to '{topic_str}'."
        if performance_eval.get("status") == "capability_gap": return "Evolve architecture/parameters for capability gap."
        if internal_state.get("boredom", 0.0) > 0.7: return "Explore new random task for boredom."
        return "Practice existing skill."

    def _handle_causal_credit(self):
        conscious_content = self.workspace.conscious_broadcast_content
        if conscious_content and isinstance(conscious_content, dict) and conscious_content.get("type") == "causal_credit":
            target_action = conscious_content.get("target_action"); credit = conscious_content.get("credit", 0.0)
            print(f"✨ Causal credit detected! Target: {target_action}, Credit: {credit}")
            if self.state.get("last_action") and target_action == f"action_{self.state['last_action']}":
                print(f"  - Applying credit to last action '{self.state['last_action']}'.")
                if self.state['last_action'] in ["practice_skill_with_rl"]:
                    self.rl_agent.learn(reward=0.0, causal_credit=credit)
                    print("  - Modulated RL agent plasticity.")

    async def _execute_action(self, action: str, internal_state: Dict[str, Any], performance_eval: Dict[str, Any]) -> tuple[Dict[str, Any], float, List[str]]: # Make async
        from snn_research.rl_env.grid_world import GridWorldEnv
        from snn_research.training.bio_trainer import BioRLTrainer
        try:
            if action == "explore_curiosity":
                topic = internal_state.get("curiosity_context"); topic_str = str(topic)[:50] if topic else None
                if not topic_str: return {"status": "skipped", "info": "No curiosity context."}, 0.0, []
                logging.info(f"🔬 Researching topic: '{topic_str}'")
                new_model_info = await self.autonomous_agent.handle_task(task_description=topic_str, unlabeled_data_path="data/sample_data.jsonl", force_retrain=True)
                success = new_model_info and "error" not in new_model_info
                return {"status": "success" if success else "failure", "info": f"Learned about '{topic_str}'." if success else f"Failed learning '{topic_str}'. Info: {new_model_info}", "model_info": new_model_info}, (1.0 if success else -0.5), ["autonomous_agent"]
            elif action.startswith("Evolve"):
                evolve_result = self.self_evolving_agent.evolve(performance_eval, internal_state)
                success = "failed" not in evolve_result.lower()
                return {"status": "success" if success else "failure", "info": evolve_result}, (0.9 if success else -0.2), ["self_evolver"]
            elif action.startswith("practice") or action.startswith("Practice"):
                env = GridWorldEnv(size=5, max_steps=20, device=self.rl_agent.device)
                trainer = BioRLTrainer(agent=self.rl_agent, env=env)
                res = trainer.train(num_episodes=10)
                reward = res.get('final_average_reward', 0.0)
                return {"status": "success", "results": res}, reward, ["rl_agent"]
            elif action.startswith("Answer"):
                 question = action.split(":")[-1].strip() if ":" in action else "What is SNN?"
                 model_info = await self.autonomous_agent.handle_task(task_description="general_qa")
                 # --- ▼ 修正: is not None チェックを明示的に ▼ ---
                 is_success = model_info is not None and "error" not in model_info
                 expert_id = 'unknown'
                 if is_success and model_info is not None: # model_info is not None を追加
                     expert_id = model_info.get('model_id','unknown')
                 # --- ▲ 修正 ▲ ---
                 response = {"response": f"Answered '{question}' using {expert_id}. (Sim)"} if is_success else {}
                 return {"status": "success" if is_success else "failure", "info": response if is_success else f"Failed QA for '{question}'. Expert missing/failed."}, (0.8 if is_success else -0.3), ["autonomous_agent", expert_id]
            else:
                 logging.warning(f"Unknown action: {action}. Idling.")
                 return {"status": "idle", "info": f"Unknown action: {action}"}, 0.0, []
        except Exception as e:
            logging.error(f"Error executing action '{action}': {e}", exc_info=True)
            return {"status": "error", "info": str(e)}, -1.0, []

    # ... (awareness_loop, explain_last_action は変更なし) ...
    async def awareness_loop(self, cycles: int): # Make async
        print(f"🧬 Awareness loop starting for {cycles} cycles.")
        self.running = True
        for i in range(cycles):
            if not self.running: break
            print(f"\n----- Cycle {i+1}/{cycles} -----")
            await self.life_cycle_step() # Use await
            await asyncio.sleep(2) # Use asyncio.sleep
        print("🧬 Awareness loop finished.")

    async def explain_last_action(self) -> Optional[str]: # Make async
        last_experience = None
        try:
            if not os.path.exists(self.memory.memory_path): return "行動履歴ファイルが見つかりません。"
            with open(self.memory.memory_path, "rb") as f:
                try: # Seek to find last line efficiently
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n': f.seek(-2, os.SEEK_CUR)
                except OSError: f.seek(0)
                last_line = f.readline().decode()
                if last_line: last_experience = json.loads(last_line)
        except Exception as e:
            logging.error(f"Error reading memory: {e}")
            return f"行動履歴読込エラー: {e}"

        if not last_experience: return "行動履歴が空または読み込めません。"

        decision_context = last_experience.get('decision_context', {})
        internal_state_log = decision_context.get('internal_state', {})
        performance_eval_log = decision_context.get('performance_eval', {})

        prompt = f"""
        あなたは自身の行動理由を説明するAIです。以下ログに基づき一人称で説明してください。
        行動: {last_experience.get('action')}
        根拠: 動機={internal_state_log}, 自己評価={performance_eval_log}, 物理評価={last_experience.get('reward', {}).get('physical_rewards', 'N/A')}
        指示: 思考プロセスを平易に説明せよ。
        """
        try:
            snn_llm = self.langchain_adapter
            explanation = await asyncio.to_thread(snn_llm._call, prompt) # Run sync _call in thread
            return explanation
        except Exception as e:
            logging.error(f"LLM explanation generation failed: {e}")
            return "エラー: 自己言及生成失敗。"