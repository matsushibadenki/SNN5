# ファイルパス: snn_research/cognitive_architecture/hierarchical_planner.py
# (修正)
# 循環インポートエラーを解消するため、TYPE_CHECKINGを使用して
# Memoryの型ヒントを解決する。

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import torch
from transformers import AutoTokenizer
import asyncio

from .planner_snn import PlannerSNN
from snn_research.distillation.model_registry import ModelRegistry
from .rag_snn import RAGSystem

# --- ▼ 修正 ▼ ---
if TYPE_CHECKING:
    from snn_research.agent.memory import Memory
# --- ▲ 修正 ▲ ---

class Plan:
    def __init__(self, goal: str, task_list: List[Dict[str, Any]]):
        self.goal = goal
        self.task_list = task_list
    def __repr__(self) -> str:
        return f"Plan(goal='{self.goal}', tasks={len(self.task_list)})"

class HierarchicalPlanner:
    def __init__(
        self,
        model_registry: ModelRegistry,
        rag_system: RAGSystem,
        # --- ▼ 修正 ▼ ---
        memory: "Memory",
        # --- ▲ 修正 ▲ ---
        planner_model: Optional[PlannerSNN] = None,
        tokenizer_name: str = "gpt2",
        device: str = "cpu"
    ):
        self.model_registry = model_registry
        self.rag_system = rag_system
        self.memory = memory
        self.planner_model = planner_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = getattr(self.tokenizer, 'model_max_length', 1024)
        self.device = device
        if self.planner_model: self.planner_model.to(self.device)
        self.SKILL_MAP: Dict[int, Dict[str, Any]] = asyncio.run(self._build_skill_map())
        print(f"🧠 Planner initialized with {len(self.SKILL_MAP)} skills and Causal Memory access.")

    async def _build_skill_map(self) -> Dict[int, Dict[str, Any]]:
        all_models = await self.model_registry.list_models()
        skill_map: Dict[int, Dict[str, Any]] = {}
        for i, model_info in enumerate(all_models):
            skill_map[i] = {"task": model_info.get("model_id"), "description": model_info.get("task_description"), "expert_id": model_info.get("model_id")}
        if not any(skill['task'] == 'general_qa' for skill in skill_map.values()):
            skill_map[len(skill_map)] = {"task": "general_qa", "description": "Answer a general question.", "expert_id": "general_snn_v3"}
        return skill_map

    def _create_rule_based_plan(self, prompt: str, skills_to_avoid: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if skills_to_avoid is None: skills_to_avoid = []
        task_list = []
        prompt_lower = prompt.lower()
        available_skills = [s for s in self.SKILL_MAP.values() if s.get('task') not in skills_to_avoid]
        for skill in available_skills:
            task_keywords = (skill.get('task') or '').lower().split('_')
            desc_keywords = (skill.get('description') or '').lower().split()
            if any(kw in prompt_lower for kw in task_keywords if kw) or any(kw in prompt_lower for kw in desc_keywords if kw):
                if skill not in task_list: task_list.append(skill)
        if not task_list and not skills_to_avoid:
            fallback = next((s for s in available_skills if "general" in (s.get("task") or "")), None)
            if fallback: task_list.append(fallback)
        return task_list

    async def create_plan(
        self,
        high_level_goal: str,
        context: Optional[str] = None,
        skills_to_avoid: Optional[List[str]] = None
    ) -> Plan:
        if skills_to_avoid is None: skills_to_avoid = []
        print(f"🌍 Creating plan for goal: {high_level_goal}, avoiding skills: {skills_to_avoid}")
        self.SKILL_MAP = await self._build_skill_map()
        task_list = self._create_rule_based_plan(high_level_goal, skills_to_avoid)
        print(f"✅ Plan created with {len(task_list)} step(s).")
        return Plan(goal=high_level_goal, task_list=task_list)

    async def refine_plan_after_failure(
        self,
        failed_plan: Plan,
        failed_task: Dict[str, Any]
    ) -> Optional[Plan]:
        print(f"🤔 Task '{failed_task.get('task')}' failed. Refining plan using causal memory...")
        causal_query = f"The action '{failed_task.get('task')}' resulted in a failure while pursuing the goal '{failed_plan.goal}'."
        similar_failures = self.memory.retrieve_similar_experiences(causal_query=causal_query, top_k=3)

        skills_to_avoid_set: set[str] = set()
        failed_task_name = failed_task.get('task')
        if failed_task_name:
            skills_to_avoid_set.add(failed_task_name)

        if similar_failures:
            print("  - Found similar past failures. Analyzing causes...")
            for failure in similar_failures:
                retrieved_text = failure.get("retrieved_causal_text", "")
                if "leads to the effect 'failure'" in retrieved_text:
                    parts = retrieved_text.split("'")
                    if len(parts) >= 4:
                        cause_event = parts[3]
                        if cause_event.startswith("action_"):
                            failed_action = cause_event.replace("action_", "")
                            print(f"    - Past data suggests that action '{failed_action}' often leads to failure in this context.")
                            skills_to_avoid_set.add(failed_action)
        
        skills_to_avoid_list = list(skills_to_avoid_set)
        print(f"  - Attempting to create a new plan avoiding: {skills_to_avoid_list}")
        new_plan = await self.create_plan(
            high_level_goal=failed_plan.goal,
            skills_to_avoid=skills_to_avoid_list
        )

        if new_plan.task_list and new_plan.task_list != failed_plan.task_list:
            print("✅ Successfully created a revised plan.")
            return new_plan
        else:
            print("❌ Could not find a viable alternative plan.")
            return None

    def execute_task(self, task_request: str, context: str) -> Optional[str]:
        plan = asyncio.run(self.create_plan(task_request, context))
        if plan.task_list:
            failed_task = plan.task_list[0]
            print(f"\n--- [SIMULATION] Task '{failed_task.get('task')}' is assumed to have failed. ---")
            
            new_plan = asyncio.run(self.refine_plan_after_failure(plan, failed_task))
            
            if new_plan:
                return f"Original plan failed. Revised plan: {new_plan.task_list}"
            else:
                return "Original plan failed and no alternative was found."
        return "Could not create an initial plan."