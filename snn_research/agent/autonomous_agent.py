# ファイルパス: snn_research/agent/autonomous_agent.py
# (更新: STRMAC状態認識ルーティングの概念を統合)
# Title: 自律エージェント ベースクラス
# Description: タスクに応じて専門家モデルを選択・学習し、推論を実行する自律エージェントの基本機能を提供します。
#              Web検索、記憶システム、スパイク通信機能などを統合します。
# 改善点: P10.1 STRMACの状態認識ルーティングの概念をfind_expertに導入。

from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union
import asyncio
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf, DictConfig
import re
from collections import Counter
from heapq import nlargest
import json
try:
    from googlesearch import search  # type: ignore
except ImportError:
    print("⚠️ 'googlesearch-python' is not installed. Web search functionality will be limited. Please run 'pip install googlesearch-python'")
    def search(*args, **kwargs):
        return iter([])

from snn_research.distillation.model_registry import ModelRegistry
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from snn_research.tools.web_crawler import WebCrawler
from .memory import Memory as AgentMemory
from snn_research.deployment import SNNInferenceEngine
from snn_research.communication.spike_encoder_decoder import SpikeEncoderDecoder
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

if TYPE_CHECKING:
    from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner


class AutonomousAgent:
    """
    自律的にタスクを実行するエージェントのベースクラス。
    """
    def __init__(
        self,
        name: str,
        planner: "HierarchicalPlanner",
        model_registry: ModelRegistry,
        memory: AgentMemory,
        web_crawler: WebCrawler,
        accuracy_threshold: float = 0.6,
        energy_budget: float = 10000.0
    ):
        self.name = name
        self.planner = planner
        self.model_registry = model_registry
        self.memory = memory
        self.web_crawler = web_crawler
        self.current_state: Dict[str, Union[str, Dict[str, Any], None]] = {
            "agent_name": name, "last_action": None, "last_result": None
        }
        self.accuracy_threshold = accuracy_threshold
        self.energy_budget = energy_budget
        self.spike_communicator = SpikeEncoderDecoder()

    def receive_and_process_spike_message(self, spike_pattern: torch.Tensor, source_agent: str):
        """
        他のエージェントから送信されたスパイクメッセージを受信し、解釈して記憶する。
        """
        print(f"📡 Agent '{self.name}' received a spike message from '{source_agent}'.")
        decoded_message = self.spike_communicator.decode_data(spike_pattern)

        if decoded_message and isinstance(decoded_message, dict) and "error" not in decoded_message:
            print(f"  - Decoded Intent: {decoded_message.get('intent')}")
            print(f"  - Decoded Payload: {decoded_message.get('payload')}")

            # 記憶に記録
            self.memory.record_experience(
                state=self.current_state,
                action="receive_communication",
                result={"decoded_message": decoded_message, "source": source_agent},
                reward={"external": 0.2}, # 通信成功ボーナス
                expert_used=["spike_communicator"],
                decision_context={"reason": "Inter-agent communication received."}
            )
            # 現在の状態を更新 (例)
            self.current_state["last_communication"] = decoded_message
        else:
            # デコード失敗またはエラーの場合
            raw_text = decoded_message if isinstance(decoded_message, str) else str(decoded_message)
            print(f"  - Failed to decode spike message. Raw content: {raw_text}")
            self.memory.record_experience(
                state=self.current_state,
                action="receive_communication_failed",
                result={"raw_content": raw_text, "source": source_agent},
                reward={"external": -0.1}, # デコード失敗ペナルティ
                expert_used=["spike_communicator"],
                decision_context={"reason": "Failed to decode incoming spike message."}
            )

    def execute(self, task_description: str) -> str:
        """
        与えられたタスクを実行する。 (handle_taskへの委譲を想定)
        """
        print(f"Agent '{self.name}' received task: {task_description}")
        # 実際には handle_task を呼び出すことが多い
        # ここでは簡易的に実行ログを返す
        result_info = asyncio.run(self.handle_task(task_description))

        expert_id_list: List[str] = []
        if result_info and result_info.get("path"):
            result = f"Task '{task_description}' handled by Agent '{self.name}'. Outcome: SUCCESS (Expert: {result_info.get('model_id')})"
            reward_val = 1.0
            if result_info.get('model_id'):
                 expert_id_list.append(str(result_info['model_id']))
        elif result_info and "error" in result_info:
             result = f"Task '{task_description}' handled by Agent '{self.name}'. Outcome: FAILURE ({result_info.get('error')})"
             reward_val = -1.0
        else:
            result = f"Task '{task_description}' handled by Agent '{self.name}'. Outcome: SKIPPED/UNKNOWN"
            reward_val = 0.0


        self.memory.record_experience(
            state=self.current_state,
            action="execute_task",
            result={"status": "SUCCESS" if reward_val > 0 else "FAILURE", "details": result},
            reward={"external": reward_val},
            expert_used=expert_id_list,
            decision_context={"reason": "Direct execution command received."},
            causal_snapshot=f"Executing task: {task_description}"
        )
        self.current_state["last_action"] = "execute_task"
        self.current_state["last_result"] = result
        return result

    # --- P10.1: 状態認識ルーティングのロジックを追加 ---
    def _encode_system_state(self) -> str:
        """現在のシステム状態（メモリ、過去の行動結果）を要約し、文字列として符号化する。"""
        # 簡易的に、過去の行動、結果、現在の目標（plannerから取得）を文字列に統合
        last_action = self.current_state.get('last_action', 'N/A')
        last_result = self.current_state.get('last_result', 'N/A')
        current_goal = self.planner.current_goal if hasattr(self.planner, 'current_goal') else 'N/A'
        
        state_str = f"Goal: {current_goal}. Last Action: {last_action}. Last Result: {last_result}."
        return state_str

    async def find_expert(self, task_description: str) -> Dict[str, Any] | None:
        """
        P10.1 STRMACの概念に基づき、現在のシステム状態を考慮して最適な専門家SNNを検索する。
        """
        safe_task_description = task_description.lower().replace(" ", "_").replace("/", "_")
        print(f"Searching for expert for task: {safe_task_description}")
        candidate_experts = await self.model_registry.find_models_for_task(safe_task_description, top_k=5)

        if not candidate_experts:
            print(f"最適な専門家が見つかりませんでした: {safe_task_description}")
            return None

        # 1. 現在のシステム状態を符号化
        system_state_str = self._encode_system_state()
        
        # 2. 状態認識ルーティングの実行
        # (簡易版: 専門家のタスク記述と現在の状態文字列の類似度を評価)
        
        # 専門家のタスク記述をリスト化 (専門知識の埋め込みベクトルを代替)
        expert_descriptions = [expert.get('task_description', '') for expert in candidate_experts]
        
        # TF-IDFを使い、状態と専門知識の「キーワード」の類似度を計算する（簡易な埋め込みの代わり）
        vectorizer = TfidfVectorizer().fit([system_state_str] + expert_descriptions)
        state_vec = vectorizer.transform([system_state_str])
        expert_vecs = vectorizer.transform(expert_descriptions)
        
        # コサイン類似度を計算 (状態と専門知識のアライメント)
        similarities = cosine_similarity(state_vec, expert_vecs).flatten()
        
        # 3. 類似度に基づいてスコアを調整し、最も適した専門家を選択
        best_expert: Optional[Dict[str, Any]] = None
        max_adjusted_score = -1.0

        for i, expert in enumerate(candidate_experts):
            metrics = expert.get("metrics", {})
            accuracy = metrics.get("accuracy", 0.0)
            spikes_value = metrics.get("avg_spikes_per_sample")
            spikes = float(spikes_value) if spikes_value is not None else float('inf')
            
            # 精度とエネルギー効率のフィルタ
            if accuracy < self.accuracy_threshold or spikes > self.energy_budget:
                continue # 必須要件を満たさない
            
            # STRMAC: 状態との類似度 (similarities[i]) で精度をブースト
            # Adjusted Score = Accuracy * (1 + Similarity * 0.5)
            adjusted_score = accuracy * (1 + similarities[i] * 0.5)
            
            if adjusted_score > max_adjusted_score:
                max_adjusted_score = adjusted_score
                best_expert = expert
                
        if best_expert:
            acc = best_expert.get("metrics", {}).get("accuracy", 0.0)
            spk = best_expert.get("metrics", {}).get("avg_spikes_per_sample", float('inf'))
            sim = similarities[candidate_experts.index(best_expert)]
            print(f"✅ STRMAC選択: {best_expert.get('model_id')} (Acc: {acc:.4f}, Sim: {sim:.2f}, Adj.Score: {max_adjusted_score:.4f})")
            return best_expert
        else:
            print(f"⚠️ 専門家は見つかりましたが、精度/エネルギー要件を満たすモデルがありませんでした。")
            return None # 条件を満たすモデルがない場合、学習を試みる

    def learn_from_web(self, topic: str) -> str:
        """
        Webクローラーを使って情報を収集し、知識を更新する。(現在は要約のみ)
        """
        print(f"Agent '{self.name}' is learning about '{topic}' from the web.")
        urls = self._search_for_urls(topic)
        task_name = f"learn_from_web_{topic.replace(' ', '_')}"
        if not urls:
            result_details = "Could not find relevant information on the web."
            self.memory.record_experience(
                state=self.current_state, action=task_name,
                result={"status": "FAILURE", "details": result_details},
                reward={"external": -0.5},
                expert_used=["web_crawler"],
                decision_context={"reason": "No relevant URLs found during web search."},
                causal_snapshot=f"Web search for '{topic}' failed."
            )
            self.current_state["last_action"] = task_name
            self.current_state["last_result"] = {"status": "FAILURE", "details": result_details}
            return result_details

        all_content = ""
        for url in urls[:2]:
            print(f"Crawling URL: {url}")
            try:
                crawled_data_path = self.web_crawler.crawl(url, max_pages=1)
                if crawled_data_path and os.path.exists(crawled_data_path):
                     with open(crawled_data_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                content_data = json.loads(line)
                                all_content += content_data.get('text', '') + "\n\n"
                            except json.JSONDecodeError:
                                print(f"Warning: Skipping invalid JSON line in {crawled_data_path}")
            except Exception as e:
                 print(f"Error crawling {url}: {e}")


        if not all_content.strip():
             result_details = "Crawled content was empty or could not be processed."
             self.memory.record_experience(
                state=self.current_state, action=task_name,
                result={"status": "FAILURE", "details": result_details},
                reward={"external": -0.3},
                expert_used=["web_crawler"],
                decision_context={"reason": "Crawled content was empty."},
                causal_snapshot=f"Crawling for '{topic}' yielded no content."
             )
             self.current_state["last_action"] = task_name
             self.current_state["last_result"] = {"status": "FAILURE", "details": result_details}
             return result_details


        summary = self._summarize(all_content)

        self.memory.record_experience(
            state=self.current_state, action=task_name,
            result={"status": "SUCCESS", "summary": summary, "source_urls": urls[:2]},
            reward={"external": 0.8},
            expert_used=["web_crawler", "summarizer"],
            decision_context={"reason": "Information successfully retrieved and summarized from the web."},
            causal_snapshot=f"Successfully learned about '{topic}' from web."
        )
        self.current_state["last_action"] = task_name
        self.current_state["last_result"] = {"status": "SUCCESS", "summary": summary}
        return f"Successfully learned about '{topic}'. Summary: {summary}"

    def _search_for_urls(self, query: str) -> list[str]:
        """
        指定されたクエリでWebを検索し、関連するURLのリストを返す。
        """
        print(f"🔍 Searching the web for: '{query}'")
        try:
            urls = list(search(query, num_results=3, lang="ja"))
            print(f"✅ Found {len(urls)} relevant URLs.")
            if not urls:
                 print("No URLs found via search, using fallback.")
                 urls = [
                     'https://www.nature.com/articles/s41583-024-00888-x',
                     'https://www.frontiersin.org/articles/10.3389/fnins.2023.1209795/full',
                 ]
            return urls
        except Exception as e:
            print(f"❌ Web search failed: {e}. Using fallback URLs.")
            return [
                'https://www.nature.com/articles/s41583-024-00888-x',
                'https://www.frontiersin.org/articles/10.3389/fnins.2023.1209795/full',
            ]

    def _summarize(self, text: str) -> str:
        """
        テキストを受け取り、要約を生成する。（現在は簡易実装）
        将来的には専門家モデルを呼び出す。
        """
        print("✍️ Summarizing content...")
        if not text:
            return "(No content to summarize)"

        summarizer_expert = asyncio.run(self.find_expert("文章要約"))

        if not summarizer_expert:
            print("⚠️ Summarization expert not found. Using basic extractive summary.")
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|。)\s', text.strip())
            sentences = [s for s in sentences if s]

            words = re.findall(r'\b\w+\b', text.lower())
            if not words: return "(Could not extract words)"

            word_freq = Counter(words)
            sentence_scores: Dict[int, float] = {}
            for i, s in enumerate(sentences):
                 s_words = re.findall(r'\b\w+\b', s.lower())
                 score = sum(word_freq[word] for word in s_words)
                 length = len(s_words)
                 sentence_scores[i] = score / (length + 1e-5)

            num_summary_sentences = min(3, len(sentences))
            highest_scoring_indices = nlargest(num_summary_sentences, sentence_scores, key=lambda k: sentence_scores[k])

            summary = " ".join([sentences[i] for i in sorted(highest_scoring_indices)])
            return summary
        else:
            print(f"✅ Found summarization expert: {summarizer_expert.get('model_id')}")
            summary_result = f"(Summary generated by expert '{summarizer_expert.get('model_id')}'): " + " ".join(text.split()[:30]) + "..."
            return summary_result


    async def handle_task(self, task_description: str, unlabeled_data_path: Optional[str] = None, force_retrain: bool = False) -> Optional[Dict[str, Any]]:
        """
        タスクを処理する中心的なメソッド。専門家を検索し、いなければ学習を試みる。
        """
        print(f"--- Handling Task: {task_description} ---")
        self.memory.record_experience(
             state=self.current_state,
             action="handle_task_start",
             result={"task": task_description},
             reward={"external": 0.0},
             expert_used=[],
             decision_context={"reason": "Task received by agent."}
        )

        expert_model: Optional[Dict[str, Any]] = None
        if not force_retrain:
            candidate_expert = await self.find_expert(task_description)
            if candidate_expert:
                expert_model = candidate_expert

        if expert_model:
             self.current_state["last_action"] = "expert_found"
             self.current_state["last_result"] = expert_model.get("model_id")
             return expert_model


        if unlabeled_data_path:
            print("- No suitable expert found or retraining forced. Initiating on-demand learning...")
            try:
                from app.containers import TrainingContainer
                container = TrainingContainer()
                container.config.from_yaml("configs/base_config.yaml")
                container.config.from_yaml("configs/models/medium.yaml")

                device = container.device()
                student_model = container.snn_model().to(device)
                optimizer = container.optimizer(params=student_model.parameters())
                scheduler = container.scheduler(optimizer=optimizer) if container.config.training.gradient_based.use_scheduler() else None

                distillation_trainer = container.distillation_trainer(
                    model=student_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    rank=-1
                )

                manager_config = container.config()

                manager = KnowledgeDistillationManager(
                    student_model=student_model,
                    trainer=distillation_trainer,
                    tokenizer_name=container.config.data.tokenizer_name(),
                    model_registry=self.model_registry,
                    device=device,
                    config=manager_config
                )

                wikitext_path = "data/wikitext-103_train.jsonl"
                learning_data_path: str
                if os.path.exists(wikitext_path):
                    print(f"✅ 大規模データセット '{wikitext_path}' を発見。本格的な学習に使用します。")
                    learning_data_path = wikitext_path
                else:
                    learning_data_path = unlabeled_data_path
                    print(f"⚠️ 大規模データセットが見つからないため、指定された '{learning_data_path}' を使用します。")

                student_model_config = container.config.model.to_dict()

                new_model_info = await manager.run_on_demand_pipeline(
                    task_description=task_description,
                    unlabeled_data_path=learning_data_path,
                    force_retrain=force_retrain,
                    student_config=student_model_config
                )

                model_id_val = new_model_info.get('model_id') if new_model_info else None
                model_id_str = str(model_id_val) if model_id_val is not None else "unknown"
                reward_val = 1.0 if new_model_info and "error" not in new_model_info else -0.8
                
                self.memory.record_experience(
                    state=self.current_state,
                    action="on_demand_learning",
                    result=new_model_info if new_model_info else {"error": "Training pipeline failed to return info"},
                    reward={"external": reward_val},
                    expert_used=[model_id_str] if model_id_str != "unknown" else [],
                    decision_context={"reason": "Attempted to create a new expert for the task."},
                    causal_snapshot=f"On-demand learning for '{task_description}' completed."
                 )
                self.current_state["last_action"] = "on_demand_learning"
                self.current_state["last_result"] = new_model_info if new_model_info else {"error": "Training pipeline failed"}
                return new_model_info

            except Exception as e:
                print(f"❌ On-demand learning failed: {e}")
                import traceback
                traceback.print_exc()
                error_info = {"error": str(e)}
                self.memory.record_experience(
                     state=self.current_state,
                     action="on_demand_learning_error",
                     result=error_info,
                     reward={"external": -1.0},
                     expert_used=[],
                     decision_context={"reason": "An unexpected error occurred during training."},
                     causal_snapshot=f"Critical error during on-demand learning for '{task_description}'."
                 )
                self.current_state["last_action"] = "on_demand_learning_error"
                self.current_state["last_result"] = error_info
                return error_info

        print("- No expert found and no unlabeled data provided for training.")
        no_expert_info = {"status": "skipped", "reason": "No expert found and no data for training."}
        self.memory.record_experience(
            state=self.current_state,
            action="handle_task_skipped",
            result=no_expert_info,
            reward={"external": -0.1},
            expert_used=[],
            decision_context={"reason": "Unable to proceed with the task."},
            causal_snapshot=f"Skipped task '{task_description}' due to lack of expert/data."
        )
        self.current_state["last_action"] = "handle_task_skipped"
        self.current_state["last_result"] = no_expert_info
        return no_expert_info

    async def run_inference(self, model_info: Dict[str, Any], prompt: str) -> None:
        """
        指定されたモデルで推論を実行する。
        """
        model_id = model_info.get('model_id', 'N/A')
        model_path = model_info.get('model_path') or model_info.get('path')
        model_config = model_info.get('config')

        print(f"\n--- Running Inference ---")
        print(f"Model ID: {model_id}")
        print(f"Model Path: {model_path}")
        print(f"Prompt: {prompt}")

        if not model_path or not os.path.exists(model_path):
            print(f"❌ Error: Model file not found at '{model_path}'. Cannot run inference.")
            self.memory.record_experience(self.current_state, "inference_error", {"error": "Model file not found"}, {"external": -0.5}, [model_id], {})
            self.current_state["last_action"] = "inference_error"
            self.current_state["last_result"] = {"error": "Model file not found"}
            return

        if not model_config:
            print("❌ Error: Model config not found in model_info. Cannot initialize inference engine.")
            self.memory.record_experience(self.current_state, "inference_error", {"error": "Model config not found"}, {"external": -0.5}, [model_id], {})
            self.current_state["last_action"] = "inference_error"
            self.current_state["last_result"] = {"error": "Model config not found"}
            return

        try:
            inference_config_dict = {
                'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                'data': {
                    'tokenizer_name': "gpt2"
                },
                'model': model_config
            }
            absolute_path = str(Path(model_path).resolve())
            inference_config_dict['model']['path'] = absolute_path

            inference_config = OmegaConf.create(inference_config_dict)

            inference_engine = SNNInferenceEngine(config=inference_config)

            full_response = ""
            print("Response Stream: ", end="", flush=True)
            for chunk, stats in inference_engine.generate(prompt, max_len=50):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n--- Inference Complete ---")

            self.memory.record_experience(
                 state=self.current_state,
                 action="inference_success",
                 result={"prompt": prompt, "response": full_response, "stats": inference_engine.last_inference_stats},
                 reward={"external": 0.5},
                 expert_used=[model_id],
                 decision_context={"reason": f"Successfully generated response using expert '{model_id}'."},
                 causal_snapshot=f"Inference using '{model_id}' for prompt '{prompt[:20]}...' succeeded."
            )
            self.current_state["last_action"] = "inference_success"
            self.current_state["last_result"] = {"response": full_response}


        except Exception as e:
            print(f"\n❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            error_info = {"error": str(e), "model_id": model_id, "prompt": prompt}
            self.memory.record_experience(
                state=self.current_state,
                action="inference_error",
                result=error_info,
                reward={"external": -0.7},
                expert_used=[model_id],
                decision_context={"reason": f"An error occurred during inference with expert '{model_id}'."},
                causal_snapshot=f"Inference using '{model_id}' failed."
            )
            self.current_state["last_action"] = "inference_error"
            self.current_state["last_result"] = error_info
