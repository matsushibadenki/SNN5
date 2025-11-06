# ファイルパス: snn_research/agent/autonomous_agent.py
# (修正)
# Title: 自律エージェント ベースクラス
# Description: タスクに応じて専門家モデルを選択・学習し、推論を実行する自律エージェントの基本機能を提供します。
#              Web検索、記憶システム、スパイク通信機能などを統合します。
# 循環インポートエラーを解消するため、TYPE_CHECKINGを使用して
# HierarchicalPlannerの型ヒントを解決する。
# 修正(v2): 知識蒸留トレーナーをインスタンス化する際に、
#          必須引数である `rank` を渡すように修正。
# 修正(mypy): [list-item], [assignment], [arg-type] エラーを解消。

from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union # Union をインポート
import asyncio
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf, DictConfig # DictConfig をインポート
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

# --- ▼ 修正 ▼ ---
if TYPE_CHECKING:
    from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
# --- ▲ 修正 ▲ ---


class AutonomousAgent:
    """
    自律的にタスクを実行するエージェントのベースクラス。
    """
    def __init__(
        self,
        name: str,
        # --- ▼ 修正 ▼ ---
        planner: "HierarchicalPlanner",
        # --- ▲ 修正 ▲ ---
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
        # --- ▼ 修正 ▼ ---
        # last_action, last_result は辞書型も受け入れるように Union を使用
        self.current_state: Dict[str, Union[str, Dict[str, Any], None]] = {
            "agent_name": name, "last_action": None, "last_result": None
        }
        # --- ▲ 修正 ▲ ---
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

        expert_id_list: List[str] = [] # 型ヒント追加
        if result_info and result_info.get("path"):
            result = f"Task '{task_description}' handled by Agent '{self.name}'. Outcome: SUCCESS (Expert: {result_info.get('model_id')})"
            reward_val = 1.0
            if result_info.get('model_id'):
                 expert_id_list.append(str(result_info['model_id'])) # model_idがNoneでないことを確認
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
            expert_used=expert_id_list, # 修正: list[str]を渡す
            decision_context={"reason": "Direct execution command received."},
            causal_snapshot=f"Executing task: {task_description}" # 因果スナップショット例
        )
        self.current_state["last_action"] = "execute_task"
        self.current_state["last_result"] = result
        return result


    async def find_expert(self, task_description: str) -> Dict[str, Any] | None:
        """
        タスクに最適な専門家モデルをモデルレジストリから検索する。
        """
        # タスク記述を正規化 (小文字化、スペースをアンダースコアに)
        safe_task_description = task_description.lower().replace(" ", "_").replace("/", "_")
        print(f"Searching for expert for task: {safe_task_description}")
        candidate_experts = await self.model_registry.find_models_for_task(safe_task_description, top_k=5)

        if not candidate_experts:
            print(f"最適な専門家が見つかりませんでした: {safe_task_description}")
            return None

        # 精度とエネルギー効率の基準でフィルタリング
        suitable_experts = []
        for expert in candidate_experts:
            metrics = expert.get("metrics", {})
            accuracy = metrics.get("accuracy", 0.0)
            # handle potential missing key or None value for spikes
            spikes_value = metrics.get("avg_spikes_per_sample")
            spikes = float(spikes_value) if spikes_value is not None else float('inf')

            if accuracy >= self.accuracy_threshold and spikes <= self.energy_budget:
                suitable_experts.append(expert)

        if suitable_experts:
            # 基準を満たすモデルの中から最も精度の高いものを選択
            best_expert = max(suitable_experts, key=lambda x: x.get("metrics", {}).get("accuracy", 0.0))
            acc = best_expert.get("metrics", {}).get("accuracy", 0.0)
            spk = best_expert.get("metrics", {}).get("avg_spikes_per_sample", float('inf'))
            print(f"✅ 条件を満たす最適な専門家を発見: {best_expert.get('model_id')} (Accuracy: {acc:.4f}, Spikes: {spk:.2f})")
            return best_expert
        else:
            # 基準を満たすモデルがない場合、最も精度の高いモデルをフォールバックとして返す
            print(f"⚠️ 専門家は見つかりましたが、精度/エネルギー要件を満たすモデルがありませんでした。")
            best_candidate = max(candidate_experts, key=lambda x: x.get("metrics", {}).get("accuracy", 0.0))
            acc = best_candidate.get("metrics", {}).get("accuracy", 0.0)
            spk = best_candidate.get("metrics", {}).get("avg_spikes_per_sample", float('inf'))
            print(f"   - 最高性能モデル (フォールバック): {best_candidate.get('model_id')} (Accuracy: {acc:.4f}, Spikes: {spk:.2f})")
            print(f"   - (要件: accuracy >= {self.accuracy_threshold}, spikes <= {self.energy_budget})")
            return best_candidate

    def learn_from_web(self, topic: str) -> str:
        """
        Webクローラーを使って情報を収集し、知識を更新する。(現在は要約のみ)
        """
        print(f"Agent '{self.name}' is learning about '{topic}' from the web.")
        urls = self._search_for_urls(topic)
        task_name = f"learn_from_web_{topic.replace(' ', '_')}" # タスク名を生成
        if not urls:
            result_details = "Could not find relevant information on the web."
            self.memory.record_experience(
                state=self.current_state, action=task_name,
                result={"status": "FAILURE", "details": result_details},
                reward={"external": -0.5}, # 少し低いペナルティ
                expert_used=["web_crawler"],
                decision_context={"reason": "No relevant URLs found during web search."},
                causal_snapshot=f"Web search for '{topic}' failed."
            )
            self.current_state["last_action"] = task_name
            # --- ▼ 修正 ▼ ---
            self.current_state["last_result"] = {"status": "FAILURE", "details": result_details}
            # --- ▲ 修正 ▲ ---
            return result_details

        all_content = ""
        # 収集するURL数を制限 (例: 最初の2つ)
        for url in urls[:2]:
            print(f"Crawling URL: {url}")
            # クロール実行 (エラーハンドリング強化)
            try:
                crawled_data_path = self.web_crawler.crawl(url, max_pages=1) # 1ページだけ取得
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
             # --- ▼ 修正 ▼ ---
             self.current_state["last_result"] = {"status": "FAILURE", "details": result_details}
             # --- ▲ 修正 ▲ ---
             return result_details


        summary = self._summarize(all_content)

        self.memory.record_experience(
            state=self.current_state, action=task_name,
            result={"status": "SUCCESS", "summary": summary, "source_urls": urls[:2]},
            reward={"external": 0.8}, # 情報収集成功ボーナス
            expert_used=["web_crawler", "summarizer"], # 仮のsummarizer
            decision_context={"reason": "Information successfully retrieved and summarized from the web."},
            causal_snapshot=f"Successfully learned about '{topic}' from web."
        )
        self.current_state["last_action"] = task_name
        # --- ▼ 修正 ▼ ---
        self.current_state["last_result"] = {"status": "SUCCESS", "summary": summary}
        # --- ▲ 修正 ▲ ---
        return f"Successfully learned about '{topic}'. Summary: {summary}"

    def _search_for_urls(self, query: str) -> list[str]:
        """
        指定されたクエリでWebを検索し、関連するURLのリストを返す。
        """
        print(f"🔍 Searching the web for: '{query}'")
        try:
            # googlesearchライブラリを使用
            # num_results を 3 に変更
            urls = list(search(query, num_results=3, lang="ja")) # 日本語検索を指定
            print(f"✅ Found {len(urls)} relevant URLs.")
            if not urls:
                 print("No URLs found via search, using fallback.")
                 # フォールバックURL (例)
                 urls = [
                     'https://www.nature.com/articles/s41583-024-00888-x',
                     'https://www.frontiersin.org/articles/10.3389/fnins.2023.1209795/full',
                 ]
            return urls
        except Exception as e:
            print(f"❌ Web search failed: {e}. Using fallback URLs.")
            # エラー時もフォールバックURLを返す
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

        summarizer_expert = asyncio.run(self.find_expert("文章要約")) # find_expertは非同期なのでawait

        if not summarizer_expert:
            print("⚠️ Summarization expert not found. Using basic extractive summary.")
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|。)\s', text.strip()) # 日本語句点も考慮
            sentences = [s for s in sentences if s] # 空の文を削除
            if not sentences: return "(Could not extract sentences)"

            words = re.findall(r'\b\w+\b', text.lower()) # 簡単な単語分割
            if not words: return "(Could not extract words)"

            word_freq = Counter(words)
            # スコア計算: 文中の単語頻度の合計 / 文の長さ (単純化)
            sentence_scores: Dict[int, float] = {}
            for i, s in enumerate(sentences):
                 s_words = re.findall(r'\b\w+\b', s.lower())
                 score = sum(word_freq[word] for word in s_words)
                 length = len(s_words)
                 sentence_scores[i] = score / (length + 1e-5) # ゼロ除算防止

            # スコア上位3文を選択
            num_summary_sentences = min(3, len(sentences))
            # --- ▼ 修正 ▼ ---
            # nlargest の key 引数には callable を渡す
            highest_scoring_indices = nlargest(num_summary_sentences, sentence_scores, key=lambda k: sentence_scores[k])
            # --- ▲ 修正 ▲ ---

            # 元の順序で結合
            summary = " ".join([sentences[i] for i in sorted(highest_scoring_indices)])
            return summary
        else:
            print(f"✅ Found summarization expert: {summarizer_expert.get('model_id')}")
            # 実際にはここで推論を実行する
            # summary_result = asyncio.run(self.run_inference(summarizer_expert, text)) # run_inferenceを呼び出す
            # ダミー応答
            summary_result = f"(Summary generated by expert '{summarizer_expert.get('model_id')}'): " + " ".join(text.split()[:30]) + "..."
            return summary_result


    async def handle_task(self, task_description: str, unlabeled_data_path: Optional[str] = None, force_retrain: bool = False) -> Optional[Dict[str, Any]]:
        """
        タスクを処理する中心的なメソッド。専門家を検索し、いなければ学習を試みる。
        """
        print(f"--- Handling Task: {task_description} ---")
        # 開始時に状態を記録
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
                # find_expert内でログ出力済みなのでここでは省略
                expert_model = candidate_expert # find_expertは最適なもの(なければNone)を返す

        if expert_model:
             # 専門家が見つかった場合
             self.current_state["last_action"] = "expert_found"
             self.current_state["last_result"] = expert_model.get("model_id")
             # 必要であればここで推論を実行するロジックを追加
             # await self.run_inference(expert_model, "some default prompt or context")
             return expert_model # モデル情報を返す


        # 専門家がいない、または再学習が強制される場合
        if unlabeled_data_path:
            print("- No suitable expert found or retraining forced. Initiating on-demand learning...")
            try:
                # DIコンテナの取得 (依存関係解決のため)
                # 注: 本来はAgent初期化時にDIコンテナか必要なコンポーネントを受け取るべき
                from app.containers import TrainingContainer # ここでのimportは理想的ではない
                container = TrainingContainer()
                # 必要な設定をロード
                container.config.from_yaml("configs/base_config.yaml")
                # モデル設定はタスクに応じて動的に選択する方が良いかもしれない
                container.config.from_yaml("configs/models/medium.yaml") # 仮にmediumを使用

                # 必要なコンポーネントを取得
                device = container.device()
                student_model = container.snn_model().to(device) # vocab_sizeは中で取得される想定
                optimizer = container.optimizer(params=student_model.parameters())
                scheduler = container.scheduler(optimizer=optimizer) if container.config.training.gradient_based.use_scheduler() else None

                distillation_trainer = container.distillation_trainer(
                    model=student_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    rank=-1  # 非分散学習のためrankを-1に設定
                )

                # Managerの初期化に必要なconfigを取得
                manager_config = container.config()

                manager = KnowledgeDistillationManager(
                    student_model=student_model,
                    trainer=distillation_trainer,
                    tokenizer_name=container.config.data.tokenizer_name(),
                    model_registry=self.model_registry,
                    device=device,
                    config=manager_config # configを渡す
                    # teacher_model_name は manager 内のフォールバックでconfigから取得される
                )

                # 学習データの選択ロジック
                wikitext_path = "data/wikitext-103_train.jsonl"
                learning_data_path: str
                if os.path.exists(wikitext_path):
                    print(f"✅ 大規模データセット '{wikitext_path}' を発見。本格的な学習に使用します。")
                    learning_data_path = wikitext_path
                else:
                    learning_data_path = unlabeled_data_path
                    print(f"⚠️ 大規模データセットが見つからないため、指定された '{learning_data_path}' を使用します。")

                # 学生モデルの設定を取得
                student_model_config = container.config.model.to_dict()

                # オンデマンド学習パイプラインを実行
                new_model_info = await manager.run_on_demand_pipeline(
                    task_description=task_description,
                    unlabeled_data_path=learning_data_path,
                    force_retrain=force_retrain, # force_retrain引数を渡す
                    student_config=student_model_config
                )

                # --- ▼ 修正: model_id を安全に文字列化 ▼ ---
                model_id_val = new_model_info.get('model_id') if new_model_info else None
                model_id_str = str(model_id_val) if model_id_val is not None else "unknown"
                reward_val = 1.0 if new_model_info and "error" not in new_model_info else -0.8
                
                self.memory.record_experience(
                    state=self.current_state,
                    action="on_demand_learning",
                    result=new_model_info if new_model_info else {"error": "Training pipeline failed to return info"},
                    reward={"external": reward_val},
                    expert_used=[model_id_str] if model_id_str != "unknown" else [], # 修正: model_id_str を使用
                    decision_context={"reason": "Attempted to create a new expert for the task."},
                    causal_snapshot=f"On-demand learning for '{task_description}' completed."
                 )
                # --- ▲ 修正 ▲ ---
                self.current_state["last_action"] = "on_demand_learning"
                # --- ▼ 修正 ▼ ---
                self.current_state["last_result"] = new_model_info if new_model_info else {"error": "Training pipeline failed"}
                return new_model_info

            except Exception as e:
                print(f"❌ On-demand learning failed: {e}")
                import traceback
                traceback.print_exc() # 詳細なエラーを出力
                error_info = {"error": str(e)}
                self.memory.record_experience(
                     state=self.current_state,
                     action="on_demand_learning_error",
                     result=error_info,
                     reward={"external": -1.0}, # 重いペナルティ
                     expert_used=[],
                     decision_context={"reason": "An unexpected error occurred during training."},
                     causal_snapshot=f"Critical error during on-demand learning for '{task_description}'."
                 )
                self.current_state["last_action"] = "on_demand_learning_error"
                # --- ▼ 修正 ▼ ---
                self.current_state["last_result"] = error_info
                # --- ▲ 修正 ▲ ---
                return error_info # エラー情報を返す

        # 専門家が見つからず、学習データもない場合
        print("- No expert found and no unlabeled data provided for training.")
        no_expert_info = {"status": "skipped", "reason": "No expert found and no data for training."}
        self.memory.record_experience(
            state=self.current_state,
            action="handle_task_skipped",
            result=no_expert_info,
            reward={"external": -0.1}, # 軽いペナルティ
            expert_used=[],
            decision_context={"reason": "Unable to proceed with the task."},
            causal_snapshot=f"Skipped task '{task_description}' due to lack of expert/data."
        )
        self.current_state["last_action"] = "handle_task_skipped"
        # --- ▼ 修正 ▼ ---
        self.current_state["last_result"] = no_expert_info
        # --- ▲ 修正 ▲ ---
        return no_expert_info # スキップ情報を返す

    async def run_inference(self, model_info: Dict[str, Any], prompt: str) -> None:
        """
        指定されたモデルで推論を実行する。
        """
        model_id = model_info.get('model_id', 'N/A')
        model_path = model_info.get('model_path') or model_info.get('path')
        model_config = model_info.get('config') # レジストリからconfigを取得

        print(f"\n--- Running Inference ---")
        print(f"Model ID: {model_id}")
        print(f"Model Path: {model_path}")
        print(f"Prompt: {prompt}")

        if not model_path or not os.path.exists(model_path):
            print(f"❌ Error: Model file not found at '{model_path}'. Cannot run inference.")
            self.memory.record_experience(self.current_state, "inference_error", {"error": "Model file not found"}, {"external": -0.5}, [model_id], {})
            self.current_state["last_action"] = "inference_error"
            # --- ▼ 修正 ▼ ---
            self.current_state["last_result"] = {"error": "Model file not found"}
            # --- ▲ 修正 ▲ ---
            return

        if not model_config:
            print("❌ Error: Model config not found in model_info. Cannot initialize inference engine.")
            self.memory.record_experience(self.current_state, "inference_error", {"error": "Model config not found"}, {"external": -0.5}, [model_id], {})
            self.current_state["last_action"] = "inference_error"
            # --- ▼ 修正 ▼ ---
            self.current_state["last_result"] = {"error": "Model config not found"}
            # --- ▲ 修正 ▲ ---
            return

        try:
            # 推論用の設定を作成
            # ベース設定の一部とモデル設定をマージ
            # ここでは DI コンテナを使わず直接設定を作成する例
            inference_config_dict = {
                'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                'data': {
                    'tokenizer_name': "gpt2" # 仮。本来はモデルに紐づくべき
                },
                'model': model_config # レジストリから取得したconfigを使用
            }
            # モデルパスを絶対パスにして設定に追加
            absolute_path = str(Path(model_path).resolve())
            inference_config_dict['model']['path'] = absolute_path

            inference_config = OmegaConf.create(inference_config_dict)

            # 推論エンジンを初期化
            inference_engine = SNNInferenceEngine(config=inference_config)

            # 推論実行とストリーミング出力
            full_response = ""
            print("Response Stream: ", end="", flush=True)
            for chunk, stats in inference_engine.generate(prompt, max_len=50): # max_lenは適宜調整
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n--- Inference Complete ---")

            # 成功記録
            self.memory.record_experience(
                 state=self.current_state,
                 action="inference_success",
                 result={"prompt": prompt, "response": full_response, "stats": inference_engine.last_inference_stats},
                 reward={"external": 0.5}, # 推論成功ボーナス
                 expert_used=[model_id],
                 decision_context={"reason": f"Successfully generated response using expert '{model_id}'."},
                 causal_snapshot=f"Inference using '{model_id}' for prompt '{prompt[:20]}...' succeeded."
            )
            self.current_state["last_action"] = "inference_success"
            # --- ▼ 修正 ▼ ---
            self.current_state["last_result"] = {"response": full_response}
            # --- ▲ 修正 ▲ ---


        except Exception as e:
            print(f"\n❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            error_info = {"error": str(e), "model_id": model_id, "prompt": prompt}
            # 失敗記録
            self.memory.record_experience(
                state=self.current_state,
                action="inference_error",
                result=error_info,
                reward={"external": -0.7}, # 推論失敗ペナルティ
                expert_used=[model_id],
                decision_context={"reason": f"An error occurred during inference with expert '{model_id}'."},
                causal_snapshot=f"Inference using '{model_id}' failed."
            )
            self.current_state["last_action"] = "inference_error"
            # --- ▼ 修正 ▼ ---
            self.current_state["last_result"] = error_info
            # --- ▲ 修正 ▲ ---
