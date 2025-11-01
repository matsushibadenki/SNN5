# ファイルパス: app/main.py
# (動的モデルロードUI 修正 v16 - Gradio 4.20.0 最終互換性修正)
# DIコンテナを利用した、Gradioリアルタイム対話UIの起動スクリプト
#
# 機能:
# - model_registry.json を読み込み、利用可能なモデルをGradioドロップダウンに表示。
# - コマンドライン引数 (argparse) を受け付け、レジストリのモデル情報を上書き/追加する。
# - ユーザーがモデルを選択すると、推論エンジンとサービスを動的に初期化する。
# - モデルのタイプ（テキスト/画像）をconfigから判断し、適切なタブにUIを表示する。

import gradio as gr  # type: ignore[import-untyped]
import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Optional, Any
from omegaconf import OmegaConf, DictConfig, Container # Container をインポート
from dependency_injector import providers
import numpy as np
from PIL import Image
import asyncio
import logging
import os 

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import AppContainer
# --- ▼ 修正: get_avatar_svgs の import を削除 ▼ ---
# from app.utils import get_avatar_svgs 
# --- ▲ 修正 ▲ ---
from app.services.chat_service import ChatService
from app.services.image_classification_service import ImageClassificationService
from snn_research.deployment import SNNInferenceEngine
from snn_research.distillation.model_registry import ModelRegistry, SimpleModelRegistry 

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- グローバル変数 ---
container = AppContainer()
available_models_dict: Dict[str, Dict[str, Any]] = {} 

# --- モデルロード関数 ---
def load_model_registry(registry_provider: providers.Provider[ModelRegistry]) -> Dict[str, Dict[str, Any]]:
    """model_registry.json を読み込み、UI用のモデル辞書を作成する"""
    print("Loading model registry...")
    try:
        registry = registry_provider()
        if not hasattr(registry, 'registry_path') or not registry.registry_path or not Path(registry.registry_path).exists():
             logger.error(f"model_registry.json not found at: {getattr(registry, 'registry_path', 'N/A')}")
             raise FileNotFoundError
        
        models_list = asyncio.run(registry.list_models()) # 同期的に実行
        
        # UIフレンドリーな形式に変換 { "model_id (display)": { "path": "...", "config": {...} } }
        ui_models_dict: Dict[str, Dict[str, Any]] = {}
        for model_info in models_list:
            model_id = model_info.get("model_id")
            model_path = model_info.get("model_path") or model_info.get("path")
            config = model_info.get("config") # configオブジェクト(辞書)
            
            if model_id and model_path and config:
                # config が OmegaConf オブジェクトの場合、辞書に変換
                if isinstance(config, Container): # OmegaConfの基底クラスをチェック
                    config_dict = OmegaConf.to_container(config, resolve=True)
                elif isinstance(config, dict):
                    config_dict = config
                else:
                    logger.warning(f"Skipping invalid model in registry for {model_id}: {type(config)}. Skipping.")
                    continue

                if not isinstance(config_dict, dict):
                     logger.warning(f"Converted config for {model_id} is not a dict: {type(config_dict)}. Skipping.")
                     continue
                     
                task_type = "image" if "spiking_cnn" in (config_dict.get("architecture_type") or "") else "text"
                ui_models_dict[model_id] = {
                    "path": model_path,
                    "config": config_dict, # <-- プレーンな辞書を保存
                    "task_type": task_type
                }
            else:
                logger.warning(f"Skipping invalid model in registry: {model_id} (Path: {model_path}, Config: {'Exists' if config else 'Missing'})")
                
        print(f"✅ Found {len(ui_models_dict)} valid models in registry.")
        return ui_models_dict
    except FileNotFoundError:
        registry_path = "config 'model_registry.file.path'"
        try:
             cfg_obj = container.config()
             if cfg_obj and hasattr(cfg_obj, 'model_registry') and hasattr(cfg_obj.model_registry, 'file') and hasattr(cfg_obj.model_registry.file, 'path'):
                 registry_path = container.config.model_registry.file.path() or "runs/model_registry.json"
        except Exception:
             pass
        logger.error(f"model_registry.json not found at path specified in config ({registry_path}). No models loaded.")
        return {}
    except Exception as e:
        logger.error(f"Error loading model registry: {e}")
        import traceback
        traceback.print_exc()
        return {}

def load_inference_services(model_id: str) -> Tuple[Optional[ChatService], Optional[ImageClassificationService], str, Dict, Dict, Dict]:
    """選択されたモデルIDに基づいて推論サービスをロードする"""
    global available_models_dict # グローバル変数を参照

    if not model_id or model_id == "Select Model":
        return None, None, "Please select a model from the dropdown.", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False) # Textタブをデフォルト表示

    try:
        model_info = available_models_dict.get(model_id) 
        if not model_info:
            raise KeyError(f"Model ID '{model_id}' not found in the loaded models dictionary.")

        relative_path = model_info.get("path")
        model_config_dict = model_info.get("config") # .get() を使用
        task_type = model_info.get("task_type")
        
        model_path: Optional[str] = None
        if relative_path:
            resolved_path = Path(relative_path).resolve()
            if resolved_path.exists():
                model_path = str(resolved_path)
            else:
                logger.warning(f"Path for '{model_id}' exists in dict ('{relative_path}') but not found on disk at '{resolved_path}'.")
        
        print(f"[DEBUG] Loading model '{model_id}':")
        print(f"  [DEBUG] - Relative Path in Dict: {relative_path}")
        print(f"  [DEBUG] - Resolved Path: {model_path}")
        print(f"  [DEBUG] - Config retrieved (type {type(model_config_dict)}): {str(model_config_dict)[:200]}...") 
        print(f"  [DEBUG] - Task type retrieved: {task_type}")

        if not model_path or model_config_dict is None or not task_type:
            missing = []
            if not model_path: missing.append(f"path (Resolved path from '{relative_path}' failed)")
            if model_config_dict is None: missing.append("config")
            if not task_type: missing.append("task_type")
            error_msg = f"Model info for '{model_id}' is incomplete. Missing: {', '.join(missing)}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # ベース設定とモデル設定をマージ
        config = container.config()
        full_config_dict = OmegaConf.merge(config, {"model": model_config_dict})
        
        # モデルパスを上書き
        OmegaConf.update(full_config_dict, "model.path", model_path, merge=True)
        
        # DIコンテナのプロバイダーを上書きしてサービスをインスタンス化
        engine_provider = container.snn_inference_engine
        
        chat_service: Optional[ChatService] = None
        image_service: Optional[ImageClassificationService] = None
        # service_instance can hold either a ChatService or ImageClassificationService depending on task
        service_instance: Optional[ChatService | ImageClassificationService] = None
        status_message = ""
        
        if task_type == "text":
            with engine_provider.override(providers.Factory(SNNInferenceEngine, config=full_config_dict)):
                service_instance = container.chat_service()
            chat_service = service_instance
            status_message = f"✅ Text Model '{model_id}' loaded."
            print(status_message)
            # テキストタブを表示し、画像タブを隠す
            return chat_service, None, status_message, gr.update(selected="text_tab"), gr.update(visible=True), gr.update(visible=False)

        elif task_type == "image":
            with engine_provider.override(providers.Factory(SNNInferenceEngine, config=full_config_dict)):
                service_instance = container.image_classification_service()
            
            # --- ▼ 修正: mypy [assignment] エラーを修正 (v24) ▼ ---
            # 196行目付近のエラー:
            chat_service = None
            
            # 204行目付近のエラー (return文での型不一致) を防ぐため、
            # image_service に正しい型 (ImageClassificationService) を代入する
            image_service = service_instance
            # --- ▲ 修正 ▲ ---

            status_message = f"✅ Image Model '{model_id}' loaded."
            print(status_message)
            # 画像タブを表示し、テキストタブを隠す
            return chat_service, image_service, status_message, gr.update(selected="image_tab"), gr.update(visible=False), gr.update(visible=True)

        else:
            status_message = f"⚠️ Unknown task type '{task_type}' for model '{model_id}'."
            print(status_message)
            # 不明な場合はテキストタブをデフォルト表示
            return None, None, status_message, gr.update(selected="text_tab"), gr.update(visible=True), gr.update(visible=False)

    except Exception as e:
        status_message = f"❌ Error loading model '{model_id}': {e}"
        print(status_message)
        import traceback
        traceback.print_exc()
         # エラー時はテキストタブをデフォルト表示
        return None, None, status_message, gr.update(selected="text_tab"), gr.update(visible=True), gr.update(visible=False)


# --- Gradio UI 構築 ---
def main():
    global available_models_dict # グローバル変数を関数内で使用宣言
    
    # 1. コマンドライン引数のパーサーを定義
    parser = argparse.ArgumentParser(description="SNN Multi-Task Interface")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Base config file path")
    # --- 動的モデルロードのための引数を追加 ---
    parser.add_argument("--chat_model_config", type=str, help="Path to the chat model config file (e.g., small.yaml)")
    parser.add_argument("--chat_model_path", type=str, help="Path to the chat model weights (.pth)")
    parser.add_argument("--cifar_model_config", type=str, help="Path to the CIFAR model config file")
    parser.add_argument("--cifar_model_path", type=str, help="Path to the CIFAR model weights (.pth)")
    parser.add_argument("--ai_tech_model_config", type=str, help="Path to the AI tech model config file")
    parser.add_argument("--ai_tech_model_path", type=str, help="Path to the AI tech model weights (.pth)")
    parser.add_argument("--summarization_model_config", type=str, help="Path to the summarization model config file")
    parser.add_argument("--summarization_model_path", type=str, help="Path to the summarization model weights (.pth)")
    
    args = parser.parse_args()

    # 2. DIコンテナと設定のロード
    container.config.from_yaml(args.config)
    container.wire(modules=[__name__])

    # 3. モデルレジストリからモデルをロード (グローバル変数に代入)
    available_models_dict = load_model_registry(container.model_registry)

    # 4. コマンドライン引数からモデルを *上書き* または *追加* (グローバル変数を更新)
    
    # (ヘルパー関数)
    def add_model_from_args(model_id, config_path, model_path):
        global available_models_dict # グローバル変数を変更する宣言
        if config_path and model_path:
            # --- ▼ 修正: ここでは .exists() チェックをしない (相対パスのため) ▼ ---
            # if not Path(model_path).exists():
            #     logger.warning(f"File not found for model '{model_id}' from command line: {model_path}. Skipping.")
            #     return
            if not Path(config_path).exists():
                logger.warning(f"Config file not found for model '{model_id}' from command line: {config_path}. Skipping.")
                return
            # --- ▲ 修正 ▲ ---
                
            try:
                config_obj = OmegaConf.load(config_path)
                model_config_block = config_obj.get('model', config_obj) 
                
                model_config_dict = OmegaConf.to_container(model_config_block, resolve=True)
                if not isinstance(model_config_dict, dict):
                    raise TypeError(f"Loaded config for {model_id} is not a dictionary.")
                    
                task_type = "image" if "spiking_cnn" in (model_config_dict.get("architecture_type") or "") else "text"
                
                print(f"[DEBUG] Preparing to add/update model '{model_id}' from args:")
                print(f"  [DEBUG] - Path: {model_path}")
                print(f"  [DEBUG] - Config dict (type {type(model_config_dict)}): {str(model_config_dict)[:200]}...")
                print(f"  [DEBUG] - Task type: {task_type}")

                available_models_dict[model_id] = {
                    "path": model_path, # <-- 相対パスのまま保存
                    "config": model_config_dict, 
                    "task_type": task_type
                }
                print(f"✅ Loaded/Updated model '{model_id}' from command line arguments.")
            except Exception as e:
                logger.error(f"Error loading model '{model_id}' from command line args ({config_path}): {e}")

    # コマンドライン引数で指定された各モデルを処理
    add_model_from_args("chat_model_default", args.chat_model_config, args.chat_model_path)
    add_model_from_args("cifar10_distilled_from_resnet18", args.cifar_model_config, args.cifar_model_path)
    add_model_from_args("最新のai技術", args.ai_tech_model_config, args.ai_tech_model_path)
    add_model_from_args("文章要約", args.summarization_model_config, args.summarization_model_path)

    model_choices = ["Select Model"] + list(available_models_dict.keys())

    # 5. Gradio UI 構築
    initial_stats_md = "**Inference Time:** `N/A`\n**Tokens/Second:** `N/A`\n---\n**Total Spikes:** `N/A`\n**Spikes/Second:** `N/A`"

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="green")) as demo:
        
        chat_service_state = gr.State(None)
        image_service_state = gr.State(None)

        gr.Markdown("# 🧠 SNN Multi-Task Interface (Dynamic Loading)")
        gr.Markdown("`runs/model_registry.json` およびコマンドライン引数から利用可能なモデルを読み込みます。")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=model_choices,
                value=model_choices[0]
            )
            status_textbox = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tabs() as tabs_container:
            
            with gr.TabItem("💬 Text / Chat", id="text_tab") as text_tab:
                gr.Markdown("テキストベースのモデル（チャット、QA、要約など）をテストします。")
                with gr.Row():
                    with gr.Column(scale=2):
                        # --- ▼ 修正: type="messages" と avatar_images を削除 ▼ ---
                        chat_chatbot = gr.Chatbot(
                            label="SNN Chat", 
                            height=500
                        )
                        # --- ▲ 修正 ▲ ---
                    with gr.Column(scale=1):
                        chat_stats_display = gr.Markdown(value=initial_stats_md, label="📊 Inference Stats")
                with gr.Row():
                    chat_msg_textbox = gr.Textbox(show_label=False, placeholder="メッセージを入力...", container=False, scale=6)
                    chat_submit_btn = gr.Button("Send", variant="primary", scale=1)
                    chat_clear_btn = gr.Button("Clear", scale=1)
                
                with gr.Accordion("Summarization", open=False):
                    gr.Markdown("チャットではなく、テキストボックスで要約を実行します。")
                    with gr.Row():
                        sum_input_textbox = gr.Textbox(label="Input Text", lines=10, placeholder="要約したい文章を入力してください...")
                        sum_output_textbox = gr.Textbox(label="Summary", lines=10, interactive=False)
                    sum_summarize_btn = gr.Button("Summarize", variant="primary")
                    sum_stats_display = gr.Markdown(value=initial_stats_md, label="📊 Inference Stats")

            with gr.TabItem("🖼️ Image Classification", id="image_tab", visible=False) as image_tab:
                gr.Markdown("画像分類モデル（SpikingCNNなど）をテストします。")
                with gr.Row():
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_output_label = gr.Label(num_top_classes=3, label="Classification Result")
                img_classify_btn = gr.Button("Classify Image", variant="primary")

        # --- イベントハンドラ ---
        
        model_dropdown.change(
            fn=load_inference_services,
            inputs=[model_dropdown], 
            outputs=[
                chat_service_state, 
                image_service_state, 
                status_textbox, 
                tabs_container,
                text_tab,
                image_tab
            ],
            queue=False # queue=False を指定
        )

        def chat_clear_all(): return [], "", initial_stats_md
        
        # --- ▼ 修正: stream_chat_wrapper を "List[List]" (タプルリスト) 形式に対応させる ▼ ---
        def stream_chat_wrapper(message: str, history: List[List[Optional[str]]], service: Optional[ChatService]):
            """
            Gradioの "List[List]" 形式 (タプルリスト) を受け取り、
            ChatServiceの List[List] 形式に渡し、
            結果を再び "List[List]" 形式で返すラッパー。
            """
            if not service:
                history.append([message, "Error: Chat service is not loaded."])
                yield history, initial_stats_md
                return
            if not message:
                 yield history, initial_stats_md
                 return

            try:
                # ChatService.stream_response は List[List[Optional[str]]] を期待
                yield from service.stream_response(message, history) 
            except Exception as e:
                 logger.error(f"Error during chat stream: {e}")
                 history.append([message, f"Error: {e}"])
                 yield history, initial_stats_md
        # --- ▲ 修正 ▲ ---

        chat_submit_event = msg_textbox.submit(fn=stream_chat_wrapper, inputs=[chat_msg_textbox, chat_chatbot, chat_service_state], outputs=[chat_chatbot, chat_stats_display], queue=False) # queue=False
        chat_submit_event.then(fn=lambda: "", inputs=None, outputs=chat_msg_textbox)
        chat_button_submit_event = chat_submit_btn.click(fn=stream_chat_wrapper, inputs=[chat_msg_textbox, chat_chatbot, chat_service_state], outputs=[chat_chatbot, chat_stats_display], queue=False) # queue=False
        chat_button_submit_event.then(fn=lambda: "", inputs=None, outputs=chat_msg_textbox)
        
        chat_clear_btn.click(fn=chat_clear_all, inputs=None, outputs=[chat_chatbot, chat_msg_textbox, chat_stats_display], queue=False)
        
        def summarize_text(text: str, service: Optional[ChatService]) -> Tuple[str, str]:
            if not service:
                return "Error: Summarization service is not loaded.", initial_stats_md
            if not text:
                return "", initial_stats_md
            full_response = ""
            stats_md_output = initial_stats_md
            try:
                iterator = service.stream_response(text, [])
                final_history: List[List[Optional[str]]] = []
                while True:
                    try:
                        current_history, stats_md_output = next(iterator)
                        final_history = current_history
                    except StopIteration:
                        if final_history and final_history[-1] and len(final_history[-1]) > 1:
                            response_content = final_history[-1][1]
                            full_response = response_content if response_content is not None else ""
                        break
            except Exception as e:
                logger.error(f"Error during summarization: {e}")
                return f"Error: {e}", initial_stats_md
            return full_response, stats_md_output

        sum_summarize_btn.click(
            fn=summarize_text,
            inputs=[sum_input_textbox, chat_service_state],
            outputs=[sum_output_textbox, sum_stats_display],
            queue=False # queue=False を指定
        )

        def classify_image(image: Any, service: Optional[ImageClassificationService]) -> Dict[str, float]:
             if not service:
                 return {"Error": 1.0, "Service not loaded": 0.0}
             if image is None:
                 return {"Error": 1.0, "No image provided": 0.0}
             try:
                 return service.predict(image)
             except Exception as e:
                  logger.error(f"Error during image classification: {e}")
                  return {"Error": 1.0, str(e): 0.0}

        img_classify_btn.click(
            fn=classify_image, 
            inputs=[img_input, image_service_state], 
            outputs=[img_output_label],
            queue=False # queue=False を指定
        )

    # 6. Webアプリケーションの起動
    config_obj = container.config()
    server_port_val = config_obj.get('app', {}).get('server_port', 7860)
    server_name_val = config_obj.get('app', {}).get('server_name', '127.0.0.1')
    
    server_port = int(server_port_val) if server_port_val is not None else 7860
    server_name = str(server_name_val) if server_name_val is not None else "127.0.0.1"

    print("\nStarting Gradio web server for Multi-Task app...")
    print(f"Please open http://{server_name}:{server_port} in your browser.")
    
    # --- ▼ 修正: .queue() を削除 ▼ ---
    demo.launch(server_name=server_name, server_port=server_port)
    # --- ▲ 修正 ▲ ---

if __name__ == "__main__":
    main()