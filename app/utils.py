# ファイルパス: app/utils.py
# (Gradio 4.20.0 互換修正 v3)
# Gradioアプリケーション用の共通ユーティリティ
#
# 機能:
# - アプリケーション間で共有される定数や関数を定義する。
# - 共通のGradio UIレイアウトを構築する関数を提供する。
# - 修正: `type="messages"` と `avatar_images` を削除。
# - 修正: `queue=False` をイベントリスナーに追加。
# - 修正: `stream_fn` の型ヒントを `List[List[Optional[str]]]` に修正。
# - 修正: `.queue().launch()` を `.launch()` に修正。
#
# 修正 (v4):
# - mypyエラー [import-untyped] を解消。

import gradio as gr  # type: ignore[import-untyped]
# --- ▼ 修正: Callable, Iterator, Tuple, List, Optional をインポート ▼ ---
from typing import Callable, Iterator, Tuple, List, Optional
# --- ▲ 修正 ▲ ---
import torch

def get_auto_device() -> str:
    """実行環境に最適なデバイスを自動的に選択する。"""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_avatar_svgs():
    """
    Gradioチャットボット用のアバターSVGアイコンのタプルを返す。
    (注: 古いGradioバージョンでは使用されない可能性がある)
    """
    user_avatar_svg = r"""
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-user"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
    """
    assistant_avatar_svg = r"""
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-zap"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
    """
    return user_avatar_svg, assistant_avatar_svg

def build_gradio_ui(
    # --- ▼ 修正: 型ヒントを List[List[Optional[str]]] に変更 ▼ ---
    stream_fn: Callable[[str, List[List[Optional[str]]]], Iterator[Tuple[List[List[Optional[str]]], str]]],
    # --- ▲ 修正 ▲ ---
    title: str,
    description: str,
    chatbot_label: str,
    theme: gr.themes.Base
) -> gr.Blocks:
    """
    共通のGradio Blocks UIを構築する (Gradio 4.20.0 互換)。

    Args:
        stream_fn: チャットメッセージを処理し、応答をストリーミングする関数。
        title: UIのメインタイトル。
        description: UIの説明文。
        chatbot_label: チャットボットコンポーネントのラベル。
        theme:適用するGradioテーマ。

    Returns:
        gr.Blocks: 構築されたGradio UIオブジェクト。
    """
    # --- ▼ 修正: アバターを使用しない ▼ ---
    # user_avatar, assistant_avatar = get_avatar_svgs()
    # --- ▲ 修正 ▲ ---

    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(f"# {title}\n{description}")
        
        initial_stats_md = """
        **Inference Time:** `N/A`
        **Tokens/Second:** `N/A`
        ---
        **Total Spikes:** `N/A`
        **Spikes/Second:** `N/A`
        """

        with gr.Row():
            with gr.Column(scale=2):
                # --- ▼ 修正: type="messages" と avatar_images を削除 ▼ ---
                chatbot = gr.Chatbot(label=chatbot_label, height=500)
                # --- ▲ 修正 ▲ ---
            with gr.Column(scale=1):
                stats_display = gr.Markdown(value=initial_stats_md, label="📊 Inference Stats")

        with gr.Row():
            msg_textbox = gr.Textbox(
                show_label=False,
                placeholder="メッセージを入力...",
                container=False,
                scale=6,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", scale=1)

        gr.Markdown("<footer><p>© 2025 SNN System Design Project. All rights reserved.</p></footer>")

        def clear_all():
            return [], "", initial_stats_md

        # `submit` アクションの定義
        submit_event = msg_textbox.submit(
            fn=stream_fn,
            inputs=[msg_textbox, chatbot],
            outputs=[chatbot, stats_display],
            queue=False # --- ▼ 修正: queue=False を追加 ▼ ---
        )
        # --- ▲ 修正 ▲ ---
        submit_event.then(fn=lambda: "", inputs=None, outputs=msg_textbox)
        
        button_submit_event = submit_btn.click(
            fn=stream_fn,
            inputs=[msg_textbox, chatbot],
            outputs=[chatbot, stats_display],
            queue=False # --- ▼ 修正: queue=False を追加 ▼ ---
        )
        # --- ▲ 修正 ▲ ---
        button_submit_event.then(fn=lambda: "", inputs=None, outputs=msg_textbox)

        # `clear` アクションの定義
        clear_btn.click(
            fn=clear_all,
            inputs=None,
            outputs=[chatbot, msg_textbox, stats_display],
            queue=False
        )
    
    return demo