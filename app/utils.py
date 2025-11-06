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
#
# 修正 (v5):
# - 循環インポート解消のため、collate_fn を train.py から移動。

import gradio as gr  # type: ignore[import-untyped]
# --- ▼ 修正: Callable, Iterator, Tuple, List, Optional, Any をインポート ▼ ---
from typing import Callable, Iterator, Tuple, List, Optional, Any
# --- ▲ 修正 ▲ ---
import torch
# --- ▼ 修正 (v5): 必要なインポートを追加 ▼ ---
from transformers import PreTrainedTokenizerBase
# --- ▲ 修正 (v5) ▲ ---


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

# --- ▼ 修正 (v5): collate_fn を train.py から移動 ▼ ---
def collate_fn(tokenizer: PreTrainedTokenizerBase, is_distillation: bool) -> Callable[[List[Any]], Any]:
    """
    データローダー用の Collate 関数。
    (train.py から app/utils.py に移動)
    """
    def collate(batch: List[Any]) -> Any:
        padding_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        inputs: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        logits: List[torch.Tensor] = [] # Only used if is_distillation

        # Handle different batch item types (dict from HF, tuple from SNNBaseDataset)
        for item in batch:
            if isinstance(item, dict):
                # Ensure keys exist and are tensors or tensor-like
                inp = item.get('input_ids')
                tgt = item.get('labels') # Assuming 'labels' key
                if inp is None or tgt is None: continue # Skip invalid items
                inputs.append(torch.tensor(inp) if not isinstance(inp, torch.Tensor) else inp)
                targets.append(torch.tensor(tgt) if not isinstance(tgt, torch.Tensor) else tgt)
                if is_distillation:
                    lg = item.get('teacher_logits')
                    if lg is not None: logits.append(torch.tensor(lg) if not isinstance(lg, torch.Tensor) else lg)
                    else: logits.append(torch.empty(0)) # Placeholder if missing

            elif isinstance(item, tuple) and len(item) >= 2:
                # Ensure elements are tensors or tensor-like
                inp = item[0]
                tgt = item[1]
                if not isinstance(inp, (torch.Tensor, list, tuple)) or not isinstance(tgt, (torch.Tensor, list, tuple)): continue
                inputs.append(torch.tensor(inp) if not isinstance(inp, torch.Tensor) else inp)
                targets.append(torch.tensor(tgt) if not isinstance(tgt, torch.Tensor) else tgt)
                if is_distillation:
                    if len(item) >= 3:
                         lg = item[2]
                         if lg is not None: logits.append(torch.tensor(lg) if not isinstance(lg, torch.Tensor) else lg)
                         else: logits.append(torch.empty(0))
                    else: logits.append(torch.empty(0))
            else:
                print(f"Warning: Skipping unsupported batch item type: {type(item)}")
                continue # Skip unsupported item types

        if not inputs or not targets: # If batch becomes empty after filtering
            # Return empty structures that match expected types
            if is_distillation:
                return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0, 0), dtype=torch.float32)
            else:
                # 辞書形式を返す (標準のcollate_fnが期待する形式)
                return {
                    "input_ids": torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_val),
                    "attention_mask": torch.nn.utils.rnn.pad_sequence([torch.ones_like(i) for i in inputs], batch_first=True, padding_value=0),
                    "labels": torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
                }

        # --- 標準 (非蒸留) の collate ロジック (辞書を返す) ---
        if not is_distillation:
            padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_val)
            padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
            attention_mask = torch.ones_like(padded_inputs)
            attention_mask[padded_inputs == padding_val] = 0
            return {
                "input_ids": padded_inputs,
                "attention_mask": attention_mask,
                "labels": padded_targets
            }
        
        # --- 蒸留 (タプルを返す) ---
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_val)
        padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
        padded_logits = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=0.0)
        attention_mask = torch.ones_like(padded_inputs)
        attention_mask[padded_inputs == padding_val] = 0
        
        # ターゲットとロジットのシーケンス長を合わせる
        seq_len = padded_inputs.shape[1]
        if padded_targets.shape[1] < seq_len:
            pad = torch.full((padded_targets.shape[0], seq_len - padded_targets.shape[1]), -100, dtype=padded_targets.dtype, device=padded_targets.device)
            padded_targets = torch.cat([padded_targets, pad], dim=1)
        if padded_logits.shape[1] < seq_len:
            pad = torch.full((padded_logits.shape[0], seq_len - padded_logits.shape[1], padded_logits.shape[2]), 0.0, dtype=padded_logits.dtype, device=padded_logits.device)
            padded_logits = torch.cat([padded_logits, pad], dim=1)
            
        return padded_inputs, attention_mask, padded_targets, padded_logits
    
    return collate
# --- ▲ 修正 (v5) ▲ ---


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
