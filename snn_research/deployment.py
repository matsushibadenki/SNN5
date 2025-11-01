# matsushibadenki/snn4/snn_research/deployment.py
# Title: SNN推論エンジン
# Description: 訓練済みSNNモデルをロードし、テキスト生成のための推論を実行するクラス。
# BugFix: モデルのパスを絶対パスに解決してからロードすることで、ファイルが見つからない問題を解消。
# BugFix: state_dictのキーから 'model.' プリフィックスを削除し、読み込みエラーを修正。
# BugFix: IndentationErrorの修正。
# BugFix: generateメソッドの max_len が None になる TypeError を修正 (最終対策 v5)。

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
# typing から time をインポート
from typing import Iterator, Optional, Dict, Any, List, Union, Tuple
import time # time をインポート
from omegaconf import DictConfig, OmegaConf
from snn_research.core.snn_core import SNNCore # SNNCoreをインポート
import logging # logging をインポート

# logging を設定 (すでにある場合は不要)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_auto_device() -> str:
    """実行環境に最適なデバイスを自動的に選択する。"""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

class SNNInferenceEngine:
    """
    学習済みSNNモデルをロードして推論を実行するエンジン。
    """
    def __init__(self, config: DictConfig):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config
        device_str = OmegaConf.select(config, "device", default="auto") # selectで安全に取得
        self.device = get_auto_device() if device_str == "auto" else device_str

        self.last_inference_stats: Dict[str, Any] = {}

        # 先にTokenizerをロードしてvocab_sizeを取得
        tokenizer_path = OmegaConf.select(config, "data.tokenizer_name", default="gpt2") # selectで安全に取得
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Could not load tokenizer from {tokenizer_path}. Error: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # vocab_sizeを渡してSNNCoreを初期化
        vocab_size = len(self.tokenizer)
        model_config = OmegaConf.select(config, "model", default=config) # selectで安全に取得
        # --- ▼ 修正: model_configがNoneの場合のフォールバックを追加 ▼ ---
        if model_config is None:
             logger.warning("Model config is None in SNNInferenceEngine init. Using empty config.")
             model_config = OmegaConf.create({}) # 空のDictConfigを作成
        # --- ▲ 修正 ▲ ---
        self.model = SNNCore(model_config, vocab_size=vocab_size)

        model_path_str = OmegaConf.select(config, "model.path", default=None) # selectで安全に取得

        if model_path_str:
            model_path = Path(model_path_str).resolve()

            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    # checkpointがstate_dictそのものか、キーを含む辞書かを確認
                    state_dict_to_load: Dict[str, Any]
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                         state_dict_to_load = checkpoint['model_state_dict']
                    elif isinstance(checkpoint, dict): # state_dict そのものの場合
                         state_dict_to_load = checkpoint
                    else:
                         raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

                    # state_dictのキーから "model." プリフィックスを削除
                    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict_to_load.items()}

                    # SNNCoreラッパーの中の実際のモデルにstate_dictをロードする
                    # strict=False を追加して、キーが一致しないエラーを回避
                    missing_keys, unexpected_keys = self.model.model.load_state_dict(new_state_dict, strict=False)
                    if missing_keys: logger.warning(f"Missing keys when loading state dict: {missing_keys}")
                    if unexpected_keys: logger.warning(f"Unexpected keys when loading state dict: {unexpected_keys}")

                    logger.info(f"✅ Model loaded from {model_path}")
                except RuntimeError as e:
                    logger.warning(f"⚠️ Warning: Failed to load state_dict, possibly due to architecture mismatch: {e}. Using an untrained model.")
                except Exception as e:
                     logger.warning(f"⚠️ Warning: An error occurred while loading model state_dict: {e}. Using an untrained model.")
            else:
                 logger.warning(f"⚠️ Warning: Model file not found at {model_path}. Using an untrained model.")
        else:
            logger.warning("⚠️ Warning: No model path specified in config. Using an untrained model.")


        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_len: Optional[int], stop_sequences: Optional[List[str]] = None) -> Iterator[Tuple[str, Dict[str, float]]]:
        """
        プロンプトに基づいてテキストと統計情報をストリーミング生成する。
        max_len が None の場合にデフォルト値を設定する。
        """
        # --- ▼ 修正: max_len が None または不正な値の場合の処理を強化 ▼ ---
        if max_len is None:
            default_max_len = 100 # デフォルト値を設定
            logger.warning(f"max_len was None, using default value: {default_max_len}")
            max_len = default_max_len
        elif not isinstance(max_len, int) or max_len <= 0:
             default_max_len = 100
             logger.warning(f"Invalid max_len value ({max_len}), using default value: {default_max_len}")
             max_len = default_max_len
        # --- ▲ 修正 ▲ ---

        tokenizer_callable = getattr(self.tokenizer, "__call__", None)
        if not callable(tokenizer_callable):
            raise TypeError("Tokenizer is not callable.")
        input_ids = tokenizer_callable(prompt, return_tensors="pt")["input_ids"].to(self.device)

        total_spikes = 0.0
        start_time = time.time() # 推論開始時間

        # --- ▼ 修正: max_len はここで整数のはず ▼ ---
        for i in range(max_len):
        # --- ▲ 修正 ▲ ---
            loop_start_time = time.time() # ループ開始時間
            with torch.no_grad():
                outputs, avg_spikes_tensor, _ = self.model(input_ids, return_spikes=True)

            avg_spikes = avg_spikes_tensor.item() if isinstance(avg_spikes_tensor, torch.Tensor) else 0.0
            total_spikes += avg_spikes * input_ids.shape[1] # スパイク数を加算

            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            if next_token_id.item() == getattr(self.tokenizer, 'eos_token_id', None):
                break

            try:
                 # decode処理でエラーが発生する可能性がある
                 new_token = self.tokenizer.decode(next_token_id.item())
            except Exception as e:
                 logger.error(f"Error decoding token ID {next_token_id.item()}: {e}")
                 new_token = "[Decode Error]" # エラーを示す代替トークン

            if stop_sequences and any(seq in new_token for seq in stop_sequences):
                break

            current_duration = time.time() - start_time
            current_stats = {"total_spikes": total_spikes}
            yield new_token, current_stats

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            loop_duration = time.time() - loop_start_time
            # print(f"  Loop {i+1} duration: {loop_duration:.4f}s") # デバッグ用

        self.last_inference_stats = {"total_spikes": total_spikes}

