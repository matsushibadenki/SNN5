# matsushibadenki/snn4/snn_research/data/datasets.py
# 各種データ形式に対応するデータセットクラス
# 
# 機能:
# - Hugging Face Tokenizer を使用するように全面的に刷新。
# - 旧来の独自Vocabularyクラスを廃止し、標準的なNLPパイプラインとの互換性を向上。
# - データ形式に応じたテキスト抽出ロジックを提供。
# - 事前計算されたロジットを読み込むDistillationDatasetを新設。
# - mypyエラーを解消するため、SNNBaseDatasetの型ヒントを修正。
# - 大規模データセットに対応するため、遅延読み込み（Lazy Loading）を実装。

import torch
from torch.utils.data import Dataset
from typing import Iterator, Dict, Any, Tuple
import os
import json
from enum import Enum
from transformers import PreTrainedTokenizerBase

# --- データローダーとデータ形式 ---
class DataFormat(Enum):
    SIMPLE_TEXT = "simple_text"
    DIALOGUE = "dialogue"
    INSTRUCTION = "instruction"

def load_jsonl_data(file_path: str) -> Iterator[Dict[str, Any]]:
    """JSONLファイルを一行ずつ読み込むジェネレータ"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# --- データセットクラス ---
class SNNBaseDataset(Dataset):
    """
    大規模なJSONLファイルをメモリ効率良く扱うための、新しいデータセット基底クラス。
    ファイル全体をメモリに読み込む代わりに、各行のオフセットをキャッシュします。
    """
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # ファイルを開き、各行の開始位置（バイトオフセット）を記録
        self.offsets = []
        with open(self.file_path, 'rb') as f:
            self.offsets.append(f.tell())
            while f.readline():
                self.offsets.append(f.tell())
        self.offsets.pop() # 最後のEOFオフセットは不要なので削除

    def __len__(self):
        return len(self.offsets)

    def _get_json_item(self, idx: int) -> Dict[str, Any]:
        """指定されたインデックスの行をファイルから読み込んでJSONとしてパースする"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            return json.loads(line)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            f"{self.tokenizer.bos_token or ''}{text}",
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors="pt"
        )

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        raise NotImplementedError

class SimpleTextDataset(SNNBaseDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self._get_json_item(idx)
        tokenized = self._encode_text(item['text'])
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids[:-1], input_ids[1:]

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path): yield item['text']

class DialogueDataset(SNNBaseDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self._get_json_item(idx)
        eos_token = self.tokenizer.eos_token or ''
        full_conversation = f" {eos_token} ".join([turn['value'] for turn in item['conversations']])
        tokenized = self._encode_text(full_conversation)
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids[:-1], input_ids[1:]

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path):
            for turn in item['conversations']: yield turn['value']

class InstructionDataset(SNNBaseDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self._get_json_item(idx)
        prompt = item['instruction']
        if 'input' in item and item['input']: prompt += f"\n{item['input']}"
        full_text = f"{prompt}\n{item['output']}"
        tokenized = self._encode_text(full_text)
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids[:-1], input_ids[1:]

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path):
            yield item['instruction']
            if 'input' in item and item['input']: yield item['input']
            yield item['output']

class DistillationDataset(SNNBaseDataset):
    """
    事前計算された教師モデルのロジットを読み込むためのデータセット。
    こちらもメモリ効率の良い読み込み方式を継承。
    """
    def __init__(self, file_path: str, data_dir: str, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        super().__init__(file_path, tokenizer, max_seq_len)
        self.data_dir = data_dir

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self._get_json_item(idx)
        
        tokenized = self._encode_text(item['text'])
        input_ids = tokenized['input_ids'].squeeze(0)
        
        student_input = input_ids[:-1]
        student_target = input_ids[1:]
        
        logits_path = os.path.join(self.data_dir, item['logits_path'])
        teacher_logits = torch.load(logits_path).to(torch.float32)

        min_len = min(student_input.size(0), teacher_logits.size(0))
        
        student_input = student_input[:min_len]
        student_target = student_target[:min_len]
        teacher_logits = teacher_logits[:min_len]
        
        return student_input, student_target, teacher_logits

def get_dataset_class(data_format: DataFormat) -> type[SNNBaseDataset]:
    format_map = {
        DataFormat.SIMPLE_TEXT: SimpleTextDataset,
        DataFormat.DIALOGUE: DialogueDataset,
        DataFormat.INSTRUCTION: InstructionDataset
    }
    return format_map[data_format]
