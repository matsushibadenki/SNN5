# matsushibadenki/snn4/snn_research/benchmark/metrics.py
# ベンチマーク評価用のメトリクス関数
#
# 修正点:
# - calculate_perplexity: ダミー実装から、クロスエントロピー損失に基づく実際のパープレキシティ計算を実装。
# - calculate_energy_consumption: ダミー実装から、スパイク数に基づくエネルギー消費の推定モデルを実装。
#
# 修正 (v2):
# - calculate_energy_consumption を snn_research/metrics/energy.py に移管。

from typing import List, Any, Dict, cast, Sized
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# calculate_energy_consumption は snn_research.metrics.energy に移動した

def calculate_accuracy(true_labels: List[int], pred_labels: List[int]) -> float:
    """分類タスクの正解率を計算する。"""
    if len(true_labels) == 0:
        return 0.0
    correct = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    return correct / len(true_labels)

def calculate_perplexity(model: nn.Module, loader: DataLoader, device: str) -> float:
    """
    言語モデルのパープレキシティを計算する。
    パープレキシティは、クロスエントロピー損失の指数として計算され、モデルがテストセットをどれだけ「驚き」をもって予測したかを示す指標。低いほど良い。
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # パディングトークンなどを無視
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Calculating Perplexity"):
            # データローダーの出力形式に応じて柔軟に対応
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                # ラベルが 'labels' または 'targets' のキーで渡されることを想定
                labels = batch.get('labels', batch.get('targets'))
                if labels is not None:
                    labels = labels.to(device)
            else:
                input_ids, labels = batch[0], batch[1]
                if input_ids is not None:
                    input_ids = input_ids.to(device)
                if labels is not None:
                    labels = labels.to(device)

            if input_ids is None or labels is None:
                continue

            # モデルの forward パス
            # SNNモデルは (logits, spikes, mem) のタプルを返す可能性がある
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # 損失計算
            # logits: [batch, seq_len, vocab_size], labels: [batch, seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # パディングされていないトークン数で加重平均
            num_tokens = (labels.view(-1) != criterion.ignore_index).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float('inf')

    # 全トークンにわたる平均クロスエントロピー損失
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity
