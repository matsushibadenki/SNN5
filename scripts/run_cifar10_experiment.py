# ファイルパス: scripts/run_cifar10_experiment.py
# (新規作成)
# Title: CIFAR-10 ベンチマーク実験スクリプト
# Description:
# snn_4_ann_parity_plan.mdの「優先実験案」に基づき、CIFAR-10データセットで
# ANNモデルとSNNモデルの訓練・評価を行い、性能を直接比較するためのスクリプト。
# 修正(mypy): [import-untyped], [name-defined]エラーを解消。

import argparse
import time
import pandas as pd  # type: ignore
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import sys
from typing import Dict

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_research.benchmark import TASK_REGISTRY
from app.utils import get_auto_device
from transformers import AutoTokenizer

def train_and_evaluate_model(
    model_type: str,
    task,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float
) -> Dict:
    """
    指定されたモデルタイプの訓練と評価を行う。
    """
    print("\n" + "="*20 + f" 🚀 Starting Experiment for: {model_type} " + "="*20)
    
    # vocab_sizeは画像タスクでは使用しないが、インターフェースを合わせるために渡す
    model = task.build_model(model_type, vocab_size=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # 訓練ループ
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{model_type} Training]")
        for batch in train_progress:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            # モデルの出力形式に合わせてlogitsを取得
            outputs = model(**inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
    # 評価
    print(f"\n--- Evaluating {model_type} model ---")
    start_time = time.time()
    metrics = task.evaluate(model, val_loader)
    duration = time.time() - start_time
    
    metrics["model"] = model_type
    metrics["eval_time_sec"] = duration
    
    print(f"  - Results: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SNN vs ANN CIFAR-10 Experiment")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs for demonstration.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training and validation batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    args = parser.parse_args()

    device = get_auto_device()
    print(f"Using device: {device}")
    
    # CIFAR-10タスクの準備
    TaskClass = TASK_REGISTRY["cifar10"]
    # tokenizerは不要だがインターフェースを合わせるためにダミーを渡す
    task = TaskClass(tokenizer=AutoTokenizer.from_pretrained("gpt2"), device=device, hardware_profile={})
    
    train_dataset, val_dataset = task.prepare_data()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn(), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn())

    all_results = []
    
    # ANNモデルの訓練と評価
    ann_metrics = train_and_evaluate_model(
        'ANN', task, train_loader, val_loader, device, args.epochs, args.learning_rate
    )
    all_results.append(ann_metrics)
    
    # SNNモデルの訓練と評価
    snn_metrics = train_and_evaluate_model(
        'SNN', task, train_loader, val_loader, device, args.epochs, args.learning_rate
    )
    all_results.append(snn_metrics)

    # 最終比較レポートの表示
    print("\n\n" + "="*25 + " 🏆 Final Comparison Summary " + "="*25)
    df = pd.DataFrame(all_results)
    
    snn_energy = df[df['model'] == 'SNN']['estimated_energy_j'].iloc[0]
    ann_energy = df[df['model'] == 'ANN']['estimated_energy_j'].iloc[0]
    if ann_energy > 0 and snn_energy is not None:
        efficiency_gain = (1 - (snn_energy / ann_energy)) * 100
        df['efficiency_gain_%'] = [f"{efficiency_gain:.2f}%" if m == 'SNN' else '-' for m in df['model']]

    print(df.to_string())
    print("="*90)


if __name__ == "__main__":
    main()