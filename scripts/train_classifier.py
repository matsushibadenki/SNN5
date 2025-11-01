# ファイルパス: scripts/train_classifier.py
# (更新)
#
# Title: 分類タスク用 SNN訓練スクリプト
#
# Description:
# - SST-2のような分類タスクでSNNモデルを訓練し、評価するための専用スクリプト。
# - DIコンテナへの依存をなくし、循環参照エラーを根本的に解決。
# - 訓練後、検証セットで最高の精度を達成したモデルを保存する。

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import sys

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 循環参照を避けるため、必要なコンポーネントを直接インポート
from snn_research.benchmark import TASK_REGISTRY
from app.utils import get_auto_device
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="SNN Classifier Training Script")
    parser.add_argument("--task", type=str, default="sst2", choices=list(TASK_REGISTRY.keys()), help="The benchmark task to train on.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--output_dir", type=str, default="runs/classifiers", help="Directory to save the trained model.")
    args = parser.parse_args()

    device = get_auto_device()
    print(f"Using device: {device}")

    # 1. タスクとデータの準備
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    TaskClass = TASK_REGISTRY.get(args.task)
    if not TaskClass:
        raise ValueError(f"Task '{args.task}' not found in registry.")

    # hardware_profileは訓練に不要なため、ダミーを渡す
    task = TaskClass(tokenizer, device, hardware_profile={}) 
    
    train_dataset, val_dataset = task.prepare_data()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn(), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn())

    # 2. モデル、損失関数、オプティマイザの準備
    model = task.build_model('SNN', tokenizer.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    best_accuracy = 0.0
    output_path = Path(args.output_dir) / args.task
    output_path.mkdir(parents=True, exist_ok=True)
    model_save_path = output_path / "best_model.pth"

    print(f"\n--- Starting training for '{args.task}' for {args.epochs} epochs ---")

    for epoch in range(args.epochs):
        # 訓練ループ
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        for batch in train_progress:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            logits, _, _ = model(**inputs)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})

        # 検証ループ
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                
                logits, _, _ = model(**inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Accuracy={accuracy:.4f}")

        # ベストモデルの保存
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"🏆 New best model saved to '{model_save_path}' with accuracy: {accuracy:.4f}")

    print("\n✅ Training complete.")

if __name__ == "__main__":
    main()