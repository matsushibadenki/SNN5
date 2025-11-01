# ファイルパス: scripts/run_benchmark_suite.py
# Title: 統合ベンチマークスイート
# Description: 複数のベンチマーク実験を体系的に実行し、結果をリーダーボード形式でレポートに追記する。
# 改善点(v2): MRPCタスクの比較実験を追加。
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
from typing import Dict, List, Any

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
    learning_rate: float,
    vocab_size: int
) -> Dict[str, Any]:
    """指定されたモデルタイプの訓練と評価を行うヘルパー関数。"""
    print("\n" + "="*20 + f" 🚀 Starting Experiment for: {model_type} on {task.__class__.__name__} " + "="*20)
    
    model = task.build_model(model_type, vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{model_type} Training]")
        for batch in train_progress:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
    print(f"\n--- Evaluating {model_type} model ---")
    start_time = time.time()
    metrics = task.evaluate(model, val_loader)
    duration = time.time() - start_time
    
    metrics["model"] = model_type
    metrics["eval_time_sec"] = duration
    
    print(f"  - Results: {metrics}")
    return metrics

def run_cifar10_comparison(args: argparse.Namespace) -> pd.DataFrame:
    """CIFAR-10でANNとSNNの性能を比較する実験を実行する。"""
    device = get_auto_device()
    TaskClass = TASK_REGISTRY["cifar10"]
    task = TaskClass(tokenizer=AutoTokenizer.from_pretrained("gpt2"), device=device, hardware_profile={})
    
    train_dataset, val_dataset = task.prepare_data(data_dir="data")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn(), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn())

    results = []
    ann_metrics = train_and_evaluate_model('ANN', task, train_loader, val_loader, device, args.epochs, args.learning_rate, vocab_size=10)
    results.append(ann_metrics)
    snn_metrics = train_and_evaluate_model('SNN', task, train_loader, val_loader, device, args.epochs, args.learning_rate, vocab_size=10)
    results.append(snn_metrics)
    return pd.DataFrame(results)

def run_sst2_comparison(args: argparse.Namespace) -> pd.DataFrame:
    """SST-2でANNとSNNの性能を比較する実験を実行する。"""
    device = get_auto_device()
    TaskClass = TASK_REGISTRY["sst2"]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    task = TaskClass(tokenizer=tokenizer, device=device, hardware_profile={})
    
    train_dataset, val_dataset = task.prepare_data(data_dir="data")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn(), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn())

    results = []
    ann_metrics = train_and_evaluate_model('ANN', task, train_loader, val_loader, device, args.epochs, args.learning_rate, vocab_size=tokenizer.vocab_size)
    results.append(ann_metrics)
    snn_metrics = train_and_evaluate_model('SNN', task, train_loader, val_loader, device, args.epochs, args.learning_rate, vocab_size=tokenizer.vocab_size)
    results.append(snn_metrics)
    return pd.DataFrame(results)

def run_mrpc_comparison(args: argparse.Namespace) -> pd.DataFrame:
    """MRPCでANNとSNNの性能を比較する実験を実行する。"""
    device = get_auto_device()
    TaskClass = TASK_REGISTRY["mrpc"]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None: tokenizer.sep_token = tokenizer.eos_token
    task = TaskClass(tokenizer=tokenizer, device=device, hardware_profile={})
    
    train_dataset, val_dataset = task.prepare_data(data_dir="data")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn(), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn())

    results = []
    ann_metrics = train_and_evaluate_model('ANN', task, train_loader, val_loader, device, args.epochs, args.learning_rate, vocab_size=tokenizer.vocab_size)
    results.append(ann_metrics)
    snn_metrics = train_and_evaluate_model('SNN', task, train_loader, val_loader, device, args.epochs, args.learning_rate, vocab_size=tokenizer.vocab_size)
    results.append(snn_metrics)
    return pd.DataFrame(results)

def save_report(df: pd.DataFrame, output_dir: str, experiment_name: str, args: argparse.Namespace):
    """実験結果をMarkdown形式でリーダーボードに追記する。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"{experiment_name}_leaderboard.md"

    if 'estimated_energy_j' in df.columns:
        snn_row = df[df['model'] == 'SNN']
        ann_row = df[df['model'] == 'ANN']
        if not snn_row.empty and not ann_row.empty:
            snn_energy = snn_row['estimated_energy_j'].iloc[0]
            ann_energy = ann_row['estimated_energy_j'].iloc[0]
            if ann_energy > 0 and snn_energy is not None:
                efficiency_gain = (1 - (snn_energy / ann_energy)) * 100
                df['efficiency_gain_%'] = [f"{efficiency_gain:.2f}%" if m == 'SNN' else '-' for m in df['model']]

    # 新しい実行結果にメタデータを追加
    df['run_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
    df['tag'] = args.tag or 'default'

    # レポートファイルに追記
    with open(report_path, 'a', encoding='utf-8') as f:
        if f.tell() == 0: # ファイルが空の場合、ヘッダーを書き込む
            f.write(f"# Benchmark Leaderboard: {experiment_name.replace('_', ' ').title()}\n\n")
        f.write(f"## 📊 Run at: {df['run_date'].iloc[0]} (Tag: {df['tag'].iloc[0]})\n\n")
        f.write(f"**Configuration:** Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.learning_rate}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n---\n\n")

    print(f"\n✅ ベンチマーク結果を '{report_path}' に追記しました。")

def main(args: argparse.Namespace):
    """ベンチマークスイートのメイン関数。"""
    if args.experiment == "all":
        cifar10_results_df = run_cifar10_comparison(args)
        save_report(cifar10_results_df, args.output_dir, "cifar10_ann_vs_snn", args)
        sst2_results_df = run_sst2_comparison(args)
        save_report(sst2_results_df, args.output_dir, "sst2_ann_vs_snn", args)
        mrpc_results_df = run_mrpc_comparison(args)
        save_report(mrpc_results_df, args.output_dir, "mrpc_ann_vs_snn", args)
    elif args.experiment == "cifar10_comparison":
        results_df = run_cifar10_comparison(args)
        save_report(results_df, args.output_dir, "cifar10_ann_vs_snn", args)
    elif args.experiment == "sst2_comparison":
        results_df = run_sst2_comparison(args)
        save_report(results_df, args.output_dir, "sst2_ann_vs_snn", args)
    elif args.experiment == "mrpc_comparison":
        results_df = run_mrpc_comparison(args)
        save_report(results_df, args.output_dir, "mrpc_ann_vs_snn", args)
    else:
        print(f"Unknown experiment: {args.experiment}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN vs ANN Benchmark Suite")
    parser.add_argument("--experiment", type=str, default="all", choices=["all", "cifar10_comparison", "sst2_comparison", "mrpc_comparison"], help="実行する実験を選択します。")
    parser.add_argument("--tag", type=str, help="実験にカスタムタグを付けます。")
    parser.add_argument("--epochs", type=int, default=3, help="訓練のエポック数。")
    parser.add_argument("--batch_size", type=int, default=32, help="訓練と評価のバッチサイズ。")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="オプティマイザの学習率。")
    parser.add_argument("--output_dir", type=str, default="benchmarks", help="結果レポートを保存するディレクトリ。")
    
    args = parser.parse_args()
    main(args)