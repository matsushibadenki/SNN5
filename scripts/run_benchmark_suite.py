# ファイルパス: scripts/run_benchmark_suite.py
# Title: 統合ベンチマークスイート
# Description: 複数のベンチマーク実験を体系的に実行し、結果をリーダーボード形式でレポートに追記する。
# 改善点(v2): MRPCタスクの比較実験を追加。
# 改善点(v3): 継続学習評価のため、--model_path と --eval_only を追加。
#             訓練済みモデルを指定し、評価のみを実行できるようにする。
#
# 修正 (v4):
# - 健全性チェック (health-check) での `omegaconf.errors.ConfigAttributeError: Missing key model` エラーを解消。
# - `train_and_evaluate_model` が `cifar10_spikingcnn_config.yaml` のような
#   `model:` キーを持たない設定ファイルをロードする際、
#   `run_distillation.py` と同様に `{'model': ...}` でラップするように修正。

import argparse
import time
import pandas as pd  # type: ignore
import torch
import torch.nn as nn
# ... existing code ...
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
# ... existing code ...
from pathlib import Path
import sys
# --- ▼ 修正: 必要な型ヒントを追加 ▼ ---
from typing import Dict, List, Any, Optional
# ... existing code ...
# --- ▲ 修正 ▲ ---

sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_research.benchmark import TASK_REGISTRY, BenchmarkTask
from app.utils import get_auto_device
from transformers import AutoTokenizer
# --- ▼ 修正: SNNCoreをインポート（モデルロード用） ▼ ---
from snn_research.core.snn_core import SNNCore
from omegaconf import OmegaConf, DictConfig # DictConfig をインポート
# --- ▲ 修正 ▲ ---

def train_and_evaluate_model(
# ... existing code ...
    model_type: str,
    task: BenchmarkTask,
    train_loader: DataLoader,
# ... existing code ...
    device: str,
    epochs: int,
    learning_rate: float,
# ... existing code ...
    vocab_size: int,
    # --- ▼ 修正: 評価専用モードの引数を追加 ▼ ---
    eval_only: bool = False,
# ... existing code ...
    model_path: Optional[str] = None,
    model_config_path: Optional[str] = None
    # --- ▲ 修正 ▲ ---
# ... existing code ...
) -> Dict[str, Any]:
    """指定されたモデルタイプの訓練と評価を行うヘルパー関数。"""
    
    model: nn.Module
# ... existing code ...
    
    # --- ▼ 修正: 評価専用モードのロジック ▼ ---
    if eval_only:
        if not model_path or not model_config_path:
# ... existing code ...
            raise ValueError("--eval_only を使用する場合、--model_path と --model_config_path の両方が必要です。")
        print("\n" + "="*20 + f" 🚀 Starting EVALUATION for: {model_path} on {task.__class__.__name__} " + "="*20)
        
        # SNNCoreコンテナ経由でモデルをロード
        try:
            # --- ▼ 修正 (v4): 'model:' キーがないコンフィグに対応 ▼ ---
            cfg_raw: DictConfig = OmegaConf.load(model_config_path)
            cfg_model: DictConfig
            if "model" in cfg_raw:
                cfg_model = cfg_raw.model
            else:
                # cifar10_spikingcnn_config.yaml のようなファイルの場合、
                # cfg_raw自体がモデル設定だと見なす
                cfg_model = cfg_raw
            # --- ▲ 修正 (v4) ▲ ---

            # モデルタイプ（SNNかANNか）に基づいてロード処理を変更
            if model_type == 'SNN':
# ... existing code ...
                # vocab_sizeはタスクに応じて設定
                num_classes = 10 if task.__class__.__name__ == "CIFAR10Task" else vocab_size
                model_container = SNNCore(config=cfg_model, vocab_size=num_classes)
                model = model_container.model # SNNCore内部の実際のモデルを取得
# ... existing code ...
            else: # ANN
                # ANNBaselineModelまたはSimpleCNNをインスタンス化
                if task.__class__.__name__ == "CIFAR10Task":
# ... existing code ...
                    model = task.build_model('ANN', vocab_size=10) # SimpleCNN
                else:
                    model = task.build_model('ANN', vocab_size=vocab_size) # ANNBaselineModel
# ... existing code ...
            
            # state_dictのロード
            checkpoint = torch.load(model_path, map_location=device)
# ... existing code ...
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if list(state_dict.keys())[0].startswith('module.'):
                 state_dict = {k[7:]: v for k, v in state_dict.items()}
# ... existing code ...
            
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            print(f"✅ 評価用モデルを '{model_path}' から正常にロードしました。")
# ... existing code ...
        except Exception as e:
            print(f"❌ 評価用モデルのロードに失敗しました: {e}")
            raise
# ... existing code ...
    else:
        # 従来の訓練モード
        print("\n" + "="*20 + f" 🚀 Starting Experiment for: {model_type} on {task.__class__.__name__} " + "="*20)
# ... existing code ...
        model = task.build_model(model_type, vocab_size=vocab_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate)
# ... existing code ...
        
        for epoch in range(epochs):
            model.train()
# ... existing code ...
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{model_type} Training]")
            for batch in train_progress:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
# ... existing code ...
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(**inputs)
# ... existing code ...
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(logits, labels)
                loss.backward()
# ... existing code ...
                optimizer.step()
                train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
    # --- ▲ 修正 ▲ ---
# ... existing code ...
            
    print(f"\n--- Evaluating {model_type} model ---")
    start_time = time.time()
# ... existing code ...
    metrics = task.evaluate(model, val_loader)
    duration = time.time() - start_time
    
    metrics["model"] = model_type
# ... existing code ...
    metrics["eval_time_sec"] = duration
    
    print(f"  - Results: {metrics}")
# ... existing code ...
    return metrics

# --- ▼ 修正: 実行関数がargs全体を受け取るように変更 ▼ ---
def run_experiment_by_name(experiment_name: str, args: argparse.Namespace) -> pd.DataFrame:
# ... existing code ...
    """実験名に基づいて適切な比較実験を実行する。"""
    device = get_auto_device()
    TaskClass = TASK_REGISTRY.get(experiment_name.split('_')[0]) #例: "cifar10"
# ... existing code ...
    if not TaskClass:
        raise ValueError(f"Task for experiment '{experiment_name}' not found.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
# ... existing code ...
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None: tokenizer.sep_token = tokenizer.eos_token
    
    task = TaskClass(tokenizer=tokenizer, device=device, hardware_profile={})
# ... existing code ...
    
    train_dataset, val_dataset = task.prepare_data(data_dir="data")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn(), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=task.get_collate_fn())

    vocab_size = tokenizer.vocab_size
# ... existing code ...
    if experiment_name == "cifar10_comparison":
        vocab_size = 10 # CIFAR10のクラス数

    results = []
# ... existing code ...
    
    # --- ▼ 修正: eval_onlyロジックを反映 ▼ ---
    # SNNモデルのパスとコンフィグ
    snn_model_path = args.model_path if args.model_type == 'SNN' else None
# ... existing code ...
    snn_model_config = args.model_config if args.model_type == 'SNN' else None
    
    # ANNモデルのパスとコンフィグ
    ann_model_path = args.model_path if args.model_type == 'ANN' else None
# ... existing code ...
    # ANNのコンフィグはSNNと同じものを使うか、別途指定が必要 (ここではSNN用を流用)
    ann_model_config = args.model_config if args.model_type == 'ANN' else None 
    
    # --model_type が指定されている場合、そのタイプのみを評価
# ... existing code ...
    if args.model_type:
        model_path = args.model_path
        model_config = args.model_config
# ... existing code ...
        
        metrics = train_and_evaluate_model(
            args.model_type, task, train_loader, val_loader, device, args.epochs, args.learning_rate, vocab_size,
# ... existing code ...
            eval_only=args.eval_only, model_path=model_path, model_config_path=model_config
        )
        results.append(metrics)
# ... existing code ...
    else:
        # 通常の比較実行
        ann_metrics = train_and_evaluate_model(
# ... existing code ...
            'ANN', task, train_loader, val_loader, device, args.epochs, args.learning_rate, vocab_size,
            eval_only=args.eval_only, model_path=ann_model_path, model_config_path=ann_model_config
        )
# ... existing code ...
        results.append(ann_metrics)
        snn_metrics = train_and_evaluate_model(
            'SNN', task, train_loader, val_loader, device, args.epochs, args.learning_rate, vocab_size,
# ... existing code ...
            eval_only=args.eval_only, model_path=snn_model_path, model_config_path=snn_model_config
        )
        results.append(snn_metrics)
# ... existing code ...
    # --- ▲ 修正 ▲ ---
    
    return pd.DataFrame(results)


def save_report(df: pd.DataFrame, output_dir: str, experiment_name: str, args: argparse.Namespace):
# ... existing code ...
    """実験結果をMarkdown形式でリーダーボードに追記する。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
# ... existing code ...
    report_path = output_path / f"{experiment_name}_leaderboard.md"

    if 'estimated_energy_j' in df.columns and 'model' in df.columns:
        snn_row = df[df['model'] == 'SNN']
# ... existing code ...
        ann_row = df[df['model'] == 'ANN']
        if not snn_row.empty and not ann_row.empty:
            snn_energy = snn_row['estimated_energy_j'].iloc[0]
# ... existing code ...
            ann_energy = ann_row['estimated_energy_j'].iloc[0]
            if ann_energy > 0 and snn_energy is not None:
                efficiency_gain = (1 - (snn_energy / ann_energy)) * 100
# ... existing code ...
                df['efficiency_gain_%'] = [f"{efficiency_gain:.2f}%" if m == 'SNN' else '-' for m in df['model']]

    # 新しい実行結果にメタデータを追加
    df['run_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
# ... existing code ...
    df['tag'] = args.tag or 'default'
    # --- ▼ 修正: 評価モードの情報を追記 ▼ ---
    if args.eval_only:
# ... existing code ...
         df['mode'] = f"EvalOnly ({Path(args.model_path).name if args.model_path else 'N/A'})" # model_pathがNoneの場合を処理
    else:
         df['mode'] = "Train+Eval"
# ... existing code ...
    # --- ▲ 修正 ▲ ---

    # レポートファイルに追記
    with open(report_path, 'a', encoding='utf-8') as f:
# ... existing code ...
        if f.tell() == 0: # ファイルが空の場合、ヘッダーを書き込む
            f.write(f"# Benchmark Leaderboard: {experiment_name.replace('_', ' ').title()}\n\n")
        f.write(f"## 📊 Run at: {df['run_date'].iloc[0]} (Tag: {df['tag'].iloc[0]})\n\n")
        
        # --- ▼ 修正: 評価モードの情報を追記 ▼ ---
# ... existing code ...
        if args.eval_only:
            f.write(f"**Configuration:** Mode: EvalOnly, Model: {args.model_path}, Config: {args.model_config}\n\n")
        else:
# ... existing code ...
            f.write(f"**Configuration:** Mode: Train+Eval, Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.learning_rate}\n\n")
        # --- ▲ 修正 ▲ ---
            
        f.write(df.to_markdown(index=False))
# ... existing code ...
        f.write("\n\n---\n\n")

    print(f"\n✅ ベンチマーク結果を '{report_path}' に追記しました。")
# ... existing code ...

def main(args: argparse.Namespace):
    """ベンチマークスイートのメイン関数。"""
    if args.experiment == "all":
# ... existing code ...
        # 'all' の場合、eval_only はサポートしない（複雑になりすぎるため）
        if args.eval_only:
            print("Error: --eval_only は 'all' 実験ではサポートされていません。個別のタスクを指定してください。")
# ... existing code ...
            return
            
        cifar10_results_df = run_experiment_by_name("cifar10_comparison", args)
        save_report(cifar10_results_df, args.output_dir, "cifar10_ann_vs_snn", args)
# ... existing code ...
        
        sst2_results_df = run_experiment_by_name("sst2_comparison", args)
        save_report(sst2_results_df, args.output_dir, "sst2_ann_vs_snn", args)
        
        mrpc_results_df = run_experiment_by_name("mrpc_comparison", args)
# ... existing code ...
        save_report(mrpc_results_df, args.output_dir, "mrpc_ann_vs_snn", args)
        
    elif args.experiment in ["cifar10_comparison", "sst2_comparison", "mrpc_comparison"]:
        results_df = run_experiment_by_name(args.experiment, args)
# ... existing code ...
        report_name = args.experiment.replace('_comparison', '_ann_vs_snn')
        save_report(results_df, args.output_dir, report_name, args)
    else:
# ... existing code ...
        print(f"Unknown experiment: {args.experiment}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN vs ANN Benchmark Suite")
# ... existing code ...
    parser.add_argument("--experiment", type=str, default="all", choices=["all", "cifar10_comparison", "sst2_comparison", "mrpc_comparison"], help="実行する実験を選択します。")
    parser.add_argument("--tag", type=str, help="実験にカスタムタグを付けます。")
    parser.add_argument("--output_dir", type=str, default="benchmarks", help="結果レポートを保存するディレクトリ。")
# ... existing code ...
    
    # 訓練モード用
    parser.add_argument("--epochs", type=int, default=3, help="[訓練モード] 訓練のエポック数。")
    parser.add_argument("--batch_size", type=int, default=32, help="[訓練モード] 訓練と評価のバッチサイズ。")
# ... existing code ...
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="[訓練モード] オプティマイザの学習率。")

    # --- ▼ 修正: 評価専用モードの引数を追加 ▼ ---
    parser.add_argument("--eval_only", action="store_true", help="[評価モード] 訓練をスキップし、指定されたモデルで評価のみを行います。")
# ... existing code ...
    parser.add_argument("--model_path", type=str, help="[評価モード] 評価する学習済みモデルのパス (.pth)。")
    parser.add_argument("--model_config", type=str, help="[評価モード] 評価するモデルのアーキテクチャ設定ファイル (.yaml)。")
    parser.add_argument("--model_type", type=str, choices=['SNN', 'ANN'], help="[評価モード] 評価するモデルのタイプ (SNNまたはANN)。")
# ... existing code ...
    # --- ▲ 修正 ▲ ---
    
    args = parser.parse_args()
    
# ... existing code ...
    # --- ▼ 修正: 評価モードの引数チェック ▼ ---
    if args.eval_only and (not args.model_path or not args.model_config):
        print("Error: --eval_only を使用する場合、--model_path と --model_config の両方を指定する必要があります。")
# ... existing code ...
        sys.exit(1)
    if args.eval_only and not args.model_type:
         # model_typeが指定されていない場合、両方を評価しようとするため、
# ... existing code ...
         # ここでは、--eval_only時は--model_typeも必須とする。
         print("Error: --eval_only を使用する場合、--model_type ('SNN' または 'ANN') も指定する必要があります。")
         sys.exit(1)
# ... existing code ...
    if args.eval_only and args.experiment == "all":
         print("Error: --eval_only は 'all' 実験ではサポートされていません。個別のタスクを指定してください。")
         sys.exit(1)
# ... existing code ...
    # --- ▲ 修正 ▲ ---

    main(args)
