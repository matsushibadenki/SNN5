# matsushibadenki/snn/scripts/run_benchmark.py
# 複数のタスクを実行可能な、新しいベンチマークスイート
#
# 変更点:
# - mypyエラーを解消するため、型ヒントの修正と、結果を格納する辞書の扱いを変更。
#
# 改善点:
# - ROADMAPフェーズ4「ハードウェア展開」に基づき、ハードウェアプロファイルの選択機能を追加。
# - SNNとANNのエネルギー効率を比較し、最終レポートに表示する機能を追加。

import argparse
import time
import pandas as pd  # type: ignore
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, Any, List, Tuple, Callable, Sized, cast

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_research.benchmark.tasks import SST2Task
from snn_research.hardware.profiles import get_hardware_profile

# タスク名とクラスのマッピング
TASK_REGISTRY = {
    "sst2": SST2Task,
}

def run_single_task(task_name: str, device: str, hardware_profile: Dict[str, Any]):
    """単一のベンチマークタスクを実行する。"""
    print("\n" + "="*20 + f" 🚀 Starting Benchmark for: {task_name.upper()} " + "="*20)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    TaskClass = TASK_REGISTRY[task_name]
    task = TaskClass(tokenizer, device, hardware_profile)

    _, val_dataset = task.prepare_data(data_dir="data")
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=task.get_collate_fn())
    
    results = []
    for model_type in ['SNN', 'ANN']:
        print(f"\n--- Evaluating {model_type} model ---")
        model = task.build_model(model_type, tokenizer.vocab_size).to(device)
        
        start_time = time.time()
        metrics = task.evaluate(model, val_loader)
        duration = time.time() - start_time
        
        result_record: Dict[str, Any] = {
            "model": model_type,
            "task": task_name,
            "eval_time_sec": duration,
        }
        result_record.update(metrics)
        
        # ANNの場合、比較用のエネルギー消費を計算
        if model_type == 'ANN':
            num_params = sum(p.numel() for p in model.parameters())
            # 1パラメータあたり1回のMAC演算と仮定
            ann_ops = num_params
            result_record['estimated_energy_j'] = ann_ops * hardware_profile["ann_energy_per_op"]
        
        results.append(result_record)
        print(f"  - Results: {result_record}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="SNN vs ANN Benchmark Suite")
    parser.add_argument(
        "--task", 
        type=str, 
        default="all", 
        choices=["all"] + list(TASK_REGISTRY.keys()),
        help="実行するベンチマークタスクを選択します。"
    )
    parser.add_argument("--model_path", type=str, help="評価する学習済みSNNモデルのパス。")
    parser.add_argument("--hardware_profile", type=str, default="default", help="使用するハードウェアプロファイル (例: 'loihi')")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    hardware_profile = get_hardware_profile(args.hardware_profile)
    print(f"Using hardware profile: {hardware_profile['name']}")

    tasks_to_run = TASK_REGISTRY.keys() if args.task == "all" else [args.task]
    
    all_results = []
    for task_name in tasks_to_run:
        all_results.extend(run_single_task(task_name, device, hardware_profile))

    print("\n\n" + "="*25 + " 🏆 Final Benchmark Summary " + "="*25)
    df = pd.DataFrame(all_results)
    
    # エネルギー効率の比較を計算
    snn_energy = df[df['model'] == 'SNN']['estimated_energy_j'].iloc[0]
    ann_energy = df[df['model'] == 'ANN']['estimated_energy_j'].iloc[0]
    if ann_energy > 0:
        efficiency_gain = (1 - (snn_energy / ann_energy)) * 100
        df['efficiency_gain_%'] = [f"{efficiency_gain:.2f}%" if m == 'SNN' else '-' for m in df['model']]

    print(df.to_string())
    print("="*90)

if __name__ == "__main__":
    main()
