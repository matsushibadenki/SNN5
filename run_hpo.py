# ファイルパス: run_hpo.py
# Title: Optunaによるハイパーパラメータ最適化スクリプト
# Description: Optunaライブラリを使用し、指定された学習スクリプト
#              (例: run_distillation.py, train.py) のハイパーパラメータを自動調整する。
#
# 修正 (v1):
# - mypyエラー [operator] を解消するため、metric_valueがNoneでないことを確認。
# - mypyエラー [var-annotated] を解消するため、Dict, Anyをインポートし、
#   params_to_save に型ヒントを追加。

import optuna
import argparse
import subprocess
import yaml
import os
import sys
import uuid
import shutil
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import json
import logging
# --- ▼ 修正: Dict, Any をインポート ▼ ---
from typing import Dict, Any
# --- ▲ 修正 ▲ ---

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# プロジェクトルートをPythonパスに追加 (run_hpo.pyがプロジェクトルートにあると仮定)
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# プロジェクト内の関数をインポート (必要に応じて)
# from app.utils import get_auto_device # 必要なら

# --- Objective Function ---
def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    """Optunaの試行ごとに呼び出される目的関数。"""
    
    # --- 1. ハイパーパラメータの提案 ---
    # ここで探索したいパラメータとその範囲を定義
    # 例: 知識蒸留 (run_distillation.py) のパラメータ
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    temperature = trial.suggest_float("temperature", 1.5, 3.5)
    ce_weight = trial.suggest_float("ce_weight", 0.1, 0.5)
    distill_weight = 1.0 - ce_weight # 合わせて1になるように
    spike_reg_weight = trial.suggest_float("spike_reg_weight", 1e-5, 1e-2, log=True)
    # 必要に応じて他のパラメータも追加 (例: batch_size, warmup_epochs, neuron params...)
    
    # --- 2. 設定の上書き ---
    # 各試行にユニークな出力ディレクトリを作成
    trial_id = str(uuid.uuid4())[:8]
    output_dir = Path(args.output_base_dir) / f"trial_{trial.number}_{trial_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    overrides = [
        f"training.gradient_based.learning_rate={lr}",
        f"training.gradient_based.distillation.loss.temperature={temperature}",
        f"training.gradient_based.distillation.loss.ce_weight={ce_weight}",
        f"training.gradient_based.distillation.loss.distill_weight={distill_weight}",
        f"training.gradient_based.distillation.loss.spike_reg_weight={spike_reg_weight}",
        # 最適化中は短いエポック数で実行
        f"training.epochs={args.eval_epochs}",
        f"training.log_dir={output_dir.as_posix()}" # ログ出力先を試行ごとに変更
    ]
    
    # --- 3. 学習スクリプトの実行 ---
    command = [
        sys.executable, # 現在のPythonインタプリタを使用
        args.target_script,
        "--config", args.base_config,
        "--model_config", args.model_config,
        "--task", args.task, # run_distillation.py が task 引数を取る場合
        # 必要に応じて他の引数を追加 (例: --teacher_model)
    ]
    if args.teacher_model:
        command.extend(["--teacher_model", args.teacher_model])
        
    for override in overrides:
        command.extend(["--override_config", override])
        
    logger.info(f"--- Starting Trial {trial.number} ---")
    logger.info(f"Parameters: lr={lr:.5f}, temp={temperature:.2f}, ce_w={ce_weight:.2f}, spike_w={spike_reg_weight:.5f}")
    logger.info(f"Command: {' '.join(command)}")
    
    try:
        # 学習スクリプトを実行 (標準出力をキャプチャ)
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info(f"Trial {trial.number} finished successfully.")
        # logger.debug(result.stdout) # 必要ならログを出力
        
        # --- 4. 結果の解析 ---
        # 学習スクリプトが最終的な検証メトリクスを標準出力するか、
        # 指定されたlog_dirに結果ファイル(例: metrics.json)を保存すると仮定
        
        # 例1: 標準出力からパース (最後の 'Final Validation Metrics:' 行を探す)
        metric_value = float('inf') # Optunaは最小化するので損失をデフォルトに
        accuracy = 0.0 # 精度も記録しておく
        for line in reversed(result.stdout.strip().split('\n')):
            if args.metric_name == "accuracy" and "accuracy:" in line and "Validation Results" in line:
                 try:
                     # 例: "Epoch 2 Validation Results: total: 2.5, ..., accuracy: 0.65"
                     acc_str = line.split("accuracy:")[1].split(",")[0].strip()
                     accuracy = float(acc_str)
                     metric_value = -accuracy # Optunaは最小化するので精度を負にする
                     logger.info(f"Trial {trial.number}: Found validation accuracy: {accuracy:.4f}")
                     break
                 except (IndexError, ValueError) as e:
                     logger.warning(f"Trial {trial.number}: Could not parse accuracy from line: '{line}'. Error: {e}")
            elif args.metric_name == "loss" and "total:" in line and "Validation Results" in line:
                 try:
                     # 例: "Epoch 2 Validation Results: total: 2.5, ..."
                     loss_str = line.split("total:")[1].split(",")[0].strip()
                     metric_value = float(loss_str)
                     logger.info(f"Trial {trial.number}: Found validation loss: {metric_value:.4f}")
                     break
                 except (IndexError, ValueError) as e:
                     logger.warning(f"Trial {trial.number}: Could not parse loss from line: '{line}'. Error: {e}")

        # 例2: 結果ファイルから読み込む (より堅牢)
        # metrics_file = output_dir / "metrics.json"
        # if metrics_file.exists():
        #     with open(metrics_file, 'r') as f:
        #         metrics = json.load(f)
        #         metric_value = metrics.get(args.metric_name, float('inf'))
        # else:
        #     logger.warning(f"Trial {trial.number}: Metrics file not found at {metrics_file}")
        #     metric_value = float('inf') # 失敗時は大きな値を返す
        
        # 不要になった試行ディレクトリを削除 (ディスク容量節約のためオプション)
        # shutil.rmtree(output_dir)
            
        return metric_value

    except subprocess.CalledProcessError as e:
        logger.error(f"Trial {trial.number} failed!")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Return Code: {e.returncode}")
        logger.error(f"Output:\n{e.stdout}")
        logger.error(f"Stderr:\n{e.stderr}")
        # 失敗した試行には非常に悪いスコアを返す
        return float('inf') # 最小化の場合
    except Exception as e:
        logger.error(f"An unexpected error occurred in trial {trial.number}: {e}", exc_info=True)
        return float('inf')

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization using Optuna")
    parser.add_argument("--target_script", type=str, default="run_distillation.py", help="学習を実行するターゲットスクリプト (例: run_distillation.py, train.py)")
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml", help="基本設定ファイル")
    parser.add_argument("--model_config", type=str, required=True, help="モデルアーキテクチャ設定ファイル (例: configs/cifar10_spikingcnn_config.yaml)")
    parser.add_argument("--task", type=str, required=True, help="ターゲットタスク (例: cifar10)")
    parser.add_argument("--teacher_model", type=str, help="教師モデル (run_distillation.py用)")
    parser.add_argument("--n_trials", type=int, default=50, help="Optunaの試行回数")
    parser.add_argument("--eval_epochs", type=int, default=3, help="各試行で実行するエポック数")
    parser.add_argument("--metric_name", type=str, default="accuracy", choices=["accuracy", "loss"], help="最適化するメトリクス ('accuracy' または 'loss')")
    parser.add_argument("--output_base_dir", type=str, default="runs/hpo_trials", help="各試行のログを保存するベースディレクトリ")
    parser.add_argument("--study_name", type=str, default="snn_hpo_study", help="Optuna Studyの名前 (DB保存用)")
    parser.add_argument("--storage", type=str, default="sqlite:///runs/hpo_study.db", help="OptunaのDB保存先 (例: sqlite:///study.db)")
    
    args = parser.parse_args()

    # 出力ディレクトリ作成
    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.storage.replace("sqlite:///", ""))).mkdir(parents=True, exist_ok=True)

    # Optuna Studyの作成と最適化の実行
    direction = "maximize" if args.metric_name == "accuracy" else "minimize"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True, # 既存の研究があれば再開
        direction=direction
    )
    
    logger.info(f"Starting Optuna optimization for {args.n_trials} trials...")
    logger.info(f"Optimizing '{args.metric_name}' ({direction})")
    logger.info(f"Study Name: {args.study_name}, Storage: {args.storage}")
    
    try:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, timeout=None) # Timeoutなし
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
        
    # 最適化結果の表示
    logger.info("--- Optimization Finished ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    metric_value = best_trial.value
    # --- ▼ 修正: mypy [operator] ▼ ---
    if args.metric_name == "accuracy" and metric_value is not None:
        metric_value = -metric_value # 精度に戻す
    # --- ▲ 修正 ▲ ---
        
    logger.info(f"Best trial (Trial {best_trial.number}):")
    logger.info(f"  Value ({args.metric_name}): {metric_value:.4f}")
    logger.info("  Best Parameters:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # 最適パラメータをYAMLファイルとして保存
    best_params_yaml = Path(args.output_base_dir) / "best_params.yaml"
    with open(best_params_yaml, 'w') as f:
        # パラメータをOmegaConfが読み込める形式で保存
        # --- ▼ 修正: mypy [var-annotated] ▼ ---
        params_to_save: Dict[str, Any] = {}
        # --- ▲ 修正 ▲ ---
        for key, value in best_trial.params.items():
            # '.'区切りでネスト構造を作成
            keys = key.split('.')
            d = params_to_save
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        yaml.dump(params_to_save, f, default_flow_style=False)
    logger.info(f"Best parameters saved to: {best_params_yaml}")