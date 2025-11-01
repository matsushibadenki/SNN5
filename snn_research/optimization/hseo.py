# ファイルパス: snn_research/optimization/hseo.py
# (新規作成)
# Title: Hybrid Swarm Evolution Optimization (HSEO) 実装
# Description:
# SelfEvolvingAgent が使用するための、微分不要な最適化アルゴリズムを実装します。
# 中核アルゴリズムとして、粒子群最適化 (Particle Swarm Optimization, PSO) を実装します。
# また、SNNの性能を評価するための目的関数 evaluate_snn_params も実装します。
# mypy --strict 準拠。

import numpy as np
import torch
import sys
import subprocess
import logging
import re
import os
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Any, Optional

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# プロジェクトルートをPythonパスに追加 (train.py を呼び出すため)
project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 1. SNN評価関数 (目的関数) ---

def evaluate_snn_params(
    model_config_path: str,
    base_training_config_path: str,
    params_to_override: Dict[str, Any],
    eval_epochs: int = 1,
    device: str = "cpu",
    task_name: str = "sst2", # 評価用のデフォルトタスク
    metric_to_optimize: str = "loss" # "loss" または "accuracy"
) -> float:
    """
    指定されたパラメータで SNN モデルを短期間訓練し、性能メトリクスを返す。
    run_hpo.py と同様に、train.py をサブプロセスとして呼び出す。

    Args:
        model_config_path (str): モデル設定ファイルのパス。
        base_training_config_path (str): 基本訓練設定ファイルのパス。
        params_to_override (Dict[str, Any]): 上書きするパラメータ (例: {"training.gradient_based.learning_rate": 0.001})。
        eval_epochs (int): 評価のために実行するエポック数。
        device (str): 使用するデバイス ('cpu' or 'cuda')。
        task_name (str): 評価に使用するタスク名 (データパスの代わり)。
        metric_to_optimize (str): 最適化対象のメトリクス ("loss" または "accuracy")。

    Returns:
        float: 評価スコア (最適化対象メトリクス)。
               Optunaと同様に、損失(loss)はそのまま、精度(accuracy)は負の値を返す。
    """
    
    # --- 1. 設定の上書き ---
    overrides: List[str] = []
    for key, value in params_to_override.items():
        overrides.append(f"{key}={value}")
    
    # 評価用のエポック数とタスク名で上書き
    overrides.append(f"training.epochs={eval_epochs}")
    overrides.append(f"task_name={task_name}") # train.py が task_name からデータをロードすることを期待
    
    # ログディレクトリを一時的な場所にする (HSEOではログは不要)
    # (ただし、train.py がログディレクトリに依存する場合は指定が必要)
    # overrides.append("training.log_dir=/tmp/hseo_eval") 

    # --- 2. 学習スクリプトの実行コマンド構築 ---
    command: List[str] = [
        sys.executable, # 現在のPythonインタプリタ
        os.path.join(project_root, "train.py"),
        "--config", base_training_config_path,
        "--model_config", model_config_path,
        # train.py が --task_name に応じてデータパスを内部で処理すると仮定
        # もし --data_path が必須なら、task_nameに応じて設定する必要がある
    ]
    
    for override in overrides:
        command.extend(["--override_config", override])

    logger.info(f"HSEO: Evaluating parameters with command: {' '.join(command)}")

    # --- 3. サブプロセス実行と結果のパース ---
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        
        # 標準出力から最後の検証メトリクスをパース
        metric_value: Optional[float] = None
        
        if metric_to_optimize == "accuracy":
            # 精度をパース (例: "Validation Results: ..., accuracy: 0.65")
            # 精度は最大化を目指すので、負の値を返す
            for line in reversed(result.stdout.strip().split('\n')):
                if "accuracy:" in line and "Validation Results" in line:
                    match = re.search(r"accuracy:\s*([0-9\.]+)", line)
                    if match:
                        metric_value = -float(match.group(1)) # 精度を最大化 = 負の精度を最小化
                        break
            if metric_value is None: metric_value = 0.0 # 精度が見つからなければ 0 (負なので最悪値)

        else: # "loss" (デフォルト)
            # 損失をパース (例: "Validation Results: total: 2.5, ...")
            # 損失は最小化を目指す
            for line in reversed(result.stdout.strip().split('\n')):
                if "total:" in line and "Validation Results" in line:
                    match = re.search(r"total:\s*([0-9\.]+)", line)
                    if match:
                        metric_value = float(match.group(1))
                        break
            if metric_value is None: metric_value = float('inf') # 損失が見つからなければ無限大

        logger.info(f"HSEO: Evaluation complete. Metric ({metric_to_optimize}): {metric_value}")
        return metric_value

    except subprocess.CalledProcessError as e:
        logger.error(f"HSEO: Parameter evaluation failed!")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Return Code: {e.returncode}")
        logger.error(f"Output:\n{e.stdout}")
        logger.error(f"Stderr:\n{e.stderr}")
        # 失敗した試行には非常に悪いスコアを返す
        return float('inf') # 最小化の場合
    except Exception as e:
        logger.error(f"HSEO: An unexpected error occurred during evaluation: {e}", exc_info=True)
        return float('inf')


# --- 2. HSEO (PSO) コアアルゴリズム ---

def optimize_with_hseo(
    objective_function: Callable[[np.ndarray], np.ndarray],
    dim: int,
    num_particles: int,
    max_iterations: int,
    exploration_range: List[Tuple[float, float]], # パラメータごとの [min, max] のリスト
    w: float = 0.5,  # 慣性係数
    c1: float = 1.5, # 個体最適解への引力
    c2: float = 1.5, # 全体最適解への引力
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    粒子群最適化 (PSO) を実行する。
    HSEO (Hybrid Swarm) の実装としてPSOを使用する。

    Args:
        objective_function (Callable): 目的関数。パーティクル集団(N, D)を受け取り、スコア配列(N,)を返す。
        dim (int): パラメータの次元数。
        num_particles (int): パーティクルの数。
        max_iterations (int): 最大イテレーション数。
        exploration_range (List[Tuple[float, float]]): 各次元の探索範囲 [min, max] のリスト。
        w (float): 慣性係数。
        c1 (float): 個体最適解への引力。
        c2 (float): 全体最適解への引力。
        seed (Optional[int]): 乱数シード。
        verbose (bool): ログ出力の有無。

    Returns:
        Tuple[np.ndarray, float]: (最適パラメータ, 最適スコア)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 探索範囲をNumPy配列に変換
    min_bounds = np.array([r[0] for r in exploration_range])
    max_bounds = np.array([r[1] for r in exploration_range])
    range_width = max_bounds - min_bounds

    # 1. 初期化
    # パーティクルの位置 (パラメータセット)
    particles_pos = min_bounds + np.random.rand(num_particles, dim) * range_width
    # パーティクルの速度
    particles_vel = np.random.randn(num_particles, dim) * 0.1
    
    # 各パーティクルのパーソナルベスト (pbest)
    pbest_pos = particles_pos.copy()
    pbest_scores = np.full(num_particles, float('inf'))
    
    # グローバルベスト (gbest)
    gbest_pos = np.zeros(dim)
    gbest_score = float('inf')

    if verbose:
        logger.info(f"HSEO (PSO): Starting optimization with {num_particles} particles for {max_iterations} iterations.")
        logger.info(f"HSEO (PSO): Dimension={dim}, Bounds={exploration_range}")

    # 2. イテレーション
    for i in range(max_iterations):
        # 目的関数で全パーティクルのスコアを評価
        scores = objective_function(particles_pos)
        
        # pbest の更新
        update_mask = scores < pbest_scores
        pbest_pos[update_mask] = particles_pos[update_mask]
        pbest_scores[update_mask] = scores[update_mask]
        
        # gbest の更新
        best_particle_idx = np.argmin(scores)
        if scores[best_particle_idx] < gbest_score:
            gbest_score = scores[best_particle_idx]
            gbest_pos = particles_pos[best_particle_idx].copy()
            
            if verbose:
                logger.info(f"HSEO (PSO) [Iter {i+1}/{max_iterations}]: New gbest score = {gbest_score:.6f}")
        
        # 3. 速度と位置の更新
        r1 = np.random.rand(num_particles, dim)
        r2 = np.random.rand(num_particles, dim)
        
        # PSO更新式
        cognitive_vel = c1 * r1 * (pbest_pos - particles_pos)
        social_vel = c2 * r2 * (gbest_pos - particles_pos)
        particles_vel = w * particles_vel + cognitive_vel + social_vel
        
        particles_pos = particles_pos + particles_vel
        
        # 探索範囲内にクリッピング
        particles_pos = np.clip(particles_pos, min_bounds, max_bounds)

    if verbose:
        logger.info(f"HSEO (PSO): Optimization finished. Final best score = {gbest_score:.6f}")
        logger.info(f"HSEO (PSO): Best parameters = {gbest_pos}")

    return gbest_pos, gbest_score