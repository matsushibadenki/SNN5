# ファイルパス: scripts/run_ecg_analysis.py
# (新規作成)
# Title: ECG異常検出 デモアプリケーション
# Description:
# Improvement-Plan.md に記載された実用アプリケーション例の一つである
# ECG異常検出のデモンストレーションを行うCLIスクリプト。
# ダミーのECGデータを生成し、SNNモデル（SpikingCNNまたはTemporalFeatureExtractor）
# をロードして異常/正常の分類を実行し、結果を表示する。

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np
import logging
import random
from omegaconf import OmegaConf
# --- ▼ 修正: Optional をインポート ▼ ---
from typing import Optional
# --- ▲ 修正 ▲ ---

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.utils import get_auto_device
from snn_research.core.snn_core import SNNCore # SNNCoreをインポート
# ダミーモデル（訓練済みモデルがない場合のフォールバック用）
from tests.test_integration_real_world import MockSNNCore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_dummy_ecg_data(batch_size: int, time_steps: int, device: str) -> torch.Tensor:
    """
    ダミーのECG時系列データを生成する。
    Args:
        batch_size (int): 生成するサンプル数。
        time_steps (int): 各サンプルの時間ステップ数。
        device (str): データ生成先のデバイス ('cpu' or 'cuda')。
    Returns:
        torch.Tensor: 生成されたECGデータ (Batch, TimeSteps, Features=1)。
    """
    logger.info(f"   -> Generating {batch_size} dummy ECG samples ({time_steps} steps)...")
    # 基礎となるノイズ
    data = torch.randn(batch_size, time_steps, 1, device=device) * 0.1
    # 周期的な心拍様パターンを追加
    time_vector = torch.arange(time_steps, device=device).float() / time_steps * (2 * torch.pi * (time_steps / 1000.0)) # 1秒周期を仮定
    heartbeat_pattern = torch.sin(time_vector).unsqueeze(0).unsqueeze(-1) * 0.5
    data += heartbeat_pattern
    # 高周波ノイズも少し追加
    data += torch.randn(batch_size, time_steps, 1, device=device) * 0.05
    # 時折、異常を示す大きなスパイクをランダムに追加 (例: 10%のサンプルに)
    for i in range(batch_size):
        if random.random() < 0.2: # 20%の確率で異常スパイク
            spike_time = random.randint(time_steps // 4, 3 * time_steps // 4)
            spike_height = (random.random() - 0.5) * 2.0 # -1.0 ~ 1.0
            data[i, spike_time:spike_time+5, 0] += spike_height
            logger.info(f"     (Injecting anomaly spike into sample {i})")

    logger.info(f"   -> Generated ECG data shape: {data.shape}")
    # SpikingCNNは (B, C, H, W) を期待するため、次元を追加・変換
    # ここでは TemporalFeatureExtractor を想定し (B, T, F) のままにする
    # SpikingCNNを使う場合はモデル入力前に要変換
    return data

def load_snn_model(model_config_path: str, model_path: Optional[str], device: str, num_classes: int) -> nn.Module:
    """
    設定ファイルとオプションの重みファイルからSNNモデルをロードする。
    """
    if not Path(model_config_path).exists():
        logger.error(f"モデル設定ファイルが見つかりません: {model_config_path}")
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    cfg = OmegaConf.load(model_config_path)
    # vocab_sizeをクラス数として渡す
    model_container = SNNCore(config=cfg.model, vocab_size=num_classes)
    model = model_container.model # SNNCore内部の実際のモデルを取得

    if model_path and Path(model_path).exists():
        logger.info(f"Loading trained model weights from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # 'module.' プレフィックスがあれば削除 (DDPで保存された場合)
            if list(state_dict.keys())[0].startswith('module.'):
                 state_dict = {k[7:]: v for k, v in state_dict.items()}
            # SNNCore内部のモデルにロード
            model.load_state_dict(state_dict, strict=False)
            logger.info("✅ Trained weights loaded successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load trained weights: {e}. Using initialized model.")
    else:
        logger.warning("⚠️ No trained model path provided or file not found. Using initialized model.")
        # フォールバックとしてダミーモデルを使用する例（必要なら）
        # logger.warning("Using MockSNNCore as fallback.")
        # model = MockSNNCore(num_classes=num_classes, features=64) # 特徴量数は合わせる必要あり

    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="SNN ECG Anomaly Detection Demo")
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/models/temporal_snn.yaml", # Temporal SNNをデフォルトに
        help="Path to the SNN model architecture configuration file (e.g., temporal_snn.yaml or cifar10_spikingcnn.yaml)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained SNN model weights (.pth). If not provided, uses an initialized model."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of ECG samples to generate and classify."
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=500,
        help="Number of time steps for each ECG sample."
    )
    args = parser.parse_args()

    device = get_auto_device()
    logger.info(f"Using device: {device}")

    num_classes = 2 # 異常(1) / 正常(0)

    # 1. モデルのロード
    try:
        model = load_snn_model(args.model_config, args.model_path, device, num_classes)
        # モデルのアーキテクチャタイプを取得 (設定ファイルから)
        cfg = OmegaConf.load(args.model_config)
        architecture_type = cfg.model.get("architecture_type", "unknown")
        logger.info(f"Successfully loaded SNN model (Type: {architecture_type}).")
    except Exception as e:
        logger.error(f"モデルのロード中にエラーが発生しました: {e}")
        sys.exit(1)

    # 2. ダミーECGデータの生成
    ecg_data = generate_dummy_ecg_data(args.num_samples, args.time_steps, device)

    # 3. 推論の実行
    logger.info("\n--- Starting ECG Classification ---")
    results = []
    with torch.no_grad():
        # モデルタイプに応じて入力形式を調整
        if architecture_type == "spiking_cnn":
            # SpikingCNNは (B, C, H, W) を期待する。ECGデータを画像のように扱う
            # (B, T, 1) -> (B, 1, T, 1) or (B, 1, 1, T) などに変換
            # ここでは簡易的に時間軸を高さとみなす (B, 1, T, 1)
            model_input = ecg_data.permute(0, 2, 1).unsqueeze(-1) # (B, 1, T, 1)
            logger.info(f"   Input reshaped for SpikingCNN: {model_input.shape}")
        elif architecture_type == "temporal_snn":
             # TemporalFeatureExtractor (RSNN) は (B, T, F) を期待
             model_input = ecg_data # (B, T, 1)
             logger.info(f"   Input shape for Temporal SNN: {model_input.shape}")
        else:
             # 他のモデルタイプの場合、適切な変換が必要
             logger.warning(f"   Input shape adjustment for model type '{architecture_type}' is not explicitly defined. Using raw shape {ecg_data.shape}.")
             model_input = ecg_data # とりあえずそのまま渡す


        # モデルのforwardを呼び出す
        # SNNCoreを使っている場合、内部のモデルが呼ばれる
        # 多くのモデルは (logits, avg_spikes, mem) を返す
        try:
            outputs = model(model_input, return_spikes=True) # kwargs経由ではなく直接渡す
            logits = outputs[0]
            avg_spikes = outputs[1].item() if outputs[1] is not None else 0.0
        except Exception as e:
             logger.error(f"推論中にエラーが発生しました: {e}")
             sys.exit(1)


        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)

        for i in range(args.num_samples):
            pred_class = predictions[i].item()
            pred_label = "Anomaly" if pred_class == 1 else "Normal"
            confidence = probabilities[i, pred_class].item()
            results.append({
                "Sample": i + 1,
                "Prediction": pred_label,
                "Confidence": f"{confidence:.2%}",
                # "Avg Spikes": f"{avg_spikes:.0f}" # バッチ全体の平均スパイク数
            })

    # 4. 結果の表示
    print("\n--- Classification Results ---")
    if results:
        # Display results in a table-like format
        headers = results[0].keys()
        print(f"{' | '.join(headers)}")
        print("-" * (len(' | '.join(headers)) + 2))
        for row in results:
            print(f"{' | '.join(map(str, row.values()))}")
        print(f"\nAverage Spikes per Sample (Batch): {avg_spikes:.0f}") # バッチ全体の平均を表示
    else:
        print("No results generated.")

    logger.info("\n✅ ECG analysis demo finished.")

if __name__ == "__main__":
    main()