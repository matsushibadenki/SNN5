# ファイルパス: scripts/ann2snn.py
# (改修)
#
# Title: ANN-SNN 変換 実験スクリプト
#
# Description:
# doc/SNN開発：基本設計思想.md (セクション6.3) に基づき、
# 1. ANNモデル (SimpleCNN) をCIFAR-10データセットで訓練し、
# 2. 訓練済みANNモデルを SNNモデル (SpikingCNN) に変換し、
# 3. 変換後のSNNモデルの精度を評価する、
# という一連のワークフローを実行するスクリプト。
#
# 改善 (v2):
# - mypy --strict 準拠のための型ヒントを追加。
# - print文を logging に置き換え。
#
# 修正 (v3):
# - mypy [attr-defined] (ANNToSNNConverter -> AnnToSnnConverter) を修正。
# - mypy [assignment] (int = float) を修正。
#
# 修正 (v4):
# - mypy [call-arg] [attr-defined] エラーを修正。
# - AnnToSnnConverter の新しいAPI (v3) に合わせて呼び出し方法を修正。
#
# 修正 (v5):
# - AnnToSnnConverter の API (v3) への適合を再確認。
# - AnnToSnnConverter は snn_model と model_config を __init__ で要求する。
# - 変換実行は convert_cnn_weights メソッドで行う。

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms # type: ignore[import-untyped]
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import logging 
from typing import Dict, Any, cast, Tuple, List 
import os 

sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_research.benchmark.ann_baseline import SimpleCNN
# --- ▼ 修正: クラス名を AnnToSnnConverter に変更 ▼ ---
from snn_research.conversion.ann_to_snn_converter import AnnToSnnConverter 
# --- ▲ 修正 ▲ ---
from snn_research.core.snn_core import SpikingCNN # 変換後のSNNモデル
from snn_research.core.neurons import AdaptiveLIFNeuron
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]

# --- ロガー設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_ann(
    model: nn.Module, 
    device: torch.device, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    epoch: int
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' # type: ignore[arg-type]
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def evaluate_snn(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader, 
    time_steps: int
) -> float:
    model.eval()
    # --- ▼ 修正: [assignment] エラー (int = float) を回避 ▼ ---
    correct: float = 0.0
    # --- ▲ 修正 ▲ ---
    total: int = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # SNN評価 (SpikingCNNは内部でTステップループ)
            SJ_F.reset_net(model)
            # SpikingCNNは (logits, avg_spikes, mem) を返す
            outputs, _, _ = model(input_images=data) # type: ignore[operator]
            
            # 最終的なロジット (時間平均済み)
            final_logits: torch.Tensor = outputs
            
            pred = final_logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    return accuracy

def main() -> None:
    # --- 1. 準備 ---
    use_cuda: bool = torch.cuda.is_available()
    device: torch.device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # CIFAR-10 データローダー
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # --- 2. ANNモデルの構築と訓練 ---
    logger.info("--- 1. ANN Training ---")
    ann_model: SimpleCNN = SimpleCNN(num_classes=10).to(device)
    optimizer: optim.Optimizer = optim.Adam(ann_model.parameters(), lr=0.001)
    
    ann_epochs: int = 3 # 簡易的な訓練
    for epoch in range(1, ann_epochs + 1):
        train_ann(ann_model, device, train_loader, optimizer, epoch)
    
    ann_model.eval()
    logger.info("✅ ANN training complete.")

    # --- 3. ANN-SNN 変換 ---
    logger.info("--- 2. ANN-to-SNN Conversion ---")
    
    # 変換後のSNNモデル（SpikingCNN）の設定
    time_steps: int = 16
    neuron_config: Dict[str, Any] = {
        'type': 'lif',
        'tau_mem': 10.0,
        'base_threshold': 1.0 # 閾値は変換時に調整される
    }
    
    # 変換後のSNNモデルのインスタンス化 (重みはダミー)
    # vocab_size=10 (num_classes)
    snn_model_skel: SpikingCNN = SpikingCNN(vocab_size=10, time_steps=time_steps, neuron_config=neuron_config)
    
    # 変換器の初期化と実行
    # ANNモデルのReLUをSNNのLIFニューロンにマッピング
    # --- ▼ 修正: AnnToSnnConverter の新しいAPI (v3) に合わせて修正 ▼ ---
    snn_config_dict: Dict[str, Any] = {
        "time_steps": time_steps,
        "neuron": neuron_config
    }
    
    converter = AnnToSnnConverter(
        snn_model=snn_model_skel, 
        model_config=snn_config_dict
    )
    
    output_dir = "runs/ann2snn_tests"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "converted_snn_model.pth")
    
    # 変換を実行 (内部で正規化、閾値調整、重みコピーが行われる)
    converter.convert_cnn_weights(
        ann_model=ann_model,
        output_path=output_path,
        calibration_loader=train_loader # 閾値調整用にデータローダーを渡す
    )
    
    # 変換器が保持しているSNNモデル（重みコピー済み）を取得
    snn_model: nn.Module = converter.snn_model
    # --- ▲ 修正 ▲ ---
    
    snn_model = snn_model.to(device)
    logger.info("✅ ANN-SNN conversion complete.")

    # --- 4. SNNモデルの評価 ---
    logger.info("--- 3. SNN Evaluation ---")
    
    snn_accuracy = evaluate_snn(snn_model, device, test_loader, time_steps)
    
    logger.info(f"--- 📊 Results ---")
    logger.info(f"Converted SNN Accuracy (T={time_steps}): {snn_accuracy:.2f}%")
    logger.info("設計思想 (セクション6.3) のワークフローを検証しました。")

if __name__ == "__main__":
    main()
