# ファイルパス: run_distillation.py
# コードの最も最初には、ファイルパス、ファイルの内容を示したタイトル、機能の説明を詳細に記述してください。 修正内容は記載する必要はありません。
# Title: 知識蒸留実行スクリプト
# Description: KnowledgeDistillationManagerを使用して、知識蒸留プロセスを開始します。
#              設定ファイルとコマンドライン引数からパラメータを読み込みます。
#
# 修正 (v6): HPO連携時の設定読み込みと適用のロジックを dependency-injector に
#             合わせて修正。OmegaConf.update の誤用を修正。

import argparse
import asyncio
import torch
import torchvision.models as models  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from typing import Any, List, Optional
import sys # sysをインポート

# プロジェクトルートをPythonパスに追加 (app.containersをインポートするため)
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import TrainingContainer
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from snn_research.benchmark import TASK_REGISTRY


async def main() -> None:
    parser = argparse.ArgumentParser(description="SNN Knowledge Distillation Runner")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Base config file path")
    parser.add_argument("--model_config", type=str, default="configs/cifar10_spikingcnn_config.yaml", help="SNN model architecture config file path")
    parser.add_argument("--task", type=str, default="cifar10", help="The benchmark task to distill.")
    parser.add_argument("--teacher_model", type=str, default="resnet18", help="The torchvision teacher model to use.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of distillation epochs.")
    parser.add_argument(
        "--override_config",
        type=str,
        action='append',
        help="Override config (e.g., 'training.epochs=5')"
    )
    args = parser.parse_args()

    # --- ▼ 修正: HPO連携のため、設定のロード順序を修正 ▼ ---

    # 1. DIコンテナのインスタンス化 (空の設定で)
    container = TrainingContainer()
    
    # 2. 基本設定をロード
    container.config.from_yaml(args.config)

    # 3. モデル設定をロード (AttributeError 修正)
    #    cifar10_spikingcnn_config.yaml には 'model:' キーがないため、
    #    'model' ノード配下にマージする
    try:
        # モデル設定を辞書としてロード
        model_cfg_dict = OmegaConf.to_container(OmegaConf.load(args.model_config), resolve=True)
        if isinstance(model_cfg_dict, dict):
            # 'model' キーでラップしてDIコンテナの設定にマージ
            container.config.from_dict({'model': model_cfg_dict})
        else:
            raise TypeError(f"Model config loaded from {args.model_config} is not a dictionary.")
    except Exception as e:
        print(f"Warning: Could not load or merge model config '{args.model_config}': {e}")
        # 'model' が設定されていない可能性があるため、空の辞書をマージしておく
        container.config.from_dict({'model': {}})


    # 4. コマンドライン引数からエポック数を上書き
    #    (override_config よりも先に適用)
    container.config.training.epochs.from_value(args.epochs)
    
    # 5. HPOからの --override_config を適用
    if args.override_config:
        print(f"Applying {len(args.override_config)} overrides from command line...")
        for override in args.override_config:
            try:
                keys, value_str = override.split('=', 1)
                # 型を推論
                value: Any
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        if value_str.lower() == 'true':
                            value = True
                        elif value_str.lower() == 'false':
                            value = False
                        else:
                            value = value_str  # 文字列として保持

                # 修正: dependency-injector の provider API を使って上書き
                key_parts = keys.split('.')
                config_provider = container.config
                for part in key_parts:
                    # providerオブジェクトを辿る
                    config_provider = getattr(config_provider, part)
                
                # 最終的な provider に .from_value() で値を設定
                config_provider.from_value(value)
                print(f"  - Applied: {keys} = {value}")
            except Exception as e:
                print(f"Error applying override '{override}': {e}")
    # --- ▲ 修正 ▲ ---

    # DIコンテナから必要なコンポーネントを正しい順序で取得・構築
    device = container.device()
    
    # (cfg.model が正しく設定されたため、SNNCoreの初期化が成功するはず)
    student_model = container.snn_model(vocab_size=10).to(device)
    optimizer = container.optimizer(params=student_model.parameters())
    scheduler = container.scheduler(optimizer=optimizer) if container.config.training.gradient_based.use_scheduler() else None

    # --- 教師モデルの構築 ---
    print(f"🧠 Initializing ANN teacher model ({args.teacher_model})...")
    if args.teacher_model == "resnet18":
        teacher_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # CIFAR-10用に最終層を変更
        num_ftrs = teacher_model.fc.in_features
        teacher_model.fc = torch.nn.Linear(num_ftrs, 10)
    else:
        raise ValueError(f"Unsupported teacher model: {args.teacher_model}")
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    distillation_trainer = container.distillation_trainer(
        model=student_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=-1
    )
    model_registry = container.model_registry()

    manager = KnowledgeDistillationManager(
        student_model=student_model,
        teacher_model=teacher_model,
        trainer=distillation_trainer,
        tokenizer_name=container.config.data.tokenizer_name(), # tokenizerはCIFARタスクでは使われないがインターフェースのため渡す
        model_registry=model_registry,
        device=device,
        config=container.config() # 修正: config を渡す
    )

    # --- データセットの準備 ---
    TaskClass = TASK_REGISTRY.get(args.task)
    if not TaskClass:
        raise ValueError(f"Task '{args.task}' not found.")
    task = TaskClass(tokenizer=container.tokenizer.provided, device=device, hardware_profile={})
    train_dataset, val_dataset = task.prepare_data()

    # 知識蒸留用にデータセットをラップ
    train_loader, val_loader = manager.prepare_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=task.get_collate_fn(),
        batch_size=container.config.training.batch_size()
    )

    # 蒸留の実行
    await manager.run_distillation(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=container.config.training.epochs(), # 設定ファイルからエポック数を取得
        model_id=f"{args.task}_distilled_from_{args.teacher_model}",
        task_description=f"An expert SNN for {args.task}, distilled from {args.teacher_model}.",
        # 修正: model.to_dict() ではなく、コンテナから辞書を取得
        student_config=container.config.model.to_dict()
    )

if __name__ == "__main__":
    asyncio.run(main())
