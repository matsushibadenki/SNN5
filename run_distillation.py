# ファイルパス: run_distillation.py
# コードの最も最初には、ファイルパス、ファイルの内容を示したタイトル、機能の説明を詳細に記述してください。 修正内容は記載する必要はありません。
# Title: 知識蒸留実行スクリプト
# Description: KnowledgeDistillationManagerを使用して、知識蒸留プロセスを開始します。
#              設定ファイルとコマンドライン引数からパラメータを読み込みます。
#
# 修正 (v_async_fix):
# - KnowledgeDistillationManager.prepare_dataset が async def に変更されたことに伴い、
#   main() 内での呼び出し時に await を追加 (L183)。
#
# 修正 (v_hpo_fix_attr_error):
# - HPO実行時に KnowledgeDistillationManager (L162) に渡される config が
#   dict だったため、OmegaConf.create() で DictConfig に変換するよう修正。
#
# 修正 (v_hpo_fix_tensor_size_mismatch):
# - OOM (Trial 224, 226) の根本原因である、cifar10 (32x32) と
#   ViT (224x224) のミスマッチを解消するため、
#   タスクが 'cifar10' の場合、データローダーが参照する
#   `config.data.img_size` も 32 に上書きするロジックを追加 (L129)。
#
# 修正 (v_hpo_fix_type_error):
# - TypeError (unexpected keyword argument 'img_size') を解消するため、
#   img_size を task.prepare_data() ではなく、
#   TaskClass() のコンストラクタに渡すよう修正 (L226-L233)。


import argparse
import asyncio
import torch
import torchvision.models as models  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from typing import Any, List, Optional, cast, Dict
import sys 
import os

# プロジェクトルートをPythonパスに追加 (run_hpo.py と同じ修正)
project_root: str = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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


    container = TrainingContainer()
    
    # 2. 基本設定をロード
    container.config.from_yaml(args.config)

    # 3. モデル設定をロード (AttributeError 修正)
    #    cifar10_spikingcnn_config.yaml には 'model:' キーがないため、
    #    'model' ノード配下にマージする
    try:
        # --- ▼ 修正 (v_hpo_fix_3): インデントエラーとExceptブロックのロジックを修正 ▼ ---
        cfg_raw = OmegaConf.load(args.model_config)
        
        # ロードした config が 'model:' キーをトップレベルに持っているか確認
        if isinstance(cfg_raw, DictConfig) and 'model' in cfg_raw:
            # 既に 'model' キーがある場合 (spiking_transformer.yaml など)
            # .model ノードを直接マージする
            container.config.model.from_dict(
                cast(Dict[str, Any], OmegaConf.to_container(cfg_raw.model, resolve=True))
            )
        elif isinstance(cfg_raw, DictConfig):
            # 'model' キーがない場合 (cifar10_spikingcnn_config.yaml など)
            # 辞書全体を 'model' キーでラップしてマージする
            model_config_dict = OmegaConf.to_container(cfg_raw, resolve=True)
            if isinstance(model_config_dict, dict):
                container.config.from_dict({'model': model_config_dict})
            else:
                 # --- ▼ 修正: インデントを修正 (21 -> 20 spaces) ▼ ---
                 raise TypeError(f"Model config loaded from {args.model_config} is not a dictionary.")
        else:
             # --- ▼ 修正: インデントを修正 (17 -> 16 spaces) ▼ ---
             raise TypeError(f"Model config loaded from {args.model_config} is not a dictionary.")
            
    except Exception as e:
        print(f"Warning: Could not load or merge model config '{args.model_config}': {e}")
        # --- ▼ 修正: 失敗時のフォールバックを except ブロック内に復元 ▼ ---
        # 'model' が設定されていない可能性があるため、空の辞書をマージしておく
        container.config.from_dict({'model': {}})
        # --- ▲ 修正 ▲ ---


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

    # --- ▼ 修正 (v_hpo_fix_tensor_size_mismatch) ▼ ---
    # HPO (spiking_transformer.yaml) と cifar10 タスクのミスマッチを修正
    if args.task == 'cifar10':
        print("INFO: Overriding data/model config for CIFAR-10 (img_size=32, patch_size=4).")
        
        # 1. モデルコンフィグ (SNNCoreが読み取る) を上書き
        try:
            container.config.model.img_size.from_value(32)
            container.config.model.patch_size.from_value(4)
        except Exception as e:
            print(f"Warning: Could not override config.model: {e}")

        # 2. データコンフィグ (CIFAR10Taskが読み取る) を上書き
        try:
            if container.config.data.img_size.provided:
                container.config.data.img_size.from_value(32)
            else:
                # 存在しない場合は作成 (base_configにdata.img_sizeがない場合)
                container.config.data.from_dict({'img_size': 32})
                
            if container.config.data.patch_size.provided:
                container.config.data.patch_size.from_value(4)
            else:
                # 存在しない場合は作成
                container.config.data.from_dict({'patch_size': 4})
                
        except Exception as e:
            print(f"Warning: Could not override config.data: {e}")
    # --- ▲ 修正 (v_hpo_fix_tensor_size_mismatch) ▲ ---


    # DIコンテナから必要なコンポーネントを正しい順序で取得・構築
    device = container.device()
    
    # (cfg.model が正しく設定されたため、SNNCoreの初期化が成功するはず)
    # --- ▼ 修正(v7): [arg-type] vocab_size=10 を明示的に渡す ▼ ---
    # (タスクが "cifar10" であることが前提)
    student_model = container.snn_model(vocab_size=10).to(device)
    # --- ▲ 修正(v7) ▲ ---
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

    # --- ▼ 修正 (v_hpo_fix_attr_error): dict を DictConfig に変換 ▼ ---
    # Managerの初期化に必要なconfigを取得
    manager_config_dict: Dict[str, Any] = container.config() # これは dict を返す
    manager_config_omegaconf: DictConfig = OmegaConf.create(manager_config_dict) # dict -> DictConfig

    manager = KnowledgeDistillationManager(
        student_model=student_model,
        teacher_model=teacher_model,
        trainer=distillation_trainer,
        tokenizer_name=container.config.data.tokenizer_name(), # tokenizerはCIFARタスクでは使われないがインターフェースのため渡す
        model_registry=model_registry,
        device=device,
        config=manager_config_omegaconf # 修正: DictConfig オブジェクトを渡す
    )
    # --- ▲ 修正 (v_hpo_fix_attr_error) ▲ ---

    # --- データセットの準備 ---
    TaskClass = TASK_REGISTRY.get(args.task)
    if not TaskClass:
        raise ValueError(f"Task '{args.task}' not found.")
        
    # --- ▼ 修正 (v_hpo_fix_type_error): img_size を __init__ に渡す ▼ ---
    task_init_kwargs: Dict[str, Any] = {
        "tokenizer": container.tokenizer(),
        "device": device,
        "hardware_profile": {}
    }
    if args.task == 'cifar10':
        # CIFAR10Task が img_size を __init__ で受け取ることを期待
        task_init_kwargs['img_size'] = container.config.data.img_size()

    task = TaskClass(**task_init_kwargs)
    # --- ▲ 修正 (v_hpo_fix_type_error) ▲ ---
    
    
    # --- ▼ 修正(v_hpo_fix_type_error): kwargs を削除 ▼ ---
    # data_kwargs = {} # 削除
    train_dataset, val_dataset = task.prepare_data(data_dir="data")
    # --- ▲ 修正(v_hpo_fix_type_error) ▲ ---

    # 知識蒸留用にデータセットをラップ
    # --- ▼ 修正 (v_async_fix): await を追加 ▼ ---
    train_loader, val_loader = await manager.prepare_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=task.get_collate_fn(),
        batch_size=container.config.training.batch_size()
    )
    # --- ▲ 修正 (v_async_fix) ▲ ---

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