# ファイルパス: run_distillation.py
# Title: 知識蒸留実行スクリプト
# Description: KnowledgeDistillationManagerを使用して、知識蒸留プロセスを開始します。
#              設定ファイルとコマンドライン引数からパラメータを読み込みます。

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

    # --- ▼▼▼ NameError 修正: containerの初期化を再配置 ▼▼▼ ---
    container = TrainingContainer()
    # --- ▲▲▲ NameError 修正 ▲▲▲ ---
    
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
                    #... (型推論ロジックは省略) ...
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
    
    # 6. 【致命的なバグ修正】 spike_rate=0 を解消するため、spike_reg_weight を強制的に低い値に固定
    #    (Optunaが探索する高すぎる値 (e.g., 2.839) をデバッグレベルでオーバーライド)
    try:
        config_provider = container.config.training.gradient_based.distillation.loss.spike_reg_weight
        DEBUG_SPIKE_REG_VALUE = 1e-6 # 以前の修正を維持
        config_provider.from_value(DEBUG_SPIKE_REG_VALUE)
        print(f"  - 【DEBUG OVERRIDE】 Forced spike_reg_weight to: {DEBUG_SPIKE_REG_VALUE}")
    except Exception as e:
        print(f"Warning: Could not force spike_reg_weight. This may cause spike_rate=0: {e}")
        
    # 7. 【最終手段】 learning_rate を強制的に高く設定 <<< ここを修正/追加
    try:
        config_provider_lr = container.config.training.gradient_based.learning_rate
        DEBUG_LR_VALUE = 1e-3 # HPOの探索範囲（例: 6.5e-5）よりも高い値を強制
        config_provider_lr.from_value(DEBUG_LR_VALUE)
        print(f"  - 【DEBUG OVERRIDE】 Forced learning_rate to: {DEBUG_LR_VALUE}")
    except Exception as e:
        print(f"Warning: Could not force learning_rate: {e}")
        
        

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

    student_model = container.snn_model(vocab_size=10).to(device)
    
    # --- ▼▼▼ 【最優先修正】重み初期化の強制 (spike_rate=0の最終防衛線) ▼▼▼ ---
    def aggressive_init(m: torch.nn.Module):
        """すべてのConv/Linear層にXavier初期化を適用し、確実に電流を流す。"""
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            # Glorot (Xavier) Uniform initializationを適用
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    print("🔥 Forcing aggressive Xavier weight initialization to ensure initial spike activity.")
    student_model.apply(aggressive_init)
    # --- ▲▲▲ 【最優先修正】重み初期化の強制 (spike_rate=0の最終防衛線) ▼▼▼ ---
    
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

    print("\n=============================================")
    print("🚨 FINAL DEBUG CHECK BEFORE STARTING TRAINING 🚨")
    print(f"  V_THRESHOLD (from YAML): {container.config.model.neuron.v_threshold()}")
    print(f"  LR (Forced): {container.config.training.gradient_based.learning_rate()}")
    print(f"  SPIKE_REG_W (Forced): {container.config.training.gradient_based.distillation.loss.spike_reg_weight()}")
    print("=============================================\n")

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
    
    # --- ▼▼▼ 環境整合性チェック: 最終オーバーライド値の確認 ▼▼▼ ---
    print("\n=============================================")
    print("🚨 FINAL DEBUG CHECK BEFORE STARTING TRAINING 🚨")
    print(f"  V_THRESHOLD (from YAML): {container.config.model.neuron.v_threshold()}")
    print(f"  LR (Forced): {container.config.training.gradient_based.learning_rate()}")
    print(f"  SPIKE_REG_W (Forced): {container.config.training.gradient_based.distillation.loss.spike_reg_weight()}")
    print("=============================================\n")
    # --- ▲▲▲ 環境整合性チェック ▲▲▲ ---
