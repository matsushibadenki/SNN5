# ファイルパス: matsushibadenki/snn5/SNN5-dbc4f9d167f9df8d0c770008428a1d2832405ddf/run_distill_hpo.py
# Title: 知識蒸留実行スクリプト (HPO専用)
# Description: KnowledgeDistillationManagerを使用して、知識蒸留プロセスを開始します。
#              【最終版】SNN起動に必要な構造的修正を前提とし、外部からのパラメータ強制を全て削除しました。

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

# --- ▼▼▼ 【最優先追加】現在の実行パスをログに出力 (環境不整合の確認用) ▼▼▼ ---
print(f"🚨 DEBUG: Currently executing script from: {os.path.abspath(__file__)}")
# --- ▲▲▲ 【最優先追加】 ▲▲▲ ---

from app.containers import TrainingContainer
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from snn_research.benchmark import TASK_REGISTRY


async def main() -> None:
    parser = argparse.ArgumentParser(description="SNN Knowledge Distillation Runner")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Base config file path")
    parser.add_argument("--model_config", type=str, default="configs/models/spiking_transformer.yaml", help="SNN model architecture config file path")
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

    # 3. モデル設定をロード (修正ロジックは変更なし)
    try:
        cfg_raw = OmegaConf.load(args.model_config)
        
        if isinstance(cfg_raw, DictConfig) and 'model' in cfg_raw:
            container.config.model.from_dict(
                cast(Dict[str, Any], OmegaConf.to_container(cfg_raw.model, resolve=True))
            )
        elif isinstance(cfg_raw, DictConfig):
            model_config_dict = OmegaConf.to_container(cfg_raw, resolve=True)
            if isinstance(model_config_dict, dict):
                container.config.from_dict({'model': model_config_dict})
            else:
                 raise TypeError(f"Model config loaded from {args.model_config} is not a dictionary.")
        else:
             raise TypeError(f"Model config loaded from {args.model_config} is not a dictionary.")
            
    except Exception as e:
        print(f"Warning: Could not load or merge model config '{args.model_config}': {e}")
        container.config.from_dict({'model': {}})


    # 4. コマンドライン引数からエポック数を上書き
    container.config.training.epochs.from_value(args.epochs)
    
    # 5. HPOからの --override_config を適用
    if args.override_config:
        print(f"Applying {len(args.override_config)} overrides from command line...")
        for override in args.override_config:
            try:
                keys, value_str = override.split('=', 1)
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
                            value = value_str

                key_parts = keys.split('.')
                config_provider = container.config
                for part in key_parts:
                    config_provider = getattr(config_provider, part)
                
                config_provider.from_value(value)
                print(f"  - Applied: {keys} = {value}")
            except Exception as e:
                print(f"Error applying override '{override}': {e}")
    
    
    # --- ▼▼▼ 【デバッグ強制オーバーライドの削除】 HPOに任せる ▼▼▼ ---
    # 以前のデバッグロジックは全て削除
    
    # --- ▲▲▲ 【デバッグ強制オーバーライドの削除】 ▲▲▲ ---
        

    # --- ▼ 修正 (v_hpo_fix_tensor_size_mismatch) ▼ ---
    if args.task == 'cifar10':
        print("INFO: Overriding data/model config for CIFAR-10 (img_size=32, patch_size=4).")
        
        try:
            container.config.model.img_size.from_value(32)
            container.config.model.patch_size.from_value(4)
        except Exception as e:
            print(f"Warning: Could not override config.model: {e}")

        try:
            if container.config.data.img_size.provided:
                container.config.data.img_size.from_value(32)
            else:
                container.config.data.from_dict({'img_size': 32})
                
            if container.config.data.patch_size.provided:
                container.config.data.patch_size.from_value(4)
            else:
                container.config.data.from_dict({'patch_size': 4})
                
        except Exception as e:
            print(f"Warning: Could not override config.data: {e}")


    # DIコンテナから必要なコンポーネントを正しい順序で取得・構築
    device = container.device()

    student_model = container.snn_model(vocab_size=10).to(device)
    
    # --- ▼▼▼ 【最小限の起動保証】 Xavier初期化のみ残す ▼▼▼ ---
    def aggressive_init(m: torch.nn.Module):
        """すべてのConv/Linear層にXavier初期化を適用し、バイアスは0に設定。"""
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # 最終バイアス注入デバッグを削除し、標準の0初期化に戻す
                torch.nn.init.constant_(m.bias, 0)
    
    print("🔥 Forcing aggressive Xavier weight initialization to ensure initial spike activity.")
    student_model.apply(aggressive_init)
    
    # V_INIT強制設定の削除
    # --- ▲▲▲ 【最小限の起動保証】 ▲▲▲ ---
    
    optimizer = container.optimizer(params=student_model.parameters())
    scheduler = container.scheduler(optimizer=optimizer) if container.config.training.gradient_based.use_scheduler() else None

    # --- 教師モデルの構築 ---
    print(f"🧠 Initializing ANN teacher model ({args.teacher_model})...")
    if args.teacher_model == "resnet18":
        teacher_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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
    manager_config_dict: Dict[str, Any] = container.config()
    manager_config_omegaconf: DictConfig = OmegaConf.create(manager_config_dict)

    manager = KnowledgeDistillationManager(
        student_model=student_model,
        teacher_model=teacher_model,
        trainer=distillation_trainer,
        tokenizer_name=container.config.data.tokenizer_name(),
        model_registry=model_registry,
        device=device,
        config=manager_config_omegaconf
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
    # --- ▲ 修正 (v_async_fix) ▲ ---
    
    # --- ▼▼▼ 環境整合性チェック: HPOの正規のパラメータをログに表示 ▼▼▼ ---
    print("\n=============================================")
    print("✅ FINAL HPO PARAMETER CHECK (CLEAN STATE) ✅")
    
    # HPOが選択した/YAMLで定義された値を表示
    print(f"  V_THRESHOLD (HPO/YAML): {container.config.model.neuron.v_threshold()}")
    print(f"  LR (HPO/YAML): {container.config.training.gradient_based.learning_rate()}")
    print(f"  SPIKE_REG_W (HPO/YAML): {container.config.training.gradient_based.distillation.loss.spike_reg_weight()}")
    print(f"  V_RESET (HPO/YAML): {container.config.model.neuron.v_reset()}")
    print(f"  V_DECAY (HPO/YAML): {container.config.model.neuron.v_decay()}")
    print(f"  BIAS (HPO/YAML): {container.config.model.neuron.bias()}")
    print("=============================================\n")
    # --- ▲▲▲ 環境整合性チェック ▲▲▲ ---

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
