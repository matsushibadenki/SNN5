# ファイルパス: run_distillation.py
# Title: 知識蒸留実行スクリプト
# Description: KnowledgeDistillationManagerを使用して、知識蒸留プロセスを開始します。
#              設定ファイルとコマンドライン引数からパラメータを読み込みます。

import argparse
import asyncio
import torch
# ... (他のimportは省略) ...


async def main() -> None:
    # ... (argparse、configロード、override_config適用は省略) ...

    # DIコンテナから必要なコンポーネントを正しい順序で取得・構築
    device = container.device()
    
    # (cfg.model が正しく設定されたため、SNNCoreの初期化が成功するはず)
    # --- ▼ 修正(v7): [arg-type] vocab_size=10 を明示的に渡す ▼ ---
    # (タスクが "cifar10" であることが前提)
    student_model = container.snn_model(vocab_size=10).to(device)
    # --- ▲ 修正(v7) ▲ ---
    
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
