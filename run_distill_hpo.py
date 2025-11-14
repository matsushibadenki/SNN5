# ファイルパス: run_distill_hpo.py
# Title: 知識蒸留実行スクリプト (HPO専用)
# Description: KnowledgeDistillationManagerを使用して、知識蒸留プロセスを開始します。
#              【デバッグ強制設定復活版】spike_rate=0 の問題を回避するため、全ての積極的な設定を強制的に適用します。
#
# 【!!! エラー修正 (HSEO module not found) !!!】
# (L17-20) sys.path の設定を、app.containers (L25) などの
#          プロジェクト内インポートよりも *前* に移動する。
#
# 【!!! エラー修正 (tokenizer_name is None) !!!】
# (L61-69) --task 引数に基づき、data config (例: configs/data/cifar10.yaml) を
#          ロードするロジックを追加。
# 修正 (v12 - HPO正常化):
# - HPO (Trial 857等) のログ分析に基づき、ネットワークが「沈黙」
#   しているのではなく、「スパイクで爆発 (損失 4.04)」していると断定。
# - 原因は、BIAS=2.0, V_init=0.499, aggressive_init などの
#   強制デバッグ設定が強すぎることにある。
# - HPOを正しく機能させるため、これらの強制設定をすべてコメントアウトする。

import argparse
import asyncio
import torch
import torchvision.models as models  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from typing import Any, List, Optional, cast, Dict
import sys 
import os

# --- ▼▼▼ 【!!! 修正 (HSEO module not found) !!!】 ▼▼▼
# sys.path の修正を、app.containers のインポートより *前* に移動する
project_root: str = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- ▲▲▲ 【!!! 修正 !!!】 ▲▲▲


# プロジェクトルートをPythonパスに追加 (run_hpo.py と同じ修正)
# project_root: str = os.path.abspath(os.path.dirname(__file__)) # 削除 (上に移動)
# if project_root not in sys.path: # 削除
#     sys.path.insert(0, project_root) # 削除

# --- ▼▼▼ 【最優先追加】現在の実行パスをログに出力 (環境不整合の確認用) ▼▼▼ ---
print(f"🚨 DEBUG: Currently executing script from: {os.path.abspath(__file__)}")
# --- ▲▲▲ 【最優先追加】 ▲▲▲ ---

from app.containers import TrainingContainer
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from snn_research.benchmark import TASK_REGISTRY


async def main() -> None:
    # デバッグログ用の変数 (try/exceptブロックの外側で定義)
    # NOTE: これらの変数は aggressive_init のクロージャで利用されます。
    DEBUG_LR_VALUE: float = 0.0
    DEBUG_SPIKE_REG_VALUE: float = 0.0
    DEBUG_V_THRESHOLD_VALUE: float = 0.0
    DEBUG_V_RESET_VALUE: float = 0.0
    DEBUG_V_DECAY_VALUE: float = 0.0
    DEBUG_BIAS_VALUE: float = 0.0 # モデルバイアス注入用
    DEBUG_V_INIT_VALUE_FORCED: float = 0.0 # 初期電位注入用

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

    # --- ▼▼▼ 【!!! エラー修正 (tokenizer_name is None) v2 !!!】 ▼▼▼
    # 2.5. データ設定をロード
    # --task "cifar10" に基づき、"configs/data/cifar10.yaml" をロードする
    
    # project_root (L.17で定義済み) を基準に絶対パスを構築する
    data_config_path = os.path.join(project_root, f"configs/data/{args.task}.yaml")

    if os.path.exists(data_config_path):
        print(f"INFO: Loading data config: {data_config_path}")
        container.config.from_yaml(data_config_path)
    else:
        # どのパスを探しに行ったか明確にするため、絶対パスをログに出力
        print(f"WARNING: Data config file not found at: {data_config_path}. 'data' config might be incomplete.")
    # --- ▲▲▲ 【!!! エラー修正 v2 !!!】 ▲▲▲

    # 3. モデル設定をロード 
    try:
        # --- ▼ 修正 (v_hpo_fix_3): ロードロジックの修正 ▼ ---
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
    
    
    # --- ▼▼▼ 【デバッグ強制オーバーライドの復活と再導入】 ▼▼▼ ---
    # 【!!! HPO修正 (v12): HPOの邪魔になるため、すべてコメントアウト !!!】

    # 6. 【デバッグ復活】 spike_reg_weight を強制的に低い値に固定
    # try:
    #     config_provider = container.config.training.gradient_based.distillation.loss.spike_reg_weight
    #     DEBUG_SPIKE_REG_VALUE = 1e-6 
    #     config_provider.from_value(DEBUG_SPIKE_REG_VALUE)
    #     print(f"  - 【DEBUG OVERRIDE】 Forced spike_reg_weight to: {DEBUG_SPIKE_REG_VALUE}")
    # except Exception as e:
    #     print(f"Warning: Could not force spike_reg_weight. This may cause spike_rate=0: {e}")
        
    # 7. 【デバッグ復活】 learning_rate を強制的に高く設定
    # try:
    #     config_provider_lr = container.config.training.gradient_based.learning_rate
    #     DEBUG_LR_VALUE = 1e-2 # 以前の修正を復活 (1e-2)
    #     config_provider_lr.from_value(DEBUG_LR_VALUE)
    #     print(f"  - 【DEBUG OVERRIDE】 Forced learning_rate to: {DEBUG_LR_VALUE}")
    # except Exception as e:
    #     print(f"Warning: Could not force learning_rate: {e}")

    # 8. 【デバッグ復活】 V_THRESHOLD を強制的に設定 (※これはHPO対象外なので残してもOK)
    #    (ただし、HPOが v_threshold を探索する場合は、ここもコメントアウトが必要)
    try:
        config_provider_v_th = container.config.model.neuron.v_threshold
        DEBUG_V_THRESHOLD_VALUE = 0.5 
        if config_provider_v_th() < 1e-5:
            config_provider_v_th.from_value(DEBUG_V_THRESHOLD_VALUE)
            print(f"  - 【DEBUG OVERRIDE】 Forced V_THRESHOLD to: {DEBUG_V_THRESHOLD_VALUE}")
        else:
             DEBUG_V_THRESHOLD_VALUE = config_provider_v_th()
    except Exception as e:
        print(f"Warning: Could not force V_THRESHOLD: {e}")
    
    # 9. 【デバッグ復活】 v_reset を強制的に 0.0 に設定 (ゼロリセット固定)
    # try:
    #     config_provider_v_reset = container.config.model.neuron.v_reset
    #     DEBUG_V_RESET_VALUE = 0.0 
    #     config_provider_v_reset.from_value(DEBUG_V_RESET_VALUE)
    #     print(f"  - 【DEBUG OVERRIDE】 Forced v_reset to: {DEBUG_V_RESET_VALUE}")
    # except Exception as e:
    #     print(f"Warning: Could not force v_reset: {e}")

    # 10. 【デバッグ復活】 v_decay を強制的に 0.999 に設定
    # try:
    #     config_provider_v_decay = container.config.model.neuron.v_decay
    #     DEBUG_V_DECAY_VALUE = 0.999 
    #     config_provider_v_decay.from_value(DEBUG_V_DECAY_VALUE)
    #     print(f"  - 【DEBUG OVERRIDE】 Forced v_decay to: {DEBUG_V_DECAY_VALUE}")
    # except Exception as e:
    #     print(f"Warning: Could not force v_decay: {e}")

    # 11. 【デバッグ復活】 bias を強制的に 2.0 に設定 (ニューロン層バイアス)
    # try:
    #     config_provider_bias = container.config.model.neuron.bias
    #     DEBUG_BIAS_VALUE = 2.0  
    #     config_provider_bias.from_value(DEBUG_BIAS_VALUE)
    #     print(f"  - 【DEBUG OVERRIDE】 Forced neuron bias to: {DEBUG_BIAS_VALUE}")
    # except Exception as e:
    #     print(f"Warning: Could not force neuron bias: {e}")
    # --- ▲▲▲ 【デバッグ強制オーバーライドの復活と再導入】 ▲▲▲ ---
        

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

    # ssn_core.py 側で vocab_size を処理するように修正したため、ここは変更不要
    student_model = container.snn_model(vocab_size=10).to(device)
    
    # --- ▼▼▼ 【!!! HPO修正 (v12): 強制バイアス注入とV_INIT強制を *無効化* !!!】 ▼▼▼ ---
    
    # V_INITの強制設定 (無効化)
    # DEBUG_V_INIT_VALUE_FORCED = 0.499 # 初期電位のデバッグ値を復活
    
    def aggressive_init(m: torch.nn.Module):
        """ (v12: この関数は呼び出されなくなるが、念のため内部も無効化) """
        # NOTE: DEBUG_BIAS_VALUE はクロージャで参照されます。
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            # Glorot (Xavier) Uniform initializationを適用
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # 【修正: バイアス項に強制的に大きな値を注入】
                # torch.nn.init.constant_(m.bias, DEBUG_BIAS_VALUE) # 2.0を直接注入 (無効化)
                torch.nn.init.constant_(m.bias, 0.0) # 通常のゼロ初期化に戻す
                # print(f"  - INJECTED BIAS: {DEBUG_BIAS_VALUE} for {m.__class__.__name__}")
                    
    print("INFO: Using standard weight initialization (Forced bias injection is DISABLED for HPO).")
    # student_model.apply(aggressive_init) # (無効化)
    
    # --- V_INITの強制設定 (無効化) ---
    # try:
    #     print(f"🧠 DEBUG: Setting initial membrane potential (V_init) to: {DEBUG_V_INIT_VALUE_FORCED} (V_TH=0.5)")
    #     for name, module in student_model.named_modules():
    #         if hasattr(module, 'v_init'):
    #              # type: ignore[attr-defined]
    #             module.v_init = DEBUG_V_INIT_VALUE_FORCED # type: ignore[attr-defined] 
    # except Exception as e:
    #     print(f"Warning: Could not set V_init on all neurons: {e}")
    
    # --- ▲▲▲ 【!!! HPO修正 (v12) !!!】 ▲▲▲ ---
    
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
    
    # --- ▼▼▼ 環境整合性チェック: コア修正の確認とデバッグ値の表示 ▼▼▼ ---
    CORE_TAU_MEM_VALUE = "NOT FOUND"
    try:
        for name, module in student_model.named_modules():
            if 'BioLIFNeuron' in module.__class__.__name__ and hasattr(module, 'tau_mem'):
                CORE_TAU_MEM_VALUE = str(getattr(module, 'tau_mem'))
                break 
    except Exception as e:
        CORE_TAU_MEM_VALUE = f"Error: {e}"
        
    # --- ▼▼▼ 【!!! HPO修正 (v12): デバッグログの表示を修正 !!!】 ▼▼▼ ---
    # ...
    print("\n=============================================")
    print("🚨 FINAL DEBUG CHECK (RE-FORCED PARAMETERS) 🚨")
    print(f"  V_THRESHOLD (HPO/YAML): {container.config.model.neuron.v_threshold()}")
    print(f"  LR (HPO/YAML): {container.config.training.gradient_based.learning_rate()}")
    print(f"  SPIKE_REG_W (HPO/YAML): {container.config.training.gradient_based.distillation.loss.spike_reg_weight()}")
    
    print("--- FORCED VALUES (v12: Most should be DISABLED) ---")
    # print(f"  LR (Forced): {DEBUG_LR_VALUE}") # 無効化
    # print(f"  V_THRESHOLD (Forced): {DEBUG_V_THRESHOLD_VALUE}") # v_threshold は残す
    # print(f"  V_RESET (Forced): {DEBUG_V_RESET_VALUE}") # 無効化
    # print(f"  V_DECAY (Forced): {DEBUG_V_DECAY_VALUE}") # 無効化
    # print(f"  NEURON_BIAS (Forced): {DEBUG_BIAS_VALUE} (Config Override)") # 無効化
    # print(f"  LAYER_BIAS (Injected): {DEBUG_BIAS_VALUE} (Direct Weight Init)") # 無効化
    
    print("--- STRUCTURAL FIX CHECK ---")
    # print(f"  V_INIT (Forced): {DEBUG_V_INIT_VALUE_FORCED}") # 無効化
    print(f"  CORE_TAU_MEM (Hardcoded in LIF.py): {CORE_TAU_MEM_VALUE}")
    print("=============================================\n")
    # --- ▲▲▲ 【!!! HPO修正 (v12) !!!】 ▲▲▲ ---

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
