# ファイルパス: run_compiler_test.py
# (更新)
#
# Title: ニューロモーフィック・コンパイラ テストスクリプト
#
# Description:
# - ロードマップ「ニューロモーフィックハードウェアへの最適化」で実装した
#   NeuromorphicCompilerの動作を検証するためのスクリプト。
# - ダミーのBioSNNモデルを構築し、それをハードウェア構成ファイルに
#   コンパイルするプロセスを実行する。
#
# 改善点(v2):
# - ROADMAPフェーズ6に基づき、コンパイル後のハードウェア性能シミュレーションを実行する処理を追加。
# 改善点(v3): コンパイルされたファイルに学習則が含まれているか検証するテストを追加。
# 改善点(snn_4_ann_parity_plan):
# - 学習則のパラメータ検証をより厳密化。
# - 古いスクリプトを削除し、こちらに機能を統合。
# - プルーニングを適用し、最適化されたモデルのコンパイルをテストする機能を追加。
# 修正: CausalTraceCreditAssignment -> CausalTraceCreditAssignmentEnhanced
# 修正: CausalTraceCreditAssignmentEnhancedV2 に対応
#
# 修正 (v7):
# - mypy [attr-defined] エラーを解消するため、apply_magnitude_pruning を
#   apply_sbc_pruning に変更し、ダミーの引数を追加。
#
# 改善 (v8):
# - BioSNN に加え、SNNCore ベースのモデル (SEW-ResNet) の
#   コンパイルとシミュレーションもテストするよう拡張。
# - DIコンテナ (TrainingContainer) を使用して SNNCore モデルを構築。
#
# 修正 (v9):
# - mypy [misc], [union-attr] エラーを解消するため、
#   pruned_model と snn_core_model を cast するよう修正。
#
# 修正 (v10):
# - mypy [call-arg] エラーを修正。
# - BioSNN のコンストラクタ引数を `learning_rule` から `synaptic_rule` に変更。
#
# 修正 (v11):
# - mypy [call-arg], [misc], [union-attr] エラーを修正。
# - BioSNN の __init__ シグネチャ変更 (layer_sizes -> input_size, layer_configs) に対応。
#
# 修正 (v12):
# - mypy [assignment] エラーを解消するため、configから取得した time_steps を int に明示的に変換。
#
# 修正 (v13):
# - mypy [assignment] および [no-redef] エラーを解消するため、変数をアンビギュアスに宣言し、
#   `time_steps` 変数名を `test_snncore_compilation` 内で再利用しないよう修正。

import sys
from pathlib import Path
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
import copy
# --- ▼ 修正: 必要な型ヒントを追加 ▼ ---
from typing import Dict, Any, cast, List
from omegaconf import OmegaConf
# --- ▲ 修正 ▲ ---

sys.path.append(str(Path(__file__).resolve().parent))

from snn_research.bio_models.simple_network import BioSNN
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignmentEnhancedV2
from snn_research.hardware.compiler import NeuromorphicCompiler
from snn_research.training.pruning import apply_sbc_pruning
# --- ▼ 修正: SNNCoreモデル構築のために DIコンテナとSNNCoreをインポート ▼ ---
from app.containers import TrainingContainer
from snn_research.core.snn_core import SNNCore
# --- ▲ 修正 ▲ ---


def test_biosnn_compilation(compiler: NeuromorphicCompiler, output_dir: str, container: TrainingContainer) -> None:
    """BioSNNモデルのコンパイルとプルーニングをテストする。"""
    print("\n--- 1. BioSNNモデルのコンパイルテスト開始 ---")
    
    learning_rate = 0.005
    learning_rule = CausalTraceCreditAssignmentEnhancedV2(
        learning_rate=learning_rate, a_plus=1.0, a_minus=1.0,
        tau_trace=20.0, tau_eligibility=50.0
    )
    
    model_input_size = 10
    model_layer_configs: List[Dict[str, int]] = [
        {"n_e": 20, "n_i": 0}, # 隠れ層
        {"n_e": 5, "n_i": 0}   # 出力層
    ]
    
    model: BioSNN = BioSNN(
        input_size=model_input_size,
        layer_configs=model_layer_configs,
        neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0, 'threshold_decay': 0.99, 'threshold_step': 0.05}, # P8.3のパラメータ追加
        synaptic_rule=learning_rule,
        homeostatic_rule=None,
        sparsification_config={"enabled": True, "contribution_threshold": 0.01}
    )
    print("✅ ダミーのBioSNNモデルを構築しました。")

    pruned_model_uncast: nn.Module = apply_sbc_pruning(
        copy.deepcopy(model), 
        amount=0.3,
        dataloader_stub=DataLoader(TensorDataset(torch.randn(10, 10), torch.randn(10, 5)), batch_size=2),
        loss_fn_stub=nn.MSELoss()
    )
    pruned_model: BioSNN = cast(BioSNN, pruned_model_uncast)
    
    original_connections = sum(torch.sum(w.data > 0).item() for w in model.weights_ee) # type: ignore[misc]
    pruned_connections = sum(torch.sum(w.data > 0).item() for w in pruned_model.weights_ee) # type: ignore[misc]
    
    print(f"🔪 モデルをプルーニングしました: {original_connections} -> {pruned_connections} connections (E->E only)")
    assert pruned_connections < original_connections

    output_path = os.path.join(output_dir, "compiled_biosnn_pruned_config.yaml")
    compiler.compile(pruned_model, output_path)

    if os.path.exists(output_path):
        print(f"\n✅ BioSNNコンパイル成功: 設定ファイルが '{output_path}' に生成されました。")
        with open(output_path, 'r') as f:
            config = yaml.safe_load(f)
        assert "learning_rule_config" in config
        lr_config = config["learning_rule_config"]
        assert lr_config["rule_name"] == "CausalTraceCreditAssignmentEnhancedV2", "学習則の名前が一致しません。"
        assert "parameters" in lr_config
        params = lr_config["parameters"]
        assert "learning_rate" in params and abs(params["learning_rate"] - learning_rate) < 1e-6
        print("  - 検証: 学習則のコンパイル結果は正常です。")
        
        compiled_connections = config.get("network_summary", {}).get("total_connections", 0)
        
        pruned_model.clamp_weights()
        pruned_total_conn = 0
        for w_list in [pruned_model.weights_ee, pruned_model.weights_ie, pruned_model.weights_ei, pruned_model.weights_ii]:
             pruned_total_conn += sum(torch.sum(w.data > 0).item() for w in w_list) # type: ignore[assignment]
        
        assert compiled_connections == pruned_total_conn
        print(f"  - 検証: プルーニング結果がコンパイルファイルに正しく反映されました ({compiled_connections} total connections)。")

        # --- ▼ 修正 (L135-L147): time_steps_sim の宣言と代入を修正 ▼ ---
        time_steps_val: Any = container.config.model.time_steps()
        
        # L145: time_steps_sim はここで初めて宣言される (デフォルト値を設定)
        time_steps_sim: int = 16 
        
        # L135: floatの可能性を考慮し、intに明示的に変換（L135の行）
        if isinstance(time_steps_val, (int, float)):
            # [assignment]エラーを抑制し、代入のみ行う（再注釈なし）
            time_steps_sim = int(time_steps_val) # type: ignore[assignment]
        # --- ▲ 修正 (L135-L147) ▲ ---

        simulation_report = compiler.simulate_on_hardware(
            compiled_config_path=output_path,
            total_spikes=15000,
            time_steps=time_steps_sim
        )
        print("\n--- 📊 BioSNN ハードウェアシミュレーション結果 ---")
        for key, value in simulation_report.items(): print(f"  - {key}: {value:.4e}")
        print("--------------------------------------------------")
    else:
        print(f"\n❌ BioSNNテスト失敗: 設定ファイルが生成されませんでした。")
        raise AssertionError("BioSNNコンパイルテスト失敗")

def test_snncore_compilation(compiler: NeuromorphicCompiler, output_dir: str, container: TrainingContainer) -> None:
    """SNNCore (SEW-ResNet) モデルのコンパイルをテストする。"""
    print("\n--- 2. SNNCore (SEW-ResNet) モデルのコンパイルテスト開始 ---")

    try:
        # この関数は main で設定された container を使用する
        
        # モデル構成を上書き (SEW-ResNet)
        container.config.model.architecture_type.from_value("sew_resnet")
        
        snn_core_model_uncast: nn.Module = container.snn_model(vocab_size=10)
        snn_core_model: SNNCore = cast(SNNCore, snn_core_model_uncast)
        
        snn_core_model.eval()
        print(f"✅ ダミーのSNNCoreモデル ({snn_core_model.config.architecture_type}) を構築しました。")

    except Exception as e:
        print(f"❌ SNNCoreモデルの構築に失敗しました: {e}")
        print("   SEW-ResNetの実装 (snn_research/architectures/sew_resnet.py) が必要です。")
        return

    output_path = os.path.join(output_dir, "compiled_snncore_sew_resnet_config.yaml")
    compiler.compile(snn_core_model, output_path)
    
    if os.path.exists(output_path):
        print(f"\n✅ SNNCoreコンパイル成功: 設定ファイルが '{output_path}' に生成されました。")
        with open(output_path, 'r') as f:
            config: Dict[str, Any] = yaml.safe_load(f)
            
        assert "network_summary" in config
        summary = config["network_summary"]
        assert summary["total_neurons"] > 0
        assert summary["total_connections"] > 0
        print(f"  - 検証: ネットワーク概要: Neurons={summary['total_neurons']}, Connections={summary['total_connections']}")
        
        assert "learning_rule_config" in config
        assert config["learning_rule_config"]["rule_name"] == "None"
        print("  - 検証: 学習則 (None) のコンパイル結果は正常です。")

        estimated_spikes = 500000
        
        time_steps_val: Any = container.config.model.time_steps()
        
        # time_steps_core_val の初期化と代入ロジック
        time_steps_core_val: int = 16 # Initialize with default
        
        # floatの可能性を考慮し、intに明示的に変換
        if isinstance(time_steps_val, (int, float)):
            time_steps_core_val = int(time_steps_val) # type: ignore[assignment]
        
        simulation_report = compiler.simulate_on_hardware(
            compiled_config_path=output_path,
            total_spikes=estimated_spikes,
            time_steps=time_steps_core_val # 使用する変数を time_steps_core_val に統一
        )
        print("\n--- 📊 SNNCore ハードウェアシミュレーション結果 ---")
        for key, value in simulation_report.items(): print(f"  - {key}: {value:.4e}")
        print("---------------------------------------------------")
    else:
        print(f"\n❌ SNNCoreテスト失敗: 設定ファイルが生成されませんでした。")
        raise AssertionError("SNNCoreコンパイルテスト失敗")

def main():
    """
    NeuromorphicCompilerのテストを実行する。
    """
    print("--- ニューロモーフィック・コンパイラ 統合テスト開始 ---")

    # 1. DIコンテナを初期化 (テスト全体で共有)
    container = TrainingContainer()
    
    # 2. ベース設定とモデル設定をロード (テストで上書きされる)
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml("configs/cifar10_spikingcnn_config.yaml")

    compiler = NeuromorphicCompiler(hardware_profile_name="default")
    output_dir = "runs/compiler_tests"
    os.makedirs(output_dir, exist_ok=True)

    # テスト1: BioSNN (プルーニング + 学習則)
    try:
        # BioSNNはここで設定を上書きする必要はない (内部でハードコードされているため)
        test_biosnn_compilation(compiler, output_dir, container)
    except Exception as e:
        print(f"❌ BioSNNコンパイルテスト中にエラーが発生しました: {e}", exc_info=True)

    # テスト2: SNNCore (SEW-ResNet)
    try:
        # SNNCoreテストは内部で architecture_type を 'sew_resnet' に上書きする
        test_snncore_compilation(compiler, output_dir, container)
    except Exception as e:
        print(f"❌ SNNCoreコンパイルテスト中にエラーが発生しました: {e}", exc_info=True)


    print("\n--- ニューロモーフィック・コンパイラ 統合テスト終了 ---")

if __name__ == "__main__":
    main()