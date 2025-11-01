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

import sys
from pathlib import Path
import os
import torch
import torch.nn as nn # ◾️◾️◾️ 追加 ◾️◾️◾️
from torch.utils.data import DataLoader, TensorDataset # ◾️◾️◾️ 追加 ◾️◾️◾️
import yaml
import copy

sys.path.append(str(Path(__file__).resolve().parent))

from snn_research.bio_models.simple_network import BioSNN
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignmentEnhancedV2
from snn_research.hardware.compiler import NeuromorphicCompiler
# --- ▼ 修正 ▼ ---
from snn_research.training.pruning import apply_sbc_pruning # apply_magnitude_pruning から変更
# --- ▲ 修正 ▲ ---

def main():
    """
    NeuromorphicCompilerのテストを実行する。
    プルーニングを適用したモデルのコンパイルも検証する。
    """
    print("--- ニューロモーフィック・コンパイラ テスト開始 ---")

    learning_rate = 0.005
    learning_rule = CausalTraceCreditAssignmentEnhancedV2(
        learning_rate=learning_rate, a_plus=1.0, a_minus=1.0,
        tau_trace=20.0, tau_eligibility=50.0
    )
    model = BioSNN(
        layer_sizes=[10, 20, 5],
        neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0},
        learning_rule=learning_rule
    )
    print("✅ ダミーのBioSNNモデルを構築しました。")

    # (プルーニング、コンパイル、検証のロジックは変更なし)
    original_connections = sum(torch.sum(w > 0).item() for w in model.weights)
    pruning_amount = 0.3
    
    # --- ▼ 修正: apply_sbc_pruning にダミーの引数を追加 ▼ ---
    # SBCのヘッセ行列計算（スタブ）に必要なダミーのデータローダーと損失関数
    dummy_dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 5))
    dummy_loader = DataLoader(dummy_dataset, batch_size=2)
    dummy_loss = nn.MSELoss() # ダミーの損失関数

    pruned_model = apply_sbc_pruning(
        copy.deepcopy(model), 
        amount=pruning_amount,
        dataloader_stub=dummy_loader, # ダミー引数を追加
        loss_fn_stub=dummy_loss       # ダミー引数を追加
    )
    # --- ▲ 修正 ▲ ---
    
    pruned_connections = sum(torch.sum(w > 0).item() for w in pruned_model.weights)
    print(f"🔪 モデルをプルーニングしました: {original_connections} -> {pruned_connections} connections")
    assert pruned_connections < original_connections

    compiler = NeuromorphicCompiler(hardware_profile_name="default")
    output_dir = "runs/compiler_tests"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "compiled_pruned_hardware_config.yaml")
    compiler.compile(pruned_model, output_path)

    if os.path.exists(output_path):
        print(f"\n✅ コンパイル成功: 設定ファイルが '{output_path}' に生成されました。")
        with open(output_path, 'r') as f:
            config = yaml.safe_load(f)
        assert "learning_rule_config" in config
        lr_config = config["learning_rule_config"]
        assert lr_config["rule_name"] == "CausalTraceCreditAssignmentEnhancedV2", "学習則の名前が一致しません。"
        assert "parameters" in lr_config
        params = lr_config["parameters"]
        assert "learning_rate" in params and abs(params["learning_rate"] - learning_rate) < 1e-6
        print("  - 検証: 学習則のコンパイル結果は正常です。")
        compiled_connections = sum(layer['num_connections'] for layer in config['synaptic_connectivity'])
        assert compiled_connections == pruned_connections
        print(f"  - 検証: プルーニング結果がコンパイルファイルに正しく反映されました ({compiled_connections} connections)。")

        simulation_report = compiler.simulate_on_hardware(
            compiled_config_path=output_path,
            total_spikes=15000,
            time_steps=100
        )
        print("\n--- 📊 ハードウェアシミュレーション結果 ---")
        for key, value in simulation_report.items(): print(f"  - {key}: {value:.4e}")
        print("------------------------------------------")
    else:
        print(f"\n❌ テスト失敗: 設定ファイルが生成されませんでした。")

    print("\n--- ニューロモーフィック・コンパイラ テスト終了 ---")

if __name__ == "__main__":
    main()
