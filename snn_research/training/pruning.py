# ファイルパス: snn_research/training/pruning.py
# (新規作成)
# Title: 構造的プルーニング (Structural Pruning)
# Description:
# snn_4_ann_parity_plan.mdのStep 3.7に基づき、モデルのスパース化と効率化のため、
# 構造的プルーニング機能を提供する。
# この実装では、最も基本的な手法の一つであるMagnitude Pruning（大きさによる枝刈り）を導入する。
# 修正(snn_4_ann_parity_plan): デバッグ用のログ出力を追加。

import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple

def apply_magnitude_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    指定されたモデルの線形層と畳み込み層に、大きさベースの構造化されていないプルーニングを適用する。

    Args:
        model (nn.Module): プルーニングを適用するモデル。
        amount (float): プルーニングする重みの割合 (0.0から1.0の間)。

    Returns:
        nn.Module: プルーニングが適用されたモデル。
    """
    if not (0.0 < amount < 1.0):
        print(f"⚠️ プルーニング量が無効です ({amount})。0.0から1.0の間の値を指定してください。プルーニングをスキップします。")
        return model

    parameters_to_prune: List[Tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))

    if not parameters_to_prune:
        print("⚠️ プルーニング対象のパラメータが見つかりませんでした。")
        return model

    print(f"✅ {len(parameters_to_prune)}個のモジュールを対象に、{amount:.2%}のグローバルプルーニングを適用します...")
    
    # デバッグ用のログ出力
    print("   - Calling prune.global_unstructured...")
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # プルーニングを恒久的に適用（元の重みパラメータを削除し、スパースな重みで置き換える）
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    print("✅ プルーニングが恒久的に適用されました。")
    return model