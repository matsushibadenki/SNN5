# ファイルパス: tests/cognitive_architecture/test_artificial_brain.py
# タイトル: 人工脳 統合テスト
# 機能説明:
# - 人工脳（ArtificialBrain）の統合的な動作を検証する。
# - DIコンテナ（BrainContainer）が全コンポーネントを正しく構築し、
#   循環参照なしにArtificialBrainインスタンスを生成できることを確認する。
# - 認知サイクルがエラーなく実行され、各認知モジュールが意図通りに連携し、
#   状態を変化させることを検証する。特に、短期記憶から長期記憶への
#   「記憶の固定」プロセスが正しく機能することを確認する。
# 改善点(v4):
# - ロードマップ フェーズ5に基づき、テストを「深化」。
# - 認知サイクル実行後、入力テキストのキーワードが長期記憶に
#   正しく関連付けられて格納されたかを具体的に検証するアサーションを追加。

import sys
from pathlib import Path
import pytest

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.containers import BrainContainer
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

@pytest.fixture(scope="module")
def brain_container():
    """DIコンテナを初期化し、テストフィクスチャとして提供する。"""
    container = BrainContainer()
    # テスト用の設定をロード
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml("configs/models/small.yaml")
    
    # RAGSystemの初回セットアップをシミュレート
    rag_system = container.agent_container.rag_system()
    if not rag_system.vector_store:
        rag_system.setup_vector_store()
    return container

def test_artificial_brain_instantiation(brain_container: BrainContainer):
    """
    BrainContainerがArtificialBrainインスタンスを正常に構築できるかテストする。
    """
    brain = brain_container.artificial_brain()
    assert brain is not None
    assert brain.pfc is not None
    assert brain.hippocampus is not None
    assert brain.motor is not None
    print("✅ ArtificialBrainインスタンスの構築に成功しました。")

def test_cognitive_cycle_runs_and_consolidates_memory(brain_container: BrainContainer):
    """
    run_cognitive_cycleが実行され、記憶の固定化が正しく行われるかテストする。
    """
    brain: ArtificialBrain = brain_container.artificial_brain()
    
    # サイクル実行前の状態
    initial_cortex_size = len(brain.cortex.get_all_knowledge())
    
    # 記憶の固定がトリガーされるまで5回サイクルを実行
    test_inputs = [
        "This is a test about system integration.",
        "Another test focused on memory and learning.",
        "A third input to populate the hippocampus.",
        "Fourth cycle continues the process.",
        "Fifth cycle should trigger consolidation."
    ]
    
    try:
        for i, text in enumerate(test_inputs):
            brain.run_cognitive_cycle(text)
            # 5サイクル目に統合が起こることを確認
            if (i + 1) % 5 != 0:
                # サイクルごとに短期記憶が増える
                assert len(brain.hippocampus.working_memory) == (i + 1) % 5
            else:
                # 5サイクル目に長期記憶へ転送され、短期記憶はクリアされる
                assert len(brain.hippocampus.working_memory) == 0

        print(f"✅ 5回の認知サイクルが正常に完了しました。")
    except Exception as e:
        pytest.fail(f"run_cognitive_cycleで予期せぬエラーが発生しました: {e}")

    # --- 実行後の状態変化を詳細に検証 ---
    # 1. 海馬（短期記憶）がクリアされたか
    assert len(brain.hippocampus.working_memory) == 0, \
        "5サイクル後に海馬のワーキングメモリがクリアされていません。"
        
    # 2. 大脳皮質（長期記憶）に新しい知識が追加されたか
    final_cortex_size = len(brain.cortex.get_all_knowledge())
    assert final_cortex_size > initial_cortex_size, \
        "大脳皮質のナレッジグラフに新しい知識が追加されていません。"
        
    # 3. 記録された知識の内容を具体的に確認
    # "A test about system integration" -> "system"と"integration"が関連付けられているはず
    knowledge_system = brain.cortex.retrieve_knowledge("system")
    assert knowledge_system is not None, "長期記憶から 'system' の知識を取得できませんでした。"
    assert any(rel['relation'] == 'co-occurred_with' and rel['target'] == 'integration' for rel in knowledge_system), \
        "'system' と 'integration' の関連性が記録されていません。"
        
    # "memory and learning" -> "memory"と"learning"が関連付けられているはず
    knowledge_memory = brain.cortex.retrieve_knowledge("memory")
    assert knowledge_memory is not None, "長期記憶から 'memory' の知識を取得できませんでした。"
    assert any(rel['relation'] == 'co-occurred_with' and rel['target'] == 'learning' for rel in knowledge_memory), \
        "'memory' と 'learning' の関連性が記録されていません。"
        
    print("✅ 記憶の固定化プロセスと、その内容が正しく記録されたことを確認しました。")