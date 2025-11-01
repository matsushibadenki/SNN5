# ファイルパス: snn_research/cognitive_architecture/hybrid_perception_cortex.py
# (更新)
# 修正: mypyエラーを解消するため、Optional型を明示的にインポート・使用。
# 改善点(v2): 「意識的認知サイクル」実装のため、GlobalWorkspaceと連携。
#            - 処理結果を返すのではなく、顕著性スコアと共にWorkspaceにアップロードする。
#            - 顕著性スコアとして、入力パターンの活動量（総スパイク数）を利用する。

import torch
from typing import Dict, Any, Optional

from .som_feature_map import SomFeatureMap
from .global_workspace import GlobalWorkspace

class HybridPerceptionCortex:
    """
    自己組織化マップ(SOM)を統合し、GlobalWorkspaceと連携する知覚野モジュール。
    """
    def __init__(self, workspace: GlobalWorkspace, num_neurons: int, feature_dim: int = 64, som_map_size=(8, 8), stdp_params: Optional[Dict[str, Any]] = None):
        """
        Args:
            workspace (GlobalWorkspace): 情報をアップロードするための中央ハブ。
            num_neurons (int): 入力スパイクパターンのニューロン数。
            feature_dim (int): SOMへの入力特徴ベクトルの次元数。
            som_map_size (tuple): SOMのマップサイズ。
            stdp_params (Optional[dict]): SOMが使用するSTDP学習則のパラメータ。
        """
        self.workspace = workspace
        self.num_neurons = num_neurons
        self.feature_dim = feature_dim
        
        self.input_projection = torch.randn((num_neurons, feature_dim))
        
        if stdp_params is None:
            stdp_params = {'learning_rate': 0.005, 'a_plus': 1.0, 'a_minus': 1.0, 'tau_trace': 20.0}
        
        self.som = SomFeatureMap(
            input_dim=feature_dim,
            map_size=som_map_size,
            stdp_params=stdp_params
        )
        print("🧠 ハイブリッド知覚野モジュールが初期化されました (SOM統合)。")

    def perceive_and_upload(self, spike_pattern: torch.Tensor) -> None:
        """
        入力スパイクを知覚・学習し、その結果と顕著性をGlobalWorkspaceにアップロードする。
        """
        if spike_pattern.shape[1] != self.num_neurons:
            raise ValueError(f"入力スパイクのニューロン数 ({spike_pattern.shape[1]}) が"
                             f"知覚野のニューロン数 ({self.num_neurons}) と一致しません。")

        # 1. 時間的プーリング
        temporal_features = torch.sum(spike_pattern, dim=0)

        # 2. 特徴射影
        feature_vector = torch.matmul(temporal_features, self.input_projection)
        feature_vector = torch.relu(feature_vector)

        # 3. SOMによる特徴分類と学習
        for _ in range(5):
            som_spikes = self.som(feature_vector)
            self.som.update_weights(feature_vector, som_spikes)
        
        final_som_activation = self.som(feature_vector)
        
        # 顕著性スコアを計算（入力スパイクの総量＝刺激の強さ）
        salience = torch.sum(spike_pattern).item() / spike_pattern.numel()
        
        perception_data = {"type": "perception", "features": final_som_activation}

        self.workspace.upload_to_workspace(
            source="perception",
            data=perception_data,
            salience=salience
        )
        print(f"  - 知覚野: 特徴を抽出し、Workspaceにアップロードしました。")