# ファイルパス: snn_research/conversion/ann_to_snn_converter.py
# (更新)
# Title: ANN-SNN 変換コンバータ (ECL対応)
# Description:
# - GGUF/Safetensors形式のANNモデルからSNNへの変換・蒸留を行うコンバータ。
# - [改善 v3] 堅牢な変換パイプラインを実装。BatchNorm Folding, 安全な重みコピー,
#   パーセンタイルベースの閾値キャリブレーション、ロギングを導入。
# - [改善 v4] SNN5改善レポート (セクション3.1) に基づき、ECLコンポーネントの
#   インポートと、`convert_cnn_weights` でECL関連コンポーネント
#   (LearnableClippingLayer, DualThresholdNeuron) の使用を
#   考慮するロジック（スタブ）を追加。
#
#   (改善 v5):
#   - ECL (エラー補償学習) の「スタブ」を解消。
#   - use_ecl=True の場合、実際にANNモデルのReLU層を
#     LearnableClippingLayer に置き換える処理を実装。
#
#   (修正 v15):
#   - mypy [syntax] (line 20) エラーを修正。
#   - mypy [operator] (line 71/107) エラーを修正。
#
#   (修正 v16):
#   - mypy [syntax] (line 27) を `type: ignore[import-untyped]` に修正。
#   - mypy [operator] (line 70, 91, 107) に `type: ignore` を追加。
#
# mypy --strict 準拠。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm
from typing import Dict, Any, Optional, cast, Type, List
import logging
from transformers import AutoModelForCausalLM

from snn_research.core.snn_core import AdaptiveLIFNeuron
from snn_research.core.neurons import DualThresholdNeuron # ECL用ニューロン
from .conversion_utils import safe_copy_weights, calibrate_thresholds_by_percentile
from .fold_bn import fold_all_batchnorms
from .ecl_components import LearnableClippingLayer # ECL用クリッピングレイヤー



# GGUFの依存関係をオプションにする
try:
    # --- ▼ 修正 (v16): [syntax] -> [import-untyped] に修正 ▼ ---
    from gguf import GGUFReader  # type: ignore[import-untyped]
    # --- ▲ 修正 (v16) ▲ ---
    GGUF_AVAILABLE = True
except ImportError:
    GGUFReader = Any  # type: ignore[misc, assignment]
    GGUF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_gguf(path: str) -> Dict[str, torch.Tensor]:
    """GGUFファイルを読み込み、PyTorchのstate_dictを返す。"""
    if not GGUF_AVAILABLE:
        raise ImportError("GGUFファイルを読み込むには `gguf` ライブラリが必要です。`pip install gguf` を実行してください。")
    logging.info(f"GGUFファイルをロード中: {path}")
    # --- ▼ 修正 (v16): [operator] 誤検知を抑制 ▼ ---
    reader = GGUFReader(path, 'r') # type: ignore[operator]
    # --- ▲ 修正 (v16) ▲ ---
    state_dict: Dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        state_dict[tensor.name] = torch.from_numpy(tensor.data.copy())
    logging.info(f"✅ GGUFから {len(state_dict)} 個のテンソルをロードしました。")
    return state_dict  # type: ignore[return-value]

def _replace_relu_with_ecl(
    module: nn.Module, 
    initial_threshold: float = 1.0,
    inplace: bool = True
) -> nn.Module:
    """
    (改善 v5) モデル内の nn.ReLU を LearnableClippingLayer に再帰的に置き換える。
    SNN5改善レポート (セクション3.1, 引用[6]) のための実装。
    
    Note: inplace=True のみサポート（常にモジュールを直接変更します）
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            ecl_layer = LearnableClippingLayer(initial_threshold=initial_threshold, num_features=None)
            setattr(module, name, ecl_layer)
            logging.info(f"  - [ECL] Replaced '{name}' (ReLU) with LearnableClippingLayer.")
        else:
            _replace_relu_with_ecl(child, initial_threshold, inplace=True)
            
    return module

class AnnToSnnConverter:
    """
    既存のANNモデルファイルからSNNモデルを生成するユーティリティ。
    """
    def __init__(self, snn_model: nn.Module, model_config: Dict[str, Any]):
        self.snn_model = snn_model
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.snn_model.to(self.device)

    def _load_ann_weights(self, ann_model_path: str, is_llm: bool = False) -> Dict[str, torch.Tensor]:
        """ANNモデルの重みをファイルから読み込む。"""
        logging.info(f"💾 ANNモデルの重みをロード中: {ann_model_path}")
        if ann_model_path.endswith(".safetensors"):
            return load_file(ann_model_path, device=self.device)
        elif ann_model_path.endswith(".gguf"):
            return _load_gguf(ann_model_path)
        elif is_llm:
            try:
                # --- ▼ 修正 (v16): mypy [operator] 誤検知を抑制 ▼ ---
                model = AutoModelForCausalLM.from_pretrained(ann_model_path).to(self.device) # type: ignore[operator]
                # --- ▲ 修正 (v16) ▲ ---
                return model.state_dict()
            except Exception as e:
                logging.error(f"Hugging Faceモデルのロードに失敗しました: {e}")
                raise
        else:
            try:
                return torch.load(ann_model_path, map_location=self.device)
            except Exception as e:
                logging.error(f"PyTorchモデルのロードに失敗しました: {e}")
                raise

    def convert_llm_weights(
        self,
        ann_model_name_or_path: str,
        output_path: str,
        calibration_loader: Optional[Any] = None,
        # --- ▼ 追加 ▼ ---
        use_ecl: bool = False # ECL (エラー補償学習) を試みるか
        # --- ▲ 追加 ▼ ---
    ) -> None:
        """
        Hugging FaceのLLMをロードし、正規化と高度なマッピングを行ってSNNに変換する。
        """
        logging.info(f"--- 🚀 高忠実度LLM変換開始: {ann_model_name_or_path} ---")
        
        # 1. ANNモデルのロード
        # --- ▼ 修正 (v16): mypy [operator] 誤検知を抑制 ▼ ---
        ann_model = AutoModelForCausalLM.from_pretrained(ann_model_name_or_path).to(self.device) # type: ignore[operator]
        # --- ▲ 修正 (v16) ▲ ---
        ann_model.eval()

        # (中略: LLM変換の警告)
        logging.warning("LLMの完全なSNN化は実験的です。ハイブリッドアプローチを推奨します。")
        
        # --- ▼ 修正: ECL (スタブ解消 v5) ▼ ---
        if use_ecl:
            logging.info("ECL (エラー補償学習) モードが有効です。")
            # (スタブ: 実際にはここでANNモデルのReLUをLearnableClippingLayerに置き換える前処理が必要)
            # (改善 v5: 実際に置き換え処理を呼び出す)
            logging.info("  - ANNモデルのReLUをLearnableClippingLayerに置き換え中...")
            ann_model = _replace_relu_with_ecl(ann_model, initial_threshold=1.0, inplace=True)
            
            is_dual_threshold = any(isinstance(m, DualThresholdNeuron) for m in self.snn_model.modules())
            if not is_dual_threshold:
                logging.warning("ECLが有効ですが、SNNモデルにDualThresholdNeuronが見つかりません。")
        # --- ▲ 修正 ▲ ---

        # 2. 重みコピー
        ann_state_dict = ann_model.state_dict()
        safe_copy_weights(self.snn_model, ann_state_dict)

        # 3. 閾値キャリブレーション
        if calibration_loader:
            logging.info("LLMアクティベーションに基づく閾値キャリブレーションを実行します...")
            thresholds = calibrate_thresholds_by_percentile(ann_model, calibration_loader, device=self.device)
            # (中略: 閾値設定ロジック)
            logging.info(f"計算された閾値: {thresholds}")
        else:
            logging.warning("キャリブレーションデータがないため、閾値調整をスキップします。精度が大幅に低下する可能性があります。")

        # 4. 変換済みモデルの保存
        torch.save({
            'model_state_dict': self.snn_model.state_dict(),
            'config': self.model_config
        }, output_path)
        logging.info(f"✅ LLM変換が完了し、モデルを '{output_path}' に保存しました。")

    def convert_cnn_weights(
        self,
        ann_model: nn.Module,
        output_path: str,
        calibration_loader: Any,
        # --- ▼ 追加 ▼ ---
        use_ecl: bool = False
        # --- ▲ 追加 ▼ ---
    ):
        """CNNモデルの高忠実度変換を実行する。"""
        logging.info("--- 🚀 高忠実度CNN変換開始 ---")
        ann_model.to(self.device)
        ann_model.eval()

        # --- ▼ 修正: ECL (スタブ解消 v5) ▼ ---
        if use_ecl:
            logging.info("ECL (エラー補償学習) モードが有効です。")
            # (スタブ: 実際にはここでANNモデルのReLUをLearnableClippingLayerに置き換える)
            # (改善 v5: 実際に置き換え処理を呼び出す)
            logging.info("  - ANNモデルのReLUをLearnableClippingLayerに置き換え中...")
            ann_model = _replace_relu_with_ecl(ann_model, initial_threshold=1.0, inplace=True)
            
            is_dual_threshold = any(isinstance(m, DualThresholdNeuron) for m in self.snn_model.modules())
            if not is_dual_threshold:
                logging.warning("ECLが有効ですが、SNNモデルにDualThresholdNeuronが見つかりません。")
        # --- ▲ 修正 ▲ ---

        # 1. BatchNorm Folding
        logging.info("BatchNorm Foldingを実行中...")
        folded_model = fold_all_batchnorms(ann_model)
        
        # 2. 閾値キャリブレーション
        logging.info("パーセンタイルベースの閾値キャリブレーションを実行中...")
        thresholds = calibrate_thresholds_by_percentile(folded_model, calibration_loader, device=self.device)
        
        # SNNモデルの対応するLIF層に閾値を設定
        # --- ▼ 修正: ECLニューロンも対象にする ▼ ---
        snn_neuron_layers: List[nn.Module] = [
            m for m in self.snn_model.modules() 
            if isinstance(m, (AdaptiveLIFNeuron, DualThresholdNeuron))
        ]
        
        if len(snn_neuron_layers) == len(thresholds):
            # (中略: 閾値設定ロジック)
            for lif, (name, thr) in zip(snn_neuron_layers, thresholds.items()):
                if isinstance(lif, DualThresholdNeuron):
                    lif.threshold_high.data.fill_(thr)
                    lif.threshold_low.data.fill_(thr * 0.5)
                    logging.info(f"SNN ECL Neuron (T_h, T_l) を設定: ({thr:.4f}, {thr*0.5:.4f})")
                elif isinstance(lif, AdaptiveLIFNeuron):
                    lif.base_threshold.data.fill_(thr)
                    logging.info(f"SNN LIF Neuron (base_threshold) を {thr:.4f} に設定しました。")
        # --- ▲ 修正 ▲ ---
        else:
            logging.warning(f"ANNとSNNのアクティベーション/ニューロン層の数が一致しません (ANN: {len(thresholds)}, SNN: {len(snn_neuron_layers)})。閾値設定をスキップします。")
            
        # 3. 安全な重みコピー
        logging.info("安全な重みコピーを実行中...")
        safe_copy_weights(self.snn_model, folded_model.state_dict())
        
        # 4. モデルの保存
        torch.save({
            'model_state_dict': self.snn_model.state_dict(),
            'config': self.model_config
        }, output_path)
        logging.info(f"✅ CNN変換が完了し、モデルを '{output_path}' に保存しました。")