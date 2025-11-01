# matsushibadenki/snn_research/conversion/ann_to_snn_converter.py
# (更新)
# GGUF/Safetensors形式のANNモデルからSNNへの変換・蒸留を行うコンバータ
#
# 機能:
# - [改善 v3] 堅牢な変換パイプラインを実装。BatchNorm Folding, 安全な重みコピー,
#   パーセンタイルベースの閾値キャリブレーション、ロギングを導入。
# - [改善 v3] LLM変換の非現実性を明確化し、ハイブリッドアプローチの重要性を強調。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm
from typing import Dict, Any, Optional
import logging
from transformers import AutoModelForCausalLM

from snn_research.core.snn_core import AdaptiveLIFNeuron
from .conversion_utils import safe_copy_weights, calibrate_thresholds_by_percentile
from .fold_bn import fold_all_batchnorms

# GGUFの依存関係をオプションにする
try:
    from gguf import GGUFReader
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_gguf(path: str) -> Dict[str, torch.Tensor]:
    """GGUFファイルを読み込み、PyTorchのstate_dictを返す。"""
    if not GGUF_AVAILABLE:
        raise ImportError("GGUFファイルを読み込むには `gguf` ライブラリが必要です。`pip install gguf` を実行してください。")
    logging.info(f"GGUFファイルをロード中: {path}")
    reader = GGUFReader(path, 'r')
    state_dict = {tensor.name: torch.from_numpy(tensor.data.copy()) for tensor in reader.tensors}
    logging.info(f"✅ GGUFから {len(state_dict)} 個のテンソルをロードしました。")
    return state_dict

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
                model = AutoModelForCausalLM.from_pretrained(ann_model_path)
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
        calibration_loader: Optional[Any] = None
    ) -> None:
        """
        Hugging FaceのLLMをロードし、正規化と高度なマッピングを行ってSNNに変換する。
        """
        logging.info(f"--- 🚀 高忠実度LLM変換開始: {ann_model_name_or_path} ---")
        
        # 1. ANNモデルのロード
        ann_model = AutoModelForCausalLM.from_pretrained(ann_model_name_or_path).to(self.device)
        ann_model.eval()

        #
        # 警告: Transformer全体の完全なスパイク化は非常に困難です。
        # 特に、LayerNormやSoftmaxを含む自己注意メカニズムの直接変換は、
        # 大幅な精度低下や、膨大なタイムステップ数を要求する原因となります。
        #
        # 現実的なアプローチは、以下のいずれかです:
        # 1. ハイブリッドモデル: Attentionなど計算が複雑な部分はアナログのまま残し、
        #    FFN層など一部のみをSNNに置き換える。
        # 2. ANN-SNN変換 + 長時間の微調整: 重みをコピーした後、代理勾配法を
        #    用いてSNNモデルを再度ファインチューニングする。
        #
        # このメソッドでは、主に互換性のある線形層の安全な重みコピーに焦点を当てます。
        logging.warning("LLMの完全なSNN化は実験的です。ハイブリッドアプローチを推奨します。")

        # 2. 重みコピー
        ann_state_dict = ann_model.state_dict()
        safe_copy_weights(self.snn_model, ann_state_dict)

        # 3. 閾値キャリブレーション
        if calibration_loader:
            logging.info("LLMアクティベーションに基づく閾値キャリブレーションを実行します...")
            thresholds = calibrate_thresholds_by_percentile(ann_model, calibration_loader, device=self.device)
            # ここで、計算された閾値をSNNモデルの各LIFニューロンに設定するロジックが必要
            # 例: for name, module in self.snn_model.named_modules(): ...
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
        calibration_loader: Any
    ):
        """CNNモデルの高忠実度変換を実行する。"""
        logging.info("--- 🚀 高忠実度CNN変換開始 ---")
        ann_model.to(self.device)
        ann_model.eval()

        # 1. BatchNorm Folding
        logging.info("BatchNorm Foldingを実行中...")
        folded_model = fold_all_batchnorms(ann_model)
        
        # 2. 閾値キャリブレーション
        logging.info("パーセンタイルベースの閾値キャリブレーションを実行中...")
        thresholds = calibrate_thresholds_by_percentile(folded_model, calibration_loader, device=self.device)
        
        # SNNモデルの対応するLIF層に閾値を設定
        lif_layers = [m for m in self.snn_model.modules() if isinstance(m, AdaptiveLIFNeuron)]
        if len(lif_layers) == len(thresholds):
            for lif, (name, thr) in zip(lif_layers, thresholds.items()):
                lif.base_threshold.data.fill_(thr)
                logging.info(f"SNNレイヤーの閾値を {thr:.4f} に設定しました。")
        else:
            logging.warning("ANNとSNNのReLU/LIF層の数が一致しません。閾値設定をスキップします。")
            
        # 3. 安全な重みコピー
        logging.info("安全な重みコピーを実行中...")
        safe_copy_weights(self.snn_model, folded_model.state_dict())
        
        # 4. モデルの保存
        torch.save({
            'model_state_dict': self.snn_model.state_dict(),
            'config': self.model_config
        }, output_path)
        logging.info(f"✅ CNN変換が完了し、モデルを '{output_path}' に保存しました。")