# ファイルパス: snn_research/training/quantization.py
# (更新)
# Title: 量子化認識学習 (Quantization-Aware Training - QAT)
# Description:
# snn_4_ann_parity_plan.mdのStep 3.7に基づき、モデルの量子化を
# 訓練中にシミュレートするための機能を提供する。
# これにより、量子化による精度低下を最小限に抑え、
# ニューロモーフィックハードウェアなど低ビット精度での展開を容易にする。
#
# 追加 (v2):
# - SNN5改善レポート (セクション4.2, 引用[20]) に基づき、
#   SNN固有の膜電位量子化 (SpQuant-SNN) をシミュレートする関数を追加。
#
# 修正 (v3): mypy [name-defined] エラーを修正。
#
# 改善 (SNN5改善レポート 4.2 対応):
# - _quantize_membrane_potential (スタブ) を、学習可能なスケールファクタを
#   持つ QuantizeMembranePotential モジュールに置き換え。
# - SpQuantWrapper が新しい量子化モジュールを使用するように修正。

import torch
import torch.nn as nn
import copy
# --- ▼ 修正 ▼ ---
from typing import Dict, Any, cast, Tuple, Optional
import logging
import torch.nn.functional as F # [name-defined] F をインポート

from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron
from spikingjelly.activation_based import base as sj_base # type: ignore
from snn_research.core.base import BaseModel # [name-defined] BaseModel をインポート

logger = logging.getLogger(__name__)
# --- ▲ 修正 ▲ ---


def apply_qat(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    モデルに量子化スタブを挿入して、QATの準備を行う。
    この関数は、線形層と畳み込み層を量子化対象とします。

    Args:
        model (nn.Module): QATを適用するモデル。
        inplace (bool): モデルを直接変更するかどうか。

    Returns:
        nn.Module: QATが適用されたモデル。
    """
    if not inplace:
        model = copy.deepcopy(model)

    # PyTorchのEager Mode QATを使用
    # 'fbgemm'はx86 CPUに最適化されたバックエンド
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # 量子化の準備
    # `torch.quantization.prepare_qat` は、指定されたqconfigに基づいて
    # 量子化対象のモジュール（Conv, Linearなど）をラップし、
    # 量子化/逆量子化の操作とオブザーバーを挿入する。
    torch.quantization.prepare_qat(model, inplace=True)
    
    print("✅ モデルにQATの準備ができました。量子化スタブが挿入されました。")
    return model

def convert_to_quantized_model(model: nn.Module) -> nn.Module:
    """
    QAT後のモデルを、実際に量子化されたモデルに変換する。
    推論時に使用する。

    Args:
        model (nn.Module): QAT済みのモデル。

    Returns:
        nn.Module: 量子化されたモデル。
    """
    model.eval()
    # `torch.quantization.convert` は、オブザーバーで収集された統計情報に基づき
    # モデルの重みとバイアスを量子化し、浮動小数点演算を整数演算に置き換える。
    quantized_model = torch.quantization.convert(model.to('cpu'), inplace=False)
    print("✅ QAT済みモデルが量子化され、推論の準備ができました。")
    return quantized_model

# --- ▼▼▼ SNN5改善レポートに基づく追加実装 (セクション4.2, [20]) ▼▼▼ ---

# --- ▼ 改善 (SNN5改善レポート 4.2 対応) ▼ ---
# 学習可能なスケールファクタを持つ3値量子化関数
class TernaryQuantizeFunction(torch.autograd.Function):
    """
    (改善) SpQuant-SNN [20] に基づく、学習可能なスケールファクタを持つ
    3値量子化のカスタム勾配。
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        順伝播: 3値 (-alpha, 0, +alpha) に量子化
        y = +alpha (if x > threshold)
        y = 0      (if |x| <= threshold)
        y = -alpha (if x < -threshold)
        """
        ctx.save_for_backward(x, alpha)
        ctx.threshold = threshold
        
        out = torch.zeros_like(x)
        out[x > threshold] = alpha.item()
        out[x < -threshold] = -alpha.item()
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        逆伝播:
        - x への勾配: Straight-Through Estimator (STE)
        - alpha への勾配: 量子化の境界に基づき計算
        """
        x, alpha = ctx.saved_tensors
        threshold: float = cast(float, ctx.threshold)
        
        # --- 入力 (x) への勾配 (STE) ---
        # |x| <= 1.0 の範囲でのみ勾配を流す (STEの標準的なクリッピング)
        grad_x = grad_output.clone()
        grad_x[torch.abs(x) > 1.0] = 0 
        
        # --- スケールファクタ (alpha) への勾配 ---
        # dL/d_alpha = dL/dy * dy/d_alpha
        # dy/d_alpha = +1 (if x > threshold), -1 (if x < -threshold), 0 (else)
        grad_alpha = torch.zeros_like(alpha)
        grad_alpha_term = (grad_output * (x > threshold).float()) + (grad_output * (x < -threshold).float() * -1.0)
        grad_alpha = grad_alpha_term.sum() # 全バッチ・全ニューロンで勾配を合計
        
        return grad_x, grad_alpha, None # threshold への勾配はなし (学習させない)

class QuantizeMembranePotential(nn.Module):
    """
    (改善) SpQuant-SNN [20] に基づく、学習可能な3値量子化モジュール。
    """
    def __init__(self, initial_alpha: float = 1.0, threshold: float = 0.5):
        super().__init__()
        # スケールファクタ (alpha) を学習可能なパラメータとして定義
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.threshold = threshold
        
    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        """膜電位(mem)を3値(-alpha, 0, +alpha)に量子化する。"""
        return TernaryQuantizeFunction.apply(mem, self.alpha, self.threshold)


@torch.no_grad()
def _apply_negative_membrane_pruning(mem_quant: torch.Tensor) -> torch.Tensor:
    """
    (改善) SpQuant-SNN [20] に基づき、量子化された膜電位の負の値をプルーニング（ゼロ化）する。
    """
    # mem_quant は (-alpha, 0, +alpha) の3値と仮定
    pruned_mem = F.relu(mem_quant) # 負の値 (-alpha) を 0 にする
    return pruned_mem

class SpQuantWrapper(nn.Module):
    """
    SpQuant-SNN [20] の「二重圧縮」をシミュレートするため、
    既存のSNNニューロンをラップするラッパーモジュール (スタブ改善)。
    """
    neuron: nn.Module
    quantizer: QuantizeMembranePotential # 型ヒントを修正

    def __init__(self, neuron_module: nn.Module):
        super().__init__()
        if not isinstance(neuron_module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron)):
            raise TypeError(f"SpQuantWrapperはLIF, Izhikevich, GLIFニューロンのみラップ可能です。Got: {type(neuron_module)}")
        
        self.neuron = neuron_module
        
        # --- ▼ 改善: 学習可能な量子化モジュールをインスタンス化 ▼ ---
        # 閾値はニューロンのベース閾値の半分などを参考に設定可能
        base_thresh: float = 1.0
        if hasattr(neuron_module, 'base_threshold'):
             # (mypy) base_threshold は nn.Parameter の可能性がある
             thresh_param = getattr(neuron_module, 'base_threshold')
             if isinstance(thresh_param, torch.Tensor):
                 base_thresh = thresh_param.mean().item()
             elif isinstance(thresh_param, float):
                 base_thresh = thresh_param
                 
        self.quantizer = QuantizeMembranePotential(
            initial_alpha=base_thresh, # スケールをベース閾値に合わせる
            threshold=base_thresh * 0.5 # 量子化の閾値
        )
        logger.info(f"SpQuantWrapper: ラッピング -> {type(neuron_module).__name__} (Quantization Alpha: {self.quantizer.alpha.item():.2f}, Thresh: {self.quantizer.threshold:.2f})")
        # --- ▲ 改善 ▲ ---

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 元のニューロンで膜電位を計算
        spike, mem_out = self.neuron(x)
        
        # 2. 膜電位を量子化 (学習可能な関数を使用)
        mem_quant = self.quantizer(mem_out)
        
        # 3. 負の膜電位をプルーニング
        mem_pruned = _apply_negative_membrane_pruning(mem_quant)
        
        # (注: 本来はこの mem_pruned をニューロンの *次のステップ* の状態として
        #  強制的に設定する必要があるが、現在のLIF実装では困難)
        
        # ここでは、元のスパイクと、デバッグ用にプルーニングされた膜電位を返す
        # (注: QAT中は、mem_quant または mem_pruned を次の層に渡すべきかもしれないが、
        #  現在のアーキテクチャでは mem は返されるだけで伝播しない)
        return spike, mem_pruned

    def set_stateful(self, stateful: bool) -> None:
        if hasattr(self.neuron, 'set_stateful'):
            cast(Any, self.neuron).set_stateful(stateful)

    def reset(self) -> None:
        if hasattr(self.neuron, 'reset'):
            cast(Any, self.neuron).reset()
# --- ▲▲▲ SNN5改善レポートに基づく修正 ▲▲▲ ---

def apply_spquant_quantization(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    SNN5改善レポート (セクション4.2, [20]) に基づき、
    モデル内のSNNニューロンをSpQuantWrapperでラップする (スタブ)。

    Args:
        model (nn.Module): 量子化対象のSNNモデル。
        inplace (bool): モデルを直接変更するかどうか。

    Returns:
        nn.Module: SpQuantWrapperが適用されたモデル。
    """
    if not inplace:
        model = copy.deepcopy(model)

    logger.info("---  applying SpQuant-SNN (Membrane Quantization Stub) ---")
    
    # モデルの階層を再帰的に探索
    for name, module in list(model.named_children()):
        # 1. 目的のニューロンかチェック
        if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron)):
            logger.info(f"  - ラッピング対象: {name} ({type(module).__name__})")
            # 2. ラッパーで置き換え
            wrapper = SpQuantWrapper(module)
            setattr(model, name, wrapper)
        # 3. コンテナモジュール（Sequential, ModuleListなど）の場合は再帰
        elif isinstance(module, (nn.ModuleList, nn.Sequential, sj_base.MemoryModule, BaseModel)):
            apply_spquant_quantization(module, inplace=True) # 子モジュールを直接変更
        # 4. SNNCoreの場合は内部モデルを探索
        elif isinstance(module, nn.Module) and hasattr(module, 'model') and isinstance(getattr(module, 'model'), nn.Module):
             apply_spquant_quantization(getattr(module, 'model'), inplace=True)


    logger.info("--- SpQuant-SNN (Stub) 適用完了 ---")
    return model
