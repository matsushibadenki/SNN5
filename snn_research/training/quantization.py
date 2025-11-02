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

import torch
import torch.nn as nn
import copy
# --- ▼ 追加 ▼ ---
from typing import Dict, Any, cast
import logging

from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron
from spikingjelly.activation_based import base as sj_base # type: ignore

logger = logging.getLogger(__name__)
# --- ▲ 追加 ▲ ---


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

@torch.no_grad()
def _quantize_membrane_potential(mem: torch.Tensor) -> torch.Tensor:
    """
    (スタブ) SpQuant-SNN [20] に基づき、膜電位(mem)を3値(-1, 0, +1)に量子化する。
    """
    # ここでは単純な閾値ベースの量子化をスタブとして実装
    # 実際には学習可能な閾値やスケールファクタが必要
    threshold_pos: float = 0.5
    threshold_neg: float = -0.5
    
    mem_quant = torch.zeros_like(mem)
    mem_quant[mem > threshold_pos] = 1.0
    mem_quant[mem < threshold_neg] = -1.0
    return mem_quant

@torch.no_grad()
def _apply_negative_membrane_pruning(mem: torch.Tensor) -> torch.Tensor:
    """
    (スタブ) SpQuant-SNN [20] に基づき、量子化された膜電位の負の値をプルーニング（ゼロ化）する。
    """
    # mem は 3値 (-1, 0, 1) に量子化済みと仮定
    pruned_mem = F.relu(mem) # 負の値を0にする
    return pruned_mem

class SpQuantWrapper(nn.Module):
    """
    SpQuant-SNN [20] の「二重圧縮」をシミュレートするため、
    既存のSNNニューロンをラップするラッパーモジュール (スタブ)。
    """
    def __init__(self, neuron_module: nn.Module):
        super().__init__()
        if not isinstance(neuron_module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron)):
            raise TypeError(f"SpQuantWrapperはLIF, Izhikevich, GLIFニューロンのみラップ可能です。Got: {type(neuron_module)}")
        
        self.neuron: nn.Module = neuron_module
        logger.info(f"SpQuantWrapper: ラッピング -> {type(neuron_module).__name__}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 元のニューロンで膜電位を計算
        # (注: 本来はLIFの内部ロジック (mem = mem * decay + x) に介入する必要がある)
        # (簡易実装: 入力xを膜電位の代理とみなす)
        
        # --- (より正確なスタブ) ---
        # 内部状態 (mem) を取得する必要があるが、forwardの戻り値は (spike, mem)
        # 実行前に mem を取得するのは困難。
        # したがって、このラッパーはニューロンの *入力電流* (x) ではなく、
        # *膜電位* (mem) を量子化する必要がある。
        
        # このスタブ実装では、ニューロンの *出力* である膜電位 (mem_out) を
        # 次のステップの入力として（ダミーで）量子化・プルーニングする
        
        spike, mem_out = self.neuron(x)
        
        # 2. 膜電位を量子化
        mem_quant = _quantize_membrane_potential(mem_out)
        
        # 3. 負の膜電位をプルーニング
        mem_pruned = _apply_negative_membrane_pruning(mem_quant)
        
        # (注: 本来はこの mem_pruned をニューロンの *次のステップ* の状態として
        #  強制的に設定する必要があるが、現在のLIF実装では困難)
        
        # ここでは、元のスパイクと、デバッグ用にプルーニングされた膜電位を返す
        return spike, mem_pruned

    def set_stateful(self, stateful: bool) -> None:
        if hasattr(self.neuron, 'set_stateful'):
            cast(Any, self.neuron).set_stateful(stateful)

    def reset(self) -> None:
        if hasattr(self.neuron, 'reset'):
            cast(Any, self.neuron).reset()

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
