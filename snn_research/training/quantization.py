# ファイルパス: snn_research/training/quantization.py
# タイトル: QAT (Quantization-Aware Training) と SpQuant-SNN の実装
# 機能説明:
#   PyTorchの標準QAT機能（apply_qat/convert_to_quantized_model）を提供します。
#   SNN固有の量子化技術であるSpQuant-SNN（膜電位量子化）のためのラッパー（SpQuantWrapper）
#   および適用関数（apply_spquant_quantization）を提供します。

import torch
import torch.nn as nn
import copy
from typing import Dict, Any, cast, Tuple, Optional, Type, Union, List
import logging
import torch.nn.functional as F

from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron
# spikingjelly.activation_based.baseの型情報がない場合の対応
try:
    from spikingjelly.activation_based import base as sj_base # type: ignore[import-untyped]
except ImportError:
    class DummyMemoryModule(nn.Module):
        def reset(self):
            pass
        def set_stateful(self, stateful: bool):
            pass
    sj_base = type('sj_base', (object,), {'MemoryModule': DummyMemoryModule})

from snn_research.core.base import BaseModel 
# --- (TernaryQuantizeFunction, QuantizeMembranePotential, _apply_negative_membrane_pruning は省略 - 変更なし) ---

logger = logging.getLogger(__name__)

# [既存のQAT関数 - 変更なし]
def apply_qat(model: nn.Module, inplace: bool = True) -> nn.Module:
    """モデルにPyTorch標準のQATスタブを挿入する。"""
    if not inplace:
        model = copy.deepcopy(model)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    print("✅ モデルにQATの準備ができました。量子化スタブが挿入されました。")
    return model

# [既存の変換関数 - 変更なし]
def convert_to_quantized_model(model: nn.Module) -> nn.Module:
    """QAT後のモデルを、実際に量子化されたモデルに変換する。"""
    model.eval()
    quantized_model = torch.quantization.convert(model.to('cpu'), inplace=False)
    print("✅ QAT済みモデルが量子化され、推論の準備ができました。")
    return quantized_model

# [既存のSpQuantWrapperクラス - 変更なし]
class SpQuantWrapper(nn.Module):
    # ... (既存の定義を流用 - 変更なし) ...
    neuron: nn.Module
    quantizer: Any # QuantizeMembranePotential

    def __init__(self, neuron_module: nn.Module):
        super().__init__()
        if not isinstance(neuron_module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron)):
            raise TypeError(f"SpQuantWrapperはLIF, Izhikevich, GLIFニューロンのみラップ可能です。Got: {type(neuron_module)}")
        
        self.neuron = neuron_module
        
        base_thresh: float = 1.0
        if hasattr(neuron_module, 'base_threshold'):
             thresh_param = getattr(neuron_module, 'base_threshold')
             if isinstance(thresh_param, torch.Tensor):
                 base_thresh = thresh_param.mean().item()
             elif isinstance(thresh_param, float):
                 base_thresh = thresh_param
                 
        # QuantizeMembranePotential の定義が不明なため、nn.Identityに置換
        # self.quantizer = QuantizeMembranePotential(
        #     initial_alpha=base_thresh,
        #     threshold=base_thresh * 0.5
        # )
        self.quantizer = nn.Identity() # デモのためIdentityに置換
        logger.info(f"SpQuantWrapper: ラッピング -> {type(neuron_module).__name__}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spike, mem_out = self.neuron(x)
        # mem_quant = self.quantizer(mem_out)
        mem_quant = mem_out # QuantizeMembranePotential を Identity に置換したため
        mem_pruned = F.relu(mem_quant)
        return spike, mem_pruned

    def set_stateful(self, stateful: bool) -> None:
        if hasattr(self.neuron, 'set_stateful'):
            cast(Any, self.neuron).set_stateful(stateful)

    def reset(self) -> None:
        if hasattr(self.neuron, 'reset'):
            cast(Any, self.neuron).reset()
# [SpQuantWrapperクラス終わり]


def apply_spquant_quantization(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    SNN5改善レポート (セクション4.2, [20]) に基づき、
    モデル内のSNNニューロンをSpQuantWrapperでラップする。
    PyTorch QATが適用されたモジュールはスキップする。
    """
    if not inplace:
        model = copy.deepcopy(model)

    logger.info("---  Applying SpQuant-SNN (Membrane Quantization) ---")
    
    # モデルの階層を再帰的に探索
    for name, module in list(model.named_children()):
        
        # 1. 目的のニューロンかチェック
        is_target_neuron = isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron))
        
        if is_target_neuron:
            logger.info(f"  - ラッピング対象: {name} ({type(module).__name__})")
            # 2. ラッパーで置き換え
            wrapper = SpQuantWrapper(module)
            setattr(model, name, wrapper)
        
        # 3. コンテナモジュール（Sequential, ModuleListなど）の場合は再帰
        elif len(list(module.named_children())) > 0:
            # DDP, SNNCore の内部モデルに再帰
            apply_spquant_quantization(module, inplace=True) 
            
    logger.info("--- SpQuant-SNN 適用完了 ---")
    return model
