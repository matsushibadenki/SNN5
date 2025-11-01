# ファイルパス: snn_research/training/quantization.py
# (新規作成)
# Title: 量子化認識学習 (Quantization-Aware Training - QAT)
# Description:
# snn_4_ann_parity_plan.mdのStep 3.7に基づき、モデルの量子化を
# 訓練中にシミュレートするための機能を提供する。
# これにより、量子化による精度低下を最小限に抑え、
# ニューロモーフィックハードウェアなど低ビット精度での展開を容易にする。

import torch
import torch.nn as nn
import copy

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