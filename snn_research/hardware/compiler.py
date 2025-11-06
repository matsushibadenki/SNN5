# ファイルパス: snn_research/hardware/compiler.py
# (更新)
#
# Title: ニューロモーフィック・コンパイラ（Lava/SpiNNakerエクスポート強化 v10）
#
# Description:
# - mypyエラーを解消するため、typing.castを使用してモジュールの型を明示的に指定。
# - ROADMAPフェーズ6に基づき、simulate_on_hardwareメソッドを実装。
# - 改善点(v3): 学習則のパラメータもコンパイルしてハードウェア構成に含める機能を追加。
# - 改善点(snn_4_ann_parity_plan): 学習則のシリアライズをより堅牢な方法に変更。
# - 改善点(v5): ニューロンパラメータもハードウェア構成に含めるように修正。
#
# 修正 (v6): NeuromorphicExporterのロジックを統合し、SNNCoreベースのモデルも解析できるようにする。
# 修正 (v7): mypy [name-defined] エラーを解消するため、Tuple をインポート。
# 修正 (v8): 【技術指令】指令1「ハードウェア協調設計」に基づき、
#             ハードウェアプロファイルから物理的制約（量子化ビット数、スパース性）を
#             読み込み、コンパイル設定に出力する機能を追加。
#
# 改善 (v9):
# - ロードマップ P4.2 / P4.3 に基づき、Lava および sPyNNaker への
#   エクスポート用メソッドスタブを追加。 (「実装があまい」点の解消)
#
# 修正 (v10):
# - mypy [name-defined] logger をインポート。
#
# 改善 (v11):
# - P4.2 / P4.3 のスタブ実装を強化。
# - hw_config をパースし、Lava/sPyNNakerの実際のAPI呼び出しを
#   含むスクリプトを生成するように改善。 (「実装があまい」点の解消)
#
# 修正 (v12):
# - mypy [operator] (Tensor not callable) エラーを解消するため、
#   model_to_compile._analyze_model_structure への誤った呼び出しを
#   self._analyze_model_structure に修正。
#
# 修正 (v13):
# - mypy [syntax] (Unmatched '}') エラーを解消するため、
#   ファイル末尾の余分な '}' を削除。
#
# 修正 (v14): SyntaxError: 末尾の余分な '}' を削除。
#
# 修正 (v15): 構文エラー解消のため、クラスを閉じる `}` を末尾に復元
#
# 修正 (v16): 構文エラー解消のため、末尾の '}' を削除。
#
# 修正 (v17): SyntaxError: 末尾の不要な '}' を削除。(v_syn)

from typing import Dict, Any, List, cast, Union, Optional, Type, Tuple
import yaml
import time
import os
import torch
import torch.nn as nn
import logging
from collections import OrderedDict
import re # スクリプト生成のためにインポート

# SNNコアコンポーネントをインポート
from snn_research.core.snn_core import SNNCore
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron, ProbabilisticLIFNeuron
from snn_research.core.base import SNNLayerNorm
from snn_research.core.attention import SpikeDrivenSelfAttention # 旧Attention (必要に応じてMultiLevelに置き換え)
from torch.nn import MultiheadAttention as StandardAttention

from snn_research.bio_models.simple_network import BioSNN
from snn_research.bio_models.lif_neuron import BioLIFNeuron
from snn_research.hardware.profiles import get_hardware_profile
from snn_research.learning_rules.base_rule import BioLearningRule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NeuromorphicCompiler:
    """
    SNNモデル(BioSNNまたはSNNCore)をニューロモーフィックハードウェア用の構成にコンパイルする。
    NeuromorphicExporterの機能を統合。
    """
    def __init__(self, hardware_profile_name: str = "default"):
        """
        Args:
            hardware_profile_name (str): 'profiles.py'で定義されたハードウェアプロファイル名。
        """
        self.hardware_profile = get_hardware_profile(hardware_profile_name)
        print(f"🔩 ニューロモーフィック・コンパイラが初期化されました (ターゲット: {self.hardware_profile['name']})。")

    def _get_neuron_type_and_params(self, module: nn.Module) -> Tuple[str, Dict[str, Any]]:
        """ニューロンモジュールからタイプ名と主要なパラメータを抽出する。"""
        params: Dict[str, Any] = {}
        neuron_type = "Unknown"

        if isinstance(module, AdaptiveLIFNeuron):
            neuron_type = "AdaptiveLIF"
            params = {
                # tau_mem は学習可能な nn.Parameter (log_tau_mem)
                "tau_mem": (torch.exp(module.log_tau_mem.data) + 1.1).mean().item(),
                "base_threshold": getattr(module, 'base_threshold').mean().item() if hasattr(module, 'base_threshold') and isinstance(getattr(module, 'base_threshold'), torch.Tensor) else getattr(module, 'base_threshold', 1.0),
                "adaptation_strength": getattr(module, 'adaptation_strength', 0.1),
                "target_spike_rate": getattr(module, 'target_spike_rate', 0.02),
                "threshold_decay": getattr(module, 'threshold_decay', 0.99),
                "threshold_step": getattr(module, 'threshold_step', 0.05),
                "noise_intensity": getattr(module, 'noise_intensity', 0.0),
            }
        elif isinstance(module, IzhikevichNeuron):
            neuron_type = "Izhikevich"
            params = { "a": getattr(module, 'a', 0.02), "b": getattr(module, 'b', 0.2), "c": getattr(module, 'c', -65.0), "d": getattr(module, 'd', 8.0), "dt": getattr(module, 'dt', 0.5) }
        elif isinstance(module, ProbabilisticLIFNeuron):
             neuron_type = "ProbabilisticLIF"
             params = { 
                 "tau_mem": (torch.exp(module.log_tau_mem.data) + 1.1).mean().item(),
                 "threshold": getattr(module, 'threshold', 1.0), 
                 "temperature": getattr(module, 'temperature', 0.5), 
                 "noise_intensity": getattr(module, 'noise_intensity', 0.0)
             }
        elif isinstance(module, BioLIFNeuron): # BioLIFもニューロン層
             neuron_type = "BioLIF"
             params = {
                 "tau_mem": module.tau_mem,
                 "v_threshold": module.v_thresh,
                 "v_reset": module.v_reset,
                 "v_rest": module.v_rest,
                 "dt": module.dt,
             }

        # パラメータをシリアライズ可能な型に変換
        serializable_params: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                serializable_params[k] = v.tolist()
            elif isinstance(v, (float, int, str, bool)):
                 serializable_params[k] = v
            else:
                 logging.warning(f"Unexpected type {type(v)} for parameter '{k}' in {neuron_type}. Converting to string.")
                 serializable_params[k] = str(v)
        return neuron_type, serializable_params

    def _analyze_model_structure(self, model: nn.Module) -> Dict[str, Any]:
        """
        モデル構造を解析し、ハードウェアマッピングに適した中間表現を生成する（SNNCore, BioSNN対応）。
        """
        structure: Dict[str, Any] = {"layers": [], "connections": [], "summary": {}}
        layer_map: Dict[str, Dict[str, Any]] = OrderedDict() # モジュール名をキーにしたレイヤー情報 (順序保持)
        neuron_count = 0
        connection_count = 0
        layer_index = 0

        # --- モデルの全モジュールをリスト化 ---
        all_modules: List[Tuple[str, nn.Module]] = list(cast(nn.Module, model).named_modules())
        module_dict: Dict[str, nn.Module] = {name: module for name, module in all_modules}

        # --- ニューロン層の解析 ---
        neuron_offset = 0
        for name, module in all_modules:
            is_neuron_layer = False
            num_neurons = 0
            n_type = "Unknown"
            n_params: Dict[str, Any] = {}

            if isinstance(module, (AdaptiveLIFNeuron, IzhikevichNeuron, ProbabilisticLIFNeuron)):
                n_type, n_params = self._get_neuron_type_and_params(module)
                num_neurons_attr = getattr(module, 'features', 0)
                num_neurons = cast(int, num_neurons_attr)
                is_neuron_layer = True
            elif isinstance(module, BioLIFNeuron):
                 n_type, n_params = self._get_neuron_type_and_params(module)
                 num_neurons_attr = getattr(module, 'n_neurons', 0)
                 num_neurons = cast(int, num_neurons_attr)
                 is_neuron_layer = True

            if is_neuron_layer and num_neurons > 0:
                layer_info: Dict[str, Any] = {
                    "name": name,
                    "module_type": type(module).__name__,
                    "type": "neuron_layer",
                    "index": layer_index,
                    "neuron_type": n_type,
                    "num_neurons": num_neurons,
                    "params": n_params,
                    "neuron_ids": list(range(neuron_offset, neuron_offset + num_neurons))
                }
                structure["layers"].append(layer_info)
                layer_map[name] = layer_info
                neuron_count += num_neurons
                layer_index += 1
                neuron_offset += num_neurons

        # --- 接続層の解析 ---
        # 入力層のニューロン数を推定
        first_conn_input_size = 0
        if isinstance(model, BioSNN):
            first_conn_input_size = model.layer_sizes[0]
        else:
            # SNNCoreモデルの入力層 (EmbeddingまたはConv) を探す
            first_module = next(iter(all_modules), None)
            if first_module:
                 name, module = first_module
                 if isinstance(module, nn.Embedding):
                     first_conn_input_size = module.embedding_dim
                 elif isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                     first_conn_input_size = cast(int, getattr(module, 'in_features', getattr(module, 'in_channels', 0)))
            
            if first_conn_input_size == 0:
                 # 見つからない場合はフォールバック
                 for name, module in all_modules:
                     if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                         first_conn_input_size = cast(int, getattr(module, 'in_features', getattr(module, 'in_channels', 0)))
                         break
        
        input_layer_info: Dict[str, Any]
        if first_conn_input_size > 0:
            input_layer_info = {"neuron_ids": list(range(first_conn_input_size)), "layer_name": "input", "name": "input", "num_neurons": first_conn_input_size, "type": "input_layer"}
            layer_map["input"] = input_layer_info
        else:
             logging.warning("Could not determine input layer size.")
             input_layer_info = {"neuron_ids": [], "layer_name": "input", "name": "input", "num_neurons": 0, "type": "input_layer"}
             layer_map["input"] = input_layer_info

        # 出力層のニューロン数を推定
        last_conn_output_size = 0
        if isinstance(model, BioSNN):
             last_conn_output_size = model.layer_sizes[-1]
        else:
             for name, module in reversed(all_modules):
                 if isinstance(module, nn.Linear): # 出力層は通常 Linear
                     last_conn_output_size = cast(int, getattr(module, 'out_features', 0))
                     if last_conn_output_size > 0:
                         break
        
        output_layer_info: Dict[str, Any]
        if last_conn_output_size > 0:
            output_layer_info = {"neuron_ids": list(range(last_conn_output_size)), "layer_name": "output", "name": "output", "num_neurons": last_conn_output_size, "type": "output_layer"}
            layer_map["output"] = output_layer_info
        else:
             logging.warning("Could not determine output layer size.")
             output_layer_info = {"neuron_ids": [], "layer_name": "output", "name": "output", "num_neurons": 0, "type": "output_layer"}
             layer_map["output"] = output_layer_info

        # SNNCoreベースのモデルの接続を解析
        if not isinstance(model, BioSNN):
            for i, (name, module) in enumerate(all_modules):
                is_connection_layer = isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, SpikeDrivenSelfAttention, StandardAttention))
                
                if is_connection_layer:
                    source_module_name: Optional[str] = None
                    target_module_name: Optional[str] = None

                    # 接続元を探す (直前のニューロン層 or 入力)
                    potential_source_name: Optional[str] = None
                    for j in range(i - 1, -1, -1):
                        prev_name, prev_module = all_modules[j]
                        # 接続層 (Linearなど) かニューロン層か入力層を探す
                        if (prev_name in layer_map and layer_map[prev_name].get("type") in ["neuron_layer", "input_layer"]) or \
                           (isinstance(prev_module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)) and prev_name != name):
                             potential_source_name = prev_name
                             break
                    source_module_name = potential_source_name or "input"
                    
                    # 接続先を探す (次のニューロン層 or 接続層 or 出力)
                    potential_target_name: Optional[str] = None
                    for j in range(i + 1, len(all_modules)):
                        next_name, next_module = all_modules[j]
                        if (next_name in layer_map and layer_map[next_name].get("type") in ["neuron_layer", "output_layer"]) or \
                           (isinstance(next_module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and next_name != name):
                            potential_target_name = next_name
                            break
                    target_module_name = potential_target_name or "output"


                    conn_type = "unknown"; in_feat = 0; out_feat = 0; num_conn = 0

                    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                        conn_type = "linear" if isinstance(module, nn.Linear) else "conv"
                        in_val: Any = getattr(module, 'in_features', getattr(module, 'in_channels', 0))
                        in_feat = in_val if isinstance(in_val, int) else 0
                        out_val: Any = getattr(module, 'out_features', getattr(module, 'out_channels', 0))
                        out_feat = out_val if isinstance(out_val, int) else 0
                        if hasattr(module, 'weight') and module.weight is not None:
                             num_conn = module.weight.numel()
                             
                        # 接続層自体も layer_map に（ニューロン層としてではなく）追加
                        if name not in layer_map:
                             layer_map[name] = {
                                 "name": name, "type": "connection_layer", 
                                 "in_features": in_feat, "out_features": out_feat
                             }
                             
                    elif isinstance(module, (SpikeDrivenSelfAttention, StandardAttention)):
                         conn_type = "attention"
                         embed_dim_val: Any = getattr(module, 'embed_dim', getattr(module, 'dim', 0))
                         if isinstance(embed_dim_val, int): in_feat = out_feat = embed_dim_val
                         for sub_m in module.modules():
                             if isinstance(sub_m, nn.Linear) and hasattr(sub_m, 'weight') and sub_m.weight is not None:
                                 num_conn += sub_m.weight.numel()
                         if name not in layer_map:
                             layer_map[name] = {"name": name, "type": "attention_layer"}

                    connection_info: Dict[str, Any] = {
                        "source_module": source_module_name,
                        "target_module": target_module_name,
                        "connection_module_name": name,
                        "type": conn_type, "in_features": in_feat, "out_features": out_feat,
                        "num_connections": num_conn,
                    }
                    structure["connections"].append(connection_info)
                    connection_count += num_conn

        # BioSNN の接続情報を追加
        if isinstance(model, BioSNN):
            for i, weight_matrix in enumerate(model.weights):
                # BioSNNのレイヤー名は 'layers.0', 'layers.1' ...
                pre_layer_name = f"layers.{i-1}" if i > 0 else "input"
                post_layer_name = f"layers.{i}"

                # layer_map から正しい情報を取得
                pre_core_config: Dict[str, Any] = layer_map[pre_layer_name] if pre_layer_name != "input" else input_layer_info
                post_core_config = layer_map.get(post_layer_name)
                
                if post_core_config is None:
                     logging.warning(f"Could not find post-synaptic layer '{post_layer_name}' in layer_map for BioSNN weights.")
                     continue

                pre_core_size: int = len(pre_core_config["neuron_ids"])
                post_core_size: int = len(post_core_config["neuron_ids"])
                
                connection_count_layer: int = 0
                for post_id_local in range(post_core_size):
                     for pre_id_local in range(pre_core_size):
                         weight: float = weight_matrix[post_id_local, pre_id_local].item()
                         if abs(weight) > 1e-9:
                             connection_count_layer += 1

                connection_info = {
                    "source_module": pre_core_config["name"],
                    "target_module": post_core_config["name"],
                    "connection_module_name": f"weights_{i}",
                    "type": "dense", "in_features": pre_core_size, "out_features": post_core_size,
                    "num_connections": connection_count_layer,
                }
                structure["connections"].append(connection_info)
                connection_count += connection_count_layer

        structure["summary"] = {
            "total_neuron_layers": len([l for l in layer_map.values() if l.get("type") == "neuron_layer"]),
            "total_neurons": neuron_count,
            "total_connections": connection_count
        }
        logging.info(f"Analyzed model structure: {structure['summary']}")
        return structure

    def _generate_hardware_config(self, model: nn.Module, target_hardware: str) -> dict:
        """
        解析されたモデル構造に基づいて、ハードウェア構成のデータを生成する。
        """
        analyzed_structure = self._analyze_model_structure(model)
        cores: List[Dict[str, Any]] = []
        connectivity: List[Dict[str, Any]] = []
        core_id_counter = 0

        layer_name_to_core_id: Dict[str, int] = {}
        neuron_layer_infos: List[Dict[str, Any]] = analyzed_structure.get("layers", [])
        for layer_info in neuron_layer_infos:
            layer_name = layer_info.get("name")
            if layer_name:
                core_id = core_id_counter
                core_data: Dict[str, Any] = {
                    "core_id": core_id,
                    "layer_name": layer_name,
                    "neuron_type": layer_info.get("neuron_type", "Unknown"),
                    "num_neurons": layer_info.get("num_neurons", 0),
                    "params": layer_info.get("params", {}),
                }
                if "neuron_ids" in layer_info:
                    core_data["neuron_ids"] = layer_info["neuron_ids"]
                cores.append(core_data)
                layer_name_to_core_id[layer_name] = core_id
                core_id_counter += 1

        connection_infos: List[Dict[str, Any]] = analyzed_structure.get("connections", [])
        for conn in connection_infos:
            source_module_name: Optional[str] = conn.get("source_module")
            target_module_name: Optional[str] = conn.get("target_module")
            connection_module_name: Optional[str] = conn.get("connection_module_name")

            source_core_id: Optional[int] = layer_name_to_core_id.get(source_module_name) if source_module_name is not None and source_module_name != "input" else -1
            target_core_id: Optional[int] = layer_name_to_core_id.get(target_module_name) if target_module_name is not None and target_module_name != "output" else -2

            is_source_valid: bool = source_core_id is not None
            is_target_valid: bool = target_core_id is not None

            if is_source_valid and is_target_valid:
                 is_input_to_layer: bool = (source_core_id == -1 and target_core_id is not None and target_core_id >= 0)
                 is_layer_to_layer: bool = (source_core_id is not None and source_core_id >= 0 and target_core_id is not None and target_core_id >= 0)
                 is_layer_to_output: bool = (source_core_id is not None and source_core_id >= 0 and target_core_id is not None and target_core_id == -2)

                 if is_input_to_layer or is_layer_to_layer or is_layer_to_output:
                    connectivity.append({
                        "source_core": source_core_id,
                        "target_core": target_core_id,
                        "connection_module_name": connection_module_name,
                        "connection_type": conn.get("type", "unknown"),
                        "num_synapses": conn.get("num_connections", 0),
                    })
                 else:
                      logging.warning(f"Skipping potentially invalid connection mapping for module '{connection_module_name}'. Source Core: {source_core_id}, Target Core: {target_core_id}")
            else:
                 logging.warning(f"Could not determine valid connection cores for module '{connection_module_name}'. Source Module: {source_module_name} (Core: {source_core_id}), Target Module: {target_module_name} (Core: {target_core_id})")

        learning_rule_config: Dict[str, Any] = {}
        if isinstance(model, BioSNN) and hasattr(model, 'learning_rule') and isinstance(model.learning_rule, BioLearningRule):
            rule: BioLearningRule = model.learning_rule
            rule_name: str = type(rule).__name__
            rule_params: Dict[str, Any] = {
                key: round(val, 6) if isinstance(val, float) else val
                for key, val in rule.__dict__.items()
                if not key.endswith('_trace') and not isinstance(val, (torch.Tensor, type(None)))
            }
            learning_rule_config = {
                "rule_name": rule_name,
                "parameters": rule_params,
                "enabled_on_hardware": self.hardware_profile.get("supports_on_chip_learning", False)
            }
            logging.info(f"Learning rule '{rule_name}' mapped.")
        else:
             learning_rule_config = { "rule_name": "None", "enabled_on_hardware": False }
             if not isinstance(model, BioSNN):
                 logging.info("Model is not BioSNN, skipping learning rule mapping.")
             else:
                 logging.info("No compatible learning rule found.")
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 指令1: ハードウェア協調設計 - 制約をプロファイルから読み込む
        hw_constraints: Dict[str, Any] = {
            "quantization_bits_activation": self.hardware_profile.get("quantization_bits_activation", 8),
            "quantization_bits_weight": self.hardware_profile.get("quantization_bits_weight", 8),
            "max_connection_sparsity": self.hardware_profile.get("max_connection_sparsity", 1.0),
            "target_synops_per_second": self.hardware_profile.get("ops_per_second", 1e9)
        }
        logging.info(f"Applying hardware constraints: {hw_constraints}")
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        config: Dict[str, Any] = {
            "target_hardware": target_hardware,
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            "compilation_constraints": hw_constraints, # ハードウェア制約を追加
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            "network_summary": analyzed_structure.get("summary", {}),
            "neuron_cores": cores,
            "synaptic_connectivity": connectivity,
            "learning_rule_config": learning_rule_config
        }
        return config


    def compile(self, model: nn.Module, output_path: str) -> None:
        """
        SNNモデル(SNNCoreまたはBioSNN)を解析し、ハードウェア構成ファイルを生成する。
        """
        print(f"⚙️ モデル '{type(model).__name__}' のコンパイルを開始...")

        # SNNCoreやBioSNNでラップされている場合は内部モデルを取得
        model_to_compile: nn.Module
        if isinstance(model, SNNCore) and hasattr(model, 'model'):
            model_to_compile = model.model
        else:
            model_to_compile = model

        config = self._generate_hardware_config(model_to_compile, self.hardware_profile['name'])
        
        # 設定ファイルの保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

        print(f"✅ コンパイル完了。ハードウェア構成を '{output_path}' に保存しました。")

    # --- ▼▼▼ 改善 (v11): P4.2 / P4.3 エクスポートスタブを強化 ▼▼▼ ---

    def _format_lava_neuron_params(self, neuron_type: str, params: Dict[str, Any]) -> str:
        """Lavaのニューロンパラメータ文字列を生成する (スタブ)"""
        if neuron_type == "AdaptiveLIF":
            # LavaのLIFプロセスのパラメータにマッピング
            return f"v_th={params.get('base_threshold', 1.0)}, du=0.0, dv=0.0" # (簡易版)
        elif neuron_type == "Izhikevich":
            # LavaにはIzhikevichの標準プロセスはないため、カスタムプロセスが必要
            return f"# Custom Izhikevich: a={params.get('a')}, b={params.get('b')}, c={params.get('c')}, d={params.get('d')}"
        elif neuron_type == "BioLIF":
            return f"v_th={params.get('v_threshold', 1.0)}, v_reset={params.get('v_reset', 0.0)}"
        return "# Unknown neuron params"

    def export_to_lava(self, model: nn.Module, output_dir: str) -> None:
        """
        (改善 v11) SNNモデルをLavaフレームワーク用の実行可能コードにエクスポートする。
        ロードマップ P4.2 に対応。
        """
        logger.info(f"--- 🌋 Lava Export (Generating Code) ---")
        
        model_to_compile: nn.Module
        if isinstance(model, SNNCore) and hasattr(model, 'model'):
            model_to_compile = model.model
        else:
            model_to_compile = model
            
        hw_config = self._generate_hardware_config(model_to_compile, "Loihi 2")
        
        # --- Lavaプロセス定義の生成 (改善 v11) ---
        lava_code_lines: List[str] = [
            f"# Auto-generated Lava Export (P4.2)",
            f"# Target: {hw_config['target_hardware']}",
            f"# Summary: {hw_config['network_summary']}",
            "",
            "import os",
            "from lava.magma.core.process.process import AbstractProcess",
            "from lava.magma.core.process.ports.ports import InPort, OutPort",
            "from lava.magma.core.run_configs import Loihi2SimCfg",
            "from lava.magma.core.run_conditions import RunSteps",
            "from lava.proc.lif.process import LIF",
            "from lava.proc.io.source import RingBuffer",
            "from lava.proc.io.sink import RingBuffer as Sink",
            "",
            "class SNN5LavaModel(AbstractProcess):",
            "    def __init__(self, **kwargs):",
            "        super().__init__(**kwargs)",
            "        # (Model inputs/outputs defined here)",
            "",
            "def build_lava_network(hw_config):",
            "    network = SNN5LavaModel()",
            "    populations = {}",
            ""
        ]

        # ニューロンコア (Population) の定義
        for core in hw_config.get("neuron_cores", []):
            core_name = re.sub(r'[^a-zA-Z0-9_]', '_', core['layer_name'])
            num_neurons = core['num_neurons']
            neuron_type = core['neuron_type']
            params_str = self._format_lava_neuron_params(neuron_type, core['params'])
            
            # LavaのLIFプロセススタブ
            if "LIF" in neuron_type:
                lava_code_lines.append(f"    # Core {core['core_id']}: {core_name}")
                lava_code_lines.append(f"    populations['{core_name}'] = LIF(shape=({num_neurons},), {params_str})")
            else:
                 lava_code_lines.append(f"    # Core {core['core_id']}: {core_name} (Type: {neuron_type}) - No standard Lava proc, skipping.")

        # 接続 (Connectivity) の定義 (スタブ)
        lava_code_lines.append("\n    # --- Synaptic Connectivity (Stub) ---")
        for conn in hw_config.get("synaptic_connectivity", []):
            src_core = conn['source_core']
            tgt_core = conn['target_core']
            conn_name = conn['connection_module_name']
            lava_code_lines.append(f"    # Connection: {conn_name} (Core {src_core} -> Core {tgt_core})")
            # 実際のLava接続ロジック (例: populations['src'].s_out.connect(populations['tgt'].a_in))
            # は、hw_configのレイヤー名マッピングが完全でないため、ここでは省略

        lava_code_lines.append("\n    return network\n")
        
        # 3. コードの保存
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "lava_model_export.py")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lava_code_lines))
            
        logger.info(f"✅ Lavaエクスポート (コード生成) が完了しました: {output_path}")

    def _format_pynn_neuron_params(self, neuron_type: str, params: Dict[str, Any]) -> str:
        """PyNN (SpiNNaker) のニューロンパラメータ文字列を生成する (スタブ)"""
        if "LIF" in neuron_type:
            # PyNNのIF_curr_expパラメータにマッピング
            return f"""
    'tau_m': {params.get('tau_mem', 10.0)},
    'v_thresh': {params.get('v_threshold', params.get('base_threshold', 1.0))},
    'v_reset': {params.get('v_reset', 0.0)},
    'v_rest': {params.get('v_rest', 0.0)},
    'i_offset': 0.0
"""
        elif neuron_type == "Izhikevich":
            # PyNNのIzhikevichモデル
            return f"""
    'a': {params.get('a', 0.02)},
    'b': {params.get('b', 0.2)},
    'c': {params.get('c', -65.0)},
    'd': {params.get('d', 8.0)}
"""
        return "    # Unknown neuron params"

    def export_to_spinnaker(self, model: nn.Module, output_dir: str) -> None:
        """
        (改善 v11) SNNモデルをSpiNNaker (sPyNNaker) 用の実行可能コードにエクスポートする。
        ロードマップ P4.3 に対応。
        """
        logger.info(f"--- 🕷️ SpiNNaker Export (Generating Code) ---")
        
        model_to_compile: nn.Module
        if isinstance(model, SNNCore) and hasattr(model, 'model'):
            model_to_compile = model.model
        else:
            model_to_compile = model
            
        hw_config = self._generate_hardware_config(model_to_compile, "SpiNNaker")

        # --- sPyNNaker スクリプトの生成 (改善 v11) ---
        spinnaker_code_lines: List[str] = [
            f"# Auto-generated sPyNNaker Export (P4.3)",
            f"# Summary: {hw_config['network_summary']}",
            "",
            "import pyNN.spiNNaker as p",
            "import numpy as np",
            "",
            "p.setup(timestep=1.0)",
            "",
            "populations = {}",
            "projections = {}",
            ""
        ]
        
        # ニューロン (Population) の定義
        spinnaker_code_lines.append("# --- 1. Define Neuron Populations ---")
        
        # 入力層 (SpikeSourceArray)
        input_layer_info = next((l for l in hw_config.get("compilation_constraints", {}).get("layers", []) if l.get("type") == "input_layer"), 
                                hw_config.get("compilation_constraints", {}).get("layers", [{}])[0]) # Fallback
        
        if not input_layer_info:
             # _analyze_model_structure から取得 (v11.1)
             # --- ▼ 修正: model_to_compile._analyze_model_structure を self._analyze_model_structure に変更 ▼ ---
             input_layer_info = next((l for n, l in self._analyze_model_structure(model_to_compile)["layer_map"].items() if l.get("type") == "input_layer"), None)
             # --- ▲ 修正 ▲ ---

        input_neurons = input_layer_info.get('num_neurons', 10) if input_layer_info else 10
        if input_neurons > 0:
            spinnaker_code_lines.append(f"populations['input'] = p.Population({input_neurons}, p.SpikeSourceArray(), label='input_source')")
        else:
            spinnaker_code_lines.append(f"# (Input layer size was 0, skipping population definition)")


        # 隠れ層と出力層 (LIF/Izhikevich)
        for core in hw_config.get("neuron_cores", []):
            core_name = re.sub(r'[^a-zA-Z0-9_]', '_', core['layer_name'])
            num_neurons = core['num_neurons']
            neuron_type = core['neuron_type']
            params_str = self._format_pynn_neuron_params(neuron_type, core['params'])
            
            pynn_model = "p.IF_curr_exp" # デフォルト
            if neuron_type == "Izhikevich":
                pynn_model = "p.Izhikevich"
            
            spinnaker_code_lines.append(f"# Core {core['core_id']}: {core_name}")
            spinnaker_code_lines.append(f"neuron_params_{core_name} = {{")
            spinnaker_code_lines.append(params_str)
            spinnaker_code_lines.append("}")
            spinnaker_code_lines.append(f"populations['{core_name}'] = p.Population({num_neurons}, {pynn_model}(**neuron_params_{core_name}), label='{core_name}')")
            spinnaker_code_lines.append("")
            
        # 出力層 (ダミー、解析が正しければneuron_coresに含まれるはず)
        
        # 接続 (Projection) の定義
        spinnaker_code_lines.append("# --- 2. Define Synaptic Projections ---")
        layer_map = hw_config.get("compilation_constraints", {}).get("layers", []) # Fallback
        if not layer_map:
             # _analyze_model_structure から取得 (v11.1)
             # --- ▼ 修正: model_to_compile._analyze_model_structure を self._analyze_model_structure に変更 ▼ ---
             layer_map = self._analyze_model_structure(model_to_compile)["layer_map"] # type: ignore[assignment]
             # --- ▲ 修正 ▲ ---
             
        # layer_map を {name: info} 辞書に変換
        layer_map_dict: Dict[str, Any] = {l['name']: l for l in layer_map if isinstance(l, dict) and 'name' in l}


        for conn in hw_config.get("synaptic_connectivity", []):
            conn_name = re.sub(r'[^a-zA-Z0-9_]', '_', conn['connection_module_name'])
            source_name_raw = conn['source_module']
            target_name_raw = conn['target_module']
            
            # layer_map からPyNNのラベルを取得
            source_label = layer_map_dict.get(source_name_raw, {}).get('name', 'unknown_src')
            target_label = layer_map_dict.get(target_name_raw, {}).get('name', 'unknown_tgt')

            # 接続タイプ (Connector)
            if conn['type'] in ['linear', 'dense']:
                connector = "p.AllToAllConnector()"
            elif conn['type'] == 'conv':
                # (sPyNNakerのConvは複雑なためスタブ)
                connector = f"p.AllToAllConnector() # (Stub for Conv2D)"
            else:
                connector = "p.AllToAllConnector() # (Stub for Attention/Other)"

            # シナプスタイプ (STDPなど)
            learning_rule_config = hw_config.get("learning_rule_config", {})
            synapse_dynamics = "p.StaticSynapse(weight=0.1, delay=1.0)" # デフォルト
            
            if learning_rule_config.get("enabled_on_hardware") and "STDP" in learning_rule_config.get("rule_name", ""):
                # (R-STDP や TripletSTDP のパラメータをPyNN形式に変換するロジック)
                synapse_dynamics = "p.STDPMechanism(...) # (STDP Stub)"
            
            spinnaker_code_lines.append(f"projections['{conn_name}'] = p.Projection(")
            spinnaker_code_lines.append(f"    populations['{source_label}'],")
            spinnaker_code_lines.append(f"    populations['{target_label}'],")
            spinnaker_code_lines.append(f"    {connector},")
            spinnaker_code_lines.append(f"    synapse_type={synapse_dynamics},")
            spinnaker_code_lines.append(f"    label='{conn_name}'")
            spinnaker_code_lines.append(")\n")
            
        spinnaker_code_lines.append("# --- 3. Run Simulation ---")
        spinnaker_code_lines.append("# p.run(1000)")
        spinnaker_code_lines.append("# p.end()")

        # 3. コードの保存
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "spinnaker_model_export.py")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(spinnaker_code_lines))
            
        logger.info(f"✅ SpiNNakerエクスポート (コード生成) が完了しました: {output_path}")

    # --- ▲▲▲ 改善 (v11) ▲▲▲ ---

    def simulate_on_hardware(self, compiled_config_path: str, total_spikes: int, time_steps: int) -> Dict[str, float]:
        """
        コンパイル済み設定に基づき、ハードウェア上での性能をシミュレートする。
        """
        logger.info(f"\n--- ⚡️ ハードウェアシミュレーション開始 ({self.hardware_profile['name']}) ---")

        if not os.path.exists(compiled_config_path):
            raise FileNotFoundError(f"コンパイル済み設定ファイルが見つかりません: {compiled_config_path}")

        with open(compiled_config_path, 'r') as f:
            config = yaml.safe_load(f)

        num_connections = config.get("network_summary", {}).get("total_connections", 0)
        num_neurons = config.get("network_summary", {}).get("total_neurons", 0)

        energy_per_synop: float = self.hardware_profile['energy_per_synop']
        energy_per_neuron_update: float = self.hardware_profile.get('energy_per_neuron_update', 1e-13) 

        avg_fan_out: float = num_connections / num_neurons if num_neurons > 0 else 100.0

        estimated_energy: float = (total_spikes * avg_fan_out * energy_per_synop) + (num_neurons * time_steps * energy_per_neuron_update)

        print(f"  - 総スパイク数: {total_spikes}")
        print(f"  - 総接続数: {num_connections}")
        print(f"  - 総ニューロン数: {num_neurons}")
        print(f"  - シナプス演算あたりのエネルギー: {energy_per_synop:.2e} J")
        print(f"  - ニューロン更新あたりのエネルギー: {energy_per_neuron_update:.2e} J")
        print(f"  -推定総エネルギー消費: {estimated_energy:.4e} J")

        ops_per_spike: float = avg_fan_out 
        total_ops: float = total_spikes * ops_per_spike + num_neurons * time_steps
        ops_per_second: float = self.hardware_profile.get('ops_per_second', 1e9) 
        parallel_cores: int = self.hardware_profile.get('parallel_cores', 128) 

        estimated_time_sec: float = total_ops / (ops_per_second * parallel_cores)
        estimated_time_ms: float = estimated_time_sec * 1000

        print(f"  - タイムステップ数: {time_steps}")
        print(f"  - 総演算数 (推定): {total_ops:.2e}")
        print(f"  - 並列秒間演算能力 (推定): {ops_per_second * parallel_cores:.2e} Ops/sec")
        print(f"  - 推定処理時間: {estimated_time_ms:.4f} ms")

        report: Dict[str, float] = {
            "estimated_energy_joules": estimated_energy,
            "estimated_processing_time_ms": estimated_time_ms,
            "total_spikes_simulated": float(total_spikes), 
            "total_operations_estimated": total_ops
        }
        print("--- ✅ シミュレーション完了 ---")
        return report
