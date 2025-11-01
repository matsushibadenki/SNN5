# ファイルパス: snn_research/hardware/compiler.py
# (更新)
#
# Title: ニューロモーフィック・コンパイラ（ハードウェア協調設計 改修版）
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

from typing import Dict, Any, List, cast, Union, Optional, Type, Tuple
import yaml
import time
import os
import torch
import torch.nn as nn
import logging
from collections import OrderedDict

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
                "tau_mem": getattr(module, 'tau_mem', 10.0),
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
             params = { "tau_mem": getattr(module, 'tau_mem', 20.0), "threshold": getattr(module, 'threshold', 1.0), "temperature": getattr(module, 'temperature', 0.5), "noise_intensity": getattr(module, 'noise_intensity', 0.0)}
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
            for name, module in all_modules:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                     first_conn_input_size = cast(int, getattr(module, 'in_features', getattr(module, 'in_channels', 0)))
                     break
        
        input_layer_info: Dict[str, Any]
        if first_conn_input_size > 0:
            input_layer_info = {"neuron_ids": list(range(first_conn_input_size)), "layer_name": "input", "name": "input"}
            layer_map["input"] = input_layer_info
        else:
             logging.warning("Could not determine input layer size.")
             input_layer_info = {"neuron_ids": [], "layer_name": "input", "name": "input"}
             layer_map["input"] = input_layer_info

        # 出力層のニューロン数を推定
        last_conn_output_size = 0
        if isinstance(model, BioSNN):
             last_conn_output_size = model.layer_sizes[-1]
        else:
             for name, module in reversed(all_modules):
                 if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                     last_conn_output_size = cast(int, getattr(module, 'out_features', getattr(module, 'out_channels', 0)))
                     break
        
        output_layer_info: Dict[str, Any]
        if last_conn_output_size > 0:
            output_layer_info = {"neuron_ids": list(range(last_conn_output_size)), "layer_name": "output", "name": "output"} # 仮のID
            layer_map["output"] = output_layer_info
        else:
             logging.warning("Could not determine output layer size.")
             output_layer_info = {"neuron_ids": [], "layer_name": "output", "name": "output"}
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
                        if prev_name in layer_map and layer_map[prev_name].get("type") == "neuron_layer":
                             potential_source_name = prev_name
                             break
                        elif prev_name == 'input':
                             potential_source_name = 'input'
                             break
                    source_module_name = potential_source_name or "input"

                    # 接続先を探す (次のニューロン層 or 出力)
                    potential_target_name: Optional[str] = None
                    for j in range(i + 1, len(all_modules)):
                        next_name, next_module = all_modules[j]
                        if next_name in layer_map and layer_map[next_name].get("type") == "neuron_layer":
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
                    elif isinstance(module, (SpikeDrivenSelfAttention, StandardAttention)):
                         conn_type = "attention"
                         embed_dim_val: Any = getattr(module, 'embed_dim', getattr(module, 'dim', 0))
                         if isinstance(embed_dim_val, int): in_feat = out_feat = embed_dim_val
                         for sub_m in module.modules():
                             if isinstance(sub_m, nn.Linear) and hasattr(sub_m, 'weight') and sub_m.weight is not None:
                                 num_conn += sub_m.weight.numel()

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


    def simulate_on_hardware(self, compiled_config_path: str, total_spikes: int, time_steps: int) -> Dict[str, float]:
        """
        コンパイル済み設定に基づき、ハードウェア上での性能をシミュレートする。
        """
        print(f"\n--- ⚡️ ハードウェアシミュレーション開始 ({self.hardware_profile['name']}) ---")

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