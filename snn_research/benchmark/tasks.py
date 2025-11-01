# ファイルパス: snn_research/benchmark/tasks.py
# Title: ベンチマークタスク定義
# Description: GLUEのSST-2, MRPCや、CIFAR-10など、各種ベンチマークタスクを定義する。
# 修正 (v2): calculate_energy_consumption を EnergyMetrics からインポートするよう修正。

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Callable, Sized, cast, Optional
from datasets import load_dataset  # type: ignore
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
from omegaconf import OmegaConf
from torchvision import datasets, transforms, models # type: ignore

from snn_research.core.snn_core import BreakthroughSNN, SNNCore
from snn_research.benchmark.ann_baseline import ANNBaselineModel, SimpleCNN
# --- ▼ 修正 ▼ ---
from snn_research.benchmark.metrics import calculate_accuracy
from snn_research.metrics.energy import EnergyMetrics # 修正: インポート元を変更
# --- ▲ 修正 ▲ ---

# --- 共通データセットクラス ---
class GenericDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

# --- ベンチマークタスクの基底クラス ---
class BenchmarkTask(ABC):
    """ベンチマークタスクの抽象基底クラス。"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: str, hardware_profile: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.device = device
        self.hardware_profile = hardware_profile

    @abstractmethod
    def prepare_data(self, data_dir: str) -> Tuple[Dataset, Dataset]:
        """データセットを準備し、train/validationのDatasetオブジェクトを返す。"""
        pass

    @abstractmethod
    def get_collate_fn(self) -> Callable:
        """タスク固有のcollate_fnを返す。"""
        pass

    @abstractmethod
    def build_model(self, model_type: str, vocab_size: int) -> nn.Module:
        """タスクに適したSNNまたはANNモデルを構築する。"""
        pass
    
    @abstractmethod
    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        """モデルを評価し、結果を辞書で返す。"""
        pass

# --- GLUE二値分類タスクの基底クラス ---
class _GLUEBinaryClassificationTask(BenchmarkTask):
    """GLUEの二値分類タスク（SST-2, MRPC）のための共通ロジックを持つ基底クラス。"""
    
    task_name: str
    sentence1_key: str
    sentence2_key: Optional[str] = None
    num_labels: int = 2

    def prepare_data(self, data_dir: str = "data") -> Tuple[Dataset, Dataset]:
        os.makedirs(data_dir, exist_ok=True)
        dataset = load_dataset("glue", self.task_name)
        
        def _load_split(split):
            data = []
            for ex in dataset[split]:
                item = {"label": ex['label']}
                if self.sentence2_key:
                    item["text"] = f"{ex[self.sentence1_key]} {self.tokenizer.sep_token} {ex[self.sentence2_key]}"
                else:
                    item["text"] = ex[self.sentence1_key]
                data.append(item)
            return GenericDataset(data)
            
        return _load_split("train"), _load_split("validation")

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch: List[Dict[str, Any]]):
            texts = [item['text'] for item in batch]
            targets = [item['label'] for item in batch]
            tokenized = self.tokenizer(
                texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            return {
                "input_ids": tokenized['input_ids'],
                "attention_mask": tokenized['attention_mask'],
                "labels": torch.tensor(targets, dtype=torch.long)
            }
        return collate_fn

    def build_model(self, model_type: str, vocab_size: int) -> nn.Module:
        class SNNClassifier(nn.Module):
            def __init__(self, snn_backbone, in_features, num_labels):
                super().__init__()
                self.snn_backbone = snn_backbone
                self.classifier = nn.Linear(in_features, num_labels)
            
            def forward(self, input_ids, **kwargs):
                hidden_states, spikes, mem = self.snn_backbone(
                    input_ids, return_spikes=True, output_hidden_states=True
                )
                pooled_output = hidden_states[:, -1, :]
                logits = self.classifier(pooled_output)
                return logits, spikes, mem

        if model_type == 'SNN':
            snn_config = {
                "architecture_type": "spiking_transformer",
                "d_model": 128, "n_head": 4, "num_layers": 4, "time_steps": 128,
                "neuron": {'type': 'lif'}
            }
            backbone = SNNCore(config=OmegaConf.create(snn_config), vocab_size=vocab_size)
            return SNNClassifier(backbone, in_features=128, num_labels=self.num_labels)
        else:
            ann_params = {'d_model': 128, 'd_hid': 256, 'nlayers': 4, 'nhead': 4, 'num_classes': self.num_labels}
            return ANNBaselineModel(vocab_size=vocab_size, **ann_params)

    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        true_labels: List[int] = []
        pred_labels: List[int] = []
        total_spikes = 0.0
        num_neurons: int = cast(int, sum(p.numel() for p in model.parameters()))
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {self.task_name.upper()}"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                targets = inputs.pop("labels")
                
                outputs = model(**inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                spikes = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None else torch.tensor(0.0)

                total_spikes += spikes.sum().item()
                preds = torch.argmax(logits, dim=1)
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())
        
        dataset_size = len(cast(Sized, loader.dataset))
        avg_spikes = total_spikes / dataset_size if total_spikes > 0 else 0.0
        
        # --- ▼ 修正 ▼ ---
        energy_j = EnergyMetrics.calculate_energy_consumption(
            avg_spikes_per_sample=avg_spikes,
            num_neurons=num_neurons, # これは総パラメータ数
            energy_per_synop=self.hardware_profile.get("energy_per_synop", 0.0)
        )
        # --- ▲ 修正 ▲ ---

        return {
            "accuracy": calculate_accuracy(true_labels, pred_labels),
            "avg_spikes": avg_spikes,
            "estimated_energy_j": energy_j,
        }

class SST2Task(_GLUEBinaryClassificationTask):
    """GLUEベンチマークのSST-2 (感情分析) タスク。"""
    task_name = "sst2"
    sentence1_key = "sentence"

class MRPCTask(_GLUEBinaryClassificationTask):
    """GLUEベンチマークのMRPC (類似文判定) タスク。"""
    task_name = "mrpc"
    sentence1_key = "sentence1"
    sentence2_key = "sentence2"

class CIFAR10Task(BenchmarkTask):
    """CIFAR-10画像分類タスク。"""
    def prepare_data(self, data_dir: str = "data") -> Tuple[Dataset, Dataset]:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return train_dataset, val_dataset

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch: List[Tuple[torch.Tensor, int]]):
            images = torch.stack([item[0] for item in batch])
            targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
            return {"input_images": images, "labels": targets}
        return collate_fn

    def build_model(self, model_type: str, vocab_size: int) -> nn.Module:
        num_classes = 10
        
        if model_type == 'SNN':
            snn_config = {
                "architecture_type": "spiking_cnn",
                "time_steps": 16,
                "neuron": {"type": "lif"}
            }
            return SNNCore(config=OmegaConf.create(snn_config), vocab_size=num_classes)
        else: # ANN
            return SimpleCNN(num_classes=num_classes)

    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        true_labels: List[int] = []
        pred_labels: List[int] = []
        total_spikes = 0.0
        num_neurons: int = cast(int, sum(p.numel() for p in model.parameters()))

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating CIFAR-10"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                targets = inputs.pop("labels")
                
                outputs = model(**inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                spikes = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None else torch.tensor(0.0)

                total_spikes += spikes.sum().item()
                preds = torch.argmax(logits, dim=1)
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())

        dataset_size = len(cast(Sized, loader.dataset))
        avg_spikes = total_spikes / dataset_size if total_spikes > 0 else 0.0
        
        # --- ▼ 修正 ▼ ---
        energy_j = EnergyMetrics.calculate_energy_consumption(
            avg_spikes_per_sample=avg_spikes,
            num_neurons=num_neurons, # これは総パラメータ数
            energy_per_synop=self.hardware_profile.get("energy_per_synop", 0.0)
        )
        # --- ▲ 修正 ▲ ---

        return {
            "accuracy": calculate_accuracy(true_labels, pred_labels),
            "avg_spikes": avg_spikes,
            "estimated_energy_j": energy_j,
        }
