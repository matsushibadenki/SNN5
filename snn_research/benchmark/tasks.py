# ファイルパス: snn_research/benchmark/tasks.py
# Title: ベンチマークタスク定義
# Description: GLUEのSST-2, MRPCや、CIFAR-10など、各種ベンチマークタスクを定義する。
# 修正 (v2): calculate_energy_consumption を EnergyMetrics からインポートするよう修正。
#
# 改善 (v3):
# - doc/SNN開発：基本設計思想.md (セクション7.1) に基づき、
#   ニューロモーフィック・データセット (CIFAR10-DVS) を扱う
#   CIFAR10DVSTask を SpikingJelly を利用して追加。
# - mypy [name-defined] [import-untyped] エラーを修正。
#
# 改善 (v4):
# - doc/SNN開発：SNN5プロジェクト改善のための情報収集.md (セクション6.1, 6.2) に基づき、
#   SHD (Spiking Heidelberg Digits) タスクを SpikingJelly を利用して追加。
#
# 修正 (v_hpo_fix_type_error_v2):
# - HPO (run_distillation.py) から 'img_size' を渡されたときに
#   TypeError が発生する問題を解消するため、CIFAR10Task に __init__ を追加。
# - prepare_data がハードコードされた 224x224 ではなく、
#   __init__ で渡された img_size (32x32) を使用するように修正。

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Callable, Sized, cast, Optional
from datasets import load_dataset  # type: ignore[import-untyped]
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
from omegaconf import OmegaConf
from torchvision import datasets, transforms # type: ignore[import-untyped]
# --- ▼ 修正 ▼ ---
from spikingjelly.activation_based import functional as SJ_F # type: ignore[import-untyped]
from spikingjelly.datasets import cifar10_dvs, shd # type: ignore[import-untyped]
# --- ▲ 修正 ▲ ---

from snn_research.core.snn_core import BreakthroughSNN, SNNCore
from snn_research.benchmark.ann_baseline import ANNBaselineModel, SimpleCNN
from snn_research.benchmark.metrics import calculate_accuracy
from snn_research.metrics.energy import EnergyMetrics 

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
        dataset = load_dataset("glue", self.task_name, cache_dir=data_dir) # cache_dir を指定
        
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
                # SNN (SNNCore) は (logits, spikes, mem) を返す
                # タイムステップループは SNNCore 内部で行われる
                outputs = self.snn_backbone(
                    input_ids, return_spikes=True, output_hidden_states=True, **kwargs
                )
                hidden_states, spikes, mem = outputs
                
                # プーリング (最後のトークンの隠れ状態を使用)
                pooled_output = hidden_states[:, -1, :]
                logits = self.classifier(pooled_output)
                return logits, spikes, mem

        if model_type == 'SNN':
            snn_config_dict = {
                "architecture_type": "spiking_transformer",
                "d_model": 128, "n_head": 4, "num_layers": 4, "time_steps": 128, # time_stepsはSNN内部で使用
                "neuron": {'type': 'lif'}
            }
            snn_config = OmegaConf.create({"model": snn_config_dict}) # SNNCoreが期待する形式
            backbone = SNNCore(config=snn_config.model, vocab_size=vocab_size)
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
        
        energy_j = EnergyMetrics.calculate_energy_consumption(
            avg_spikes_per_sample=avg_spikes,
            num_neurons=num_neurons, # これは総パラメータ数
            energy_per_synop=self.hardware_profile.get("energy_per_synop", 0.0)
        )

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

    # --- ▼ 修正 (v_hpo_fix_type_error_v2) ▼ ---
    # img_size を __init__ で受け取れるようにし、
    # prepare_data でハードコードされていた 224x224 を置き換える
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        device: str, 
        hardware_profile: Dict[str, Any],
        img_size: int = 224 # デフォルトは 224
    ):
        super().__init__(tokenizer, device, hardware_profile)
        self.img_size = img_size
        if img_size != 224:
            print(f"INFO (CIFAR10Task): Using custom img_size: {self.img_size}")
    # --- ▲ 修正 (v_hpo_fix_type_error_v2) ▲ ---

    def prepare_data(self, data_dir: str = "data") -> Tuple[Dataset, Dataset]:
        transform = transforms.Compose([
            # --- ▼ 修正 (v_hpo_fix_type_error_v2) ▼ ---
            transforms.Resize((self.img_size, self.img_size)), # ハードコード(224)を修正
            # --- ▲ 修正 (v_hpo_fix_type_error_v2) ▲ ---
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
            snn_config_dict = {
                "architecture_type": "spiking_cnn",
                "time_steps": 16,
                "neuron": {"type": "lif"}
            }
            snn_config = OmegaConf.create({"model": snn_config_dict})
            return SNNCore(config=snn_config.model, vocab_size=num_classes)
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
        
        energy_j = EnergyMetrics.calculate_energy_consumption(
            avg_spikes_per_sample=avg_spikes,
            num_neurons=num_neurons, # これは総パラメータ数
            energy_per_synop=self.hardware_profile.get("energy_per_synop", 0.0)
        )

        return {
            "accuracy": calculate_accuracy(true_labels, pred_labels),
            "avg_spikes": avg_spikes,
            "estimated_energy_j": energy_j,
        }

# --- ▼ 修正: CIFAR10DVSTask を追加 ▼ ---
class CIFAR10DVSTask(BenchmarkTask):
    """
    CIFAR10-DVS（ニューロモーフィック）画像分類タスク。
    設計思想.md (セクション7.1) に基づく。
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: str, hardware_profile: Dict[str, Any], time_steps: int = 16):
        super().__init__(tokenizer, device, hardware_profile)
        self.time_steps = time_steps

    def prepare_data(self, data_dir: str = "data") -> Tuple[Dataset, Dataset]:
        # SpikingJellyのデータセットローダーを使用
        # CIFAR10-DVSは時間ステップ (T) が可変長
        # ここでは固定長 (self.time_steps) にリサンプル（またはパディング）する前処理を定義
        
        # 空間的な前処理 (SpikingCNNが224x224を期待する場合)
        spatial_transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        
        # 時間的な前処理 (固定長 T に変換)
        def temporal_transform(x: torch.Tensor) -> torch.Tensor:
            # x は (T_orig, C, H, W)
            T_orig = x.shape[0]
            if T_orig > self.time_steps:
                # ダウンサンプリング
                indices = torch.linspace(0, T_orig - 1, self.time_steps).long()
                x = x[indices]
            elif T_orig < self.time_steps:
                # パディング
                padding = torch.zeros(self.time_steps - T_orig, *x.shape[1:], dtype=x.dtype)
                x = torch.cat([x, padding], dim=0)
            return x # (T, C, H, W)

        # SpikingJellyのCIFAR10DVSデータセット
        train_dataset = cifar10_dvs.CIFAR10DVS(
            root=data_dir,
            train=True,
            data_type='frame', # 'frame' (時間ビン) または 'event'
            frames_number=self.time_steps, # フレーム数 (T)
            split_by='number',
            transform=spatial_transform,
            target_transform=None
        )
        val_dataset = cifar10_dvs.CIFAR10DVS(
            root=data_dir,
            train=False,
            data_type='frame',
            frames_number=self.time_steps,
            split_by='number',
            transform=spatial_transform,
            target_transform=None
        )
        return train_dataset, val_dataset

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch: List[Tuple[torch.Tensor, int]]):
            # batch[i][0] は (T, C, H, W)
            # SpikingCNN (snn_core.py) は (B, C, H, W) を入力とし、
            # 内部で T (time_steps) ループを回す
            
            # ここでは、SpikingJellyの流儀に従い、
            # (T, B, C, H, W) の形状でモデルに渡すことを試みる
            
            # --- または、SpikingCNNの (B, C, H, W) 入力に合わせる ---
            # (T, B, C, H, W) -> (B, T, C, H, W) に変換
            frames = torch.stack([item[0] for item in batch]).permute(1, 0, 2, 3, 4) # (T, B, C, H, W) -> (B, T, C, H, W)
            
            # SpikingCNNは (B, C, H, W) を期待し、内部でT回ループする
            # だが、DVSデータは (B, T, C, H, W) の時系列データそのもの
            
            # --- 回避策: SpikingCNNの入力 (B, C, H, W) に合わせる ---
            # (B, T, C, H, W) の時間軸を平均化して (B, C, H, W) にする
            images = frames.mean(dim=1) # (B, C, H, W)
            
            targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
            # SpikingCNNが期待するキー 'input_images' で返す
            return {"input_images": images, "labels": targets}
        return collate_fn

    def build_model(self, model_type: str, vocab_size: int) -> nn.Module:
        # CIFAR10-DVS は 10 クラス
        num_classes = 10
        
        if model_type == 'SNN':
            snn_config_dict = {
                "architecture_type": "spiking_cnn",
                "time_steps": self.time_steps, # DVSデータのTと合わせる
                "neuron": {"type": "lif"}
            }
            snn_config = OmegaConf.create({"model": snn_config_dict})
            # vocab_size は num_classes で上書き
            return SNNCore(config=snn_config.model, vocab_size=num_classes)
        else: # ANN
            # DVSデータは時間軸を持つため、ANNベースラインは本来 (3D-CNN or RNN) であるべき
            # ここでは静止画CIFAR10と同じ SimpleCNN を流用（時間軸は平均化）
            return SimpleCNN(num_classes=num_classes)

    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        # (CIFAR10Taskと同じ評価ロジックを流用)
        model.eval()
        true_labels: List[int] = []
        pred_labels: List[int] = []
        total_spikes = 0.0
        num_neurons: int = cast(int, sum(p.numel() for p in model.parameters()))
        
        # DVSデータセットの評価では、モデルのリセットが重要
        SJ_F.reset_net(model)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating CIFAR10-DVS"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                targets = inputs.pop("labels")
                
                # inputs['input_images'] は (B, C, H, W) (時間平均済み)
                outputs = model(**inputs) # SpikingCNNが内部で T ループ
                
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                spikes = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None else torch.tensor(0.0)

                total_spikes += spikes.sum().item()
                preds = torch.argmax(logits, dim=1)
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())
        
        dataset_size = len(cast(Sized, loader.dataset))
        # avg_spikes は (総スパイク数 / (サンプル数 * T)) ではなく、
        # モデルが内部Tステップで処理した「サンプルあたりの総スパイク数」
        avg_spikes = total_spikes / dataset_size if total_spikes > 0 else 0.0
        
        energy_j = EnergyMetrics.calculate_energy_consumption(
            avg_spikes_per_sample=avg_spikes, # サンプルあたりの総スパイク数
            num_neurons=num_neurons,
            energy_per_synop=self.hardware_profile.get("energy_per_synop", 0.0)
        )

        return {
            "accuracy": calculate_accuracy(true_labels, pred_labels),
            "avg_spikes": avg_spikes,
            "estimated_energy_j": energy_j,
        }
# --- ▲ 修正 ▲ ---

# --- ▼ 改善 (v4): SHDTask を追加 ▼ ---
class SHDTask(BenchmarkTask):
    """
    SHD (Spiking Heidelberg Digits) オーディオ分類タスク。
    doc/SNN開発：SNN5プロジェクト改善のための情報収集.md (セクション6.1) に基づく。
    TSkipsSNN や SpikingSSM の評価に最適。
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: str, hardware_profile: Dict[str, Any], time_steps: int = 100):
        super().__init__(tokenizer, device, hardware_profile)
        self.time_steps = time_steps # SHDのデータは可変長だが、ここでは固定長にリサンプル
        self.num_classes = 20 # SHDは20クラス (0-9のドイツ語読み)
        self.input_features = 700 # SHDの入力特徴量

    def prepare_data(self, data_dir: str = "data") -> Tuple[Dataset, Dataset]:
        
        # 時間的な前処理 (固定長 T に変換)
        def temporal_transform(x: torch.Tensor) -> torch.Tensor:
            # x は (T_orig, C=700)
            T_orig = x.shape[0]
            if T_orig > self.time_steps:
                indices = torch.linspace(0, T_orig - 1, self.time_steps).long()
                x = x[indices]
            elif T_orig < self.time_steps:
                padding = torch.zeros(self.time_steps - T_orig, *x.shape[1:], dtype=x.dtype)
                x = torch.cat([x, padding], dim=0)
            return x # (T, C=700)

        # SpikingJellyのSHDデータセット
        train_dataset = shd.SHD(
            root=data_dir,
            train=True,
            data_type='frame',
            frames_number=self.time_steps, # フレーム数 (T)
            split_by='number',
            transform=temporal_transform, # 時間軸の変形のみ
            target_transform=None
        )
        val_dataset = shd.SHD(
            root=data_dir,
            train=False,
            data_type='frame',
            frames_number=self.time_steps,
            split_by='number',
            transform=temporal_transform,
            target_transform=None
        )
        return train_dataset, val_dataset

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch: List[Tuple[torch.Tensor, int]]):
            # batch[i][0] は (T, C=700)
            # モデル (例: TSkipsSNN) は (B, T, F_in) を期待する
            frames = torch.stack([item[0] for item in batch]) # (B, T, C=700)
            targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
            
            # TSkipsSNNが期待するキー 'input_sequence' で返す
            return {"input_sequence": frames, "labels": targets}
        return collate_fn

    def build_model(self, model_type: str, vocab_size: int) -> nn.Module:
        # vocab_size は num_classes で上書き
        num_classes = self.num_classes
        
        if model_type == 'SNN':
            # SHDタスクには TSkipsSNN や SpikingSSM が適している
            # ここでは tskips_snn をデフォルトとして構築
            snn_config_dict = {
                "architecture_type": "tskips_snn",
                "input_features": self.input_features,
                "hidden_features": 256,
                "num_layers": 3,
                "time_steps": self.time_steps,
                "forward_delays_per_layer": [[1, 2], [1, 2], [1, 2]],
                "backward_delays_per_layer": [[1], [1, 2], [1, 2]],
                "neuron": {"type": "lif"}
            }
            snn_config = OmegaConf.create({"model": snn_config_dict})
            return SNNCore(config=snn_config.model, vocab_size=num_classes)
        else: # ANN
            # ANNベースライン (例: LSTM or GRU)
            class ANN_RNN_Baseline(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
                    super().__init__()
                    self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_dim, num_classes)
                
                def forward(self, input_sequence, **kwargs):
                    # input_sequence: (B, T, F_in)
                    rnn_out, _ = self.rnn(input_sequence)
                    # 最後のタイムステップの出力をプーリング
                    pooled = rnn_out[:, -1, :]
                    logits = self.fc(pooled)
                    return logits, None, None # SNN互換のタプル
            
            return ANN_RNN_Baseline(
                input_dim=self.input_features,
                hidden_dim=256,
                num_layers=3,
                num_classes=self.num_classes
            )

    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        # (CIFAR10Taskの評価ロジックとほぼ同じだが、入力キーが異なる)
        model.eval()
        true_labels: List[int] = []
        pred_labels: List[int] = []
        total_spikes = 0.0
        num_neurons: int = cast(int, sum(p.numel() for p in model.parameters()))
        
        SJ_F.reset_net(model) # 時系列モデルのためリセット

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating SHD"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                targets = inputs.pop("labels")
                
                # inputs['input_sequence'] は (B, T, C=700)
                outputs = model(**inputs) # TSkipsSNNが内部で T ループ
                
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                spikes = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None else torch.tensor(0.0)

                total_spikes += spikes.sum().item()
                preds = torch.argmax(logits, dim=1)
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())
        
        dataset_size = len(cast(Sized, loader.dataset))
        # avg_spikes は (総スパイク数 / (サンプル数 * T)) ではなく、
        # モデルが内部Tステップで処理した「サンプルあたりの総スパイク数」
        avg_spikes = total_spikes / dataset_size if total_spikes > 0 else 0.0
        
        energy_j = EnergyMetrics.calculate_energy_consumption(
            avg_spikes_per_sample=avg_spikes, # サンプルあたりの総スパイク数
            num_neurons=num_neurons,
            energy_per_synop=self.hardware_profile.get("energy_per_synop", 0.0)
        )

        return {
            "accuracy": calculate_accuracy(true_labels, pred_labels),
            "avg_spikes": avg_spikes,
            "estimated_energy_j": energy_j,
        }
# --- ▲ 改善 (v4) ▲ ---