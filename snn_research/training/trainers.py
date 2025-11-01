# ファイルパス: snn_research/training/trainers.py
# (更新)
# (省略...)
# 修正点(v6): 継続学習(EWC)のためのFisher行列計算・保存機能を追加。
# 改善点(snn_4_ann_parity_plan): EWCデータのロード機能を追加。
# 修正点(TCL): return_full_hiddensフラグの伝搬とSelfSupervisedTrainerの修正。
# 修正点(mypy): Tensor型とfloat型の代入不一致エラーを解消。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import os
import collections
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional, cast
import shutil
import time
from torch.optim import Adam
from spikingjelly.activation_based import functional # type: ignore
from pathlib import Path

from snn_research.training.losses import CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss, PlannerLoss, ProbabilisticEnsembleLoss
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from torch.utils.tensorboard import SummaryWriter
from snn_research.visualization.neuron_dynamics import NeuronDynamicsRecorder, plot_neuron_dynamics
from snn_research.core.neurons import AdaptiveLIFNeuron

from snn_research.bio_models.simple_network import BioSNN
import copy


class BreakthroughTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], device: str,
                 grad_clip_norm: float, rank: int, use_amp: bool, log_dir: str,
                 astrocyte_network: Optional[AstrocyteNetwork] = None,
                 meta_cognitive_snn: Optional[MetaCognitiveSNN] = None,
                 enable_visualization: bool = True): # 可視化フラグを追加
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.grad_clip_norm = grad_clip_norm
        self.rank = rank
        self.use_amp = use_amp and self.device != 'mps'
        self.astrocyte_network = astrocyte_network
        self.meta_cognitive_snn = meta_cognitive_snn
        
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.best_metric = float('inf')
        
        if self.rank in [-1, 0]:
            self.writer = SummaryWriter(log_dir)
            print(f"✅ TensorBoard logging enabled. Log directory: {log_dir}")

        self.enable_visualization = enable_visualization
        if self.enable_visualization and self.rank in [-1, 0]:
            self.recorder = NeuronDynamicsRecorder(max_timesteps=100)
    
    def load_ewc_data(self, path: str):
        """事前計算されたFisher行列と最適パラメータをEWCのためにロードする。"""
        if not os.path.exists(path):
            print(f"⚠️ EWCデータファイルが見つかりません: {path}。EWCなしで学習を開始します。")
            return

        ewc_data = torch.load(path, map_location=self.device)
        if isinstance(self.criterion, CombinedLoss):
            self.criterion.fisher_matrix = ewc_data['fisher_matrix']
            self.criterion.optimal_params = ewc_data['optimal_params']
            print(f"✅ EWCデータを '{path}' からロードしました。")
        else:
            print("⚠️ 警告: EWCデータはロードされましたが、現在の損失関数はCombinedLossではありません。EWCは適用されません。")


    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        functional.reset_net(self.model)
        start_time = time.time()
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        
        hooks = []
        if not is_train and self.enable_visualization and self.rank in [-1, 0] and hasattr(self, 'recorder'):
            self.recorder.clear()
            model_to_run = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            
            def record_hook(module, input, output):
                spike, mem = output
                if hasattr(module, 'adaptive_threshold') and module.adaptive_threshold is not None:
                    threshold = module.adaptive_threshold
                else:
                    threshold = module.base_threshold.unsqueeze(0).expand_as(mem)

                self.recorder.record(
                    membrane=mem[0:1].detach(), 
                    threshold=threshold[0:1].detach(), 
                    spikes=spike[0:1].detach()
                )

            for module in model_to_run.modules():
                if isinstance(module, AdaptiveLIFNeuron):
                    hooks.append(module.register_forward_hook(record_hook))
                    break 
        
        # SelfSupervisedLoss の場合は full_hiddens を要求
        return_full_hiddens_flag = isinstance(self.criterion, SelfSupervisedLoss)
        
        with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                # outputs: (logits/hiddens, spikes, mem) のタプルを期待
                outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=return_full_hiddens_flag)
                
                logits_or_hiddens, spikes, mem = outputs
                
                if return_full_hiddens_flag:
                    # SelfSupervisedLossの場合は、hiddensを最初の引数として渡す
                    loss_dict = self.criterion(logits_or_hiddens, target_ids, spikes, mem, self.model)
                    logits = None # ロジットは存在しない
                else:
                    # 通常の損失の場合は、logitsを最初の引数として渡す
                    logits = logits_or_hiddens
                    loss_dict = self.criterion(logits, target_ids, spikes, mem, self.model)
        
        for hook in hooks:
            hook.remove()

        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            if self.meta_cognitive_snn:
                end_time = time.time()
                computation_time = end_time - start_time
                
                accuracy_val = 0.0
                if logits is not None:
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=-1)
                        if hasattr(self.criterion, 'ce_loss_fn') and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                            ignore_idx = self.criterion.ce_loss_fn.ignore_index
                            mask = target_ids != ignore_idx
                            num_masked_elements = cast(torch.Tensor, mask).sum()
                            accuracy_tensor = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                            accuracy_val = accuracy_tensor.item()
                            loss_dict['accuracy'] = accuracy_tensor # Tensorを保存
                    
                self.meta_cognitive_snn.update_metadata(
                    loss=loss_dict['total'].item(),
                    computation_time=computation_time,
                    accuracy=accuracy_val # floatを渡す
                )
        else:
            with torch.no_grad():
                accuracy_tensor = torch.tensor(0.0, device=self.device)
                if logits is not None:
                    if 'accuracy' not in loss_dict:
                        preds = torch.argmax(logits, dim=-1)
                        if hasattr(self.criterion, 'ce_loss_fn') and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                            ignore_idx = self.criterion.ce_loss_fn.ignore_index
                            mask = target_ids != ignore_idx
                            num_masked_elements = cast(torch.Tensor, mask).sum()
                            accuracy_tensor = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                            loss_dict['accuracy'] = accuracy_tensor # Tensorを保存
                else:
                    loss_dict['accuracy'] = accuracy_tensor # float(0.0)をTensorとして保存
                
                # floatに変換してから返す (最後のreturn文で自動変換される)
                if 'accuracy' in loss_dict and isinstance(loss_dict['accuracy'], torch.Tensor):
                    pass
                else:
                    loss_dict['accuracy'] = torch.tensor(loss_dict.get('accuracy', 0.0), device=self.device) # 安全なTensor化
        
        # 最終的な返却値ではTensorは.item()でfloatに変換される
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        self.model.train()
        for batch in progress_bar:
            metrics = self._run_step(batch, is_train=True)
            for key, value in metrics.items(): total_metrics[key] += value
            progress_bar.set_postfix({k: v / (progress_bar.n + 1) for k, v in total_metrics.items()})

        if self.scheduler:
            self.scheduler.step()
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        if self.rank in [-1, 0]:
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            if self.scheduler:
                self.writer.add_scalar('Train/learning_rate', self.scheduler.get_last_lr()[0], epoch)
            else:
                self.writer.add_scalar('Train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        return avg_metrics

    def evaluate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                metrics = self._run_step(batch, is_train=False)
                for key, value in metrics.items(): total_metrics[key] += value
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        if self.rank in [-1, 0]:
            print(f"Epoch {epoch} Validation Results: " + ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)
            
            if self.enable_visualization and hasattr(self, 'recorder') and self.recorder.history['membrane']:
                try:
                    save_path = Path(self.writer.log_dir) / f"neuron_dynamics_epoch_{epoch}.png"
                    plot_neuron_dynamics(self.recorder.history, save_path=save_path)
                    print(f"📊 Neuron dynamics visualization saved to {save_path}")
                except Exception as e:
                    print(f"⚠️ Failed to generate neuron dynamics plot: {e}")

        return avg_metrics

    def save_checkpoint(self, path: str, epoch: int, metric_value: float, **kwargs: Any):
        if self.rank in [-1, 0]:
            model_to_save_container = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            actual_model = cast(nn.Module, model_to_save_container.model if hasattr(model_to_save_container, 'model') else model_to_save_container)
            
            buffers_to_exclude = {
                name for name, buf in actual_model.named_buffers() 
                if any(keyword in name for keyword in ['mem', 'spikes', 'adaptive_threshold'])
            }
            model_state = {k: v for k, v in actual_model.state_dict().items() if k not in buffers_to_exclude}

            state = {
                'epoch': epoch, 'model_state_dict': model_state, 
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_metric': self.best_metric
            }
            if self.use_amp: state['scaler_state_dict'] = self.scaler.state_dict()
            if self.scheduler: state['scheduler_state_dict'] = self.scheduler.state_dict()
            state.update(kwargs)
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, path)
            print(f"✅ チェックポイントを '{path}' に保存しました (Epoch: {epoch})。")
            
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
                temp_state_for_best = {'model_state_dict': model_state, **kwargs}
                torch.save(temp_state_for_best, best_path)
                print(f"🏆 新しいベストモデルを '{best_path}' に保存しました (Metric: {metric_value:.4f})。")

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            print(f"⚠️ チェックポイントファイルが見つかりません: {path}。最初から学習を開始します。")
            return 0
            
        checkpoint = torch.load(path, map_location=self.device)
        model_to_load_container = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        actual_model = cast(nn.Module, model_to_load_container.model if hasattr(model_to_load_container, 'model') else model_to_load_container)
        actual_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if 'optimizer_state_dict' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in checkpoint: self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_metric = checkpoint.get('best_metric', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✅ チェックポイント '{path}' を正常にロードしました。Epoch {start_epoch} から学習を再開します。")
        return start_epoch
    
    def _compute_ewc_fisher_matrix(self, dataloader: DataLoader, task_name: str):
        """EWCのためのFisher情報行列を計算し、損失関数に設定する。"""
        print(f"🧠 Computing Fisher Information Matrix for EWC (task: {task_name})...")
        self.model.eval()
        
        fisher_matrix: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_matrix[name] = torch.zeros_like(param.data)

        for batch in tqdm(dataloader, desc=f"Computing Fisher Matrix for {task_name}"):
            self.model.zero_grad()
            input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
            
            logits, _, _ = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_matrix[name] += param.grad.data.pow(2) / len(dataloader)

        if isinstance(self.criterion, CombinedLoss):
            self.criterion.fisher_matrix.update(fisher_matrix)
            for name, param in self.model.named_parameters():
                if name in fisher_matrix:
                    self.criterion.optimal_params[name] = param.data.clone()
            
            ewc_data_path = Path(self.writer.log_dir) / f"ewc_data_{task_name}.pt"
            torch.save({
                'fisher_matrix': self.criterion.fisher_matrix,
                'optimal_params': self.criterion.optimal_params
            }, ewc_data_path)
            print(f"✅ EWC Fisher matrix and optimal parameters for '{task_name}' saved to '{ewc_data_path}'.")

class DistillationTrainer(BreakthroughTrainer):
    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        functional.reset_net(self.model)
        if is_train: self.model.train()
        else: self.model.eval()
            
        student_input, attention_mask, student_target, teacher_logits = [t.to(self.device) for t in batch]

        with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                # 修正: return_full_hiddens を False で渡し、通常のロジット出力を期待する
                outputs = self.model(student_input, return_spikes=True, return_full_mems=True, return_full_hiddens=False)
                
                student_logits, spikes, mem = outputs
                
                assert isinstance(self.criterion, DistillationLoss)
                loss_dict = self.criterion(
                    student_logits=student_logits, teacher_logits=teacher_logits, targets=student_target,
                    spikes=spikes, mem=mem, model=self.model, attention_mask=attention_mask
                )
        
        if is_train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss_dict['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        with torch.no_grad():
            preds = torch.argmax(student_logits, dim=-1)
            ignore_idx = self.criterion.ce_loss_fn.ignore_index
            mask = student_target != ignore_idx
            
            num_valid_tokens = mask.sum()
            if num_valid_tokens > 0:
                accuracy = (preds[mask] == student_target[mask]).float().sum() / num_valid_tokens
            else:
                accuracy = torch.tensor(0.0, device=self.device)
            loss_dict['accuracy'] = accuracy
        
        return {k: v.cpu().item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

# 修正: SelfSupervisedTrainer の _run_step を修正
class SelfSupervisedTrainer(BreakthroughTrainer):
    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        functional.reset_net(self.model)
        start_time = time.time()
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        
        # SelfSupervisedLoss (TCL) は full_hiddens を必要とするため、Trueで呼び出す
        with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                # outputs: (full_hiddens, spikes, mem) のタプルを期待
                outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=True)
                
                full_hiddens, spikes, mem = outputs
                
                assert isinstance(self.criterion, SelfSupervisedLoss)
                # full_hiddens を損失関数に渡す
                loss_dict = self.criterion(full_hiddens, target_ids, spikes, mem, self.model)
        
        # 以下のロジックはBreakthroughTrainerと同じだが、logitsが存在しないため注意
        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

        # TCLではAccuracyは意味がないため、計算をスキップし0.0として返す
        loss_dict['accuracy'] = torch.tensor(0.0, device=self.device) # Tensorとして保存

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

class PhysicsInformedTrainer(BreakthroughTrainer):
    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        functional.reset_net(self.model)
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        
        # 修正: return_full_hiddens を False で渡し、通常のロジット出力を期待する
        with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=False)
                logits, spikes, mem_sequence = outputs
                loss_dict = self.criterion(logits, target_ids, spikes, mem_sequence, self.model)
        
        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            if hasattr(self.criterion, "ce_loss_fn") and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                ignore_idx = self.criterion.ce_loss_fn.ignore_index
                mask = target_ids != ignore_idx
                num_masked_elements = cast(torch.Tensor, mask).sum()
                accuracy = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                loss_dict['accuracy'] = accuracy

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

class ProbabilisticEnsembleTrainer(BreakthroughTrainer):
    def __init__(self, ensemble_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.ensemble_size = ensemble_size

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        
        # アンサンブルのために複数回フォワードパスを実行
        ensemble_logits = []
        for _ in range(self.ensemble_size):
            functional.reset_net(self.model)
            with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
                with torch.set_grad_enabled(is_train):
                    # 修正: return_full_hiddens を False で渡し、通常のロジット出力を期待する
                    outputs = self.model(input_ids, return_spikes=True, return_full_mems=True, return_full_hiddens=False)
                    logits, _, _ = outputs
                    ensemble_logits.append(logits)
        
        ensemble_logits_tensor = torch.stack(ensemble_logits)
        
        # 損失計算 (アンサンブル全体で)
        loss_dict = self.criterion(ensemble_logits_tensor, target_ids, torch.tensor(0.0), torch.tensor(0.0), self.model)

        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

        with torch.no_grad():
            mean_logits = ensemble_logits_tensor.mean(dim=0)
            preds = torch.argmax(mean_logits, dim=-1)
            if hasattr(self.criterion, 'ce_loss_fn') and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                ignore_idx = self.criterion.ce_loss_fn.ignore_index
                mask = target_ids != ignore_idx
                num_masked_elements = cast(torch.Tensor, mask).sum()
                accuracy = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                loss_dict['accuracy'] = accuracy

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

class PlannerTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        progress_bar = tqdm(dataloader, desc=f"Planner Training Epoch {epoch}")
        
        for batch in progress_bar:
            input_ids, target_plan = [t.to(self.device) for t in batch]

            self.optimizer.zero_grad()
            
            skill_logits, _, _ = self.model(input_ids)
            
            assert isinstance(self.criterion, PlannerLoss)
            loss_dict = self.criterion(skill_logits, target_plan)
            loss = loss_dict['total']
            
            loss.backward()
            self.optimizer.step()
            
            progress_bar.set_postfix({"loss": loss.item()})
            
class BPTTTrainer:
    def __init__(self, model: nn.Module, config: DictConfig):
        self.model = model
        self.config = config
        self.optimizer = Adam(self.model.parameters(), lr=config.training.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_type = self.config.model.get("type", "simple")

    def _calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.model_type == "spiking_transformer":
            return self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        else: # simple SNN
            T, B, V = outputs.shape
            S = targets.shape[1]
            assert T == S, f"Time dimension mismatch: {T} != {S}"
            return self.criterion(outputs.permute(1, 0, 2).reshape(-1, V), targets.reshape(-1))

    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        if self.model_type == "spiking_transformer":
             outputs, _, _ = self.model(data)
        else:
             outputs = self.model(data)

        loss = self._calculate_loss(outputs, targets)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class ParticleFilterTrainer:
    """
    逐次モンテカルロ法（パーティクルフィルタ）を用いて、微分不可能なSNNを学習するトレーナー。
    CPU上での実行を想定し、GPU依存から脱却するアプローチ。
    """
    def __init__(self, base_model: BioSNN, config: Dict[str, Any], device: str):
        self.base_model = base_model.to(device)
        self.device = device
        self.config = config
        self.num_particles = config['training']['biologically_plausible']['particle_filter']['num_particles']
        self.noise_std = config['training']['biologically_plausible']['particle_filter']['noise_std']
        
        # 複数のモデル（パーティクル）をアンサンブルとして保持
        self.particles = [copy.deepcopy(self.base_model) for _ in range(self.num_particles)]
        self.particle_weights = torch.ones(self.num_particles, device=self.device) / self.num_particles
        print(f"🌪️ ParticleFilterTrainer initialized with {self.num_particles} particles.")

    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        """1ステップの学習（予測、尤度計算、再サンプリング）を実行する。"""
        
        # 1. 予測 & ノイズ付加 (各パーティクル)
        for particle in self.particles:
            # パラメータに少量のノイズを加えて多様性を維持
            with torch.no_grad():
                for param in particle.parameters():
                    param.add_(torch.randn_like(param) * self.noise_std)
        
        # 2. 尤度計算
        # 各パーティクルがターゲットをどれだけうまく予測できたかを評価
        log_likelihoods = []
        for particle in self.particles:
            particle.eval()
            with torch.no_grad():
                if data.dim() > 1:
                    squeezed_data = data.squeeze(0)
                else:
                    squeezed_data = data

                input_spikes = (torch.rand_like(squeezed_data) > 0.5).float()
                outputs, _ = particle(input_spikes)
                
                if targets.dim() > 1:
                    squeezed_targets = targets.squeeze(0)
                else:
                    squeezed_targets = targets
                
                loss = F.mse_loss(outputs, squeezed_targets)
                log_likelihoods.append(-loss)
        
        # 3. 重みの更新と正規化
        log_likelihoods_tensor = torch.tensor(log_likelihoods, device=self.device)
        self.particle_weights *= torch.exp(log_likelihoods_tensor - log_likelihoods_tensor.max())
        
        if self.particle_weights.sum() > 0:
            self.particle_weights /= self.particle_weights.sum()
        else:
            self.particle_weights.fill_(1.0 / self.num_particles)

        # 4. 再サンプリング (Resampling)
        if 1. / (self.particle_weights**2).sum() < self.num_particles / 2:
            indices = torch.multinomial(self.particle_weights, self.num_particles, replacement=True)
            new_particles = [copy.deepcopy(self.particles[i]) for i in indices]
            self.particles = new_particles
            self.particle_weights.fill_(1.0 / self.num_particles)
        
        best_particle_loss = -log_likelihoods_tensor.max().item()
        return best_particle_loss