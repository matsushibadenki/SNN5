# ファイルパス: snn_research/training/losses.py
# コードの最も最初には、ファイルパス、ファイルの内容を示したタイトル、機能の説明を詳細に記述してください。 修正内容は記載する必要はありません。
# Title: 損失関数定義
# Description: CombinedLoss, DistillationLoss, SelfSupervisedLoss (TCL), PhysicsInformedLoss, PlannerLoss, ProbabilisticEnsembleLoss を含む各種損失関数を定義します。
# 改善点(v1): 継続学習のためのElastic Weight Consolidation (EWC) 損失を追加。
# 改善点(v2): スパース性を促す正則化項(sparsity_reg_weight)を追加し,汎化性能を向上。
# 改善点(v3):
# - temporal/latency codingの導入として、時間的圧縮を促す正則化項 (temporal_compression_weight)を追加。
# - Spiking Transformerの自己注意メカニズムのスパース性を適応的に学習させるための 正則化項(sparsity_threshold_reg_weight)を追加。
# 改善点(v4): SelfSupervisedLossをTemporal Contrastive Loss (TCL)に再実装。
# 修正(v5): DistillationLossのマスク処理を画像分類に対応。

import torch
import torch.nn as nn
import torch.nn.functional as F
# --- ▼ 修正 ▼ ---
from typing import Dict, Optional, cast # cast をインポート
# --- ▲ 修正 ▲ ---
from transformers import PreTrainedTokenizerBase

from snn_research.core.snn_core import MultiLevelSpikeDrivenSelfAttention

class CombinedLoss(nn.Module):
    """クロスエントロピー損失、各種正則化、EWC損失を組み合わせた損失関数。"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ce_weight: float = 1.0, spike_reg_weight: float = 0.0, mem_reg_weight: float = 0.0, sparsity_reg_weight: float = 0.0, temporal_compression_weight: float = 0.0, sparsity_threshold_reg_weight: float = 0.0, target_spike_rate: float = 0.02, ewc_weight: float = 0.0, **kwargs):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {
            'ce': ce_weight,
            'spike_reg': spike_reg_weight,
            'mem_reg': mem_reg_weight,
            'sparsity_reg': sparsity_reg_weight,
            'temporal_compression': temporal_compression_weight,
            'sparsity_threshold_reg': sparsity_threshold_reg_weight,
            'ewc': ewc_weight
        }
        self.target_spike_rate = target_spike_rate
        # EWCのためのFisher情報行列と最適パラメータを保持
        self.fisher_matrix: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, **kwargs) -> dict:
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))

        sparsity_loss = torch.mean(torch.abs(spikes))

        mem_reg_loss = torch.mean(mem**2)

        temporal_compression_loss = torch.tensor(0.0, device=spikes.device)
        if self.weights['temporal_compression'] > 0 and spikes.ndim > 1 and spikes.shape[1] > 1:
            time_steps = spikes.shape[1]
            time_weights = torch.linspace(0, 1, time_steps, device=spikes.device).view(1, -1, 1)
            if spikes.ndim > 3:
                time_weights = time_weights.view(1, time_steps, 1, 1)
            temporal_compression_loss = (spikes * time_weights).mean()

        # スパース性閾値の正則化
        sparsity_threshold_reg_loss = torch.tensor(0.0, device=logits.device)
        if self.weights['sparsity_threshold_reg'] > 0:
            threshold_sum = torch.tensor(0.0, device=logits.device)
            count = 0
            for module in model.modules():
                if isinstance(module, MultiLevelSpikeDrivenSelfAttention):
                    threshold_sum += module.sparsity_threshold
                    count += 1
            if count > 0:
                # 閾値が大きくなることを奨励（損失を減らすため負の項にする）
                sparsity_threshold_reg_loss = - (threshold_sum / count)

        ewc_loss = torch.tensor(0.0, device=logits.device)
        if self.weights['ewc'] > 0 and self.fisher_matrix:
            for name, param in model.named_parameters():
                if name in self.fisher_matrix and param.requires_grad:
                    fisher = self.fisher_matrix[name].to(param.device)
                    opt_param = self.optimal_params[name].to(param.device)
                    ewc_loss += (fisher * (param - opt_param)**2).sum()

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss +
                      self.weights['mem_reg'] * mem_reg_loss +
                      self.weights['temporal_compression'] * temporal_compression_loss +
                      self.weights['sparsity_threshold_reg'] * sparsity_threshold_reg_loss +
                      self.weights['ewc'] * ewc_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss, 'sparsity_loss': sparsity_loss,
            'mem_reg_loss': mem_reg_loss, 'spike_rate': spike_rate,
            'temporal_compression_loss': temporal_compression_loss,
            'sparsity_threshold_reg_loss': sparsity_threshold_reg_loss,
            'ewc_loss': ewc_loss
        }

class DistillationLoss(nn.Module):
    """知識蒸留のための損失関数（各種正則化付き）。"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ce_weight: float = 0.3, distill_weight: float = 0.7,
                 spike_reg_weight: float = 0.01, mem_reg_weight: float = 0.0, sparsity_reg_weight: float = 0.00001,
                 temporal_compression_weight: float = 0.0, sparsity_threshold_reg_weight: float = 0.0,
                 temperature: float = 2.0, target_spike_rate: float = 0.02, **kwargs):
        super().__init__()
        student_pad_id = tokenizer.pad_token_id
        self.temperature = temperature
        self.weights = {
            'ce': ce_weight, 'distill': distill_weight, 'spike_reg': spike_reg_weight,
            'mem_reg': mem_reg_weight, 'sparsity_reg': sparsity_reg_weight,
            'temporal_compression': temporal_compression_weight,
            'sparsity_threshold_reg': sparsity_threshold_reg_weight
        }
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=student_pad_id if student_pad_id is not None else -100)
        self.distill_loss_fn = nn.KLDivLoss(reduction='none', log_target=True)
        self.target_spike_rate = target_spike_rate

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:

        is_classification = student_logits.ndim == 2 # (batch_size, num_classes)
        is_sequence = student_logits.ndim == 3 # (batch_size, seq_len, vocab_size)

        if is_classification:
            assert student_logits.shape == teacher_logits.shape, \
                f"Shape mismatch! Student: {student_logits.shape}, Teacher: {teacher_logits.shape}"
            assert targets.ndim == 1 and targets.shape[0] == student_logits.shape[0], \
                f"Target shape mismatch for classification: {targets.shape}, expected ({student_logits.shape[0]},)"
        elif is_sequence:
            assert student_logits.shape == teacher_logits.shape, \
                f"Shape mismatch! Student: {student_logits.shape}, Teacher: {teacher_logits.shape}"
            assert targets.ndim == 2 and targets.shape == student_logits.shape[:2], \
                f"Target shape mismatch for sequence: {targets.shape}, expected {student_logits.shape[:2]}"
        else:
            raise ValueError(f"Unsupported student_logits shape: {student_logits.shape}")

        # クロスエントロピー損失
        if is_classification:
            ce_loss = self.ce_loss_fn(student_logits, targets)
        else: # is_sequence
            ce_loss = self.ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))

        # 蒸留損失 (KLダイバージェンス)
        soft_student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)

        distill_loss_unreduced = self.distill_loss_fn(soft_student_log_probs, soft_teacher_log_probs).sum(dim=-1) # (batch,) or (batch, seq_len)

        # マスクの作成と適用
        num_valid_tokens: torch.Tensor # 型ヒントを追加
        if is_classification:
            mask = torch.ones(targets.shape[0], dtype=torch.bool, device=targets.device)
            num_valid_tokens = torch.tensor(targets.shape[0], device=targets.device)
            masked_distill_loss = distill_loss_unreduced
        else: # is_sequence
            if attention_mask is None:
                mask = (targets != self.ce_loss_fn.ignore_index)
            else:
                mask = attention_mask.bool()
            num_valid_tokens = cast(torch.Tensor, mask).sum() # castを追加
            masked_distill_loss = distill_loss_unreduced.where(mask, torch.tensor(0.0, device=distill_loss_unreduced.device))

        # 損失の平均化
        if num_valid_tokens.item() > 0: # Check tensor value using .item()
            distill_loss = masked_distill_loss.sum() / num_valid_tokens.item() # Use .item() here
        else:
            distill_loss = torch.tensor(0.0, device=student_logits.device)

        distill_loss = distill_loss * (self.temperature ** 2)

        # (正則化項の計算)
        spike_rate = spikes.mean()
        target_spike_rate = torch.tensor(self.target_spike_rate, device=spikes.device)
        spike_reg_loss = F.mse_loss(spike_rate, target_spike_rate)

        sparsity_loss = torch.mean(torch.abs(spikes))

        mem_reg_loss = torch.mean(mem**2)

        temporal_compression_loss = torch.tensor(0.0, device=spikes.device)
        if self.weights['temporal_compression'] > 0 and spikes.ndim > 1 and spikes.shape[1] > 1:
            time_steps = spikes.shape[1]
            time_weights = torch.linspace(0, 1, time_steps, device=spikes.device).view(1, -1, 1)
            if spikes.ndim > 3:
                time_weights = time_weights.view(1, time_steps, 1, 1)
            temporal_compression_loss = (spikes * time_weights).mean()

        sparsity_threshold_reg_loss = torch.tensor(0.0, device=student_logits.device)
        if self.weights['sparsity_threshold_reg'] > 0:
            threshold_sum = torch.tensor(0.0, device=student_logits.device)
            count = 0
            for module in model.modules():
                if isinstance(module, MultiLevelSpikeDrivenSelfAttention):
                    threshold_sum += module.sparsity_threshold
                    count += 1
            if count > 0:
                sparsity_threshold_reg_loss = - (threshold_sum / count)

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['distill'] * distill_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss +
                      self.weights['mem_reg'] * mem_reg_loss +
                      self.weights['temporal_compression'] * temporal_compression_loss +
                      self.weights['sparsity_threshold_reg'] * sparsity_threshold_reg_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'distill_loss': distill_loss, 'spike_reg_loss': spike_reg_loss,
            'sparsity_loss': sparsity_loss, 'mem_reg_loss': mem_reg_loss,
            'temporal_compression_loss': temporal_compression_loss,
            'sparsity_threshold_reg_loss': sparsity_threshold_reg_loss
        }


class SelfSupervisedLoss(nn.Module):
    """
    Temporal Contrastive Learning (TCL)のための損失関数。
    時間的に隣接する隠れ状態をポジティブペアとして学習する。
    """
    def __init__(self, prediction_weight: float, spike_reg_weight: float, mem_reg_weight: float, tokenizer: PreTrainedTokenizerBase, target_spike_rate: float = 0.02, tcl_weight: float = 1.0, tcl_temperature: float = 0.1, **kwargs):
        super().__init__()
        self.weights = {
            'prediction': prediction_weight,
            'spike_reg': spike_reg_weight,
            'mem_reg': mem_reg_weight,
            'tcl': tcl_weight
        }
        self.tcl_temperature = tcl_temperature
        self.target_spike_rate = target_spike_rate
        pad_id = tokenizer.pad_token_id
        self.prediction_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)

    def forward(self, full_hiddens: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, **kwargs) -> dict:
        B, S, T, D = full_hiddens.shape
        hiddens_flat = full_hiddens.permute(0, 1, 2, 3).reshape(B * S * T, D)
        hiddens_norm = F.normalize(hiddens_flat, p=2, dim=1)
        hiddens_st = full_hiddens.reshape(B * S, T, D)
        anchors = hiddens_st[:, :-1, :].reshape(-1, D)
        positives = hiddens_st[:, 1:, :].reshape(-1, D)

        ignore_index = self.prediction_loss_fn.ignore_index
        # Make sure targets are expanded correctly if needed, or use broadcasting
        # Assuming targets are (B, S)
        # We need a mask of shape (B*S*(T-1))
        # Create mask based on original sequence positions, then repeat/reshape
        valid_mask_bs = (targets != ignore_index) # Shape (B, S)
        # Repeat for each anchor time step (T-1) and flatten
        # We need to map anchor index back to (b, s, t_anchor) to check target validity at (b, s)
        # Instead, let's reshape targets to match the structure before flattening anchors
        # targets expanded: (B, S, T-1) - repeats the validity for each time step
        targets_expanded_time = targets.unsqueeze(2).repeat(1, 1, T-1) # Shape (B, S, T-1)
        # Now reshape this mask to match the flattened anchor shape
        valid_mask = (targets_expanded_time != ignore_index).reshape(-1) # Shape (B*S*(T-1))


        similarity_matrix = torch.matmul(anchors, hiddens_norm.T) / self.tcl_temperature
        positive_indices = torch.arange(anchors.size(0), device=anchors.device)
        tcl_loss_unmasked = F.cross_entropy(similarity_matrix, positive_indices, reduction='none')

        # Ensure valid_mask has the same number of elements as tcl_loss_unmasked
        if valid_mask.numel() != tcl_loss_unmasked.numel():
             # This indicates a shape mismatch, likely in mask creation. Add debugging or error.
             print(f"Warning: Mask shape {valid_mask.shape} mismatch with loss shape {tcl_loss_unmasked.shape}")
             # Fallback: don't apply mask if shapes don't match
             tcl_loss = tcl_loss_unmasked.mean()
        else:
            num_valid = valid_mask.sum().clamp(min=1)
            tcl_loss = (tcl_loss_unmasked * valid_mask.float()).sum() / num_valid


        spike_rate = spikes.mean()
        target_spike_rate = torch.tensor(self.target_spike_rate, device=spikes.device)
        spike_reg_loss = F.mse_loss(spike_rate, target_spike_rate)

        mem_reg_loss = torch.mean(mem**2)

        total_loss = (self.weights['tcl'] * tcl_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['mem_reg'] * mem_reg_loss)

        return {
            'total': total_loss,
            'tcl_loss': tcl_loss,
            'spike_reg_loss': spike_reg_loss,
            'mem_reg_loss': mem_reg_loss,
            'spike_rate': spike_rate
        }


class PhysicsInformedLoss(nn.Module):
    """
    物理法則（膜電位の滑らかさ）を制約として組み込んだ損失関数。
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ce_weight: float = 1.0, spike_reg_weight: float = 0.0, mem_smoothness_weight: float = 0.0, target_spike_rate: float = 0.02):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {
            'ce': ce_weight,
            'spike_reg': spike_reg_weight,
            'mem_smoothness': mem_smoothness_weight,
        }
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem_sequence: torch.Tensor, model: nn.Module) -> dict:
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))

        mem_smoothness_loss = torch.tensor(0.0, device=logits.device) # Initialize
        if isinstance(mem_sequence, torch.Tensor) and mem_sequence.numel() > 1 and mem_sequence.ndim > 0:
             # Try to infer time dimension (often the last dim before features, or first after batch)
             # This depends heavily on how `return_full_mems=True` returns the tensor
             # Assuming shape like (Batch, SeqLen, TimeSteps, Features) or (Batch, TimeSteps, Features)
             # Let's assume time_dim = -2 if ndim >= 2, otherwise 0
             time_dim = -2 if mem_sequence.ndim >= 3 else 0 # Educated guess
             if mem_sequence.shape[time_dim] > 1:
                 try:
                     mem_diff = torch.diff(mem_sequence, dim=time_dim)
                     mem_smoothness_loss = torch.mean(mem_diff**2)
                 except RuntimeError as e:
                     print(f"Warning: Could not compute diff for mem_smoothness_loss on dim {time_dim}, shape {mem_sequence.shape}: {e}")
             # else: print(f"Skipping mem_smoothness_loss: time dim size is <= 1") # Debug
        # else: print(f"Skipping mem_smoothness_loss: mem_sequence invalid") # Debug


        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['mem_smoothness'] * mem_smoothness_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss,
            'mem_smoothness_loss': mem_smoothness_loss,
            'spike_rate': spike_rate
        }

class PlannerLoss(nn.Module):
    """
    プランナーSNNの学習用損失関数。
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, predicted_logits: torch.Tensor, target_plan: torch.Tensor) -> Dict[str, torch.Tensor]:
        target = target_plan.view(-1)
        loss = self.loss_fn(predicted_logits, target)
        return {'total': loss, 'planner_loss': loss}

class ProbabilisticEnsembleLoss(nn.Module):
    """
    確率的アンサンブル学習のための損失関数。
    出力のばらつきを抑制する正則化項を持つ。
    """
    def __init__(self, ce_weight: float, variance_reg_weight: float, tokenizer: PreTrainedTokenizerBase, **kwargs):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {'ce': ce_weight, 'variance_reg': variance_reg_weight}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, **kwargs) -> dict:
        mean_logits = logits.mean(dim=0)
        ce_loss = self.ce_loss_fn(mean_logits.view(-1, mean_logits.size(-1)), targets.view(-1))

        probs = F.softmax(logits, dim=-1)
        variance = probs.var(dim=0).mean()
        variance_reg_loss = variance

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['variance_reg'] * variance_reg_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'variance_reg_loss': variance_reg_loss
        }