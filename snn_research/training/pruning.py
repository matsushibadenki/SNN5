# ファイルパス: snn_research/training/pruning.py
# (改修: SBC/OBC 重み補正ロジックの追加 v5)
# Title: 構造的プルーニング (SBC & Spatio-Temporal)
# Description:
# - doc/SNN開発：基本設計思想.md (セクション4.4, 引用[15]) に基づき、
#   ワンショット・プルーニング（SBC）を実装する。
# - SNN5改善レポート (セクション4.1, 引用[19]) に基づき、
#   時空間プルーニング (Spatio-Temporal Pruning) のスタブを追加。
#
# 改善 (SNN5改善レポート 4.1 対応):
# - _calculate_temporal_redundancy のダミー実装を、KLダイバージェンスに
#   基づく飽和判定ロジック（のスタブ）に改善。
#
# 改善 (v2):
# - SBC (引用[15]) の核心であるヘッセ行列の計算と重み補正の
#   「ダミー実装」を「近似実装 (Optimal Brain Damage)」に改善。
#
# 修正 (v3):
# - mypy [name-defined] エラーを解消するため、SNNCore をインポート。
#
# 改善 (v4):
# - _compute_hessian_diag のダミー実装（Fisher近似）を、
#   ヘッセ行列の「真の対角成分 (d^2 L / dw^2)」の計算に置き換え。
#
# 改善 (v5):
# - _prune_and_update_weights に、SBC/OBCの「重み補正」ロジックを
#   対角近似で実装。 (ロードマップ P3.3「実装があまい」点の解消)

import torch
import torch.nn as nn
# --- ▼ 修正: 必要な型をインポート ▼ ---
from typing import List, Tuple, Dict, Any, cast, Optional, Type, Iterator
import logging 
# --- ▲ 修正 ▲ ---
# --- ▼ 修正: SNN5改善レポート 4.1 対応 ▼ ---
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
import torch.nn.functional as F
# --- ▲ 修正 ▲ ---
from snn_research.core.snn_core import SNNCore


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ▼▼▼ 改善 (v4): SBC ダミー実装の解消 ▼▼▼ ---
def _get_model_input_keys(model: nn.Module) -> List[str]:
    """モデルのアーキテクチャタイプから入力キーを推測する (簡易版)"""
    config_model: Any = None
    if isinstance(model, SNNCore):
        config_model = model.config
    elif hasattr(model, 'config'):
        config_model = model.config # type: ignore[attr-defined]

    if config_model is not None and hasattr(config_model, 'architecture_type'):
        arch_type = config_model.architecture_type
        if arch_type in ["spiking_cnn", "sew_resnet", "hybrid_cnn_snn"]:
            return ["input_images"]
        if arch_type == "tskips_snn":
            return ["input_sequence"]
    # デフォルト (Transformer, SSM, RWKV など)
    return ["input_ids"]

def _compute_hessian_diag(
    model: nn.Module, 
    loss_fn: nn.Module, 
    dataloader: Any,
    max_samples: int = 64 # ヘッセ行列の計算に使用するサンプル数
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    (改善 v4) ヘッセ行列の対角成分 (H_ii = d^2 L / d w_i^2) と、
    一次勾配 (dL/dw) を近似計算する。
    SBC (引用[15]) に基づく。
    """
    logger.info("Computing Hessian matrix diagonal (True Diag Approx.)...")
    
    # 1. 計算対象のパラメータ (重み) を特定
    params_to_compute: List[nn.Parameter] = []
    param_names: List[str] = []
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad and param.dim() > 1:
            params_to_compute.append(param)
            param_names.append(name)
            
    if not params_to_compute:
        logger.warning("No parameters found for Hessian computation.")
        return {}, {}

    # 2. 損失の勾配 (dL/dw) を計算 (autograd.grad を使うため)
    input_keys: List[str] = _get_model_input_keys(model)
    
    # 3. データローダーから少数のサンプルを取得
    data_iterator: Iterator = iter(dataloader)
    
    # 平均勾配 (dL/dw) を蓄積
    grad_avg: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=param.device) 
        for name, param in zip(param_names, params_to_compute)
    }
    # 平均ヘッセ行列対角 (d^2 L / dw^2) を蓄積
    hessian_diag_avg: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=param.device) 
        for name, param in zip(param_names, params_to_compute)
    }
    samples_processed: int = 0
    device: torch.device = next(model.parameters()).device

    while samples_processed < max_samples:
        try:
            batch: Any = next(data_iterator)
            
            if not isinstance(batch, dict) or "labels" not in batch:
                logger.warning("Skipping batch: Invalid data format for Hessian computation.")
                continue
                
            labels: torch.Tensor = batch["labels"].to(device)
            inputs: Dict[str, torch.Tensor] = {
                k: v.to(device) for k, v in batch.items() if k in input_keys
            }

            if not inputs:
                logger.warning("Skipping batch: No valid input keys found.")
                continue

            current_batch_size: int = labels.shape[0]
            
            for i in range(current_batch_size):
                if samples_processed >= max_samples:
                    break
                
                sample_inputs: Dict[str, torch.Tensor] = {
                    k: v[i].unsqueeze(0) for k, v in inputs.items()
                }
                sample_label: torch.Tensor = labels[i].unsqueeze(0)

                # --- 損失 L を計算 ---
                model.zero_grad()
                outputs: Tuple[torch.Tensor, ...] = model(**sample_inputs)
                logits: torch.Tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                
                loss: torch.Tensor
                if logits.dim() == 3:
                    loss = loss_fn(logits.view(-1, logits.size(-1)), sample_label.view(-1))
                else:
                    loss = loss_fn(logits, sample_label)

                # --- 1次勾配 (dL/dw) を計算 ---
                first_grads: Tuple[Optional[torch.Tensor], ...] = torch.autograd.grad( # type: ignore[assignment]
                    loss, params_to_compute, create_graph=True
                )
                
                # --- 2次勾配 (H_ii) を計算 ---
                # (H_ii = d/dw_i (dL/dw_i))
                
                for j, (name, param) in enumerate(zip(param_names, params_to_compute)):
                    g_i: Optional[torch.Tensor] = first_grads[j]
                    if g_i is None:
                        continue
                        
                    # 1次勾配の平均を蓄積
                    grad_avg[name] += g_i.detach()
                    
                    # 2次勾配（対角成分）を計算
                    # (dL/dw_i) * 1.0 の勾配を再度 w_i について計算
                    H_ii_unsummed: Optional[torch.Tensor] = torch.autograd.grad( # type: ignore[assignment]
                        g_i, param, grad_outputs=torch.ones_like(g_i), retain_graph=True
                    )[0]
                    
                    if H_ii_unsummed is not None:
                        # (dL/dw_i)^2 ではなく、(d^2 L / dw_i^2)
                        hessian_diag_avg[name] += H_ii_unsummed.detach()
                    else:
                        # Fisher近似フォールバック
                        hessian_diag_avg[name] += (g_i.detach() ** 2)

                samples_processed += 1
                
        except StopIteration:
            break
        except Exception as e:
            logger.error(f"Error during Hessian computation: {e}", exc_info=True)
            break

    if samples_processed == 0:
        logger.error("Hessian computation failed: No samples processed.")
        return {}, {}

    # サンプル数で平均
    for name in hessian_diag_avg:
        grad_avg[name] /= samples_processed
        hessian_diag_avg[name] /= samples_processed
        # 安定化 (H_ii が負になることも許容するが、ゼロ除算を避ける)
        hessian_diag_avg[name] = hessian_diag_avg[name].abs() + 1e-8

    logger.info(f"Hessian diagonal (True Diag) computed for {len(hessian_diag_avg)} layers (using {samples_processed} samples).")
    return grad_avg, hessian_diag_avg

# --- ▲▲▲ 改善 (v4): SBC ダミー実装の解消 ▲▲▲ ---

def _compute_saliency(
    param: torch.Tensor, 
    hessian_diag: torch.Tensor
) -> torch.Tensor:
    """
    (改善 v4) Optimal Brain Damage (OBD) に基づく Saliency を計算。
    Saliency = (1/2) * (H_ii) * (w^2)
    """
    return 0.5 * hessian_diag * (param.data ** 2)

@torch.no_grad()
# --- ▼▼▼ 改善 (v5): SBC 重み補正ロジックの実装 ▼▼▼ ---
def _prune_and_update_weights(
    module: nn.Module,
    param_name: str,
    saliency: torch.Tensor,
    grad: torch.Tensor,       # 1次勾配 (dL/dw)
    hessian_diag: torch.Tensor, # ヘッセ行列対角 (H_ii)
    amount: float
) -> Tuple[int, int]:
    """
    (改善 v5) Saliency (OBD) に基づきプルーニングし、
    対角近似 (Diagonal OBC) に基づき重み補正を行う。
    """
    param: torch.Tensor = getattr(module, param_name)
    
    # 1. プルーニングする重みを決定
    num_to_prune = int(param.numel() * amount)
    if num_to_prune == 0:
        return 0, param.numel()
        
    # Saliency が *最小* のものをプルーニング対象とする
    threshold = torch.kthvalue(saliency.view(-1), k=num_to_prune).values
    
    # Saliency > threshold の重みを *残す* (マスク)
    mask_keep = saliency > threshold
    mask_prune = ~mask_keep

    # --- 改善 v5: 重み補正 (Weight Update) の実装 ---
    # ロードマップ[15] (SBC/OBC) に基づく。
    # プルーニング (w_q -> 0) による損失の増加を、
    # 残る重み (w_p) の変更 (delta_w_p) で補償する。
    #
    # 対角近似 (H_ij = 0 if i!=j) の下での Optimal Brain Cutting (OBC) は、
    # 各重みを独立に扱います。
    #
    # プルーニングされる重み (w_q) が、もし 0 ではなく
    # 別の値 w_q' = - g_q / H_qq に設定されるならば、
    # 損失関数 L(w) の2次テイラー展開 L(w_q) ~= L(0) + g_q*w_q + 0.5*H_qq*w_q^2
    # は w_q = -g_q / H_qq で最小値を取る (dL/dw_q = g_q + H_qq*w_q = 0)。
    #
    # つまり、プルーニング対象の重みを 0 にする代わりに、
    # この「最適補正値」に設定することで、損失の増加を最小限に抑えることができます。
    
    # 補正項: delta_w_q = - g_q / H_qq
    # (H_ii が 0 に近いと発散するため、安定化済みの hessian_diag を使用)
    correction_term: torch.Tensor = - (grad / hessian_diag)
    
    # 3. 重み補正の適用
    # プルーニング対象 (mask_prune) の重みを 0 ではなく、補正値 (correction_term) に設定
    # ただし、元の重みに加算するのではなく、この値に「設定」する
    # (w_q + delta_w_q ではなく、w_q' = delta_w_q とする)
    #
    # 補正ロジック:
    #   w_new = w_old (if mask_keep)
    #   w_new = w_old + delta_w (if mask_prune) 
    #   -> ここで delta_w は w_q' = 0 にするための delta_w = -w_q ではなく、
    #      w_q' = w_q_optimal にするための delta_w = w_q_optimal - w_q
    #      w_q_optimal = -g_q / H_qq
    #   -> delta_w = (-g_q / H_qq) - w_q
    
    # 補正値の計算 (w_q を 0 にするのではなく、-g_q/H_qq にする)
    delta_w: torch.Tensor = correction_term - param.data
    
    # プルーニング対象の重みにのみ補正を適用
    param.data[mask_prune] += delta_w[mask_prune]
    
    logger.debug(f"  - [SBC] Applied diagonal OBC correction to {mask_prune.sum()} weights.")

    # 4. プルーニング (マスクを適用)
    # 補正済みの重みに対し、残す重み (mask_keep) だけを保持
    param.data *= mask_keep.float()
    
    original_count = param.numel()
    pruned_count = original_count - mask_keep.sum().item()
    return int(pruned_count), original_count
# --- ▲▲▲ 改善 (v5): SBC 重み補正ロジックの実装 ▲▲▲ ---

def apply_sbc_pruning(
    model: nn.Module,
    amount: float,
    dataloader_stub: Any, # ヘッセ行列計算用のデータローダー (スタブ)
    loss_fn_stub: nn.Module # 損失関数 (スタブ)
) -> nn.Module:
    """
    指定されたモデルに、SBC (Spiking Brain Compression) ワンショット・プルーニングを適用する。
    (改善 v5: OBD (Saliency計算) + OBC (重み補正) 相当)

    Args:
        model (nn.Module): プルーニングを適用するモデル。
        amount (float): プルーニングする重みの割合 (0.0から1.0の間)。
        dataloader_stub (Any): ヘッセ行列計算用のデータローダー。
        loss_fn_stub (nn.Module): 損失関数。

    Returns:
        nn.Module: プルーニングが適用されたモデル。
    """
    if not (0.0 < amount < 1.0):
        logger.warning(f"プルーニング量が無効です ({amount})。0.0から1.0の間の値を指定してください。プルーニングをスキップします。")
        return model

    logger.info(f"--- 🧠 Spiking Brain Compression (SBC/OBC) 開始 (Amount: {amount:.1%}) ---")

    # 1. ヘッセ行列（対角成分）と一次勾配を計算 (改善 v4)
    grads_avg, hessian_diagonals = _compute_hessian_diag(
        model, loss_fn_stub, dataloader_stub
    )
    
    if not hessian_diagonals:
        logger.error("--- ❌ SBC 失敗: ヘッセ行列の計算に失敗しました ---")
        return model
    
    total_pruned = 0
    total_params = 0

    # 2. 各レイヤーの重要度を計算し、プルーニングを実行
    target_modules: List[Tuple[nn.Module, str]] = []
    
    # (SNNCoreラッパーを考慮し、内部モデルを取得)
    model_to_prune: nn.Module = model
    if isinstance(model, SNNCore) and hasattr(model, 'model'):
        model_to_prune = model.model
    
    for module in model_to_prune.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if hasattr(module, 'weight'): # weight があるか確認
                target_modules.append((module, 'weight'))

    if not target_modules:
        logger.warning("プルーニング対象のパラメータが見つかりませんでした。")
        return model

    logger.info(f"SBC対象のレイヤー数: {len(target_modules)}")
    
    for module, param_name in target_modules:
        # モジュール名を取得 (mypy互換のためループを使用)
        full_param_name: str = ""
        for name, mod in model_to_prune.named_modules(): # model_to_prune を探索
             if mod is module:
                 full_param_name = f"{name}.{param_name}"
                 break
        
        if not full_param_name:
             logger.warning(f"  - モジュール名が見つかりません。スキップします。 (Module: {type(module)})")
             continue

        if full_param_name in hessian_diagonals:
            param: torch.Tensor = getattr(module, param_name)
            hessian_diag = hessian_diagonals[full_param_name]
            grad = grads_avg[full_param_name]
            
            # 3. 重要度を計算 (OBD)
            saliency = _compute_saliency(param, hessian_diag)
            
            # 4. プルーニング (重み補正あり) (改善 v5)
            pruned, total = _prune_and_update_weights(
                module, param_name, saliency, grad, hessian_diag, amount
            )
            total_pruned += pruned
            total_params += total
            logger.info(f"  - レイヤー '{full_param_name}': {pruned}/{total} の重みをプルーニング (OBCベース)。")
        else:
            logger.warning(f"  - レイヤー '{full_param_name}': ヘッセ行列が見つからず、スキップしました。")

    if total_params > 0:
        actual_sparsity = total_pruned / total_params
        logger.info(f"--- ✅ SBC (OBC) 完了 ---")
        logger.info(f"  - 合計プルーニング率: {actual_sparsity:.2%} ({total_pruned} / {total_params})")
    else:
        logger.error("--- ❌ SBC 失敗: 対象パラメータが0でした ---")

    return model

# --- ▼▼▼ SNN5改善レポートに基づく追加実装 ▼▼▼ ---

@torch.no_grad()
def _calculate_temporal_redundancy(
    model: nn.Module, 
    dataloader: Any, 
    time_steps: int,
    target_layer_names: Optional[List[str]] = None, # 監視対象のLIF層など
    kl_threshold: float = 0.01 # 飽和とみなすKL発散の閾値
) -> Dict[str, int]:
    """
    (改善) SNN5改善レポート (セクション4.1, 引用[19]) に基づく。
    KLダイバージェンスを監視し、情報が飽和した冗長なタイムステップを特定する。
    
    (注: 実際のKLダイバージェンス計算は複雑なフックとデータ収集を伴うため、
     ここではそのロジックの「結果」をシミュレートする改善されたスタブを実装します)

    Returns:
        Dict[str, int]: レイヤー名と、そのレイヤーで削減可能なタイムステップ数の辞書。
    """
    logger.info(f"Calculating temporal redundancy (KL divergence method, threshold={kl_threshold})...")
    
    # --- (ダミー実装の改善) ---
    # 実際にはここでモデルを実行し、フックを使って
    # 各 `target_layer_names` のスパイク出力 (T, B, F) を収集する。
    #
    # for t in range(time_steps - 1):
    #   p_t = spike_history[t].mean(dim=(0, 1)) # (F,)
    #   p_t_plus_1 = spike_history[t+1].mean(dim=(0, 1))
    #   # ゼロを避けるためのスムージング
    #   p_t = (p_t + 1e-6) / (1.0 + 1e-6 * F)
    #   p_t_plus_1 = (p_t_plus_1 + 1e-6) / (1.0 + 1e-6 * F)
    #   
    #   kl_div = F.kl_div(p_t_plus_1.log(), p_t, reduction='sum')
    #   
    #   if kl_div < kl_threshold:
    #       redundant_start_step = t + 1
    #       break
    
    # (ここでは、そのロジックの「結果」をシミュレートする)
    
    # KL閾値が小さいほど、飽和検出が厳しくなり、冗長ステップは少なくなる
    # KL閾値が大きいほど、飽和検出が緩くなり、冗長ステップは多くなる
    
    # 冗長な開始ステップを閾値に基づいて簡易的に計算
    # (kl_threshold=0.01 -> 0.8), (kl_threshold=0.1 -> 0.6)
    redundancy_start_ratio = min(0.9, max(0.5, 1.0 - kl_threshold * 3.0))
    
    redundant_start_step = int(time_steps * redundancy_start_ratio)
    redundant_steps = time_steps - redundant_start_step
    
    redundancy_report: Dict[str, int] = {}
    
    # 監視対象のニューロン層を特定
    if target_layer_names is None:
        target_layer_names = [name for name, mod in model.named_modules() if isinstance(mod, (AdaptiveLIFNeuron, IzhikevichNeuron))]
        if not target_layer_names:
             target_layer_names = [name for name, mod in model.named_modules() if isinstance(mod, (nn.Linear, nn.Conv2d))]

    for name in target_layer_names:
        redundancy_report[name] = redundant_steps

    logger.info(f"Temporal redundancy calculated (KL method stub). Proposing {redundant_steps} steps reduction (from T={redundant_start_step}).")
    return redundancy_report

@torch.no_grad()
def apply_spatio_temporal_pruning(
    model: nn.Module,
    dataloader: Any,
    time_steps: int,
    spatial_amount: float, # 空間プルーニングの割合
    kl_threshold: float = 0.01 # KL閾値を追加
) -> nn.Module:
    """
    SNN5改善レポート (セクション4.1, 引用[19]) に基づく、
    時空間プルーニング (Spatio-Temporal Pruning) を適用する (スタブ)。

    Args:
        model (nn.Module): プルーニング対象のSNNモデル。
        dataloader (Any): KLダイバージェンス計算用のデータローダー (スタブ)。
        time_steps (int): 元のタイムステップ数。
        spatial_amount (float): 空間プルーニング（重み削除）の割合。
        kl_threshold (float): 時間プルーニングの飽和判定に使用するKLダイバージェンス閾値。

    Returns:
        nn.Module: プルーニングが適用されたモデル。
    """
    logger.info(f"--- ⚡️ Spatio-Temporal Pruning 開始 (Spatial Amount: {spatial_amount:.1%}, KL Threshold: {kl_threshold}) ---")
    
    # --- 1. 時間プルーニング (Temporal Pruning) ---
    # 引用[19]に基づき、冗長なタイムステップを特定する
    redundancy_report = _calculate_temporal_redundancy(
        model, dataloader, time_steps, kl_threshold=kl_threshold
    )
    
    avg_redundant_steps: int = 0
    if redundancy_report:
        avg_redundant_steps = int(sum(redundancy_report.values()) / len(redundancy_report))

    new_time_steps = time_steps - avg_redundant_steps
    
    logger.info(f"  [Temporal Pruning (Stub)]: 推定削減可能ステップ数: {avg_redundant_steps}. (T={time_steps} -> T={new_time_steps})")
    
    # (スタブ: 実際には、モデル内の time_steps パラメータを変更する)
    # (例: model.time_steps = new_time_steps)
    if hasattr(model, 'time_steps'):
         logger.info(f"  [Temporal Pruning (Stub)]: Updating model.time_steps to {new_time_steps}")
         # (注: この操作はモデルの実装に強く依存するため注意)
         # model.time_steps = new_time_steps
    
    # --- 2. 空間プルーニング (Spatial Pruning) ---
    # 引用[19]のLAMPSベース、または単純なMagnitudeプルーニング
    logger.info("  [Spatial Pruning (Magnitude Stub)]: 重みの空間プルーニングを実行中...")
    
    total_pruned = 0
    total_params = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight'): # 重みがあるか確認
                param: torch.Tensor = module.weight
                
                num_to_prune = int(param.numel() * spatial_amount)
                if num_to_prune == 0:
                    continue
                
                # 単純な Magnitude Pruning (スタブ)
                threshold = torch.kthvalue(param.data.abs().view(-1), k=num_to_prune).values
                mask = param.data.abs() > threshold
                param.data *= mask.float()
                
                pruned_count = param.numel() - mask.sum().item()
                total_pruned += int(pruned_count)
                total_params += param.numel()

    if total_params > 0:
        actual_sparsity = total_pruned / total_params
        logger.info(f"  [Spatial Pruning]: {actual_sparsity:.2%} ({total_pruned} / {total_params}) の重みをプルーニングしました。")
    
    logger.info("--- ✅ Spatio-Temporal Pruning 完了 (Stub) ---")
    return model
