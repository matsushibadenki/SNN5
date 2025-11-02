# ファイルパス: snn_research/training/pruning.py
# (改修: SBC プルーニング実装)
# Title: 構造的プルーニング (SBC - Spiking Brain Compression)
# Description:
# doc/SNN開発：基本設計思想.md (セクション4.4, 引用[15]) に基づき、
# 高コストな反復プルーニングではなく、ヘッセ行列（損失の二次微分）を利用した
# ワンショット・プルーニング（SBC）を実装する。
#
# 実装概要 (スタブ):
# 1. (apply_sbc_pruning): プルーニング対象の層を特定する。
# 2. (_compute_hessian_diag): 損失の二次微分（ヘッセ行列の対角成分）を計算する。(ダミー実装)
# 3. (_compute_saliency): ヘッセ行列に基づき、各重みの重要度（Saliency）を計算する。
# 4. (prune_and_update_weights): 重要度が低い重みを削除（プルーニング）し、
#    残った重みを補正（Update）して損失の増加を最小限に抑える。
#
# 追加 (v2):
# - SNN5改善レポート (セクション4.1, 引用[19]) に基づき、
#   時空間プルーニング (Spatio-Temporal Pruning) のスタブを追加。
#
# 改善 (SNN5改善レポート 4.1 対応):
# - _calculate_temporal_redundancy のダミー実装を、KLダイバージェンスに
#   基づく飽和判定ロジック（のスタブ）に改善。

import torch
import torch.nn as nn
# --- ▼ 修正: 必要な型をインポート ▼ ---
from typing import List, Tuple, Dict, Any, cast, Optional, Type
import logging 
# --- ▲ 修正 ▲ ---
# --- ▼ 修正: SNN5改善レポート 4.1 対応 ▼ ---
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
import torch.nn.functional as F
# --- ▲ 修正 ▲ ---


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _compute_hessian_diag(model: nn.Module, loss_fn: nn.Module, dataloader: Any) -> Dict[str, torch.Tensor]:
    """
    ヘッセ行列の対角成分を計算する (スタブ)。
    実際には、データローダーからの少量のサンプルを使い、
    バックプロパゲーションを2回行うなどの手法（例: L-BFGS）が必要。
    """
    logger.info("Computing Hessian matrix diagonal (Stub)...")
    hessian_diag: Dict[str, torch.Tensor] = {}
    
    # --- ダミー実装 ---
    # 実際にはここでデータローダーを数バッチ回し、
    # 各パラメータの (d^2 L / d w^2) を計算する。
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad and param.dim() > 1:
            # 対角成分はパラメータと同じ形状を持つ
            # ダミーとして、パラメータの大きさに応じたランダムな正の値を設定
            hessian_diag[name] = torch.rand_like(param) * 0.1 + (param.data.abs() * 0.5) + 1e-6
    # --- ダミー実装終了 ---
    
    logger.info(f"Hessian diagonal computed (dummy) for {len(hessian_diag)} layers.")
    return hessian_diag

def _compute_saliency(param: torch.Tensor, hessian_diag: torch.Tensor) -> torch.Tensor:
    """
    SBC (Optimal Brain Compression) に基づく重みの重要度（Saliency）を計算する。
    Saliency = (1/2) * (w^2) * (H_ii)
    """
    return 0.5 * (param.data ** 2) * hessian_diag

@torch.no_grad()
def _prune_and_update_weights(
    module: nn.Module,
    param_name: str,
    saliency: torch.Tensor,
    amount: float
) -> Tuple[int, int]:
    """
    指定されたモジュールのパラメータをプルーニングし、重みを補正する (スタブ)。
    """
    param: torch.Tensor = getattr(module, param_name)
    
    # 1. プルーニングする重みを決定
    num_to_prune = int(param.numel() * amount)
    if num_to_prune == 0:
        return 0, param.numel()
        
    threshold = torch.kthvalue(saliency.view(-1), k=num_to_prune).values
    mask = saliency > threshold
    
    # 2. 重み補正 (SBCの核心部 - ダミー実装)
    # 実際には、SBCは削除する重み (w_j) が残りの重み (w_i) に
    # どのような影響を与えるか (H_ij) を考慮して w_i を更新する。
    # delta_w_i = - (H_ii)^-1 * H_ij * w_j
    # ここでは簡易的に、残った重みをスケーリングする（ダミー）
    
    # 簡易補正: 削除される重みの総和を残りの重みで割った値を、
    # 学習率でスケールして加算する（生物学的可塑性に近いダミー補正）
    # update_factor = (param.data * ~mask).sum() / (param.data * mask).sum().clamp(min=1e-6)
    # param.data[mask] += param.data[mask] * update_factor * 0.01 # 1%補正
    
    # 3. プルーニング (マスクを適用)
    param.data *= mask.float()
    
    original_count = param.numel()
    pruned_count = original_count - mask.sum().item()
    return int(pruned_count), original_count

def apply_sbc_pruning(
    model: nn.Module,
    amount: float,
    dataloader_stub: Any, # ヘッセ行列計算用のデータローダー (スタブ)
    loss_fn_stub: nn.Module # 損失関数 (スタブ)
) -> nn.Module:
    """
    指定されたモデルに、SBC (Spiking Brain Compression) ワンショット・プルーニングを適用する。

    Args:
        model (nn.Module): プルーニングを適用するモデル。
        amount (float): プルーニングする重みの割合 (0.0から1.0の間)。
        dataloader_stub (Any): ヘッセ行列計算用のデータローダー (現在は未使用)。
        loss_fn_stub (nn.Module): 損失関数 (現在は未使用)。

    Returns:
        nn.Module: プルーニングが適用されたモデル。
    """
    if not (0.0 < amount < 1.0):
        logger.warning(f"プルーニング量が無効です ({amount})。0.0から1.0の間の値を指定してください。プルーニングをスキップします。")
        return model

    logger.info(f"--- 🧠 Spiking Brain Compression (SBC) 開始 (Amount: {amount:.1%}) ---")

    # 1. ヘッセ行列（対角成分）を計算 (スタブ)
    hessian_diagonals = _compute_hessian_diag(model, loss_fn_stub, dataloader_stub)
    
    total_pruned = 0
    total_params = 0

    # 2. 各レイヤーの重要度を計算し、プルーニングと重み補正を実行
    # (グローバルプルーニングではなく、レイヤーごとに指定された割合をプルーニング)
    target_modules: List[Tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            target_modules.append((module, 'weight'))

    if not target_modules:
        logger.warning("プルーニング対象のパラメータが見つかりませんでした。")
        return model

    logger.info(f"SBC対象のレイヤー数: {len(target_modules)}")
    
    for module, param_name in target_modules:
        # モジュール名を取得 (mypy互換のためループを使用)
        full_param_name: str = ""
        for name, mod in model.named_modules():
             if mod is module:
                 full_param_name = f"{name}.{param_name}"
                 break
        
        if not full_param_name:
             logger.warning(f"  - モジュール名が見つかりません。スキップします。")
             continue

        if full_param_name in hessian_diagonals:
            param: torch.Tensor = getattr(module, param_name)
            hessian_diag = hessian_diagonals[full_param_name]
            
            # 3. 重要度を計算
            saliency = _compute_saliency(param, hessian_diag)
            
            # 4. プルーニングと重み補正 (スタブ)
            pruned, total = _prune_and_update_weights(module, param_name, saliency, amount)
            total_pruned += pruned
            total_params += total
            logger.info(f"  - レイヤー '{full_param_name}': {pruned}/{total} の重みをプルーニング (補正実行済スタブ)。")
        else:
            logger.warning(f"  - レイヤー '{full_param_name}': ヘッセ行列が見つからず、スキップしました。")

    if total_params > 0:
        actual_sparsity = total_pruned / total_params
        logger.info(f"--- ✅ SBC 完了 ---")
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
