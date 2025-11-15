# ファイルパス: snn_research/training/losses.py
# Title: 損失関数定義 (Distillation, Regularization)
#
# 機能の説明: 知識蒸留 (KD)、スパイク正則化、スパース性正則化など、
# SNNの学習に使用するカスタム損失関数を定義する。
#
# 【修正内容 v32: ImportError (cannot import name 'CombinedLoss') の修正】
# - health-check 実行時に 'ImportError: cannot import name 'CombinedLoss''
#   が発生する問題に対処します。
# - v31.2 の循環インポート修正時に、'CombinedLoss' クラス
#   がファイルから欠落していました。
# - (L: 186) 'CombinedLoss'
#   の定義をファイル末尾に追加しました。
# - (v31.2の修正も維持) 'BaseLoss' (L: 24)
#   および 'CombinedLoss' (L: 186)
#   が 'SNNCore' ではなく
#   'nn.Module' を継承するようにし、循環インポートを回避します。
#
# 【修正内容 v31.2: 循環インポート (Circular Import) の修正】
# - (v32でも維持) 'SNNCore'
#   へのインポート (L: 21) を削除。
# - (v32でも維持) 継承元を 'nn.Module' に変更 (L: 24, 76, 122)。
#
# 【修正内容 v_health_check_fix: 'SelfSupervisedLoss' 等の ImportError 修正】
# - (L: 236 以降) health-check 実行時に 'trainers.py' (L27) が
#   インポートしようとして失敗する 'SelfSupervisedLoss', 'PhysicsInformedLoss',
#   'PlannerLoss', 'ProbabilisticEnsembleLoss' のクラス定義を追加しました。
# - (L: 19) typing のインポートを整理しました。

import torch
import torch.nn as nn
import torch.nn.functional as F
# (v32: 'CombinedLoss' の 'List' のために修正)
from typing import Dict, Any, Optional, Tuple, List, Union

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 (v32でも維持) !!!】 ▼▼▼
# (from snn_research.core.snn_core import SNNCore を削除)
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲


# === 1. 基底損失クラス (v17) ===

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 (v32でも維持) !!!】 ▼▼▼
class BaseLoss(nn.Module): # 'SNNCore' -> 'nn.Module' に変更
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲
    """
    (v17)
    カスタム損失関数の基底クラス。
    重み (weight) の管理と、'forward' のインターフェースを定義する。
    
    (v31.2) SNNCore ではなく nn.Module を継承する。
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        # (v17) 損失の重みを float として登録
        self.weight = float(weight)
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        (v17)
        損失を計算して返す (スカラーのテンソル)。
        """
        raise NotImplementedError
        
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        (v17)
        損失を計算し、重みを乗算して返す。
        """
        if self.weight == 0:
            return torch.tensor(0.0, 
                                device=self._get_device_from_args(*args, **kwargs),
                                dtype=torch.float32)
            
        # (v17) 'BaseLoss.forward' ではなく 'super().__call__' (nn.Module.forward) を呼ぶ
        loss = super().__call__(*args, **kwargs)
        
        # (v_health_check_fix) 損失が辞書の場合 (仮実装)
        if isinstance(loss, dict):
             if "total" not in loss:
                 # 'total' がなければ、仮に 0.0 を返す
                 return torch.tensor(0.0, 
                                     device=self._get_device_from_args(*args, **kwargs),
                                     dtype=torch.float32)
             
             # (v_health_check_fix) 'CombinedLoss' でない場合、
             # 'total' を返すべき (trainers.py L311 など)
             if not isinstance(self, CombinedLoss):
                 return loss["total"] * self.weight
             
             # CombinedLoss は辞書を返さず 'total' を返す (L229)
             # ここには到達しないはず
             return loss["total"] * self.weight

        return loss * self.weight

    def _get_device_from_args(self, *args, **kwargs) -> torch.device:
        """ (v17) 引数からデバイスを推測する """
        if args and isinstance(args[0], torch.Tensor):
            return args[0].device
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                return v.device
        # (v_health_check_fix) 'model' kwargs からの推測 (trainers.py L256)
        if 'model' in kwargs and isinstance(kwargs['model'], nn.Module):
            try:
                # 'model' の最初のパラメータからデバイスを取得
                return next(kwargs['model'].parameters()).device
            except StopIteration:
                pass # パラメータがない場合
        return torch.device('cpu')


# === 2. 知識蒸留 (KD) 損失 (v17) ===

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 (v32でも維持) !!!】 ▼▼▼
class DistillationLoss(BaseLoss): # BaseLoss (nn.Module) を継承
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲
    """
    (v17)
    知識蒸留 (Knowledge Distillation) 損失。
    (HPO (Turn 5) (L:341) の 'distill_loss' (KD) と 
     'ce_loss' を計算するために使用)
    """
    def __init__(
        self,
        ce_weight: float = 0.5,
        distill_weight: float = 0.5,
        temperature: float = 2.0,
        **kwargs # (v17: BaseLoss の weight を吸収)
    ):
        # (v17) BaseLoss の weight は 1.0 (固定)
        super().__init__(weight=1.0) 
        
        self.ce_weight = float(ce_weight)
        self.distill_weight = float(distill_weight)
        self.temperature = float(temperature)
        
        # (v17) KLダイバージェンス損失 (KD用)
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        # (v_health_check_fix) trainers.py L645 の呼び出しに対応
        spikes: Optional[torch.Tensor] = None, 
        mem: Optional[torch.Tensor] = None, 
        model: Optional[nn.Module] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]: # (v_health_check_fix) 辞書を返すよう変更
        """
        (v17)
        Args:
            student_logits: 生徒モデルの出力 (B, NumClasses)
            teacher_logits: 教師モデルの出力 (B, NumClasses)
            targets: 正解ラベル (B,)

        Returns:
            (v_health_check_fix) 
            Dict[str, torch.Tensor]: {"total": ..., "ce_loss": ..., "kd_loss": ...}
        """
        
        # 1. CrossEntropy (CE) 損失
        ce_loss = F.cross_entropy(student_logits, targets)
        
        # 2. Kullback-Leibler (KL) 損失 (KD)
        kd_loss = self.kl_div_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2) # (T^2 スケーリング)
        
        # 3. 総損失 (重み付け)
        total_loss = (self.ce_weight * ce_loss) + \
                     (self.distill_weight * kd_loss)
                     
        # (v_health_check_fix) 辞書形式で返す
        return {"total": total_loss, "ce_loss": ce_loss, "kd_loss": kd_loss}


# === 3. スパイク正則化損失 (v17) ===

# --- ▼▼▼ 【!!! 修正 v31.2: 循環インポート修正 (v32でも維持) !!!】 ▼▼▼
class SpikeRegularizationLoss(BaseLoss): # BaseLoss (nn.Module) を継承
# --- ▲▲▲ 【!!! 修正 v31.2】 ▲▲▲
    """
    (v17)
    スパイク発火率 (平均スパイク数) に対する L2 正則化損失。
    (HPO (Turn 5) (L:341) の 'spike_reg_loss' を計算)
    """
    def __init__(self, weight: float = 1e-4, target_rate: float = 0.0):
        super().__init__(weight=weight)
        self.target_rate = float(target_rate)
        
    def forward(self, 
                # (v_health_check_fix) trainers.py L256 の呼び出しに対応
                avg_spikes: Optional[torch.Tensor] = None,
                logits: Optional[torch.Tensor] = None, 
                targets: Optional[torch.Tensor] = None, 
                spikes: Optional[torch.Tensor] = None, 
                mem: Optional[torch.Tensor] = None, 
                model: Optional[nn.Module] = None
                ) -> torch.Tensor:
        """
        (v17 / v_health_check_fix)
        Args:
            avg_spikes (torch.Tensor): 平均スパイク数 (スカラー)
                                     (CombinedLoss 経由で渡される)
        Returns:
            torch.Tensor: L2 損失 (スカラー)
        """
        # (v_health_check_fix) avg_spikes が渡されない場合 (CombinedLoss 外)
        # 'spikes' から計算を試みる (仮)
        if avg_spikes is None:
            if spikes is not None:
                # (T, B, N) or (B, T, N)
                avg_spikes = spikes.mean()
            else:
                # スパイク情報がなければ 0.0
                return torch.tensor(0.0, 
                                    device=self._get_device_from_args(
                                        logits, targets, spikes, mem),
                                    dtype=torch.float32)

        loss = (avg_spikes - self.target_rate) ** 2
        return loss


# === 4. スパース性正則化損失 (v17) ===

class SparsityRegularizationLoss(BaseLoss):
    """
    (v17)
    スパイク発火率 (平均スパイク数) に対する L1 正則化損失。
    (HPO (Turn 5) (L:341) の 'sparsity_loss' を計算)
    """
    def __init__(self, weight: float = 1e-4):
        super().__init__(weight=weight)
        
    def forward(self, 
                # (v_health_check_fix) trainers.py L256 の呼び出しに対応
                avg_spikes: Optional[torch.Tensor] = None,
                logits: Optional[torch.Tensor] = None, 
                targets: Optional[torch.Tensor] = None, 
                spikes: Optional[torch.Tensor] = None, 
                mem: Optional[torch.Tensor] = None, 
                model: Optional[nn.Module] = None
                ) -> torch.Tensor:
        """
        (v17 / v_health_check_fix)
        Args:
            avg_spikes (torch.Tensor): 平均スパイク数 (スカラー)
        Returns:
            torch.Tensor: L1 損失 (スカラー)
        """
        # (v_health_check_fix) avg_spikes が渡されない場合 (CombinedLoss 外)
        if avg_spikes is None:
            if spikes is not None:
                avg_spikes = spikes.mean()
            else:
                return torch.tensor(0.0, 
                                    device=self._get_device_from_args(
                                        logits, targets, spikes, mem),
                                    dtype=torch.float32)

        loss = torch.abs(avg_spikes)
        return loss


# --- ▼▼▼ 【!!! 修正 v32: 欠落していた 'CombinedLoss' を追加 !!!】 ▼▼▼
class CombinedLoss(BaseLoss):
    """
    (v17)
    複数の損失関数 (BaseLoss のサブクラス) を
    辞書またはリストで受け取り、合計するラッパー。
    
    (v31.2 / v32) BaseLoss (nn.Module) を継承
    
    (v_health_check_fix)
    trainers.py (L256) から呼び出される際、
    (logits, target_ids, spikes, mem, self.model) を受け取る。
    内部の各損失関数 (SpikeRegularizationLoss など) は、
    これらの引数から必要なものだけを選んで利用する。
    """
    def __init__(
        self, 
        loss_functions: Union[Dict[str, BaseLoss], List[BaseLoss]],
        **kwargs # (v17: BaseLoss の weight を吸収)
    ):
        super().__init__(weight=1.0) # (v17: ラッパー自体は重み 1.0)
        
        if isinstance(loss_functions, dict):
            self.loss_functions = nn.ModuleDict(loss_functions)
        elif isinstance(loss_functions, list):
            # (v17) 辞書に変換 (ModuleList は辞書を返せないため)
            self.loss_functions = nn.ModuleDict(
                {f"loss_{i}": fn for i, fn in enumerate(loss_functions)}
            )
        else:
            raise TypeError(
                f"loss_functions must be a dict or list, "
                f"got {type(loss_functions)}"
            )

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]: # 辞書を返す
        """
        (v17)
        全ての損失関数を実行し、合計 (スカラー) と
        個別の損失 (辞書) を返す。
        
        (v_health_check_fix)
        trainers.py (L311) が辞書 (loss_dict['total']) を期待するため、
        {"total": ..., "loss_name_1": ...} 形式の辞書を返す。
        
        Returns:
            Dict[str, torch.Tensor]: 
            {"total": 合計損失, "loss_name_1": 個別損失1, ...}
        """
        total_loss = torch.tensor(0.0, 
                                  device=self._get_device_from_args(*args, **kwargs),
                                  dtype=torch.float32)
        
        loss_dict: Dict[str, torch.Tensor] = {}
        
        # (v17) ModuleDict をイテレート
        for name, loss_fn in self.loss_functions.items():
            # (v17) BaseLoss.__call__ (重み付け) を実行
            # (v_health_check_fix)
            # DistillationLoss は辞書を返し、他はテンソルを返す
            loss_result = loss_fn(*args, **kwargs) 
            
            if isinstance(loss_result, dict):
                # DistillationLoss などの場合
                # "total" (重み付け済み) を合計に追加
                if "total" in loss_result:
                    total_loss += loss_result["total"]
                # 個別の損失も辞書に追加 (名前が重複しないように)
                for k, v in loss_result.items():
                    loss_dict[f"{name}_{k}"] = v
            else:
                # SpikeRegularizationLoss などの場合 (重み付け済み)
                loss = loss_result
                total_loss += loss
                loss_dict[name] = loss # 個別の損失を記録
            
        loss_dict["total"] = total_loss
        return loss_dict
# --- ▲▲▲ 【!!! 修正 v32】 ▲▲▲


# --- ▼▼▼ 【!!! 修正 v_health_check_fix: 欠落していたクラスを追加 !!!】 ▼▼▼

# === 5. 自己教師あり学習損失 (v_health_check_fix) ===
class SelfSupervisedLoss(BaseLoss):
    """
    (v_health_check_fix)
    自己教師あり学習 (TCLなど) のための損失。
    trainers.py (L27) でインポートエラーが発生していたため追加。
    """
    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(weight=weight)
        # 仮実装: CE損失を内部で持つ（TCLなどがCEベースの場合が多いため）
        # trainers.py (L353) で ignore_index を参照するため ce_loss_fn とする
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, 
                full_hiddens: torch.Tensor, 
                targets: torch.Tensor, 
                spikes: Optional[torch.Tensor] = None, 
                mem: Optional[torch.Tensor] = None, 
                model: Optional[nn.Module] = None
                ) -> Dict[str, torch.Tensor]: # 辞書を返す
        """
        (v_health_check_fix)
        仮実装。
        trainers.py (L722) の呼び出し (full_hiddens) に合わせる。
        ここでは hiddens の最後のタイムステップを使うと仮定。
        
        Args:
            full_hiddens (torch.Tensor): (T, B, S, H) または (B, S, H)
            targets (torch.Tensor): (B, S)
        """
        logits: torch.Tensor
        # (T, B, S, H) -> (B, S, H)
        if full_hiddens.dim() == 4:
            logits = full_hiddens[-1] 
        # (B, S, H)
        else:
            logits = full_hiddens

        # (B, S, H) -> (B*S, H)
        logits_flat = logits.reshape(-1, logits.size(-1))
        # (B, S) -> (B*S,)
        targets_flat = targets.reshape(-1)
        
        loss = self.ce_loss_fn(logits_flat, targets_flat)
        return {"total": loss, "ce_loss": loss}


# === 6. 物理情報損失 (v_health_check_fix) ===
class PhysicsInformedLoss(BaseLoss):
    """
    (v_health_check_fix)
    物理情報ニューラルネットワーク (PINN) のための損失。
    trainers.py (L27) でインポートエラーが発生していたため追加。
    """
    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(weight=weight)
        # (trainers.py L353) の ignore_index アクセスに対応 (仮)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse_loss = nn.MSELoss()

    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor, 
                spikes: Optional[torch.Tensor] = None, 
                mem: Optional[torch.Tensor] = None, 
                model: Optional[nn.Module] = None
                ) -> Dict[str, torch.Tensor]: # 辞書を返す
        """ 
        (v_health_check_fix) 
        仮実装。PINN は通常 MSE を使うが、
        trainers.py (L256) が CE と同じシグネチャで呼び出すため、
        CE (logits vs targets) としても仮実装する。
        """
        try:
            # CE損失 (分類タスク) を試行
            loss = self.ce_loss_fn(logits.reshape(-1, logits.size(-1)), 
                                 targets.reshape(-1))
            return {"total": loss, "ce_loss": loss}
        except (RuntimeError, ValueError):
            # CEが失敗した場合 (回帰タスクなど)、MSE にフォールバック
            loss = self.mse_loss(logits, targets)
            return {"total": loss, "physics_loss": loss}


# === 7. プランナー損失 (v_health_check_fix) ===
class PlannerLoss(BaseLoss):
    """
    (v_health_check_fix)
    プランナー (PlannerTrainer) のための損失。
    trainers.py (L27) でインポートエラーが発生していたため追加。
    """
    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(weight=weight)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, 
                skill_logits: torch.Tensor, 
                target_plan: torch.Tensor,
                # (v_health_check_fix) BaseLoss.forward との互換性
                spikes: Optional[torch.Tensor] = None, 
                mem: Optional[torch.Tensor] = None, 
                model: Optional[nn.Module] = None
                ) -> Dict[str, torch.Tensor]: # 辞書を返す
        """ 
        (v_health_check_fix) 
        trainers.py (L913) の呼び出しに対応
        """
        loss = self.ce_loss(skill_logits.reshape(-1, skill_logits.size(-1)), 
                            target_plan.reshape(-1))
        return {"total": loss, "planner_loss": loss}


# === 8. 確率的アンサンブル損失 (v_health_check_fix) ===
class ProbabilisticEnsembleLoss(BaseLoss):
    """
    (v_health_check_fix)
    確率的アンサンブル (ProbabilisticEnsembleTrainer) のための損失。
    trainers.py (L27) でインポートエラーが発生していたため追加。
    """
    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(weight=weight)
        self.nll_loss = nn.NLLLoss(ignore_index=-100)
        # (trainers.py L853) の ignore_index アクセスに対応
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)


    def forward(self, 
                ensemble_logits_tensor: torch.Tensor, # (Ensemble, B, S, V) or (Ensemble, B, V)
                targets: torch.Tensor, 
                spikes: Optional[torch.Tensor] = None, 
                mem: Optional[torch.Tensor] = None, 
                model: Optional[nn.Module] = None
                ) -> Dict[str, torch.Tensor]: # 辞書を返す
        """ (v_health_check_fix) 仮実装 """
        
        # (Ensemble, B, ...) -> (B, ...)
        mean_logits = ensemble_logits_tensor.mean(dim=0) 
        log_probs = F.log_softmax(mean_logits, dim=-1)
        
        # (B, S, V) or (B, V)
        V = log_probs.size(-1)
        
        if log_probs.dim() == 3: # (B, S, V)
            loss = self.nll_loss(log_probs.reshape(-1, V), 
                                 targets.reshape(-1))
        else: # (B, V)
            loss = self.nll_loss(log_probs, targets)
            
        return {"total": loss, "nll_loss": loss}

# --- ▲▲▲ 【!!! 修正 v_health_check_fix】 ▲▲▲
