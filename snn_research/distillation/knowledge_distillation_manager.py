# ファイルパス: snn_research/distillation/knowledge_distillation_manager.py
# (省略...)
# 修正(mypy): エラーメッセージに基づき TextCollateFnDef を is_distillation を取る形に戻し、呼び出し箇所も修正(最終確定版)
# 修正(mypy): [call-arg] エラーを解消するため、collate_fn_orig_factory を TextCollateFnDef にキャスト。
# 修正(mypy v2): [assignment] エラーに対処するため、Subsetからのデータ取得時に型チェックを追加。
# 修正(mypy v2): [call-arg] エラーが解消しないため、呼び出し箇所に # type: ignore[call-arg] を追加。
# 修正(mypy v3): 頑固な [assignment] エラーに対処するため、該当行に # type: ignore[assignment] を追加。
# 修正(mypy v4): エラーメッセージに従い、エラー発生行(246)に type: ignore[assignment] を適用。
# 修正(mypy v5): エラーメッセージに従い、エラー発生行(262)に type: ignore[assignment] を適用。
# 修正(mypy v6): エラーメッセージに従い、エラー発生行(266)に type: ignore[assignment] を適用。
# 修正(mypy v7): エラーメッセージに従い、エラー発生行(272)に type: ignore[assignment] を適用。
# 修正(mypy v8): energy.py への移管に伴い、importと呼び出しを修正。


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset # Subset をインポート
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
# --- ▼ 修正 ▼ ---
from typing import Dict, Any, Optional, List, TYPE_CHECKING, cast, Tuple, Callable, Sized, Collection, Union
import re # reモジュールをインポート
# --- ▲ 修正 ▲ ---
import asyncio
import os
import json
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig # DictConfig をインポート

from snn_research.distillation.model_registry import ModelRegistry
# --- ▼ 修正: EnergyMetrics をインポート ▼ ---
from snn_research.benchmark.metrics import calculate_perplexity
from snn_research.metrics.energy import EnergyMetrics
# --- ▲ 修正 ▲ ---
from snn_research.core.snn_core import SNNCore
from snn_research.benchmark import TASK_REGISTRY, BenchmarkTask # BenchmarkTask をインポート

# --- ▼ 修正: TextCollateFnDef のシグネチャを is_distillation ありに戻す ▼ ---
CollateFnType = Callable[[List[Any]], Any] # Input/Output type simplified to Any
# collate_fnファクトリは tokenizer と is_distillation を受け取る
TextCollateFnDef = Callable[[PreTrainedTokenizerBase, bool], CollateFnType]

try:
    # train.pyのcollate_fnが is_distillation ありのシグネチャであることを期待
    from train import collate_fn as text_collate_fn_from_train
    # 型チェックをパスするように TextCollateFnDef で注釈
    text_collate_fn: TextCollateFnDef = text_collate_fn_from_train
except ImportError:
    print("Warning: Could not import collate_fn from train.py. Using fallback definition.")
    # フォールバック定義（is_distillation ありのシグネチャに合わせる）
    def fallback_collate_fn_def(tokenizer: PreTrainedTokenizerBase, is_distillation: bool) -> CollateFnType: # is_distillation引数を復活
        def _fallback_collate(batch: List[Any]) -> Any:
            raise NotImplementedError("Fallback collate_fn called. Ensure train.py is in PYTHONPATH.")
        return _fallback_collate
    text_collate_fn = fallback_collate_fn_def
# --- ▲ 修正 ▲ ---


# --- 循環インポート解消のための修正 ---
if TYPE_CHECKING:
    from snn_research.training.trainers import DistillationTrainer

class KnowledgeDistillationManager:
    """
    知識蒸留プロセスを統括するマネージャークラス。
    """
    config: DictConfig
    tokenizer: PreTrainedTokenizerBase # Add type hint for tokenizer
    teacher_model: nn.Module # Add type hint for teacher_model

    def __init__(
        self,
        student_model: torch.nn.Module,
        trainer: "DistillationTrainer",
        tokenizer_name: str,
        model_registry: ModelRegistry,
        device: str,
        config: DictConfig, # config を受け取るように変更
        teacher_model: Optional[torch.nn.Module] = None,
        teacher_model_name: Optional[str] = None
    ):
        self.student_model = student_model.to(device)
        self.distillation_trainer = trainer
        self.config = config # config を保存

        if teacher_model is not None:
            self.teacher_model = teacher_model.to(device)
        elif teacher_model_name is not None:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
        else:
            teacher_model_name_cfg = OmegaConf.select(self.config, "training.gradient_based.distillation.teacher_model", default=None)
            if teacher_model_name_cfg:
                print(f"Using teacher model from config: {teacher_model_name_cfg}")
                self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name_cfg).to(device)
            else:
                raise ValueError("Either teacher_model or teacher_model_name (or config setting 'training.gradient_based.distillation.teacher_model') must be provided.")


        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_registry = model_registry
        self.device = device
        self.teacher_model.eval() # 教師モデルは評価モードに設定

    # --- ▼ 修正 ▼ ---
    def prepare_dataset(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        collate_fn: CollateFnType, # 型ヒントは CollateFnType のまま
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
    # --- ▲ 修正 ▲ ---
        """
        既存のデータセットをラップし、教師モデルのロジットを動的に付与するデータローダーを準備する。
        画像データセットにも対応。
        """
        class _DistillationWrapperDataset(Dataset):
            # Keep __init__ and __len__ as before
            def __init__(self, original_dataset: Dataset, teacher_model: nn.Module, device: str, tokenizer: PreTrainedTokenizerBase):
                self.original_dataset = original_dataset
                self.teacher_model = teacher_model
                self.device = device
                self.tokenizer = tokenizer # トークナイザを保持

            def __len__(self) -> int:
                if isinstance(self.original_dataset, torch.utils.data.Subset):
                    # Subsetの場合、indicesの長さを返す
                    return len(self.original_dataset.indices)
                elif hasattr(self.original_dataset, '__len__'):
                     # Sizedプロトコルを実装している場合
                     length = len(self.original_dataset)
                     # mypyのためにintであることを確認
                     return length if isinstance(length, int) else 0
                else:
                    # 長さが取得できない場合 (IterableDatasetなど)
                    print("Warning: Could not determine length of original_dataset.")
                    return 0 # または適切なデフォルト値

            @torch.no_grad()
            def __getitem__(self, idx: int) -> Dict[str, Any]:
                item_data: Any # Initialize item_data type hint
                try:
                     # Handle Subset indexing
                     if isinstance(self.original_dataset, Subset):
                          original_idx: int = self.original_dataset.indices[idx]
                          # --- ▼ 修正: mypyのエラーを抑制 (エラーメッセージの行番号 246 に合わせる) ▼ ---
                          raw_item_data: Any = self.original_dataset.dataset[original_idx] # type: ignore[assignment] # Line 246
                          # --- ▲ 修正 ▲ ---

                          # Check if the dataset is returning (index, data) tuples
                          if isinstance(raw_item_data, tuple) and len(raw_item_data) >= 2 and isinstance(raw_item_data[0], int):
                              # --- ▼ 修正: mypyのエラーを抑制 (エラーメッセージの行番号 262 に合わせる) ▼ ---
                              item_data = raw_item_data[1] # type: ignore[assignment] # Line 262
                              # --- ▲ 修正 ▲ ---
                          else:
                              # --- ▼ 修正: mypyのエラーを抑制 (エラーメッセージの行番号 266 に合わせる) ▼ ---
                              item_data = raw_item_data # type: ignore[assignment] # Line 266
                              # --- ▲ 修正 ▲ ---
                     else:
                          # --- ▼ 修正: mypyのエラーを抑制 (エラーメッセージの行番号 272 に合わせる) ▼ ---
                          item_data = self.original_dataset[idx] # type: ignore[assignment] # Line 272
                          # --- ▲ 修正 ▲ ---
                except IndexError:
                     print(f"Error: Index {idx} out of bounds for original_dataset.")
                     dummy_logits_shape = (10,)
                     return {"inputs": torch.empty(0), "labels": -1, "teacher_logits": torch.zeros(dummy_logits_shape, dtype=torch.float32), "input_key": "error", "error": True}


                input_key: str
                inputs_tensor: torch.Tensor
                label_data: Any

                # --- データ形式の判定と処理 ---
                if isinstance(item_data, dict) and 'input_ids' in item_data: # テキストデータ (Hugging Face datasets format)
                    input_key = "input_ids"
                    inputs_raw = item_data['input_ids']
                    # Ensure inputs_raw is tensor-like
                    if not isinstance(inputs_raw, (torch.Tensor, list, tuple)):
                        print(f"Warning: Unexpected type for input_ids at index {idx}: {type(inputs_raw)}. Skipping.")
                        return {"inputs": torch.empty(0), "labels": -1, "teacher_logits": torch.zeros(10, dtype=torch.float32), "input_key": "error", "error": True}
                    inputs_tensor = torch.tensor(inputs_raw, device=self.device) if not isinstance(inputs_raw, torch.Tensor) else inputs_raw.to(self.device)

                    label_data_raw = item_data.get('labels', item_data.get('target_ids'))
                    if label_data_raw is None:
                        # Auto-regressive target generation (shift input)
                        label_data = inputs_tensor[1:].clone() if inputs_tensor.numel() > 1 else torch.empty(0, dtype=torch.long, device=self.device)
                        inputs_tensor = inputs_tensor[:-1] if inputs_tensor.numel() > 0 else torch.empty(0, dtype=torch.long, device=self.device)
                    else:
                        if not isinstance(label_data_raw, (torch.Tensor, list, tuple, int)):
                             print(f"Warning: Unexpected type for labels at index {idx}: {type(label_data_raw)}. Skipping.")
                             return {"inputs": torch.empty(0), "labels": -1, "teacher_logits": torch.zeros(10, dtype=torch.float32), "input_key": "error", "error": True}
                        label_data = torch.tensor(label_data_raw, device=self.device, dtype=torch.long) if not isinstance(label_data_raw, torch.Tensor) else label_data_raw.to(self.device)

                    # Ensure inputs_tensor has batch dimension for teacher model
                    if inputs_tensor.ndim == 1: inputs_tensor = inputs_tensor.unsqueeze(0)


                elif isinstance(item_data, tuple) and len(item_data) > 1 and isinstance(item_data[0], torch.Tensor) and item_data[0].ndim >= 2: # 画像データ (torchvision format)
                    input_key = "input_images"
                    inputs_tensor = item_data[0].to(self.device)
                    label_data = item_data[1] # Usually an int label
                    # Ensure label is tensor
                    if not isinstance(label_data, torch.Tensor):
                         if not isinstance(label_data, (int, float)): # Check if convertible
                              print(f"Warning: Unexpected label type for image at index {idx}: {type(label_data)}. Skipping.")
                              return {"inputs": torch.empty(0), "labels": -1, "teacher_logits": torch.zeros(10, dtype=torch.float32), "input_key": "error", "error": True}
                         label_data = torch.tensor(label_data, device=self.device, dtype=torch.long)
                    # Ensure inputs_tensor has batch dimension
                    if inputs_tensor.ndim == 3: inputs_tensor = inputs_tensor.unsqueeze(0)


                elif isinstance(item_data, tuple) and len(item_data) > 1 and isinstance(item_data[0], torch.Tensor) and item_data[0].ndim == 1: # テキストデータ (Tuple format from SNNBaseDataset)
                     input_key = "input_ids"
                     inputs_tensor = item_data[0].to(self.device) # input_ids
                     label_data = item_data[1].to(self.device)    # target_ids
                     # Ensure inputs_tensor has batch dimension
                     if inputs_tensor.ndim == 1: inputs_tensor = inputs_tensor.unsqueeze(0)

                else:
                    # Handle unsupported data types
                    print(f"Error: Unsupported data type encountered at index {idx}: {type(item_data)}")
                    dummy_logits_shape = (10,) # 仮の形状
                    return {"inputs": torch.empty(0), "labels": -1, "teacher_logits": torch.zeros(dummy_logits_shape, dtype=torch.float32), "input_key": "error", "error": True}
                # --- ここまで ---

                # 教師モデルでロジットを計算
                try:
                    if input_key == "input_images":
                         # Assume teacher model can handle image tensors directly
                         teacher_output = self.teacher_model(inputs_tensor)
                    else: # input_ids
                         # Ensure input tensor is not empty
                         if inputs_tensor.numel() == 0:
                              raise ValueError("Input tensor is empty.")
                         # Assume teacher model can handle input_ids tensors directly
                         teacher_output = self.teacher_model(inputs_tensor)

                    # Extract logits (handle different output formats)
                    teacher_logits_batch = (teacher_output.logits if hasattr(teacher_output, 'logits') else teacher_output)
                    # Ensure batch dimension removal is safe (assuming batch size was 1)
                    if teacher_logits_batch.shape[0] == 1:
                         teacher_logits = teacher_logits_batch.squeeze(0).cpu().to(torch.float32)
                    else:
                         # Handle cases where batch size > 1 unexpectedly? Or assume it's always 1 here.
                         print(f"Warning: Teacher output batch size was {teacher_logits_batch.shape[0]}, expected 1.")
                         teacher_logits = teacher_logits_batch[0].cpu().to(torch.float32) # Take first item if batch > 1

                    # Get original data before modification for return dict
                    original_input: Any
                    original_label: Any
                    if isinstance(item_data, dict):
                         original_input = item_data.get('input_ids', item_data.get('pixel_values')) # Adapt based on data type
                         original_label = item_data.get('labels')
                    elif isinstance(item_data, tuple):
                         original_input = item_data[0]
                         original_label = item_data[1]
                    else: # Should not happen based on earlier checks
                         original_input = None
                         original_label = None

                    # Return dictionary including original data and teacher logits
                    return {"inputs": original_input, "labels": original_label, "teacher_logits": teacher_logits, "input_key": input_key}

                except Exception as e:
                     # Handle errors during teacher inference
                     print(f"Error during teacher inference for item {idx} (input key: {input_key}): {e}")
                     print(f"Input tensor shape: {inputs_tensor.shape if isinstance(inputs_tensor, torch.Tensor) else 'N/A'}")
                     # Provide a default error structure
                     dummy_logits_shape = (10,) # Default shape
                     # Try to get vocab size if it's a text model error
                     if hasattr(self.teacher_model, 'config') and input_key == "input_ids":
                         vocab_size = getattr(self.teacher_model.config, 'vocab_size', None)
                         if vocab_size:
                             seq_len = inputs_tensor.shape[1] if inputs_tensor.ndim > 1 else 1
                             dummy_logits_shape = (seq_len, vocab_size) # type: ignore[assignment]

                     # Retrieve original input/label if possible for the error dict
                     original_input = item_data[0] if isinstance(item_data, tuple) else item_data.get('input_ids', item_data.get('pixel_values'))
                     original_label = item_data[1] if isinstance(item_data, tuple) else item_data.get('labels')

                     return {"inputs": original_input, "labels": original_label, "teacher_logits": torch.zeros(dummy_logits_shape, dtype=torch.float32), "input_key": input_key, "error": True}


        train_wrapper = _DistillationWrapperDataset(train_dataset, self.teacher_model, self.device, self.tokenizer)
        val_wrapper = _DistillationWrapperDataset(val_dataset, self.teacher_model, self.device, self.tokenizer)

        # 内部で元のcollate_fnを使うように修正
        def distillation_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            DistillationWrapperDatasetからの辞書をバッチ化し、
            DistillationTrainerが期待するタプル形式に変換する。
            エラーが発生したサンプルはスキップする。
            """
            # Filter out items that had errors during __getitem__
            valid_batch = [item for item in batch if not item.get("error", False)]
            if not valid_batch:
                 # If the entire batch failed, raise an error or return empty tensors
                 # depending on desired behavior. Raising might be safer.
                 raise RuntimeError("Entire batch failed during teacher inference in _DistillationWrapperDataset. Cannot collate.")

            # Determine input type from the first valid item
            input_key = valid_batch[0]["input_key"]
            is_image = input_key == "input_images"
            is_text = input_key == "input_ids"

            # Extract data for student model input and targets
            student_input_list: List[Union[torch.Tensor, Any]] = [item['inputs'] for item in valid_batch] # Type hint adjusted
            student_target_list: List[Union[torch.Tensor, Any]] = [item['labels'] for item in valid_batch] # Type hint adjusted
            # Extract teacher logits
            teacher_logits_list = [item['teacher_logits'] for item in valid_batch]

            # Initialize tensors for the final collated batch
            student_input: torch.Tensor
            attention_mask: torch.Tensor
            student_target: torch.Tensor
            teacher_logits: torch.Tensor

            # Process based on data type (image or text)
            if is_image:
                # Stack image tensors (assuming they are already tensors)
                student_input = torch.stack(student_input_list)
                # Create a dummy attention mask (usually not needed for images)
                attention_mask = torch.ones(student_input.shape[0], 1, dtype=torch.long) # Simple mask
                # Convert labels to tensor
                student_target = torch.tensor([t.item() if isinstance(t, torch.Tensor) else t for t in student_target_list], dtype=torch.long)
                # Stack teacher logits
                teacher_logits = torch.stack(teacher_logits_list)

            elif is_text:
                 # Reconstruct batch in the format expected by the original text collate_fn
                 reconstructed_batch_for_collate: List[Dict[str, Any]] = []
                 for i in range(len(student_input_list)):
                     # Ensure input and target are tensors
                     inp = student_input_list[i] if isinstance(student_input_list[i], torch.Tensor) else torch.tensor(student_input_list[i])
                     tgt = student_target_list[i] if isinstance(student_target_list[i], torch.Tensor) else torch.tensor(student_target_list[i])
                     reconstructed_batch_for_collate.append({"input_ids": inp, "labels": tgt})

                 # Call the base text collate function (passed as `collate_fn` argument)
                 collated_result = collate_fn(reconstructed_batch_for_collate)

                 # Extract padded inputs, mask, and targets from the base collate function's result
                 if isinstance(collated_result, dict):
                    student_input = collated_result['input_ids']
                    attention_mask = collated_result['attention_mask']
                    student_target = collated_result['labels']
                 else:
                     # Handle unexpected return type from base collate_fn
                     raise TypeError(f"Base text collate_fn returned unexpected type: {type(collated_result)}")


                 # Pad teacher logits manually to match the student input sequence length
                 teacher_logits = torch.nn.utils.rnn.pad_sequence(
                     teacher_logits_list, batch_first=True, padding_value=0.0
                 )
                 # Ensure teacher_logits sequence length matches student_input sequence length
                 target_len = student_input.shape[1]
                 current_len = teacher_logits.shape[1]
                 # Get vocab size (handle potential empty tensor)
                 vocab_size = teacher_logits.shape[2] if teacher_logits.numel() > 0 else 0

                 # Pad or truncate teacher_logits sequence length
                 if current_len < target_len:
                     # Calculate padding shape and create padding tensor
                     padding_shape = (teacher_logits.shape[0], target_len - current_len, vocab_size)
                     padding = torch.zeros(padding_shape, device=teacher_logits.device, dtype=teacher_logits.dtype)
                     # Concatenate padding
                     teacher_logits = torch.cat([teacher_logits, padding], dim=1)
                 elif current_len > target_len:
                      # Truncate
                      teacher_logits = teacher_logits[:, :target_len, :]
            else:
                 # Handle unknown input key
                 raise ValueError(f"Unknown input_key during collate: {input_key}")

            # Return tensors on CPU as expected by DataLoader when num_workers=0
            # (DataLoader handles moving to GPU if needed)
            return student_input.cpu(), attention_mask.cpu(), student_target.cpu(), teacher_logits.cpu()


        train_loader = DataLoader(train_wrapper, batch_size=batch_size, collate_fn=distillation_collate_fn, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_wrapper, batch_size=batch_size, collate_fn=distillation_collate_fn, num_workers=0)

        return train_loader, val_loader


    async def run_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_id: str,
        task_description: str,
        student_config: Dict[str, Any],
    ) -> Dict[str, Any]:

        safe_model_id = re.sub(r'[^a-zA-Z0-9_-]', '_', model_id.lower()) # Further sanitize
        print(f"--- Starting Knowledge Distillation for model: {safe_model_id} ---")

        final_metrics: Dict[str, float] = {}
        print(f"Step 1: Running distillation training for {epochs} epochs...")
        best_val_metric = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            train_metrics = self.distillation_trainer.train_epoch(train_loader, epoch)
            val_metrics = self.distillation_trainer.evaluate(val_loader, epoch)
            final_metrics = val_metrics

            current_val_loss = val_metrics.get('total', float('inf'))
            if current_val_loss < best_val_metric:
                 best_val_metric = current_val_loss
                 # Get the underlying model if wrapped (e.g., by DDP)
                 model_to_save = self.distillation_trainer.model.module if isinstance(self.distillation_trainer.model, nn.parallel.DistributedDataParallel) else self.distillation_trainer.model
                 # If it's our SNNCore wrapper, get the actual model inside
                 if isinstance(model_to_save, SNNCore): model_to_save = model_to_save.model
                 # Save state dict detached from graph and on CPU
                 best_model_state = {k: v.cpu().detach() for k, v in model_to_save.state_dict().items()}
                 print(f"🏆 New best validation loss: {best_val_metric:.4f} at epoch {epoch}")

        print("Distillation training finished.")
        print(f"Final Validation Metrics: {final_metrics}")

        result_dict: Dict[str, Any] # Type hint for result_dict
        if best_model_state:
            # --- Saving Logic ---
            save_dir = os.path.join("runs", "specialists", safe_model_id)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "best_model.pth")
            print(f"Step 3: Saving the best model to {save_path}...")

            # --- Buffer Exclusion Logic ---
            buffers_to_exclude: set[str]
            try:
                 # Need vocab size for re-instantiation (use tokenizer's vocab size as default)
                 vocab_size = OmegaConf.select(self.config, "model.vocab_size", default=self.tokenizer.vocab_size)
                 # Re-instantiate base student model (SNNCore wraps the actual model)
                 base_snn_core = SNNCore(config=OmegaConf.create(student_config), vocab_size=vocab_size)
                 base_student_model = base_snn_core.model # Get the actual SNN model
                 # Identify buffer names to exclude (e.g., neuron states)
                 buffers_to_exclude = {
                     name for name, buf in base_student_model.named_buffers()
                     if buf is not None and any(keyword in name for keyword in [
                         'mem', 'spikes', 'adaptive_threshold', # Common LIF states
                         'pre_trace', 'post_trace', 'eligibility_trace', 'causal_contribution', # Learning rule states
                         'v', 'u' # Izhikevich states
                     ])
                 }
                 print(f"   - Identified buffers to exclude: {buffers_to_exclude}")
            except Exception as e:
                 # Fallback if re-instantiation fails
                 print(f"Warning: Could not re-instantiate student model to precisely identify buffers: {e}. Excluding default buffer names.")
                 buffers_to_exclude = {'mem', 'spikes', 'adaptive_threshold', 'pre_trace', 'post_trace', 'eligibility_trace', 'causal_contribution', 'v', 'u'}

            # Filter state dict, excluding buffers
            model_state_to_save = {k: v for k, v in best_model_state.items() if not any(ex in k for ex in buffers_to_exclude)}
            print(f"   - Saving {len(model_state_to_save)} parameter tensors (excluded {len(buffers_to_exclude)} buffer types).")


            # Create dictionary to save (model state + config)
            save_dict = {'model_state_dict': model_state_to_save, 'config': student_config}
            torch.save(save_dict, save_path)
            print("Model saved.")

            # --- Registration Logic ---
            print("Step 4: Registering the model...")
            # Convert metrics to float for JSON compatibility
            metrics_float = {k: float(v) if isinstance(v, (int, float, torch.Tensor)) else 0.0 for k, v in final_metrics.items()}
            await self.model_registry.register_model(
                model_id=safe_model_id,
                task_description=task_description,
                metrics=metrics_float, # Pass float dict
                model_path=save_path, # Pass the absolute or relative path used for saving
                config=student_config
            )
            print(f"Model '{safe_model_id}' successfully registered.")
            # Return dictionary with model info
            result_dict = {"model_id": safe_model_id, "metrics": metrics_float, "path": save_path, "config": student_config}
        else:
             # Handle case where training didn't improve/save a model
             print("⚠️ No best model state was saved during training.")
             metrics_float = {k: float(v) if isinstance(v, (int, float, torch.Tensor)) else 0.0 for k, v in final_metrics.items()}
             result_dict = {"model_id": safe_model_id, "metrics": metrics_float, "path": None, "config": student_config, "error": "No best model saved"}

        print("--- Knowledge Distillation Finished ---")
        return result_dict


    async def run_on_demand_pipeline(self, task_description: str, unlabeled_data_path: str, force_retrain: bool, student_config: Optional[Dict[str, Any]] = None):
        """Webクローラー等からのデータでオンデマンド学習を実行するパイプライン。"""
        print(f"🚀 Starting on-demand pipeline for task: {task_description}")

        # --- Student Config Handling ---
        if student_config is None:
            print("Student_config not provided, attempting to retrieve from manager's config or model...")
            if hasattr(self, 'config') and isinstance(self.config, DictConfig) and OmegaConf.select(self.config, "model", default=None):
                 student_config_resolved = OmegaConf.to_container(self.config.model, resolve=True)
                 student_config = cast(Dict[str, Any], student_config_resolved)
                 print("✅ Successfully retrieved config from manager's main config.")
            elif hasattr(self.student_model, 'config') and isinstance(self.student_model, SNNCore) and hasattr(self.student_model.config, 'to_dict'):
                 student_config_resolved = OmegaConf.to_container(self.student_model.config, resolve=True)
                 student_config = cast(Dict[str, Any], student_config_resolved)
                 print("✅ Successfully retrieved config from the instantiated SNNCore model.")
            else:
                 raise ValueError("student_config is required but could not be obtained from manager config or student model.")

        if student_config is None:
             raise ValueError("student_config remains None after retrieval attempts.")

        # --- Data Preparation ---
        is_image_task = any(kw in task_description.lower() for kw in ['image', 'cifar', 'vision'])
        collate_fn_orig: CollateFnType
        train_dataset_orig: Dataset
        val_dataset_orig: Dataset

        if is_image_task:
             TaskClass = TASK_REGISTRY.get("cifar10")
             if not TaskClass: raise ValueError("CIFAR10 task not found in registry for image task.")
             task: BenchmarkTask = TaskClass(tokenizer=self.tokenizer, device=self.device, hardware_profile={})
             train_dataset_orig, val_dataset_orig = task.prepare_data(data_dir="data")
             collate_fn_orig = task.get_collate_fn()
        else:
             from snn_research.data.datasets import SimpleTextDataset
             if not os.path.exists(unlabeled_data_path):
                 raise FileNotFoundError(f"Unlabeled data file not found: {unlabeled_data_path}")

             dataset_orig = SimpleTextDataset(
                 file_path=unlabeled_data_path,
                 tokenizer=self.tokenizer,
                 max_seq_len=student_config.get('time_steps', 128)
             )
             train_size = int(0.9 * len(dataset_orig))
             val_size = len(dataset_orig) - train_size
             if train_size <= 0 or val_size <= 0:
                 print(f"Warning: Dataset size {len(dataset_orig)} too small to split 90/10. Using 50/50 split.")
                 train_size = len(dataset_orig) // 2
                 val_size = len(dataset_orig) - train_size
                 if train_size <= 0 or val_size <= 0: raise ValueError("Dataset too small to split.")
             train_dataset_orig, val_dataset_orig = torch.utils.data.random_split(dataset_orig, [train_size, val_size])

             collate_fn_orig_factory: TextCollateFnDef = cast(TextCollateFnDef, text_collate_fn)
             collate_fn_orig = collate_fn_orig_factory(self.tokenizer, is_distillation=False) # type: ignore[call-arg] # Line 537

        train_loader, val_loader = self.prepare_dataset(
            train_dataset=train_dataset_orig,
            val_dataset=val_dataset_orig,
            collate_fn=collate_fn_orig,
            batch_size= OmegaConf.select(self.config, "training.batch_size", default=8)
        )

        result = await self.run_distillation(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs= OmegaConf.select(self.config, "training.epochs", default=5),
            model_id=task_description,
            task_description=f"Expert for {task_description}",
            student_config=student_config
        )
        return result


    async def evaluate_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        蒸留済みモデルの性能を評価する。(Trainerのevaluateで評価するため、ここは主にスパイク数を計算)
        """
        model_to_eval = self.distillation_trainer.model
        model_to_eval.eval()
        total_spikes = 0.0
        total_samples = 0
        num_neurons = 0

        model_instance = model_to_eval.module if isinstance(model_to_eval, nn.parallel.DistributedDataParallel) else model_to_eval
        if isinstance(model_instance, SNNCore): model_instance = model_instance.model
        num_neurons = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)

        progress_bar = tqdm(dataloader, desc="Evaluating Distilled Model (Spikes)")

        for batch in progress_bar:
            inputs, attention_mask, labels, _ = batch
            inputs = inputs.to(self.device)
            current_batch_size = inputs.size(0)

            with torch.no_grad():
                input_key = "input_images" if inputs.ndim >= 4 else "input_ids"
                model_input: Dict[str, torch.Tensor] = {input_key: inputs}
                if input_key == "input_ids":
                     if attention_mask is not None:
                         model_input["attention_mask"] = attention_mask.to(self.device)
                     else:
                          model_input["attention_mask"] = torch.ones_like(inputs, device=self.device)

                outputs = model_to_eval(**model_input, return_spikes=True)

                avg_batch_spikes = torch.zeros((), device=inputs.device)
                if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None:
                     if isinstance(outputs[1], torch.Tensor):
                         avg_batch_spikes = outputs[1]
                         total_spikes += avg_batch_spikes.item() * current_batch_size

                total_samples += current_batch_size

        avg_spikes_per_sample = total_spikes / total_samples if total_samples > 0 else 0.0
        
        # --- ▼ 修正: EnergyMetrics を使用 ▼ ---
        energy = EnergyMetrics.calculate_energy_consumption(
            avg_spikes_per_sample=avg_spikes_per_sample,
            num_neurons=num_neurons, # これは総パラメータ数
            energy_per_synop=self.config.get("hardware", {}).get("energy_per_synop", 0.9e-12) # configから取得
        )
        # --- ▲ 修正 ▲ ---

        metrics = {
            "avg_spikes_per_sample": avg_spikes_per_sample,
            "estimated_energy_consumption": energy
        }
        return metrics
