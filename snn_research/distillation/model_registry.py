# ファイルパス: snn_research/distillation/model_registry.py
# (修正)
#
# タイトル: モデルレジストリ
# 機能説明: find_models_for_taskメソッドの末尾にあった余分なコロンを削除し、SyntaxErrorを修正。
#
# 改善点:
# - ROADMAPフェーズ8に基づき、マルチエージェント間の知識共有を可能にする
#   分散型モデルレジストリ(DistributedModelRegistry)を実装。
# - ファイルロック機構を導入し、複数プロセスからの同時書き込みによる
#   レジストリファイルの破損を防止する。
#
# 改善点 (v2):
# - ROADMAPフェーズ4「社会学習」に基づき、エージェントがスキル（モデル）を
#   共有するための`publish_skill`および`download_skill`メソッドを実装。
#
# 改善点 (v3):
# - 複数プロセスからの同時書き込みの堅牢性を向上させるため、
#   一時ファイルへの書き込みとアトミックなリネーム処理を導入。
#
# 修正 (v4):
# - mypyエラー[import-not-found]を解消するため、loggingをインポート。
# - mypyエラー[name-defined]を解消するため、Optionalをインポート。
# - __init__ で registry_path が None の場合のフォールバックを強化。
#
# 修正 (v5):
# - SyntaxError: 196行目の `except` ブロックのインデントを修正。
#
# 修正 (v6):
# - mypyエラー [union-attr] [arg-type] を解消するため、
#   self.registry_path を使用するメソッドに `assert self.registry_path is not None` を追加。
#
# 修正 (v7):
# - mypyエラー [return] を解消するため、_execute_with_lock に戻り値の型ヒント `-> Any` を追加。

from abc import ABC, abstractmethod
# --- ▼ 修正 ▼ ---
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import fcntl
import time
import shutil
import os 
import logging # logging をインポート
# --- ▲ 修正 ▲ ---

logger = logging.getLogger(__name__) # ロガーを設定

class ModelRegistry(ABC):
    """
    専門家モデルを管理するためのインターフェース。
    """
    registry_path: Optional[Path] # registry_path 属性を定義

    @abstractmethod
    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        """新しいモデルをレジストリに登録する。"""
        pass

    @abstractmethod
    async def find_models_for_task(self, task_description: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """特定のタスクに最適なモデルを検索する。"""
        pass

    @abstractmethod
    async def get_model_info(self, model_id: str) -> Dict[str, Any] | None:
        """モデルIDに基づいてモデル情報を取得する。"""
        pass

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """登録されているすべてのモデルのリストを取得する。"""
        pass


class SimpleModelRegistry(ModelRegistry):
    """
    JSONファイルを使用したシンプルなモデルレジストリの実装。
    """
    # --- ▼ 修正: __init__ の型ヒントを Optional[str] にし、フォールバックを強化 ▼ ---
    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            logger.warning("Registry path is None, falling back to default 'runs/model_registry.json'")
            registry_path = "runs/model_registry.json"
        
        self.registry_path = Path(registry_path)
        self.project_root = self.registry_path.resolve().parent.parent
        self.models: Dict[str, List[Dict[str, Any]]] = self._load()
    # --- ▲ 修正 ▲ ---

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        # --- ▼ 修正: mypy [union-attr] [arg-type] ▼ ---
        assert self.registry_path is not None, "registry_path is not initialized"
        # --- ▲ 修正 ▲ ---
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content:
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def _save(self) -> None:
        # --- ▼ 修正: mypy [union-attr] [arg-type] ▼ ---
        assert self.registry_path is not None, "registry_path is not initialized"
        # --- ▲ 修正 ▲ ---
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # 改善: アトミックな書き込み処理
        temp_path = self.registry_path.with_suffix(f"{self.registry_path.suffix}.tmp")
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.models, f, indent=4, ensure_ascii=False)
            # ファイルをアトミックにリネームして上書き
            os.rename(temp_path, self.registry_path)
        except Exception as e:
            logger.error(f"Failed to save model registry atomically: {e}")
            if temp_path.exists():
                os.remove(temp_path) # 一時ファイルを削除


    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        if task_description not in self.models:
            self.models[task_description] = []
        
        model_info = {
            "model_path": model_path,
            "metrics": metrics,
            "config": config,
            "task_description": task_description, # タスク説明も保存
            "registration_date": time.time()
        }
        
        # 既存のモデルを上書きするか、リストに追加するか（ここでは最新のものを先頭に追加）
        self.models[task_description].insert(0, model_info)
        
        self._save()
        print(f"Model for task '{model_id}' registered at '{model_path}'.")

    async def find_models_for_task(self, task_description: str, top_k: int = 1) -> List[Dict[str, Any]]:
        if task_description in self.models:
            models_for_task = self.models[task_description]
            
            # メトリクス（例：accuracy）でソート
            models_for_task.sort(
                key=lambda x: x.get("metrics", {}).get("accuracy", 0),
                reverse=True
            )

            resolved_models = []
            for model_info in models_for_task[:top_k]:
                # パスを絶対パスに解決
                relative_path_str = model_info.get('model_path') or model_info.get('path')
                
                if relative_path_str:
                    absolute_path = Path(relative_path_str).resolve()
                    model_info['model_path'] = str(absolute_path)

                model_info['model_id'] = task_description # 検索キーをmodel_idとして追加
                resolved_models.append(model_info)
            
            return resolved_models
        return []


    async def get_model_info(self, model_id: str) -> Dict[str, Any] | None:
        # このシンプルな実装では、model_id = task_description と仮定
        models = self.models.get(model_id)
        if models:
            # 最初の（通常は最新または最高の）モデル情報を返す
            model_info = models[0] 
            relative_path_str = model_info.get('model_path') or model_info.get('path')
            if relative_path_str:
                absolute_path = Path(relative_path_str).resolve()
                model_info['model_path'] = str(absolute_path)
            return model_info
        return None

    async def list_models(self) -> List[Dict[str, Any]]:
        all_models = []
        for model_id, model_list in self.models.items():
            for model_info in model_list:
                model_info_with_id = {'model_id': model_id, **model_info}
                all_models.append(model_info_with_id)
        return all_models


class DistributedModelRegistry(SimpleModelRegistry):
    """
    ファイルロックを使用して、複数のプロセスからの安全なアクセスを保証する
    分散環境向けのモデルレジストリ。社会学習機能も持つ。
    """
    def __init__(self, registry_path: Optional[str] = None, timeout: int = 10, shared_skill_dir: str = "runs/shared_skills"):
        # --- ▼ 修正: registry_path が None の場合のフォールバック ▼ ---
        if registry_path is None:
             registry_path = "runs/model_registry.json"
        # --- ▲ 修正 ▲ ---
        super().__init__(registry_path)
        self.timeout = timeout
        self.shared_skill_dir = Path(shared_skill_dir)
        self.shared_skill_dir.mkdir(parents=True, exist_ok=True)

    # --- ▼ 修正: mypy [return] (戻り値の型ヒント `-> Any` を追加) ▼ ---
    def _execute_with_lock(self, mode: str, operation, *args, **kwargs) -> Any:
    # --- ▲ 修正 ▲ ---
        """ファイルロックを取得して操作を実行する。"""
        # --- ▼ 修正: mypy [union-attr] [arg-type] ▼ ---
        assert self.registry_path is not None, "registry_path is not initialized"
        # --- ▲ 修正 ▲ ---
        start_time = time.time()
        # 'a+' モードは読み書き可能で、ファイルが存在しない場合は作成する
        with open(self.registry_path, 'a+', encoding='utf-8') as f:
            while time.time() - start_time < self.timeout:
                try:
                    lock_type = fcntl.LOCK_EX if mode == 'w' else fcntl.LOCK_SH
                    fcntl.flock(f, lock_type | fcntl.LOCK_NB) # 非ブロッキングモード
                    f.seek(0) # ファイルの先頭に戻る
                    result = operation(f, *args, **kwargs)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return result
                except (IOError, BlockingIOError):
                    # ロックが取得できない場合は少し待つ
                    time.sleep(0.1)
        # --- ▼ 修正: mypy [return] (raise を with ブロックの外に移動) ▼ ---
        raise IOError(f"レジストリの{'書き込み' if mode == 'w' else '読み取り'}ロックの取得に失敗しました。")
        # --- ▲ 修正 ▲ ---

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        """ロックを取得してレジストリファイルを読み込む。"""
        def read_operation(f) -> Dict[str, List[Dict[str, Any]]]:
            try:
                content = f.read()
                if not content:
                    return {}
                return json.loads(content)
            # --- ▼ 修正: インデントを修正 ▼ ---
            except json.JSONDecodeError:
                return {}
            # --- ▲ 修正 ▲ ---
        # --- ▼ 修正: self.registry_path が存在しない場合、空の辞書を返す ▼ ---
        # --- ▼ 修正: mypy [union-attr] ▼ ---
        assert self.registry_path is not None, "registry_path is not initialized"
        # --- ▲ 修正 ▲ ---
        if not self.registry_path.exists():
            return {}
        # --- ▲ 修正 ▲ ---
        return self._execute_with_lock('r', read_operation)

    def _save(self) -> None:
        """ロックを取得してレジストリファイルに書き込む（アトミック処理）。"""
        # --- ▼ 修正: mypy [union-attr] ▼ ---
        assert self.registry_path is not None, "registry_path is not initialized"
        # --- ▲ 修正 ▲ ---
        models_to_save = self.models # 保存する現在の状態

        def write_operation(f, models_data):
            # アトミック書き込みのために一時ファイルを使用
            temp_path = self.registry_path.with_suffix(f"{self.registry_path.suffix}.tmp")
            try:
                with open(temp_path, 'w', encoding='utf-8') as temp_f:
                    json.dump(models_data, temp_f, indent=4, ensure_ascii=False)
                # アトミックにリネーム
                os.rename(temp_path, self.registry_path)
            except Exception as e:
                logger.error(f"Failed to save model registry atomically: {e}")
                if temp_path.exists():
                    os.remove(temp_path) # 一時ファイルを削除
            
            # ロックファイル（f）の内容を更新（seek/truncateは不要）
            f.seek(0)
            f.truncate()
            json.dump(models_data, f, indent=4, ensure_ascii=False)

        self._execute_with_lock('w', write_operation, models_to_save)


    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        """ロックを使用してモデルを登録する。"""
        # 登録ロジック自体はSimpleModelRegistryと同じだが、
        # _loadと_saveがロックされるため、操作全体がアトミックになる。
        self.models = self._load() # 最新の状態を読み込む
        await super().register_model(model_id, task_description, metrics, model_path, config)
        # _save() は super().register_model() 内で呼ばれる

    async def publish_skill(self, model_id: str) -> bool:
        """
        学習済みモデル（スキル）を共有ディレクトリに公開する。
        """
        # ロックを取得してレジストリを読み書き
        self.models = self._load()
        model_info_list = self.models.get(model_id)
        if not model_info_list:
            print(f"❌ 公開失敗: モデル '{model_id}' は登録されていません。")
            return False
        
        model_info = model_info_list[0]
        src_path = Path(model_info['model_path'])
        if not src_path.exists():
            print(f"❌ 公開失敗: モデルファイルが見つかりません: {src_path}")
            return False

        dest_path = self.shared_skill_dir / f"{model_id}.pth"
        shutil.copy(src_path, dest_path)
        
        model_info['published'] = True
        model_info['shared_path'] = str(dest_path)
        self._save() # ロックして保存
        print(f"🌍 スキル '{model_id}' を共有ディレクトリに公開しました: {dest_path}")
        return True

    async def download_skill(self, model_id: str, destination_dir: str) -> Dict[str, Any] | None:
        """
        共有ディレクトリからスキルをダウンロードし、自身のレジストリに登録する。
        """
        # ロックして読み取り
        self.models = self._load()
        # 他のエージェントが公開したスキルを探す
        # ここでは簡略化のため、自身のレジストリからpublished=Trueのものを探す
        all_published = [
            {'model_id': mid, **info}
            for mid, info_list in self.models.items()
            for info in info_list if info.get('published')
        ]
        
        target_skill = next((s for s in all_published if s['model_id'] == model_id), None)

        if not target_skill or not target_skill.get('shared_path'):
            print(f"❌ ダウンロード失敗: 共有スキル '{model_id}' が見つかりません。")
            return None

        src_path = Path(target_skill['shared_path'])
        if not src_path.exists():
            print(f"❌ ダウンロード失敗: 共有ファイルが見つかりません: {src_path}")
            return None

        dest_path = Path(destination_dir) / f"{model_id}.pth"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, dest_path)

        # ダウンロードしたスキルを自身のレジストリに登録
        # (register_modelが内部で_load/_saveをロック付きで行う)
        new_local_info = target_skill.copy()
        new_local_info['model_path'] = str(dest_path)
        
        await self.register_model(
            model_id=model_id,
            task_description=new_local_info['task_description'],
            metrics=new_local_info['metrics'],
            model_path=new_local_info['model_path'],
            config=new_local_info['config']
        )
        print(f"✅ スキル '{model_id}' をダウンロードし、ローカルに登録しました: {dest_path}")
        return new_local_info