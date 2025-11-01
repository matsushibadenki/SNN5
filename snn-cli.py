# ファイルパス: snn-cli.py
# Title: SNNプロジェクト 統合CLIツール
# Description:
#   学習、推論、自己進化、人工脳シミュレーションなど、プロジェクトの全機能を
#   単一のインターフェースから制御するためのコマンドラインツール。
#   Typerライブラリを使用し、サブコマンド形式で機能を提供する。
# 改善点(v2): 各サブコマンドの引数を、呼び出し先のスクリプトと完全に一致するように修正・統一。
# 改善点(v3): benchmark runコマンドにmrpc_comparisonを追加。
# 改善点(v4): ann2snn-cnnコマンドがscripts/convert_model.pyを呼び出すように修正。
# 改善点(v5): HPO (Hyperparameter Optimization) コマンドを追加。
import typer
from typing import Optional, List
import subprocess
import sys

app = typer.Typer()
agent_app = typer.Typer()
app.add_typer(agent_app, name="agent")

benchmark_app = typer.Typer()
app.add_typer(benchmark_app, name="benchmark")

convert_app = typer.Typer()
app.add_typer(convert_app, name="convert")

# --- HPOコマンドを追加 ---
hpo_app = typer.Typer()
app.add_typer(hpo_app, name="hpo")
# --- ここまで ---

def _run_command(command: List[str]):
    """コマンドを実行し、出力をストリーミングする。"""
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8')
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
        process.wait()
        if process.returncode != 0:
            typer.echo(f"Error: Command failed with exit code {process.returncode}")
    except FileNotFoundError:
        typer.echo(f"Error: Command '{command[0]}' not found.")
    except Exception as e:
        typer.echo(f"An unexpected error occurred: {e}")

# --- (agent, life-form, planner, brain コマンドは変更なし) ---
@agent_app.command("solve")
def agent_solve(
    task: str = typer.Argument(..., help="解決したいタスクの自然言語による説明"),
    prompt: Optional[str] = typer.Option(None, help="実行するプロンプト"),
    unlabeled_data_path: Optional[str] = typer.Option(None, help="オンデマンド学習用の非ラベルデータ"),
    force_retrain: bool = typer.Option(False, help="強制的に再学習を行う"),
    model_config: str = typer.Option("configs/models/small.yaml", help="使用するモデルの設定ファイル"), # model_config追加
):
    """自律エージェントにタスクを解決させる。"""
    command = ["python", "run_agent.py", "--task_description", task, "--model_config", model_config] # model_config追加
    if prompt:
        command.extend(["--prompt", prompt])
    if unlabeled_data_path:
        command.extend(["--unlabeled_data_path", unlabeled_data_path])
    if force_retrain:
        command.append("--force_retrain")
    _run_command(command)

@agent_app.command("evolve")
def agent_evolve(
    task_description: str = typer.Argument(..., help="自己評価を開始するタスク"),
    model_config: str = typer.Option("configs/models/small.yaml", help="進化のベースとなるモデル設定"),
    training_config: str = typer.Option("configs/base_config.yaml", help="進化のベースとなる学習設定"),
    initial_accuracy: float = typer.Option(0.75, help="初期精度"), # initial_accuracy追加
    initial_spikes: float = typer.Option(1500.0, help="初期平均スパイク数"), # initial_spikes追加
):
    """自己進化エージェントを実行する。"""
    command = ["python", "run_evolution.py", "--task_description", task_description, "--model_config", model_config, "--training_config", training_config, "--initial_accuracy", str(initial_accuracy), "--initial_spikes", str(initial_spikes)]
    _run_command(command)

@agent_app.command("rl")
def agent_rl(
    episodes: int = typer.Option(1000, help="学習エピソード数"),
    output_dir: str = typer.Option("runs/rl_results", help="結果保存ディレクトリ"), # output_dir追加
):
    """強化学習エージェントを実行する。"""
    command = ["python", "run_rl_agent.py", "--episodes", str(episodes), "--output_dir", output_dir]
    _run_command(command)

@app.command("life-form")
def life_form(
    duration: int = typer.Option(60, help="実行時間（秒）。0を指定すると無限に実行します。"),
    model_config: str = typer.Option("configs/models/small.yaml", help="使用するモデルの設定ファイル"),
):
    """デジタル生命体を起動する。"""
    command = ["python", "run_life_form.py", "--duration", str(duration), "--model_config", model_config]
    _run_command(command)

@app.command("planner")
def planner(
    task_request: str = typer.Argument(..., help="プランナーへのタスク要求"),
    context_data: str = typer.Argument(..., help="タスクの文脈データ"),
):
    """階層的プランナーを実行する。"""
    _run_command(["python", "run_planner.py", "--task_request", task_request, "--context_data", context_data])

@app.command("brain")
def brain(
    prompt: Optional[str] = typer.Option(None, help="人工脳への単一の入力テキスト"),
    loop: bool = typer.Option(False, help="対話形式でシミュレーションを繰り返し実行する"),
    model_config: str = typer.Option("configs/models/small.yaml", help="モデル設定ファイル"), # model_config追加
):
    """人工脳シミュレーションを実行する。"""
    command_base = ["python"]
    if loop:
        script = "scripts/observe_brain_thought_process.py"
        command = command_base + [script, "--model_config", model_config]
    elif prompt:
        script = "run_brain_simulation.py"
        command = command_base + [script, "--prompt", prompt, "--model_config", model_config]
    else:
        typer.echo("Error: --prompt <text> または --loop のいずれかを指定してください。")
        raise typer.Exit()
    _run_command(command)

@app.command("gradient-train")
def gradient_train(
    model_config: str = typer.Option(..., help="モデルアーキテクチャ設定ファイル"), # Required option
    data_path: str = typer.Option(..., help="学習データパス"), # Required option
    base_config: str = typer.Option("configs/base_config.yaml", help="基本設定ファイル"), # Add base_config
    override_config: Optional[List[str]] = typer.Option(None, "--override_config", help="設定を上書き (例: 'training.epochs=5')"),
    resume_path: Optional[str] = typer.Option(None, help="チェックポイントから学習を再開"), # Add resume_path
    load_ewc_data: Optional[str] = typer.Option(None, help="EWCデータをロード"), # Add load_ewc_data
    task_name: Optional[str] = typer.Option(None, help="EWC用のタスク名"), # Add task_name
    use_astrocyte: bool = typer.Option(False, help="アストロサイトネットワークを有効化"), # Add use_astrocyte
):
    """train.pyを直接呼び出して勾配ベースの学習を行う。"""
    command = ["python", "train.py", "--config", base_config, "--model_config", model_config, "--data_path", data_path]
    if override_config:
        for oc in override_config:
            command.extend(["--override_config", oc])
    if resume_path:
        command.extend(["--resume_path", resume_path])
    if load_ewc_data:
        command.extend(["--load_ewc_data", load_ewc_data])
    if task_name:
        command.extend(["--task_name", task_name])
    if use_astrocyte:
        command.append("--use_astrocyte")
    _run_command(command)


@app.command("train-ultra")
def train_ultra(override_config: Optional[List[str]] = typer.Option(None, "--override_config")):
    """データ準備からUltraモデルの学習までを自動実行する。"""
    typer.echo("--- Starting Ultra Training Pipeline ---")
    _run_command(["python", "scripts/data_preparation.py"])
    train_command = ["python", "train.py", "--model_config", "configs/models/ultra.yaml"]
    if override_config:
        for oc in override_config:
            train_command.extend(["--override_config", oc])
    _run_command(train_command)
    typer.echo("--- Ultra Training Pipeline Finished ---")

@app.command("ui")
def ui(
    model_config: str = typer.Option("configs/models/small.yaml", help="使用するモデルの設定ファイル"),
    base_config: str = typer.Option("configs/base_config.yaml", help="基本設定ファイル"), # base_config追加
    start_langchain: bool = typer.Option(False, "--start-langchain", help="LangChain連携版のUIを起動する"),
):
    """Gradio UIを起動する。"""
    script = "app/langchain_main.py" if start_langchain else "app/main.py"
    # app/main.py や app/langchain_main.py は --config と --model_config を取る
    command = ["python", script, "--config", base_config, "--model_config", model_config]
    _run_command(command)

# --- (benchmark, convert コマンドは変更なし) ---
@benchmark_app.command("run")
def benchmark_run(
    experiment: str = typer.Option("all", help="実行する実験 (all, cifar10_comparison, sst2_comparison, mrpc_comparison)"),
    tag: Optional[str] = typer.Option(None, help="実験にカスタムタグを付ける"),
    epochs: int = typer.Option(3, help="訓練のエポック数"),
    batch_size: int = typer.Option(32, help="バッチサイズ"),
    learning_rate: float = typer.Option(1e-4, help="学習率"),
    output_dir: str = typer.Option("benchmarks", help="結果レポートの保存ディレクトリ"), # output_dir追加
):
    """ANN vs SNNの性能比較ベンチマークを実行する。"""
    command = ["python", "scripts/run_benchmark_suite.py", "--experiment", experiment, "--epochs", str(epochs), "--batch_size", str(batch_size), "--learning_rate", str(learning_rate), "--output_dir", output_dir]
    if tag:
        command.extend(["--tag", tag])
    _run_command(command)

@benchmark_app.command("continual")
def benchmark_continual(
    epochs_task_a: int = typer.Option(3, help="タスクAの訓練エポック数"),
    epochs_task_b: int = typer.Option(3, help="タスクBの訓練エポック数"),
    model_config: str = typer.Option("configs/models/small.yaml", help="モデル設定ファイル"),
    output_dir: str = typer.Option("benchmarks/continual_learning", help="結果保存ディレクトリ"), # output_dir追加
):
    """継続学習（破局的忘却の克服）の実験を実行する。"""
    command = ["python", "scripts/run_continual_learning_experiment.py", "--epochs_task_a", str(epochs_task_a), "--epochs_task_b", str(epochs_task_b), "--model_config", model_config, "--output_dir", output_dir]
    _run_command(command)

@convert_app.command("ann2snn-cnn")
def convert_ann2snn_cnn(
    ann_model_path: str = typer.Argument(..., help="変換元の学習済みSimpleCNNモデルのパス (.pth)"),
    output_snn_path: str = typer.Argument(..., help="変換後のSpikingCNNモデルの保存先パス (.pth)"),
    snn_model_config: str = typer.Option("configs/cifar10_spikingcnn_config.yaml", help="SpikingCNNのモデル設定ファイル"),
):
    """学習済みCNN (ANN) をSpikingCNN (SNN) に変換する。"""
    # scripts/convert_model.py を呼び出すように修正
    command = ["python", "scripts/convert_model.py", "--method", "cnn-convert", "--ann_model_path", ann_model_path, "--output_snn_path", output_snn_path, "--snn_model_config", snn_model_config]
    _run_command(command)

# --- HPOコマンド定義 ---
@hpo_app.command("run")
def hpo_run(
    target_script: str = typer.Option("run_distillation.py", help="最適化対象の学習スクリプト"),
    base_config: str = typer.Option("configs/base_config.yaml", help="基本設定ファイル"),
    model_config: str = typer.Argument(..., help="モデルアーキテクチャ設定ファイル"),
    task: str = typer.Argument(..., help="ターゲットタスク"),
    teacher_model: Optional[str] = typer.Option(None, help="教師モデル (run_distillation.py用)"),
    n_trials: int = typer.Option(50, help="Optunaの試行回数"),
    eval_epochs: int = typer.Option(3, help="各試行で実行するエポック数"),
    metric_name: str = typer.Option("accuracy", help="最適化するメトリクス ('accuracy' or 'loss')"),
    output_base_dir: str = typer.Option("runs/hpo_trials", help="各試行ログのベースディレクトリ"),
    study_name: str = typer.Option("snn_hpo_study", help="Optuna Studyの名前"),
    storage: str = typer.Option("sqlite:///runs/hpo_study.db", help="OptunaのDB保存先"),
):
    """Optunaを使ってハイパーパラメータ最適化を実行する。"""
    command = [
        "python", "run_hpo.py",
        "--target_script", target_script,
        "--base_config", base_config,
        "--model_config", model_config,
        "--task", task,
        "--n_trials", str(n_trials),
        "--eval_epochs", str(eval_epochs),
        "--metric_name", metric_name,
        "--output_base_dir", output_base_dir,
        "--study_name", study_name,
        "--storage", storage,
    ]
    if teacher_model:
        command.extend(["--teacher_model", teacher_model])
    _run_command(command)
# --- ここまで ---

if __name__ == "__main__":
    app()
