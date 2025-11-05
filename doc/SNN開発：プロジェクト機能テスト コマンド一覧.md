# **SNN プロジェクト: コマンドリファレンス (v10.0)**

## **1\. はじめに**

このドキュメントは、統合CLIツール snn-cli.py を用いてSNNプロジェクトの全機能を実行するためのリファレンスガイドです。各コマンドは、その目的別に分類されています。

**基本構文:**

python snn-cli.py

$$CATEGORY$$$$COMMAND$$$$OPTIONS$$

* **ヘルプの表示:** \* 全体: python snn-cli.py \--help  
  * カテゴリ別: python snn-cli.py$$CATEGORY$$  
    \--help  
  * コマンド別: python snn-cli.py$$CATEGORY$$$$COMMAND$$  
    \--help  
* **個別スクリプト:** 一部の機能は scripts/ ディレクトリ内のスクリプトを直接実行します。これらのヘルプは python scripts/$$script\\\_name$$  
  .py \--help で確認できます。

## **2\. コマンド一覧**

### **2.1. 🛠️ 環境設定とテスト (Environment Setup & Tests)**

プロジェクトのセットアップ、依存関係の確認、基本的な動作検証を行います。

| アクション | コマンド/方法 | 説明 |
| :---- | :---- | :---- |
| **依存関係インストール** | pip install \-r requirements.txt | プロジェクトに必要なPythonライブラリをインストールします。 |
| **健全性チェック** | pytest \-v | すべてのユニットテストおよび統合テスト（認知コンポーネント、実世界シナリオ含む）を実行します。 |
| **スモークテスト** | pytest \-v tests/test\_smoke\_all\_paradigms.py | 主要な学習パラダイムが最小データでエラーなく動作するか高速に検証します。 |
| **Colabセットアップ** | bash setup\_colab.sh (Colab環境) | Google Colab上で環境をセットアップします。 (SNN\_Project\_Colab\_Quickstart.ipynb参照) |

### **2.2. 📚 データ準備 (Data Preparation)**

モデル学習に必要なデータセットの前処理や準備を行います。

| アクション | コマンド/スクリプト | 説明 |
| :---- | :---- | :---- |
| **大規模データ準備** | python scripts/data\_preparation.py | WikiText-103のような大規模コーパスをダウンロードし、学習用に前処理します。 |
| **知識蒸留データ準備** | python scripts/prepare\_distillation\_data.py \--help | 知識蒸留のために、教師モデルのロジットを事前計算してデータセットを作成します。 |
| **知識ベース構築** | python scripts/build\_knowledge\_base.py \--help | RAGシステム用のベクトルストアを、プロジェクト内のドキュメント(doc/等)から構築します。 |

### **2.3. 🧠 モデル学習 (Model Training)**

SNNモデルの訓練や知識転移を行います。

| CLIコマンド | 個別スクリプト (直接実行) | 説明 |
| :---- | :---- | :---- |
| gradient-train | train.py (内部呼び出し) | **(標準)** 代理勾配法（SG）による直接学習を実行。設定ファイルで学習パラダイム（標準, EWC, TCL等）を指定。 |
| train-ultra | train.py (内部呼び出し) | データ準備から最大規模Ultraモデルの学習までを自動実行。 |
| (N/A) | run\_distillation.py | **(推奨)** ANN教師モデル（例: ResNet18）からの知識蒸留を行い、高性能なSNN専門家モデルを育成。 |
| (N/A) | scripts/train\_classifier.py | SST-2などの分類タスク専用のSNNモデル訓練。 |
| (N/A) | train\_planner.py | 階層的思考プランナー (PlannerSNN) を訓練。 |
| (N/A) | scripts/run\_stdp\_learning.py | **(実験的)** 生物学的学習則（STDP）による非監督学習を実行（doc/ROADMAP.md v10.0では非推奨）。 |

**主なオプション (gradient-train, train.py 共通):**

* \--config \<path\>: 基本設定ファイル (デフォルト: configs/base\_config.yaml)  
* \--model-config \<path\>: モデルアーキテクチャ設定ファイル (例: configs/models/medium.yaml)  
* \--data-path \<path\>: 学習データパス (configを上書き)  
* \--override\_config "\<key\>=\<value\>": 設定値をコマンドラインから上書き (例: "training.epochs=10")  
* \--resume\_path \<path\>: チェックポイントから学習を再開  
* \--distributed: 分散学習を有効化 (torchrun推奨: scripts/run\_distributed\_training.sh 参照)  
* \--task\_name \<name\>: EWC（継続学習）計算時にタスク名を指定 (例: sst2)  
* \--load\_ewc\_data \<path\>: 事前計算されたEWCデータをロード

**コマンド例:**

\# 中規模モデル(medium.yaml)を5エポック学習

python snn-cli.py gradient-train \\

\--model-config configs/models/medium.yaml \\

\--data-path data/sample\_data.jsonl \\

\--override\_config "training.epochs=5"

\# 知識蒸留の実行 (個別スクリプト)

python run\_distillation.py \\

\--task cifar10 \\

\--teacher\_model resnet18 \\

\--model-config configs/cifar10\_spikingcnn\_config.yaml \\

\--epochs 15

### **2.4. 🔄 モデル変換 (Model Conversion)**

学習済みのANNモデルをSNNモデルに変換します。

| CLIコマンド | 個別スクリプト (直接実行) | 説明 |
| :---- | :---- | :---- |
| convert ann2snn-cnn | scripts/convert\_model.py | 学習済みCNN (ANN) をSpikingCNN (SNN) に変換 (BatchNorm Folding, 閾値調整含む)。 |
| (N/A) | scripts/ann2snn.py | SimpleCNN から SpikingCNN への変換ワークフロー（検証用スクリプト）。 |

**コマンド例:**

\# 学習済みCNNをSpikingCNNに変換 (高忠実度)

python snn-cli.py convert ann2snn-cnn \\

runs/ann\_cifar\_baseline/cifar10/best\_model.pth \\

runs/converted/spiking\_cnn\_from\_ann.pth \\

\--snn-model-config configs/cifar10\_spikingcnn\_config.yaml

### **2.5. 📊 評価・ベンチマーク (Evaluation & Benchmarks)**

モデルの性能（精度、速度、エネルギー効率）を評価・比較します。

| CLIコマンド | 個別スクリプト (直接実行) | 説明 |
| :---- | :---- | :---- |
| benchmark run | scripts/run\_benchmark\_suite.py | CIFAR-10, SST-2, MRPC, CIFAR10-DVS, SHD等でSNNとANNの精度/速度/エネルギー効率を比較し、レポートに追記。 |
| benchmark continual | scripts/run\_continual\_learning\_experiment.py | 継続学習メカニズム(EWC)が「破局的忘却」を抑制できるか検証（benchmark run \--eval\_onlyを使用）。 |
| (N/A) | run\_compiler\_test.py | 訓練済みSNN（BioSNN, SNNCore/SEW-ResNet）をコンパイルし、ターゲットHW上での推定エネルギー消費/処理時間をシミュレート。 |

**主なオプション (benchmark run):**

* \--experiment \<name\>: 実行する実験 (all, cifar10\_comparison, sst2\_comparison, mrpc\_comparison, shd\_comparison 等)  
* \--epochs \<num\>:$$訓練モード$$  
  訓練エポック数 (デフォルト: 3\)  
* \--eval\_only:$$評価モード$$  
  訓練をスキップし、指定モデルで評価のみ実行。  
* \--model\_path \<path\>:$$評価モード$$  
  評価する学習済みモデルのパス (.pth)。  
* \--model\_config \<path\>:$$評価モード$$  
  評価するモデルのアーキテクチャ設定 (.yaml)。  
* \--model\_type \<SNN|ANN\>:$$評価モード$$  
  評価するモデルのタイプ。

**コマンド例:**

\# CIFAR-10でANN/SNN比較 (5エポック)

python snn-cli.py benchmark run \\

\--experiment cifar10\_comparison \\

\--epochs 5 \\

\--tag "AccuracyTest\_CIFAR10"

\# 継続学習実験 (EWC vs Finetune)

python snn-cli.py benchmark continual \--epochs\_task\_a 3 \--epochs\_task\_b 3

### **2.6. 🤖 エージェント・認知システム (Agents & Cognitive Systems)**

自律エージェントや認知アーキテクチャ全体を動作させます。

| CLIコマンド | 個別スクリプト (直接実行) | 説明 |
| :---- | :---- | :---- |
| agent solve | run\_agent.py | タスクを与え、最適な専門家モデルを自律的に検索・学習(オンデマンド学習)・推論させる。 |
| agent evolve | run\_evolution.py | エージェントが自身の性能を評価し、モデル構造や学習パラメータ/パラダイムを自律的に改善（HSEO最適化含む）。 |
| agent rl | run\_rl\_agent.py | 生物学的学習則を持つエージェントをGridWorld環境で訓練。 |
| planner | run\_planner.py | 複雑な要求に対し、知識ベースとスキルマップに基づき実行ステップを立案。 |
| brain \--prompt \<text\> | run\_brain\_simulation.py | 統合された認知アーキテクチャ(ArtificialBrain)を起動し、単一の認知サイクルをシミュレート。 |
| brain \--loop | scripts/observe\_brain\_thought\_process.py | 人工脳と対話し、内部状態(感情,記憶等)の変化をリアルタイム観察。 |
| life-form | run\_life\_form.py | AIを内発的動機に基づき、自律活動(思考,学習,進化)させるループを実行 (--duration)。 |

**コマンド例:**

\# 対話形式で人工脳を起動し、思考プロセスを観察

python snn-cli.py brain \--loop

\# デジタル生命体を30秒間実行

python snn-cli.py life-form \--duration 30

\# "文章要約"タスクをエージェントに依頼 (必要ならWebデータで学習)

python snn-cli.py agent solve \--task "文章要約" \--unlabeled\_data data/sample\_data.jsonl

### **2.7. 📱 アプリケーションデモ (Application Demos)**

SNN技術の実用的な応用例を示すデモスクリプトを実行します。

| アクション | コマンド/スクリプト | 説明 |
| :---- | :---- | :---- |
| **ECG異常検出デモ** | python scripts/run\_ecg\_analysis.py | ダミーECGデータを生成し、Temporal SNN (RSNN) モデルで異常/正常を分類するデモを実行。 |

**コマンド例:**

\# Temporal SNNモデルでECG異常検出デモを実行

python scripts/run\_ecg\_analysis.py \\

\--model-config configs/models/ecg\_temporal\_snn.yaml \\

\--num\_samples 10

### **2.8. 🛠️ 最適化 (Optimization)**

モデルのハイパーパラメータ最適化（HPO）を実行します。

| CLIコマンド | 個別スクリプト (直接実行) | 説明 |
| :---- | :---- | :---- |
| hpo run | run\_hpo.py | Optunaを使用し、指定された学習スクリプトのハイパーパラメータ（学習率、正則化係数等）を自動最適化。 |

**コマンド例:**

\# run\_distillation.py のパラメータを最適化 (10回試行)

python snn-cli.py hpo run \\

configs/cifar10\_spikingcnn\_config.yaml \\

cifar10 \\

\--target-script run\_distillation.py \\

\--teacher-model resnet18 \\

\--n-trials 10 \\

\--eval-epochs 1 \\

\--metric-name accuracy

### **2.9. 🩺 モデル診断 (Diagnostics)**

モデルの効率指標（スパース性、レイテンシ）を診断します。

| アクション | コマンド/スクリプト | 説明 |
| :---- | :---- | :---- |
| **効率レポート** | python scripts/report\_sparsity\_and\_T.py | モデルの「T (タイムステップ/レイテンシ)」と「s (スパース性/スパイク率)」を計測し、SOTA基準と比較。 |

**コマンド例:**

\# 学習済み中規模モデルの効率を診断

python scripts/report\_sparsity\_and\_T.py \\

\--model-config configs/models/medium.yaml \\

\--model-path runs/snn\_experiment/best\_model.pth \\

\--data-path data/smoke\_test\_data.jsonl

### **2.10. 🖥️ UI (User Interface)**

ユーザーインターフェース（Gradio）を起動します。

| CLIコマンド | 個別スクリプト (直接実行) | 説明 |
| :---- | :---- | :---- |
| ui | app/main.py | 標準のSNNモデル対話UI (Gradio) を起動。動的モデルロードに対応。 |
| ui \--start-langchain | app/langchain\_main.py | SNNモデルをLangChainアダプタ経由で利用するUIを起動。 |

**コマンド例:**

\# 動的モデルロードUIを起動 (レジストリ \+ CLI引数からモデルをロード)

python snn-cli.py ui \\

\--chat\_model\_config configs/models/medium.yaml \\

\--chat\_model\_path runs/snn\_experiment/best\_model.pth \\

Check if the user wants to write a cover letter.

\--cifar\_model\_config configs/cifar10\_spikingcnn\_config.yaml \\

\--cifar\_model\_path runs/converted/spiking\_cnn\_from\_ann.pth

\# LangChain連携UIを起動 (smallモデル使用)

python snn-cli.py ui \--start-langchain \--model-config configs/models/small.yaml

### **2.11. 🧹 プロジェクト・クリーンアップ (Project Cleanup)**

プロジェクトディレクトリ内に蓄積されたログやキャッシュファイルをクリーンアップします。

| CLIコマンド | 個別スクリプト (直接実行) | 説明 |
| :---- | :---- | :---- |
| clean | (N/A) | runs, precomputed\_data, workspace 内の一時ログやキャッシュを削除。デフォルトではモデル(.pth)やDB(.db, .jsonl)は保護。 |

**主なオプション (clean):**

* \-y, \--yes: 削除実行の確認プロンプトをスキップします。  
* \--delete-models: 保護を解除し、モデルファイル (.pth, .pt) も削除対象に含めます。  
* \--delete-data: 保護を解除し、データファイル (.jsonl, .json, .db, .csv) も削除対象に含めます。

**コマンド例:**

\# 不要なログを削除（モデルとデータは保持）

python snn-cli.py clean

\# モデルもデータも含むすべての中間生成物を削除（確認あり）

python snn-cli.py clean \--delete-models \--delete-data

\# 確認なしで全てのログを強制削除（モデルとデータは保持）

python snn-cli.py clean \-y
