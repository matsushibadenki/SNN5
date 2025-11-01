# **SNN プロジェクト: コマンドリファレンス**

## **1\. はじめに**

このドキュメントは、統合CLIツール snn-cli.py を用いてSNNプロジェクトの全機能を実行するためのリファレンスガイドです。各コマンドは、その目的別に分類されています。

**基本構文:**

python snn-cli.py \[CATEGORY\] \[COMMAND\] \[OPTIONS\]

* **ヘルプの表示:**  
  * 全体: python snn-cli.py \--help  
  * カテゴリ別: python snn-cli.py \[CATEGORY\] \--help  
  * コマンド別: python snn-cli.py \[CATEGORY\] \[COMMAND\] \--help  
* **個別スクリプト:** 一部の機能は scripts/ ディレクトリ内のスクリプトを直接実行します。これらのヘルプは python scripts/\[script\_name\].py \--help で確認できます。

## **2\. コマンド一覧**

### **2.1. 🛠️ 環境設定とテスト (Environment Setup & Tests)**

プロジェクトのセットアップ、依存関係の確認、基本的な動作検証を行います。

| アクション | コマンド/方法 | 説明 |
| :---- | :---- | :---- |
| **依存関係インストール** | pip install \-r requirements.txt | プロジェクトに必要なPythonライブラリをインストールします。 |
| **健全性チェック** | pytest \-v | すべてのユニットテストおよび統合テストを実行し、システム基盤を確認します。 |
| **スモークテスト** | pytest \-v tests/test\_smoke\_all\_paradigms.py | 主要な学習パラダイムが最小データでエラーなく動作するか高速に検証します。 |
| **Colabセットアップ** | bash setup\_colab.sh (Colab環境) | Google Colab上で環境をセットアップします。 ([Colabノートブック](https://www.google.com/search?q=SNN_Project_Colab_Quickstart.ipynb)参照) |

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
| gradient-train | train.py | 勾配ベースの学習（代理勾配法）を実行。設定ファイルで学習パラダイム等を指定。 |
| train-ultra | train.py (内部呼び出し) | データ準備から最大規模Ultraモデルの学習までを自動実行。 |
| (N/A) | run\_distillation.py | ANN教師モデルからの知識蒸留を行い、高性能なSNN専門家モデルを育成。 |
| (N/A) | scripts/train\_classifier.py | SST-2などの分類タスク専用のSNNモデル訓練。 |
| (N/A) | train\_planner.py | 階層的思考プランナー (PlannerSNN) を訓練。 |

**主なオプション (gradient-train, train.py 共通):**

* \--config \<path\>: 基本設定ファイル (デフォルト: configs/base\_config.yaml)  
* \--model-config \<path\>: モデルアーキテクチャ設定ファイル (例: configs/models/medium.yaml)  
* \--data-path \<path\>: 学習データパス (configを上書き)  
* \--override\_config "\<key\>=\<value\>": 設定値をコマンドラインから上書き (例: "training.epochs=10")  
* \--resume\_path \<path\>: チェックポイントから学習を再開  
* \--distributed: 分散学習を有効化 (torchrun推奨: scripts/run\_distributed\_training.sh 参照)  
* \--task\_name \<name\>: EWC計算時にタスク名を指定 (例: sst2)  
* \--load\_ewc\_data \<path\>: 事前計算されたEWCデータをロード

**コマンド例:**

\# 中規模モデル(medium.yaml)を5エポック学習  
python snn-cli.py gradient-train \\  
    \--model-config configs/models/medium.yaml \\  
    \--data-path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=5"

\# Spiking Transformerモデルで学習 (ログディレクトリ指定)  
python snn-cli.py gradient-train \\  
    \--model-config configs/models/spiking\_transformer.yaml \\  
    \--data-path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=10" \\  
    \--override\_config "training.log\_dir=runs/transformer\_test"

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
| (N/A) | scripts/convert\_model.py | \--method llm-convert でLLM変換 (実験的)。変換後のファインチューニング推奨。 |

**主なオプション (convert ann2snn-cnn):**

* ANN\_MODEL\_PATH: 変換元の学習済みANNモデルパス (.pth)  
* OUTPUT\_SNN\_PATH: 変換後のSNNモデル保存先パス (.pth)  
* \--snn-model-config \<path\>: 変換先のSNNモデル設定ファイル (デフォルト: configs/cifar10\_spikingcnn\_config.yaml)

**コマンド例:**

\# 学習済みCNNをSpikingCNNに変換  
python snn-cli.py convert ann2snn-cnn \\  
    runs/ann\_cifar\_baseline/cifar10/best\_model.pth \\  
    runs/converted/spiking\_cnn\_from\_ann.pth \\  
    \--snn-model-config configs/cifar10\_spikingcnn\_config.yaml

### **2.5. 📊 評価・ベンチマーク (Evaluation & Benchmarks)**

モデルの性能（精度、速度、エネルギー効率）を評価・比較します。

| CLIコマンド | 個別スクリプト (直接実行) | 説明 |
| :---- | :---- | :---- |
| benchmark run | scripts/run\_benchmark\_suite.py | CIFAR-10, SST-2, MRPC等でSNNとANNの精度/速度/エネルギー効率を比較し、レポートに追記。 |
| benchmark continual | scripts/run\_continual\_learning\_experiment.py | 継続学習メカニズム(EWC等)が「破局的忘却」を抑制できるか検証 (概念実証レベル)。 |
| (N/A) | run\_compiler\_test.py | 訓練済みSNNをコンパイルし、ターゲットHW上での推定エネルギー消費/処理時間をシミュレート (BioSNNベース)。 |

**主なオプション (benchmark run):**

* \--experiment \<name\>: 実行する実験 (all, cifar10\_comparison, sst2\_comparison, mrpc\_comparison)  
* \--epochs \<num\>: 訓練エポック数 (デフォルト: 3\)  
* \--batch\_size \<num\>: バッチサイズ (デフォルト: 32\)  
* \--learning\_rate \<float\>: 学習率 (デフォルト: 1e-4)  
* \--tag \<tag\>: レポートに付けるカスタムタグ

**コマンド例:**

\# CIFAR-10でANN/SNN比較 (5エポック)  
python snn-cli.py benchmark run \\  
    \--experiment cifar10\_comparison \\  
    \--epochs 5 \\  
    \--tag "AccuracyTest\_CIFAR10"

\# 継続学習実験 (概念実証)  
python snn-cli.py benchmark continual \--epochs\_task\_a 3 \--epochs\_task\_b 3

### **2.6. 🤖 エージェント・認知システム (Agents & Cognitive Systems)**

自律エージェントや認知アーキテクチャ全体を動作させます。

| CLIコマンド | 個別スクリプト (直接実行) | 説明 |
| :---- | :---- | :---- |
| agent solve | run\_agent.py | タスクを与え、最適な専門家モデルを自律的に検索・学習(オンデマンド学習)・推論させる。 |
| agent evolve | run\_evolution.py | エージェントが自身の性能を評価し、モデル構造や学習パラメータ/パラダイムを自律的に改善するサイクルを実行。 |
| agent rl | run\_rl\_agent.py | 生物学的学習則を持つエージェントをGridWorld環境で訓練。 |
| planner | run\_planner.py | 複雑な要求に対し、知識ベースとスキルマップに基づき実行ステップを立案 (現在は概念実証レベル)。 |
| brain | run\_brain\_simulation.py | 統合された認知アーキテクチャ(ArtificialBrain)を起動し、認知サイクルをシミュレート (--prompt or \--loop)。 |
| life-form | run\_life\_form.py | AIを内発的動機に基づき、自律活動(思考,学習,進化)させるループを実行 (--duration)。 |
| (N/A) | scripts/observe\_brain\_thought\_process.py | 人工脳と対話し、内部状態(感情,記憶等)の変化をリアルタイム観察。 |

**コマンド例:**

\# 対話形式で人工脳を起動し、思考プロセスを観察  
python snn-cli.py brain \--loop

\# デジタル生命体を30秒間実行  
python snn-cli.py life-form \--duration 30

\# "文章要約"タスクをエージェントに依頼 (必要ならWebデータで学習)  
python snn-cli.py agent solve \--task "文章要約" \--unlabeled\_data data/sample\_data.jsonl

### **2.7. 🧪 アプリケーションデモ (Application Demos)**

SNN技術の実用的な応用例を示すデモスクリプトを実行します。

| アクション | コマンド/スクリプト | 説明 |
| :---- | :---- | :---- |
| **ECG異常検出デモ** | python scripts/run\_ecg\_analysis.py | ダミーECGデータを生成し、SNNモデルで異常/正常を分類するデモを実行。 |

**コマンド例:**

\# Temporal SNNモデルでECG異常検出デモを実行  
python scripts/run\_ecg\_analysis.py \\  
    \--model-config configs/models/ecg\_temporal\_snn.yaml \\  
    \--num\_samples 10

### **2.8. 🖥️ UI (User Interface)**

ユーザーインターフェース（Gradio）を起動します。

| CLIコマンド | 個別スクリプト (直接実行) | 説明 |
| :---- | :---- | :---- |
| ui | app/main.py | 標準のSNNモデル対話UI (Gradio) を起動。 |
| ui \--start-langchain | app/langchain\_main.py | SNNモデルをLangChainアダプタ経由で利用するUIを起動。 |

**主なオプション (ui):**

* \--model-config \<path\>: 使用するモデルの設定ファイル (デフォルト: configs/models/small.yaml)

**注意:** snn-cli.py ui コマンドは \--model\_path オプションをサポートしていません。学習済みモデルを読み込む場合は、app/main.py または app/langchain\_main.py を直接実行し、--model\_path オプションを使用してください。

**コマンド例:**

\# デフォルト(small)モデルで標準UIを起動  
python snn-cli.py ui

\# 学習済みの中規模モデルで標準UIを起動 (直接実行)  
python app/main.py \\  
    \--config configs/base\_config.yaml \\  
    \--model-config configs/models/medium.yaml \\  
    \--model\_path runs/snn\_experiment/best\_model.pth \# 学習済みモデルのパス

\# LangChain連携UIを起動  
python snn-cli.py ui \--start-langchain  
