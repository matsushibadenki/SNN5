# **Project SNN: 自己進化するデジタル生命体の構築**

## **1\. プロジェクト概要**

本プロジェクトは、**スパイキングニューラルネットワーク(SNN)** の研究開発フレームワークです。生物の脳に着想を得て、エネルギー効率と時間情報処理能力に優れた次世代AIの実現を目指します。

**主な特徴:**

* **多様なSNNアーキテクチャ:** SpikingCNN, Temporal SNN (RSNN), Spiking Transformer (SDSA適応型ハイブリッド版含む), Spiking Mamba, TRM, Hybrid CNN-SNN など、基礎から最新モデルまで実装。  
* **先進的な学習手法:** 代理勾配法、ANN-SNN変換、知識蒸留、生物学的学習則（STDP, R-STDP, 因果追跡）をサポート。  
* **認知アーキテクチャ:** 人工脳の構成要素（大脳皮質、海馬、扁桃体など）をモジュール化し、統合シミュレーションが可能。  
* **自己進化:** エージェントが自身のモデル構造や学習パラメータを自律的に改善するプロトタイプ機能。  
* **ツール群:** 統合CLI、対話型UI、ベンチマークスイート、ニューロモーフィックハードウェアへのエクスポート機能。

**目標:**

単にANNの性能に追いつくだけでなく、エネルギー効率、リアルタイム性、継続学習、因果推論といったSNN固有の強みを活かし、自己進化する**デジタル生命体**としてのAIを構築することを目指します。

## **2\. なぜSNNか**

* **⚡ 超低消費電力:** イベント駆動計算により、理論上最大100倍のエネルギー効率。  
* **⏱️ 時間情報処理:** スパイクのタイミングが情報を符号化し、時系列データ（音声、センサー、イベントカメラ等）に強い。  
* **🧠 生物学的妥当性:** 脳の計算原理に近いモデルであり、神経科学的知見との連携が可能。

## **3\. 主な機能と実装状況**

プロジェクトの主要な機能と現在の実装ステータスです。（詳細は [ROADMAP.md](https://www.google.com/search?q=doc/ROADMAP.md) を参照）

| 機能カテゴリ | 詳細 | ステイタス |
| :---- | :---- | :---- |
| **基礎モデル** | SpikingCNN, Temporal SNN (RSNN) | ✅ 完了 |
| **学習: 代理勾配** | train.py による直接学習 (SG Track) | ✅ 完了 |
| **学習: ANN-SNN変換** | CNN (BatchNorm Folding対応) → SpikingCNN | ✅ 完了 |
| **学習: 知識蒸留** | run\_distillation.py (ANN教師 → SNN生徒) | ✅ 完了 |
| **学習: 生物学的** | STDP, R-STDP, BCM, CausalTrace V2 (Bio/LNN Track) | ✅ 完了 |
| **先進アーキテクチャ (SG)** | Spiking Transformer (v2), Spiking-MAMBA, Spiking-RWKV, SEW-ResNet, SpikingSSM, TSkipsSNN | 🔄 実装済 (P1 計画) |
| **先進アーキテクチャ (SFN)** | SFN変換パイプライン (SFormer) | 📋 計画中 (P1.5) |
| **先進ニューロン** | LIF, Izhikevich, BIF, TC-LIF, GLIF, DualThreshold, SFN | ✅ 完了 |
| **自己進化システム** | HSEO, 自己評価, 認知アーキテクチャ | ✅ 完了 |
| **ベンチマーク** | CIFAR-10, SST-2, MRPC (ANN vs SNN比較) | ✅ 完了 |
| **ニューロモーフィック対応** | SpikingJelly移行, Lava/SpiNNakerコード生成 | 📋 計画中 (P5, P7) |
| **ツール: CLI** | snn-cli.py (主要機能へのアクセス) | ✅ 完了 |
| **ツール: UI** | Gradio 対話インターフェース (動的ロード対応) | ✅ 完了 |
| **応用デモ** | ECG異常検出 (RSNN/TSkipsSNN対応) | ✅ 完了 |
| **オンライン/継続学習** | EWC (実験的), Bio/LNN Track (オンチップ学習) | 📋 計画中 (P8.5) |

**凡例:** ✅: 完了, 🔄: 実装済（ロードマップで継続評価・利用）, 📋: 計画中

## **4\. システムアーキテクチャ**

```mermaid
graph LR

%% スタイル定義
classDef Agent fill:#f9f,stroke:#333,stroke-width:2px;
classDef Cognitive fill:#ccf,stroke:#333,stroke-width:2px;
classDef Execution fill:#cfc,stroke:#333,stroke-width:2px;
classDef Foundation fill:#ffc,stroke:#333,stroke-width:2px;
classDef IO fill:#eee,stroke:#333,stroke-width:2px;
classDef UI fill:#ddd,stroke:#333,stroke-width:2px;

%% レイヤー    
subgraph Layer_UI["User Interface Layer"]    
    direction LR    
    CLI["snn-cli.py (Typer)"]:::UI    
    GradioUI["Gradio Web UI"]:::UI    
end

subgraph Layer_Orchestration["Orchestration Layer (Agents)"]    
    direction TB    
    LifeForm["DigitalLifeForm"]:::Agent    
    Autonomous["AutonomousAgent"]:::Agent    
    Evolving["SelfEvolvingAgentMaster"]:::Agent    
    RL["ReinforcementLearnerAgent"]:::Agent    
end

subgraph Layer_Cognitive["Cognitive Layer"]    
    direction TB    
    subgraph Cognitive_Executive["Executive & Planning"]    
        Planner["HierarchicalPlanner"]:::Cognitive    
        PFC["PrefrontalCortex"]:::Cognitive    
    end    
    subgraph Cognitive_Memory["Memory & Knowledge"]    
        Memory["Memory (Hippocampus/Cortex)"]:::Cognitive    
        RAG["RAGSystem"]:::Cognitive    
        SymbolGrounding["SymbolGrounding"]:::Cognitive    
    end    
    subgraph Cognitive_Core["Core Cognitive Processing"]    
        Brain["ArtificialBrain"]:::Cognitive    
        GWS["GlobalWorkspace"]:::Cognitive    
        Causal["CausalInferenceEngine"]:::Cognitive    
        Motivation["IntrinsicMotivation"]:::Cognitive    
        Amygdala["Amygdala"]:::Cognitive    
        BasalGanglia["BasalGanglia"]:::Cognitive    
        Perception["HybridPerceptionCortex"]:::Cognitive    
    end    
    subgraph Cognitive_Motor["Motor Control"]    
         Cerebellum["Cerebellum"]:::Cognitive    
         MotorCortex["MotorCortex"]:::Cognitive    
    end    
end

subgraph Layer_Execution["Execution Layer"]    
    direction TB    
    Training["train.py / Trainers"]:::Execution    
    Inference["SNNInferenceEngine"]:::Execution    
    Benchmark["Benchmark Suite"]:::Execution    
    Conversion["ANN-SNN Converter"]:::Execution    
    Deployment["NeuromorphicExporter/Compiler"]:::Execution    
end

subgraph Layer_Foundation["Foundation Layer"]    
    direction TB    
    Core["SNN Models (core)"]:::Foundation    
    Neurons["Neuron Models"]:::Foundation    
    Attention["Attention Mechanisms"]:::Foundation    
    Rules["BioLearningRules"]:::Foundation    
end

subgraph Layer_IO["Input/Output Layer"]    
     direction TB    
     SensoryReceptor["SensoryReceptor"]:::IO    
     SpikeEncoder["SpikeEncoder"]:::IO    
     SpikeDecoder["SpikeDecoder"]:::IO    
     Actuator["Actuator"]:::IO    
end

%% 主要な接続    
CLI --> Layer_Orchestration    
CLI --> Layer_Cognitive    
CLI --> Layer_Execution    
GradioUI --> Inference

SensoryReceptor --> SpikeEncoder    
SpikeEncoder --> Perception    
Brain --> GWS    
GWS -- Broadcasts --> PFC    
GWS -- Broadcasts --> BasalGanglia    
GWS -- Broadcasts --> Memory    
GWS -- Broadcasts --> Causal

PFC -- Goal --> Planner    
Planner -- Subtasks --> Layer_Orchestration    
Layer_Orchestration -- Execute --> Training    
Layer_Orchestration -- Execute --> Inference    
Layer_Orchestration -- Uses --> Memory    
Layer_Orchestration -- Uses --> RAG

Autonomous --> WebCrawler[Web Crawler Tool]    
WebCrawler -- Data --> Training

BasalGanglia -- ActionSelection --> Cerebellum    
Cerebellum -- RefinedCommands --> MotorCortex    
MotorCortex --> Actuator    
Actuator -- Action --> ExternalWorld([External World])

Training --> Core    
Inference --> Core    
Benchmark --> Core    
Conversion --> Core    
Deployment --> Core

Core --> Neurons    
Core --> Attention    
Training --> Rules

%% Global Workspace中心の連携 (より詳細)    
Perception -- "Upload (Salience)" --> GWS    
Amygdala -- "Upload (Salience)" --> GWS    
Memory -- "Upload (Salience)" --> GWS    
Causal -- "Upload (Salience)" --> GWS

%% 強調表示 (例)    
style GWS fill:#f9a,stroke:#f00,stroke-width:3px
```


## **5\. クイックスタート**

### **ステップ1: 環境設定**

\# 必要なライブラリをインストール

pip install \-r requirements.txt

\# (Mac Mシリーズ向け Pytorchバグ対策)

export PYTORCH\_ENABLE\_MPS\_FALLBACK=1

### **ステップ2: 健全性チェック**

プロジェクト全体のテストを実行し、環境が正しくセットアップされていることを確認します。

pytest \-v

### **ステップ3: 基本的な学習の実行 (スモークテスト)**

最小構成のモデル (micro.yaml) を使って学習パイプラインが動作するか確認します。

python snn-cli.py gradient-train \\

\--model-config configs/models/micro.yaml \\

\--data-path data/smoke\_test\_data.jsonl \\

\--override\_config "training.epochs=3"

学習ログとモデルは runs/smoke\_tests ディレクトリに保存されます。

## **6\. 使い方**

プロジェクトの主要な機能は、統合CLIツール snn-cli.py から利用できます。

各コマンドの詳細なオプションや使用例は、コマンドリファレンス を参照してください。

### **基本的な流れ**

1. **学習:** gradient-train, train-ultra, run\_distillation.py 等でモデルを学習させる。  
2. **評価:** benchmark run でANNと比較したり、benchmark continual で継続学習能力を評価する。  
3. **推論:** ui コマンドでGradio UIを起動し、学習済みモデルと対話する。  
4. **その他:** convert, agent, brain などのコマンドで、モデル変換や高度な認知機能のシミュレーションを行う。

### **主要コマンド例**

**中規模モデルの学習 (5エポック):**

python snn-cli.py gradient-train \\

\--model-config configs/models/medium.yaml \\

\--data-path data/sample\_data.jsonl \\

\--override\_config "training.epochs=5"

**学習済みモデルでUIを起動:**

\# app/main.py を直接実行し、--model\_path で学習済みモデルを指定

python app/main.py \\

\--config configs/base\_config.yaml \\

\--model\_config configs/models/medium.yaml \\

\--model\_path runs/snn\_experiment/best\_model.pth \# 学習済みモデルのパスを指定

**CIFAR-10でのANN/SNN比較ベンチマーク (1エポック):**

python snn-cli.py benchmark run \\

\--experiment cifar10\_comparison \\

\--epochs 1

**人工脳の対話シミュレーション:**

python snn-cli.py brain \--loop

**ECG異常検出デモ:**

python scripts/run\_ecg\_analysis.py \\

\--model\_config configs/models/ecg\_temporal\_snn.yaml \\

\--num\_samples 10

**(その他のコマンド詳細は [コマンドリファレンス](https://www.google.com/search?q=doc/%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89%E3%83%AA%E3%83%95%E3%82%A1%E3%83%AC%E3%83%B3%E3%82%B9.md) へ)**

## **7\. コントリビューション**

### **貢献を歓迎する領域**

* 新しいニューロンモデル、アーキテクチャの実装  
* ベンチマークデータセットの追加  
* テストカバレッジの向上  
* ドキュメントの改善  
* バグ報告・修正  
* 実用アプリケーション例の実装 (DVS連携など)  
* ニューロモーフィックハードウェア対応の強化

### **開発ガイドライン**

* コードスタイル: PEP 8準拠, 型ヒント推奨 (mypy)  
* Docstring: Google形式推奨  
* テスト: 新機能にはユニットテスト/統合テストを追加  
* Issue/Pull Request: 歓迎します！
