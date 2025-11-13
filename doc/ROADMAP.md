# **SNN5プロジェクト 統合ロードマップ**

SNN5プロジェクトの目的（Objective.md）を達成し、高度な認知能力を持つ「デジタル生命体」を創造するための、技術的・戦略的ロードマップ。  
（roadmap2.md のフェーズ構造と ROADMAP.md の戦略的タスクを融合）

## **Phase 1: 基盤構築とSNNコアの実装**

**目標:** スケーラブルで柔軟なSNNシミュレーションの核となる基盤を確立し、最新の学習理論を導入する。

* **1-1: コア・ニューロンモデルの実装**  
  * \[x\] 基本的なLIFニューロン  
  * \[x\] Adaptive LIF (ALIF) ニューロン  
  * \[x\] Izhikevichニューロン（より複雑な発火パターン用）  
  * \[ \] （TBD）BIF (Binaural Integrate-and-Fire) ニューロンなど、特定のタスク用モデル  
* **1-2: 基本的な学習則の実装と理論的深化 (最優先)**  
  * \[x\] STDP (Spike-Timing-Dependent Plasticity) (P3.1 関連)  
  * \[x\] 報酬変調型STDP (R-STDP) (P3.2 関連)  
  * \[ \] （TBD）BPTT (Backpropagation Through Time) のSNN版（Surrogate Gradientなど）  
  * \[ \] **(新規・最優先 P1.4)** **学習則の理論的深化 (NL-DSAM):** NL-DSAM（KLダイバージェンス最小化）の理論に基づき、プロジェクトの三因子学習則（CausalTrace）の誤差信号を改良し、STDPとの整合性を高める。(関連: learning\_rules/causal\_trace.py)  
* **1-3: ネットワーク構築とシミュレーション・エンジン**  
  * \[x\] torch.nn.Module をベースにした柔軟なネットワーク定義  
  * \[x\] 効率的な時間ステップ実行エンジン  
* **1-4: エンコーディング/デコーディング**  
  * \[x\] レートエンコーディング  
  * \[ \] （TBD）テンポラルエンコーディング（例: TTFS \- Time To First Spike）  
  * \[ \] (P3.6) 報酬/エラーのスパイク符号化: 内発的動機や外部報酬の連続値フィードバックを、SNNが理解可能なスパイク信号に変換するモジュール。(関連: spike\_encoder.py)  
* **1-5: 実験管理と最適化**  
  * \[x\] Hydraによる設定管理（configs/）  
  * \[x\] 基本的な学習/評価スクリプト（train.py）  
  * \[ \] (P1.5) T=1 低遅延推論の最適化: 演算ステップ数Tを最小限に抑える学習手法（DTTFS、ASR）の適用。(関連: trainer.py)  
  * \[ \] (P3.4) ハイパーパラメータ最適化（HPO）: Optuna等によるニューロンパラメータ（閾値、膜時定数等）の自動最適化。(関連: hseo.py)

## **Phase 2: 基本的な認知アーキテクチャの構築**

**目標:** 脳の主要なコンポーネントをモジュール化し、基本的な情報処理フローを実現する。

* **2-1: 感覚入力野（Sensory Cortex）**  
  * \[x\] 感覚情報（画像、時系列データ）をスパイクに変換する入力層。(関連: snn\_research/io/spike\_encoder.py)  
* **2-2: 特徴抽出（Perception Cortex）**  
  * \[x\] スパイク版の畳み込み層（Spiking CNN）による基本的な特徴抽出。(関連: snn\_research/core/models/spiking\_cnn\_model.py)  
* **2-3: 記憶（Hippocampus）**  
  * \[x\] STDPによる短期的なパターン記憶。(関連: snn\_research/cognitive\_architecture/hippocampus.py)  
  * \[ \] **(新規 P2.6)** **経験（エピソード）記憶の符号化:** 新しい体験（行動、結果、主観的クラリティ）を符号化し、時系列で保存する。(関連: hippocampus.py, memory.py)  
* **2-4: 意思決定と行動選択（Basal Ganglia）**  
  * \[x\] 報酬（強化学習）に基づいて最適な行動を選択するモジュール。(関連: snn\_research/cognitive\_architecture/basal\_ganglia.py)  
* **2-5: 運動出力野（Motor Cortex）**  
  * \[x\] 選択された行動を具体的な出力（例: クラス分類）に変換する層。(関連: snn\_research/cognitive\_architecture/motor\_cortex.py)  
* **2-6: 感情・動機付け（Amygdala / Intrinsic Motivation）**  
  * \[x\] 報酬や罰（感情価）を評価し、学習信号（ドーパミンなど）を生成する。(関連: snn\_research/cognitive\_architecture/amygdala.py)  
  * \[ \] (P2.3) 内発的動機: 好奇心（情報ゲイン）と達成感（予測誤差解消）に基づく内発的動機システムの設計と統合。(関連: intrinsic\_motivation.py)

## **Phase 3: 自己組織化と適応学習**

**目標:** データ駆動でネットワークが自律的に構造と機能を最適化し、継続的に学習する能力を獲得する。

* **3-1: 自己組織化マップ（SOM）**  
  * \[x\] 競合学習（Lateral Inhibition）による特徴マップの自己組織化。(関連: snn\_research/cognitive\_architecture/som\_feature\_map.py)  
* **3-2: 予測コーディング（Predictive Coding）**  
  * \[x\] トップダウンの予測とボトムアップの感覚入力の誤差を最小化する学習。(関連: snn\_research/core/layers/predictive\_coding.py)  
* **3-3: 強化学習エージェント (SNN-RL)**  
  * \[x\] 脳型アーキテクチャ全体を強化学習エージェントとして環境と相互作用させる。(関連: snn\_research/agent/reinforcement\_learner\_agent.py) (P3.2 関連)  
* **3-4: 知識蒸留（Knowledge Distillation）**  
  * \[x\] ANNや大規模SNN（教師モデル）から小型SNN（生徒モデル）への知識転移。(関連: snn\_research/distillation/knowledge\_distillation\_manager.py)  
* **3-5: 継続学習（Continual Learning）**  
  * \[ \] (P3.3) 破滅的忘却を抑制し、新しいタスクを継続的に学習するメカニズム（例: EWC on SNN, Replay Buffer）。(関連: scripts/run\_continual\_learning\_experiment.py)  
* **3-6: オンライン学習パイプライン**  
  * \[ \] (P3.5) 実環境からのデータストリームをリアルタイムで取り込み、モデルを非同期的に更新する堅牢なパイプラインの構築。(関連: trainers.py)

## **Phase 4: 複雑な認知機能とメタ学習**

**目標:** 高次の思考、プランニング、自己言及的なプロセスを実現し、汎用性を高める。

* **4-1: 大脳皮質 (Cortex) の階層化と時系列処理 (最優先)**  
  * \[x\] 複数の皮質領域（視覚、聴覚など）をモジュールとして統合し、階層的な処理を実現する。(関連: snn\_research/cognitive\_architecture/cortex.py)  
  * \[ \] **(新規・最優先 P1.3)** **階層的時系列アーキテクチャ (HOPE):** HOPEアーキテクチャに基づき、モチーフ（短時間）と構文（長時間）の処理を階層的に分離する構造を導入し、長期依存性を効率的に処理する。(関連: tskips\_snn.py)  
* **4-2: 前頭前野 (PFC) と高次実行機能**  
  * \[x\] ワーキングメモリ、注意制御、タスク切り替えなど、高次の実行機能を担うPFCモジュール。(関連: snn\_research/cognitive\_architecture/prefrontal\_cortex.py)  
  * \[ \] (P2.2) 階層的プランナー: PFCとBasal Ganglia (BG)を統合した階層的プランニングの実現。報酬予測と行動選択のSNN化。(関連: prefrontal\_cortex.py, basal\_ganglia.py)  
* **4-3: グローバル・ワークスペース理論 (GWT) と意識**  
  * \[x\] (P2.1) 複数の専門モジュールが情報を「ブロードキャスト」し、意識的な処理を実現するGWTアーキテクチャ。(関連: snn\_research/cognitive\_architecture/global\_workspace.py)  
* **4-4: アストロサイトによる動的ニューロン進化と学習変調**  
  * \[x\] ネットワーク全体の活動を監視し、恒常性を維持するアストロサイト・ネットワーク。  
  * \[x\] 活動が低いニューロン層を、より表現力の高いモデル（例: LIF \-\> Izhikevich）に動的に置き換える「自己進化」機能。(関連: snn\_research/cognitive\_architecture/astrocyte\_network.py)  
  * \[ \] (新規) アストロサイトによる学習変調 (Ca2+シグナリング): カルシウム動力学モデルを実装し、メタ可塑性のトリガーとして学習強度とタイミングを変調（ゲート）する。  
* **4-5: 自己認識と内的シミュレーション**  
  * \[ \] (P2.5) 自己言及と主観的経験シグナル: SNNが自身の内部状態を報告する「主観的経験」シグナルを生成・利用するメカニズム。(関連: meta\_cognitive\_snn.py)  
  * \[ \] ネットワークが自身の内部状態を監視し、行動の結果を予測する「内的シミュレーション」機能。(関連: snn\_research/cognitive\_architecture/meta\_cognitive\_snn.py)  
* **4-6: 因果推論 (Causal Inference)**  
  * \[ \] (P2.4) 過去の行動と結果の因果関係をSNNで学習し、プランニングにフィードバックする機構。世界モデルの構築。(関連: causal\_inference\_engine.py)  
  * \[ \] **(新規 P3.7)** **NL-DSAMと因果推論の連携:** 改良された学習則(P1.4)を因果推論エンジン(P2.4)に適用し、学習精度と効率を検証する。

## **Phase 5: 大規模化と現実世界への応用**

**目標:** シミュレーションを大規模化し、ANNの性能を超え、実用的なAIエージェントやシステムに応用する。

* **5-1: 大規模シミュレーションと最適化**  
  * \[ \] 分散学習（Distributed Training）による大規模モデルのサポート。(関連: scripts/run\_distributed\_training.sh)  
  * \[ \] (P1.1) SOTA変換パイプライン: ANNモデル（ResNet, ViT, MoE）からSNNへの低損失変換。最適な正規化（BN、LN）の統合。(関連: ann\_to\_snn\_converter.py)  
  * \[ \] (P1.2) SNN-MoE/Transformer 実装: Llama 4/Gemma 3クラスのMoE構造をSNNに移植。スパイク駆動型MoEルーティングと注意機構の効率化。(関連: spiking\_moe.py)  
  * \[ \] (P1.6) ニューロモルフィック・ハードウェア（Loihi, SpiNNaker, TPU, ASIC）へのコンパイル対応とベンチマーク。(関連: compiler.py)  
* **5-2: 自律型エージェント（Autonomous Agent）**  
  * \[x\] 認知アーキテクチャ全体を搭載し、複雑な環境で自律的にタスクを遂行するエージェントの基盤。(関連: snn\_research/agent/autonomous\_agent.py, run\_agent.py)  
* **5-3: デジタル生命体（Digital Life Form）**  
  * \[x\] 環境からエネルギーを獲得し、自己複製・進化するデジタル生命体のシミュレーション基盤。(関連: snn\_research/agent/digital\_life\_form.py, run\_life\_form.py)  
* **5-4: ハイブリッド・インテリジェンス**  
  * \[x\] SNNの高速・低電力な処理と、LLMの持つ豊富な言語知識を統合するアダプタ。(関連: app/adapters/snn\_langchain\_adapter.py, app/langchain\_main.py)
