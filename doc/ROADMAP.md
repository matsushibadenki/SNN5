# **SNN5プロジェクト：戦略的ロードマップ**

このロードマップは、ニューロモルフィックコンピューティングとスパイキングニューラルネットワーク（SNN）を用いて、ANNの性能を凌駕し、高度な認知能力を持つ「デジタル生命体」を創造するための戦略的な指針です。

## **P1: パフォーマンスと効率 (ANN同等性/超越) (最優先)**

| ID | タスク概要 | 詳細内容 | 関連モジュール | 優先度 | ステータス |
| :---- | :---- | :---- | :---- | :---- | :---- |
| P1.1 | SOTA変換パイプラインの確立 | ANNモデル（ResNet, ViT, MoE）からSNNへの損失の少ない変換（ANN-to-SNN）を実現する。最適な正規化（BN、LN）の統合。 | ann\_to\_snn\_converter.py, fold\_bn.py | 中 | 進行中 |
| P1.2 | SNN-MoE/Transformer 実装 | Llama 4/Gemma 3クラスのMoE構造をSNNに移植。スパイク駆動型MoEルーティングと注意機構の効率化。 | spiking\_moe.py, spiking\_transformer\_v2.py | 中 | 計画済み |
| **P1.3** | **階層的時系列アーキテクチャ (HOPE)** | **（新規 \- 最優先）HOPEアーキテクチャ(Nested Learning: The Illusion of Deep Learning
Architectures)に基づき、モチーフ（短時間）と構文（長時間）の処理を階層的に分離する構造を**TSkipsSNN**に導入し、長期依存性を効率的に処理する。** | **tskips\_snn.py, temporal\_snn.py** | **極めて高** | **計画済み** |
| **P1.4** | **学習則の理論的深化 (NL-DSAM)** | **（新規 \- 最優先）NL-DSAM（KLダイバージェンス最小化）の理論に基づき、プロジェクトの三因子学習則（CausalTrace）の誤差信号を改良し、STDPとの整合性を高める。** | **learning\_rules/causal\_trace.py, bio\_models/simple\_network.py** | **極めて高** | **計画済み** |
| P1.5 | T=1 低遅延推論の最適化 | 演算時間のステップ数 $T$ を最小限に抑えるための学習手法（DTTFS、ASR）の適用。 | trainer.py, losses.py | 中 | 進行中 |
| P1.6 | ハードウェアコンパイルとベンチマーク | 開発したSNNモデルをニューロモルフィックハードウェア（ Loihi, TPU, ASIC ）向けに最適化し、エネルギー効率を検証。 | compiler.py, metrics.py | 中 | 計画済み |

## **P2: 認知層アーキテクチャと自己認識 (目標⑨, ⑥の達成)**

P2トラックでは、単なる性能追求を超え、脳の認知機能（ワーキングメモリ、プランニング、感情、自己認識）をSNNで実現するための、認知アーキテクチャコンポーネントを開発します。

| ID | タスク概要 | 詳細内容 | 関連モジュール | 優先度 | ステータス |
| :---- | :---- | :---- | :---- | :---- | :---- |
| P2.1 | Global Workspace (GWS) | 全脳統合のためのGWS実装。SNN間の情報共有とブロードキャスト機構の確立。 | global\_workspace.py | 高 | 完了 |
| P2.2 | Hierarchical Planner | Prefrontal Cortex (PFC)とBasal Ganglia (BG)を統合した階層的プランニングの実現。報酬予測と行動選択のSNN化。 | prefrontal\_cortex.py, basal\_ganglia.py | 中 | 進行中 |
| P2.3 | Intrinsic Motivation | 好奇心（情報ゲイン）と達成感（予測誤差解消）に基づく内発的動機システムの設計と統合。行動の多様性を促進。 | intrinsic\_motivation.py | 中 | 進行中 |
| P2.4 | Causal Inference Engine | 過去の行動と結果の因果関係をSNNで学習し、プランニングにフィードバックする機構。世界モデルの構築。 | causal\_inference\_engine.py | 高 | 計画済み |
| P2.5 | 自己言及と主観的経験シグナル | 論文アイデアに基づき、SNNが自身の内部状態を報告する「主観的経験」シグナルを生成・利用するメカニズムを実装する。 | meta\_cognitive\_snn.py, self\_evolving\_agent.py | 高 | 新規 |
| **P2.6** | **経験（エピソード）記憶の符号化** | **新しい体験（行動、結果、主観的クラリティ）を符号化し、Hippocampus (海馬)に時系列で保存する。P2.5の修正ロジックの参照先を構築。** | **hippocampus.py, memory.py** | **高** | **計画済み** |

## **P3: 学習と最適化**

P3トラックは、SNNの学習効率と汎用性を高めるための、基礎的な学習規則と最適化手法を開発・検証します。

| ID | タスク概要 | 詳細内容 | 関連モジュール | 優先度 | ステータス |
| :---- | :---- | :---- | :---- | :---- | :---- |
| P3.1 | STDPベースの教師なし学習 | 時間的局所性を持つSTDP（Spike-Timing-Dependent Plasticity）を活用した教師なし特徴抽出層を実装。ニューロモルフィック学習の効率化を目指す。 | stdp.py, bio\_trainer.py | 中 | 計画済み |
| P3.2 | 強化学習への統合 (SNN-RL) | SNNをポリシー/バリューネットワークとして統合し、環境フィードバック（報酬）を考慮した学習則（R-STDP、報酬変調型STDPなど）を適用。 | reinforcement\_learner\_agent.py, reward\_modulated\_stdp.py | 中 | 進行中 |
| P3.3 | 継続学習と忘却防止 | 新しいタスク学習時の旧知識のカタストロフィック・フォゲッティングを防ぐ機構（Replay Buffer、Elastic Weight ConsolidationのSNN版）。 | continual\_learning\_experiment.py, memory.py | 中 | 計画済み |
| P3.4 | ハイパーパラメータ最適化（HPO） | Optunaなどを利用したSNNのニューロンパラメータ（閾値、膜時定数、シナプス遅延）の自動最適化。 | hseo.py, run\_optuna\_hpo.py | 中 | 進行中 |
| P3.5 | オンライン/連続学習パイプライン | 実環境からのデータストリーム（センサー入力、Webクローリング結果）をリアルタイムで取り込み、学習モデルを非同期的に更新する堅牢なパイプラインを構築する。 | trainers.py, data\_preparation.py | 中 | 計画済み |
| P3.6 | 報酬/エラーのスパイク符号化 | 内発的動機（P2.3）や外部報酬（P3.2）の連続値フィードバックを、SNNが理解可能なバイナリ/レート符号化されたスパイク信号に変換するモジュールを開発する。 | spike\_encoder.py, spike\_decoder.py | 中 | 計画済み |
| **P3.7** | **NL-DSAMと因果推論の連携** | **（新規）NL-DSAM (P1.4) で改良された学習則を、Causal Inference Engine (P2.4) のSNN層に適用し、因果関係学習の精度と効率を検証する。** | **causal\_inference\_engine.py, causal\_trace.py** | **高** | **計画済み** |

