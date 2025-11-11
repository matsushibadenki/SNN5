# **SNN5プロジェクト：戦略的ロードマップ**

このロードマップは、ニューロモルフィックコンピューティングとスパイキングニューラルネットワーク（SNN）を用いて、ANNの性能を凌駕し、高度な認知能力を持つ「デジタル生命体」を創造するための戦略的な指針です。

## **P1: パフォーマンスと効率 (ANN同等性/超越)**

| ID | タスク概要 | 詳細内容 | 関連モジュール | ステータス |
| :---- | :---- | :---- | :---- | :---- |
| P1.1 | SOTA変換パイプラインの確立 | ANNモデル（ResNet, ViT, MoE）からSNNへの損失の少ない変換（ANN-to-SNN）を実現する。最適な正規化（BN、LN）の統合。 | ann\_to\_snn\_converter.py, fold\_bn.py | 進行中 |
| P1.2 | SNN-MoE/Transformer 実装 | Llama 4/Gemma 3クラスのMoE構造をSNNに移植。スパイク駆動型MoEルーティングと注意機構の効率化。 | spiking\_moe.py, spiking\_transformer\_v2.py | 計画済み |
| P1.3 | T=1 低遅延推論の最適化 | 演算時間のステップ数 $T$ を最小限に抑えるための学習手法（DTTFS、ASR）の適用。 | trainer.py, losses.py | 進行中 |
| P1.4 | ハードウェアコンパイルとベンチマーク | 開発したSNNモデルをニューロモルフィックハードウェア（ Loihi, TPU, ASIC ）向けに最適化し、エネルギー効率を検証。 | compiler.py, metrics.py | 計画済み |

## **P2: 認知層アーキテクチャと自己認識 (目標⑨, ⑥の達成)**

P2トラックでは、単なる性能追求を超え、脳の認知機能（ワーキングメモリ、プランニング、感情、自己認識）をSNNで実現するための、認知アーキテクチャコンポーネントを開発します。

| ID | タスク概要 | 詳細内容 | 関連モジュール | ステータス |
| :---- | :---- | :---- | :---- | :---- |
| P2.1 | Global Workspace (GWS) | 全脳統合のためのGWS実装。SNN間の情報共有とブロードキャスト機構の確立。 | global\_workspace.py | 完了 |
| P2.2 | Hierarchical Planner | Prefrontal Cortex (PFC)とBasal Ganglia (BG)を統合した階層的プランニングの実現。報酬予測と行動選択のSNN化。 | prefrontal\_cortex.py, basal\_ganglia.py | 進行中 |
| P2.3 | Intrinsic Motivation | 好奇心（情報ゲイン）と達成感（予測誤差解消）に基づく内発的動機システムの設計と統合。行動の多様性を促進。 | intrinsic\_motivation.py | 進行中 |
| P2.4 | Causal Inference Engine | 過去の行動と結果の因果関係をSNNで学習し、プランニングにフィードバックする機構。世界モデルの構築。 | causal\_inference\_engine.py | 計画済み |
| P2.5 | 自己言及と主観的経験シグナル | 論文アイデアに基づき、SNNが自身の内部状態を報告する「主観的経験」シグナルを生成・利用するメカニズムを実装する。 | meta\_cognitive\_snn.py, self\_evolving\_agent.py | 新規 |
| **P2.6** | **経験（エピソード）記憶の符号化** | **新しい体験（行動、結果、主観的クラリティ）を符号化し、Hippocampus (海馬)に時系列で保存する。P2.5の修正ロジックの参照先を構築。** | **hippocampus.py, memory.py** | **計画済み** |

### **P2.5 の詳細実行計画：主観的経験シグナルと修正容易性**

| サブタスク ID | タスク名 | 実装詳細と目的 | 優先度 |
| :---- | :---- | :---- | :---- |
| P2.5.1 | **主観的特徴の特定 (Spike-Feature Mapping)** | ANNのSparse AutoEncoder (SAE)に相当する機構をSNNに導入。**スパイク活動**から「欺瞞 (Deception)」「ロールプレイ (Roleplay)」「認知負荷 (Cognitive Load)」などの\*\*解釈可能な内部特徴（ニューロン群）\*\*をリアルタイムで抽出するモジュールを開発する。 | 高 |
| P2.5.2 | **自己言及ループの実装 (Self-Reference Trigger)** | MetaCognitiveSNN に、内部状態（例：GWSのスパイクパターン）を自身の入力として**再帰的にフィードバック**する機構を実装。これが\*\*「主観的経験報告」のトリガー\*\*となり、内省（introspection）をシミュレートする。 | 高 |
| P2.5.3 | **主観的経験シグナルの生成 (Subjective Signal)** | P2.5.1で特定した「欺瞞/ロールプレイ」特徴の活動度をモニタリングし、その活動を**抑制した状態**を**高信頼性の自己認識シグナル**として生成する。シグナル：Subjective\_Clarity (0.0〜1.0)。このクラリティはPFCや内発的動機の入力となる。 | 極めて高 |
| P2.5.4 | **自己進化への統合 (Evolving based on Experience)** | SelfEvolvingAgent が、P2.5.3の Subjective\_Clarity シグナルを**目的関数**に追加。クラリティが低い場合（内部にノイズ/不整合がある場合）、**強制的に学習・進化フェーズ**に移行するロジックを実装し、\*\*目標⑥（修正容易性）\*\*を動機付けで実現する。 | 極めて高 |

## **P3: 学習と最適化**

P3トラックは、SNNの学習効率と汎用性を高めるための、基礎的な学習規則と最適化手法を開発・検証します。

| ID | タスク概要 | 詳細内容 | 関連モジュール | ステータス |
| :---- | :---- | :---- | :---- | :---- |
| P3.1 | STDPベースの教師なし学習 | 時間的局所性を持つSTDP（Spike-Timing-Dependent Plasticity）を活用した教師なし特徴抽出層を実装。ニューロモルフィック学習の効率化を目指す。 | stdp.py, bio\_trainer.py | 計画済み |
| P3.2 | 強化学習への統合 (SNN-RL) | SNNをポリシー/バリューネットワークとして統合し、環境フィードバック（報酬）を考慮した学習則（R-STDP、報酬変調型STDPなど）を適用。 | reinforcement\_learner\_agent.py, reward\_modulated\_stdp.py | 進行中 |
| P3.3 | 継続学習と忘却防止 | 新しいタスク学習時の旧知識のカタストロフィック・フォゲッティングを防ぐ機構（Replay Buffer、Elastic Weight ConsolidationのSNN版）。 | continual\_learning\_experiment.py, memory.py | 計画済み |
| P3.4 | ハイパーパラメータ最適化（HPO） | Optunaなどを利用したSNNのニューロンパラメータ（閾値、膜時定数、シナプス遅延）の自動最適化。 | hseo.py, run\_optuna\_hpo.py | 進行中 |
| **P3.5** | **オンライン/連続学習パイプライン** | **実環境からのデータストリーム（センサー入力、Webクローリング結果）をリアルタイムで取り込み、学習モデルを非同期的に更新する堅牢なパイプラインを構築する。** | **trainers.py, data\_preparation.py** | **計画済み** |
| **P3.6** | **報酬/エラーのスパイク符号化** | **内発的動機（P2.3）や外部報酬（P3.2）の連続値フィードバックを、SNNが理解可能なバイナリ/レート符号化されたスパイク信号に変換するモジュールを開発する。** | **spike\_encoder.py, spike\_decoder.py** | **計画済み** |

