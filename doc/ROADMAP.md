

# **SNN5 高度化ロードマップ：SOTA精度とエネルギー効率の実現に向けた戦略的実装計画**


## **基本設計思想**

・SNNをベースにする  
・誤差逆伝播法は用いない  
・微分を用いない  
・GPUに依存しない  
・高効率高精度を追求する


## **I. ベースラインの再定義と「ダミー実装」の解消 (フェーズ1)**

SNN5プロジェクト（matsushibadenki 1）の現状の実装、特に「未実装」または「ダミー実装」として機能しているコンポーネントを特定し、それらを2025年以降のSOTA（State-of-the-Art）標準に引き上げることは、本ロードマップの最優先事項です。GitHubリポジトリ（SNN5 2）へのアクセスが不能であるため、一般的なSNNプロジェクトの初期段階で見られる典型的なプレースホルダー実装を「ダミー実装」と推定し、その即時置換を提言します。

### **1.1. 推定されるベースラインとSOTAへの道筋**

SNN5の現在の実装は、小規模な畳み込みアーキテクチャ、単純なLeaky Integrate-and-Fire (LIF) ニューロン 3、そして生物学的妥当性を重視したSpike-Timing-Dependent Plasticity (STDP) ベースの学習ルール 4 に依存していると推定されます。これらはSNNの基本原理を学ぶ上では有用ですが、現代的なディープラーニングのタスクでANN（Artificial Neural Network）に匹敵する精度を達成する上では、「ダミー実装」に他なりません。

以下の表1は、本レポート全体のアクションプランを要約したものです。推定される「ダミー実装」の欠点を明らかにし、本ロードマップが目指すSOTAへのアップグレードパス、および関連する本文中の参照セクションを示します。

**表1：ベースラインコンポーネント vs SOTAアップグレードパス**

| コンポーネント | 推定されるダミー実装 | 欠点 | SOTAアップグレードパス | 参照 |
| :---- | :---- | :---- | :---- | :---- |
| **学習ルール** | STDP (Spike-Timing-Dependent Plasticity) | 精度が低い。ディープネットワークでの勾配ベース最適化が不可能 \[5, 6\]。 | **代理勾配法 (Surrogate Gradient: SG)** によるエンドツーエンドの直接訓練。 | 1.2, 1.3 |
| **ニューロンモデル** | 単純なLIF (固定パラメータ) | 表現力が乏しい。全ニューロンが同じダイナミクスを持つ \[7\]。 | **PLIF / GLIF** (学習可能な膜時定数 $\\tau$ やゲート機構) \[7, 8\]。 | 3.1 |
| **アーキテクチャ** | 小規模な浅いCNN | ANNの精度に匹敵しない。勾配消失により深層化が困難。 | **SEW-ResNet** (深層化) 9, **Spikingformer** (Transformer) 9。 | 2.1, 5.1 |
| **時空間処理** | (未実装) | SNN固有の時間情報を活用できていない。 | **時空間アテンション** (DTA-SNN \[10\], STAA-SNN \[11\])。 | 2.2 |
| **省エネ最適化** | (未実装 / 幻想) | 「SNNは本質的に省エネ」という神話 12。高スパイクレートと高レイテンシ。 | **スパース性正則化** 13, **QAT (QUEST)** 14, **プルーニング (SBC)** 15。 | 4.2-4.4 |
| **推論方法** | 固定タイムステップ実行 | レイテンシが固定され、リアルタイム性が低い。 | **動的推論 (SNN Cutoff)** \[16\]。入力毎にレイテンシを最適化。 | 6.1 |
| **評価メトリクス** | 精度 (Accuracy) のみ | エネルギー効率を測定できない \[17\]。 | **Synaptic Operations (SynOps)** 18, **スパイクレート** 19。 | 7.1 |

### **1.2. ロードマップ・アクション 1.1: 学習ルールの移行（STDPから代理勾配法へ）**

**欠陥の特定:** 多くの初期SNNプロジェクトでは、生物学的妥当性を名目にSTDPが学習ルールとして「ダミー実装」されています 4。

**分析:** STDPは、ニューロン間のローカルなスパイクタイミングのみに依存する教師なし、または半教師あり学習ルールです 6。これは、最新のディープラーニングにおいて業界標準となっている、誤差逆伝播法（Backpropagation）によるエンドツーエンドの勾配ベース最適化（例: Adam, SGD）と根本的に互換性がありません。\[5\]や\[85\]が示唆するように、STDPベースの手法は、複雑なタスクにおける教師あり学習において、勾配ベースの手法に性能面で大きく劣ります。

**アクション:** SNN5の学習基盤を、STDP実装から「代理勾配（Surrogate Gradient: SG）法」へ全面的に移行します。SG法は、SNNの非微分可能なスパイク生成関数（ステップ関数）を、逆伝播時のみ微分可能な関数（代理関数）で置き換える技術です 21。

**実装:** snnTorch 23 や SpikingJelly 24 などのSOTAフレームワークは、PyTorchの自動微分（autograd）と完全に互換性のある、高性能なSGモジュールを標準で提供しています。これらのライブラリを導入することで、SNN5を最新の勾配ベース最適化の軌道に乗せます。

### **1.3. ロードマップ・アクション 1.2: 代理勾配（SG）の最適化**

**欠陥の特定:** SG法を導入するだけでは不十分です。単純な矩形関数や線形関数を代理勾配として「ダミー実装」した場合、学習が不安定になるか、最適解に収束しないリスクがあります。

**分析:** 代理勾配の「形状」（例: 関数の幅や傾斜）は、SNNの学習性能に決定的な影響を与えます 25。25および25の研究は、代理勾配の幅が広すぎると勾配の不一致（mismatch）を、狭すぎると勾配消失（vanishing）を引き起こす可能性があると警告しています。

**SOTAインサイト:** 2024年以降の研究 22 では、なぜSGが経験的に成功するのか、その理論的基盤の解明が進んでいます。タスクやニューロンモデルのダイナミクスに応じて、最適なSG形状を適応的に選択することが、SOTA性能を達成するための重要な研究課題となっています 25。

**アクション:** SNN5の学習基盤に、複数のSG形状（例: Sigmoidの導関数、ArcTangent、Gaussianの導関数）をテストし、最適なものを選択するハイパーパラメータ調整プロセスを導入します。Zenke Lab 22 や関連する最新の理論的研究 25 の成果をベンチマークとして参照し、SNN5のアーキテクチャに最適なSGを特定します。

フェーズ1の成果:  
SNN5は、STDPという「ダミー実装」から脱却し、最新の勾配ベース最適化（Backpropagation）が可能な、SOTA SNN開発の標準的基盤を獲得します。

---

## **II. 精度向上のためのアーキテクチャ進化 (フェーズ2A)**

学習基盤（フェーズ1）の確立後、次のステップは「精度向上」のためのアーキテクチャの根本的な近代化です。本セクションでは、ANNでSOTA精度を達成した実証済みのアーキテクチャ（ResNet, Attention）をSNNに導入する具体的な方法論を詳述します。

### **2.1. ロードマップ・アクション 2.1: 深層化のための残差学習 (SEW-ResNet)**

**未実装機能:** 30層を超えるような深層SNNの学習機能。

**分析:** 従来のSNNは、ANNと同様に、層を深くすると勾配消失または勾配爆発の問題に直面し、深層化による精度向上の恩恵を受けられませんでした。9および26で分析されているSEW (Spike-Element-Wise) ResNetは、SNNにおいて残差接続（Residual Connection）を導入することでこの問題を解決し、SNNの直接訓練（Direct Training）を100層以上にまでスケールさせることを可能にした画期的なアーキテクチャです。

**アクション:** SNN5にSEW-ResNetアーキテクチャを実装します。これにより、ANNと同様に、ネットワークを深くすることで精度をスケールさせるという、現代のディープラーニングにおける最も基本的な戦略が実行可能になります 26。

**SOTAインサイト (ハードウェア非互換性のリスク):** ここで、matsushibadenki（ロボティクス企業）にとって極めて重要な戦略的岐路が存在します。9, 9, \[27\]は、SEW-ResNetの残差接続（ADDゲート）が「非スパイク計算（non-spike computations）」、具体的には整数と浮動小数点数の乗算を発生させるという重大な欠点を指摘しています。これは、GPU上での計算では問題になりませんが、純粋なスパイク演算（加算のみ）を前提とする多くのニューロモーフィック・ハードウェア・アクセラレータ 9 へのデプロイを不可能にする可能性があります。

したがって、SEW-ResNetは「GPU上での高精度モデル（フェーズ2A）」として実装を進める一方、この欠点を根本的に解決する「ハードウェア互換」の次世代アーキテクチャ（セクションVのSpikingformer）の研究（フェーズ2B）を並行して開始する必要があります。

### **2.2. ロードマップ・アクション 2.2: 時空間アテンションの統合**

**未実装機能:** SNN固有の時空間情報を活用するアテンションメカニズム。

**分析:** SNNは、空間（ニューロン）と時間（スパイクタイミング）の情報を同時に処理します。ANNにおけるTransformerの成功以降、この時空間情報を動的に活用するアテンション機構がSNNの精度向上に不可欠であるという認識が広まっています。2025年の最新研究 10 は、このアプローチの有効性を強く示しています。

**SOTA技術 (DTA-SNN):** \[10\]および\[86\], \[87\], \[88\]で提案されているDTA-SNN (Dual Temporal-channel-wise Attention) は、SNNの性能向上のために「同一（identical）」および「非同一（non-identical）」アテンション戦略を単一のブロックに統合する新しいメカニズムです。\[10\]によれば、これにより時間とチャネル間の複雑な相関関係と依存関係を効果的に捉え、スパイク表現（spike representation）を大幅に向上させることができます。

**SOTA技術 (STAA-SNN):** \[11\]で提案されているSTAA-SNN (Spatial-Temporal Attention Aggregator) は、SNN向けに特別に設計されたスパイク駆動の自己アテンション（spike-driven self-attention）と位置エンコーディングを導入します。これにより、空間的依存関係と時間的依存関係の両方を動的に捕捉します。

**アクション:** SNN5のアーキテクチャ（例: SEW-ResNetのブロック間）に、DTA-SNNまたはSTAA-SNNの概念に基づいた、スパイクベースの時空間アテンション・モジュールを実装します。\[10\]と\[11\]の報告によれば、これらのアテンション機構の導入により、静的データセット（CIFAR100で81%超）および動的ニューロモーフィック・データセット（CIFAR10-DVS）の両方でSOTAの精度が達成されています。

フェーズ2Aの成果:  
SNN5は、深層学習（ResNet）と高度な時空間処理（Attention）を備え、GPU上でANNに匹敵する高精度を実現するSOTAアーキテクチャを獲得します。

---

## **III. 精度向上のためのニューロン・ダイナミクスの強化 (フェーズ2C)**

アーキテクチャ（マクロ）の近代化（フェーズ2A）に続き、ニューロンモデル（ミクロ）の高度化に着手します。これは「精度向上」の第2の柱であり、SNN5に「単純なLIF」という「ダミー実装」が残っている場合の、表現力に関する根本的な欠陥を解消します。

### **3.1. ロードマップ・アクション 3.1: ニューロンモデルのアップグレード（PLIF / GLIF）**

**欠陥の特定:** 標準的なLIFニューロンモデルは、膜電位の減衰率を決定する膜時定数 ($\\tau$) や発火閾値 ($V\_{th}$) といった、ニューロンのダイナミクスを決定する重要なパラメータが、手動でチューニングされた固定値（ハイパーパラメータ）です 3。

**分析:** ネットワーク内のすべてのニューロンが、層や機能に関わらず同一のダイナミクスを持つことは、ネットワーク全体の「表現力（expressiveness）」を著しく制限します 7。例えば、入力層に近いニューロンは速い応答（小さい $\\tau$）を、後段の層は時間的な積分（大きい $\\tau$）を必要とするかもしれませんが、LIFではこれに適応できません。

**SOTA技術 (PLIF):** Parametric Leaky Integrate-and-Fire (PLIF) ニューロンは、この問題を解決するために、膜時定数 $\\tau$ を「学習可能なパラメータ」として導入します 7。これにより、ネットワークは逆伝播を通じて、各ニューロン（あるいは各層）に最適な時間スケールを自動で学習できます。

**SOTA技術 (GLIF):** Gated Leaky Integrate-and-Fire (GLIF) ニューロンは、さらに一歩進んだモデルです 8。GLIFは、複数の生物学的特徴（例: 異なる時定数、適応的な閾値、異なるリセット機構）を、学習可能な「ゲート」によって動的に融合させます。\[89\]の研究では、GLIFがPLIFやLIFよりも高い学習能力を持つことが示されています。

**アクション:** SNN5の標準LIFニューロンを、まずはPLIFに置き換えます。\[7\]によれば、これにより学習が高速化し、初期値への感度が低下し、より少ないタイムステップ（$T$）でSOTAの精度を達成することが期待されます。さらなる性能向上のため、GLIF 28 へのアップグレードを試みます。

ただし、\[90\]が警告するように、これらの高度なモデルは追加のハイパーパラメータと複雑さを導入するため、省エネ化（フェーズ3）の目標とはトレードオフの関係にある可能性を認識する必要があります。これは純粋な精度追求トラックのアクションです。

### **3.2. ロードマップ・アクション 3.2: 時間コーディング方式の最適化**

**欠陥の特定:** おそらく入力エンコーディングとして「レートコーディング（rate coding）」がダミー実装されていると推定されます。レートコーディングは、入力値の強度をスパイクの「頻度」に変換しますが、これはSNNの時間ダイナミクスを全く活用できておらず、高い精度を得るためには非常に多くのタイムステップ（高レイテンシ）を必要とします。

**分析:** SNNの真価は、スパイクの「タイミング」で情報を表現する時間コーディングにあります 30。30は、最も単純な時間コーディングの一つであるTime-to-first-spike (TTFS) コーディング（入力強度が強いほど、早くスパイクする）の効率性を強調しています。\[84\]で行われた比較実験では、時間コーディングを採用したネットワークが、レートコーディングのネットワークに対し、**5.68倍高速**（低レイテンシ）でありながら**15.12倍の低消費電力**を達成したと報告されています。

**アクション:** SNN5の入力エンコーディング方式を、レートコーディングから、TTFS 30 やその他の時間ベースのコーディング 31 へと移行する研究開発を行います。これにより、精度を維持、あるいは向上させつつ、レイテンシとエネルギー効率を劇的に改善できる可能性があります。

---

## **IV. 省エネ化（エネルギー効率）の徹底追求 (フェーズ3)**

本セクションは、クエリの第二の目標である「省エネ化向上」に正面から応えます。これは、matsushibadenkiの（ロボティクス 1）応用を想定した場合、SNN5プロジェクトにおいて**最も重要なフェーズ**となります。

### **4.1. ロードマップ・アクション 4.1: 省エネの再定義（ハードウェアの現実の直視）**

SOTAインサイト (根本的トレードオフ):  
SNN5の省エネ化戦略を立案する上で、まず2024年から2025年にかけて明らかになった「ハードウェアの現実」を直視する必要があります。

1. **神話:** 「SNNは（ANNより）本質的に省エネである」という神話が存在します 12。これは、SNNが乗算（MAC）の代わりに加算（AC）を使用し、イベント駆動型であることに起因します。  
2. **現実:** しかし、12, 32, 19などの最新のハードウェア指向の研究は、この神話を厳しく再評価しています。SNNが同等のANN（特に量子化されたANN、QNN）よりも真にエネルギー効率が高くなるのは、「（1）極端に短いタイムステップ（例: $T \\le 5$ または $T \\le 6$）」と「（2）極端に高いスパース性（例: 全ニューロンのスパイクレートが 6.4% 未満 32、あるいはニューロンのスパース性 $\> 93\\%$ 19）」という2つの条件を**同時に**達成した場合のみです。  
3. **危機:** ここで重大なジレンマが発生します。フェーズ2で追求した高精度モデル（SEW-ResNet, Attention, GLIF）は、一般的に情報を時間軸で積分するために*多くの*タイムステップ（例: $T=16$ 3）と*高い*スパイクレートを要求する傾向があり、省エネの目標と*真っ向から対立*します。

**アクション:** SNN5の主要業績評価指標（KPI）を、「精度」単体から、「精度 vs. スパース性 vs. タイムステップ」の三次元での多目的最適化に切り替えます。以降のアクション（4.2 \- 4.4）は、この厳しいトレードオフを達成するために不可欠な、SOTAの圧縮・最適化技術です。

### **4.2. ロードマップ・アクション 4.2: スパース性の強制的導入（正則化）**

**未実装機能:** スパース性を能動的に強制するメカニズム。

**分析:** アクション4.1で定義したように、SNNの省エネ性はスパースな活性化（スパイク）に依存します 33。ネットワークが密に発火すれば、SNNはANNよりもエネルギー効率が悪化します 12。13と\[58\]は、スパース性を高めるために「正則化（regularization）」が不可欠であることを示しています。12の研究では、損失関数に正則化項を導入して重みと活性化を制約することで、VGG16においてANNの69%のエネルギー消費を（高精度を維持したまま）達成しました。

**アクション:** SNN5の学習ループ（PyTorch/SNNtorch）の損失関数に、スパース性を誘発するペナルティ項を追加します。これは、単純なL1正則化 35 や、ニューロンの平均スパイクレート自体を直接ペナルティとする正則化 13 によって実現可能です。これは比較的低コストで実装可能でありながら、エネルギー効率に直結する重要なステップです。

### **4.3. ロードマップ・アクション 4.3: 量子化アウェア・トレーニング (QAT)**

**未実装機能:** 低ビット（例: 2ビット、4ビット）での学習および推論。

**分析:** エネルギー効率の追求は、活性化のスパース性（アクション4.2）だけでなく、パラメータ（重み）の低ビット化（量子化）も必要とします 37。\[91\]はSNNのためのQAT（Quantization-Aware Training）の基礎的な実装を示しています。

**SOTA技術 (QUEST):** 2025年のSOTAフレームワーク「QUEST」 14 は、このSNNの量子化を、ニューロモーフィック・デバイス（特に多状態抵抗デバイス）の物理特性と結びつけて最適化する「アルゴリズム・ハードウェア協調設計」の最先端です。14と\[40\]によれば、QUESTは「2ビット」という極端な低ビット精度でトレーニングを行い、CIFAR-10で89.6%の高い精度を達成しつつ、ANN比で**93倍**という驚異的なエネルギー効率の向上を実現します。

**アクション:** SNN5のトレーニングパイプラインにQUESTフレームワークの概念を導入し、QATを実装します。14で示されているように、これは単にビット数を落とすだけでなく、ターゲットデバイスの特性（例: ハードリセット機構の採用、LIFではなくIFニューロンの使用）に合わせて、SNNのアルゴリズム自体を最適化するプロセスを含みます。

### **4.4. ロードマップ・アクション 4.4: ワンショット・プルーニング (SBC)**

**未実装機能:** ネットワークの剪定（プルーニング）によるパラメータ削減。

**分析:** 従来のSNNプルーニング手法は、ANNと同様に、高コストな「反復的な（iterative）」プロセス（プルーニング $\\rightarrow$ 再学習 $\\rightarrow$ プルーニング $\\rightarrow$...）を必要とします 15。これは、大規模なモデルの開発サイクルを著しく遅延させます。

**SOTA技術 (SBC):** 2025年のSOTA技術「Spiking Brain Compression (SBC)」 15 は、この問題を解決する画期的な手法です。SBCは、古典的なOBC（Optimal Brain Compression）をSNNに拡張したものです。15によれば、SBCは従来の電流ベースの損失関数の代わりに、「スパイク列ベース（Van Rossum Distance）」の損失関数を使用し、そのヘッセ行列（損失の二次微分）を効率的に計算します。

**アクション:** 反復的プルーニングの代わりに、SBCを実装します。SBCの最大の利点は、ヘッセ行列を用いて「どのシナプスを削除すべきか」と「残ったシナプスの重みをどう補正すべきか」を解析的に決定できるため、\*\*再学習なしの「ワンショット」\*\*でプルーニングと量子化を実行できる点です 15。15によれば、SBCは従来の高コストな反復手法に匹敵する精度を達成しつつ、圧縮に要する計算時間を「2～3桁」削減します。これは、SNN5の開発サイクルの劇的な高速化を意味します。

### **4.5. （代替パス）ロードマップ・アクション 4.6: 高度なANN-to-SNN変換**

**分析:** フェーズ1～4で詳述した「直接訓練（Direct Training）」は、SNNの性能を最大化する上での本命ですが、学習の調整が困難な場合があります 25。その場合、SOTAのANN（例: ResNet, ViT）を訓練してからSNNに変換する「ANN-to-SNN変換」が、実用的な代替パスとなります。

**SOTA技術:** かつてのANN-to-SNN変換は、精度の大幅な低下や、精度を出すために膨大なタイムステップ（高レイテンシ）を必要とするという欠点がありました 43。しかし、2024-2025年の最新技術 44 は、これらの問題を克服しつつあります。特に44の「Adaptive-firing Neuron Model (AdaFire)」や\[45\]の「T=1（タイムステップ1）」での高性能変換技術は、超低レイテンシと高精度を両立する可能性を示しています。

**アクション:** 直接訓練が難航する場合のバックアッププランとして、これらの最新（2025年）のANN-to-SNN変換技術（例: AdaFire 44）を調査し、SNN5の高速デプロイメント・オプションとして実装を検討します。

### **4.6. フェーズ3の成果と提案テーブル**

フェーズ3の成果:  
SNN5は、高精度を追求する（フェーズ2）だけでなく、「低タイムステップ」と「高スパース性」というハードウェアの現実に即した制約下で動作可能な、真にエネルギー効率の高いモデルへと変貌します。  
以下の表2は、matsushibadenkiがSNN5の省エネ化戦略を決定する上で役立つ、主要な圧縮技術の比較です。

**表2：SNN圧縮・省エネ技術の戦略的比較**

| 技術 | 参照SOTA | 開発コスト | 精度への影響 | 省エネ効果 (SynOps/電力) | タイムステップ(T)依存性 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **スパース性正則化** | \[12, 13\] | 低 | 軽微（調整次第） | 高。スパイクレートを直接削減。 | $T$ が短くても効果あり。 |
| **QAT (量子化)** | QUEST 14 | 高 | 軽微（QATにより維持） | 極めて高い。メモリと演算コストを削減 (例: 93x 14)。 | $T$ が短い場合（$T \\le 5$）に特に有効。 |
| **プルーニング** | SBC 15 | 中 (SBCの場合) | 軽微（SBCにより維持） | 高。パラメータ数とSynOpsを削減。 | $T$ に依存しないが、スパース性向上に寄与。 |
| **ANN-SNN変換** | AdaFire 44 | 中 | SOTA ANNに依存 | 高（$T=1$ \[45\] が可能）。 | $T$ を極小化する（$T=1$）ことを目指す。 |

---

## **V. 次世代アーキテクチャの導入 (フェーズ2B: 並行トラック)**

フェーズ2A (SEW-ResNet) が既存のANNアーキテクチャをSNNに「適応」させる試みであったのに対し、本セクションはSNNの特性（イベント駆動）に最適化された次世代アーキテクチャ、特にTransformerへのロードマップを提示します。これは、フェーズ2Aで特定された「非スパイク計算」の問題 9 を根本的に解決するため、フェーズ2Aと並行して進めるべき、より挑戦的な研究開発トラックです。

### **5.1. ロードマップ・アクション 5.1: スパイクベース・トランスフォーマー (Spikingformer)**

**未実装機能:** SOTAのTransformerアーキテクチャ。

**分析:** ANNの世界では、CNN（ResNet）からTransformer（ViT）へのパラダイムシフトが完了しました。SNNにおいても、Transformer 50 やVision Transformer (ViT) 51 への適応が活発化しています。\[92\]が指摘するように、最大の課題は、Transformerの中核であるSoftmaxや高密な「Query-Key-Value」のドット積計算を、バイナリのスパイクでいかに効率的に実装するかです。

**SOTA技術 (Spikingformer):** 9, 9, 9, \[27\]は、Spikingformerと呼ばれる画期的なアーキテクチャを提案しています。これは、フェーズ2AのSEW-ResNetが抱えていた「非スパイク計算」問題を解決するために設計された、**純粋なイベント駆動**のTransformerです。9によれば、Spikingformerは「ハードウェアフレンドリーなスパイク駆動残差学習」アーキテクチャを採用しています。

**成果:** Spikingformerは、ImageNetにおいて先行研究（Spikformer）を上回るSOTAの精度（75.85%）を達成しつつ、エネルギー消費を**57.34%削減**しました 9。

**アクション:** フェーズ2AのSEW-ResNet（非スパイク計算あり）の「代替」または「次世代機」として、純粋なイベント駆動型であるSpikingformerの実装に着手します。これは、将来のニューロモーフィック・ハードウェア 9 へのデプロイを見据えた、SNN5の「本命」アーキテクチャとなり得ます。

### **5.2. ロードマップ・アクション 5.2: ハイブリッド・アーキテクチャ (HsVT)**

**分析:** 純粋なSNN Transformer（Spikingformer）の学習・実装が困難な場合の「中間解」として、ANNとSNNのコンポーネントを組み合わせたハイブリッドモデルが注目されています。

**SOTA技術 (HsVT):** 2025年のICML論文 53 は、Hybrid Spiking Vision Transformer (HsVT) を提案しています。HsVTは、ANNの強力なコンポーネント（自己アテンションと畳み込み）を利用して空間特徴を効率的に抽出し、エネルギー効率の高いSNNモジュールで時間特徴を抽出する、両者の利点を融合させたハイブリッド型です 54。

**アクション:** Spikingformerと並行し、より実装が容易で実用的なソリューションとしてHsVTを評価します。\[56\]によれば、HsVTは少ないパラメータで高い性能向上を達成しており、matsushibadenkiの製品への迅速な機能搭載に適している可能性があります。

### **5.3. 提案テーブル： SOTAアーキテクチャ戦略的比較**

以下の表3は、SNN5が採用すべきSOTAアーキテクチャに関する戦略的な意思決定を支援します。

**表3：SNN SOTAアーキテクチャ戦略的比較**

| アーキテクチャ | 主要革新 | 参照論文 | 精度 (ImageNet例) | ハードウェア互換性（非スパイク計算） |
| :---- | :---- | :---- | :---- | :---- |
| **SEW-ResNet** | SNNの深層化 (100+層) | \[9, 26\] | 70.02% (ResNet-34) 3 | **低い**（残差接続で非スパイク計算が発生 9）。 |
| **Spikingformer** | **純粋な**イベント駆動型Transformer。ハードウェアフレンドリーな残差学習。 | 9 | **75.85%** 9 | **高い**（非スパイク計算を排除するように設計 9）。 |
| **HsVT** | ANN(空間)とSNN(時間)のハイブリッドTransformer。 | \[53, 57\] | (非公開/データセット依存) | **中**（ANN部が非スパイク計算を含むが、明確に分離されている）。 |

---

## **VI. 実装とデプロイメント戦略 (フェーズ4)**

最適化されたSNN5を、matsushibadenki（ロボティクス企業 1）の製品（エッジデバイス）にデプロイするための具体的なロードマップです。

### **6.1. ロードマップ・アクション 6.1: 動的推論 (SNN Cutoff)**

**未実装機能:** 適応型（Adaptive）推論。現在のSNNの多くは、推論のために固定のタイムステップ（例: $T=100$）を実行する必要があり、入力に関わらずレイテンシが固定されています。

**分析:** ロボティクスやリアルタイム処理では、平均的なレイテンシの低さよりも、必要なタスクを即座に（例: $T=5$ で）完了できる応答性が求められます。\[16\], \[93\], \[94\], \[95\]が指摘するように、SNNは原理的に、スパイク列の途中の「いつでも（anytime）」予測が可能です。

**SOTA技術 (SNN Cutoff):** 2025年のSOTA技術「SNN Cutoff」 16 は、このSNNの特性を最大限に活用する技術です。これは、推論の途中で「出力層の確信度が十分高くなった」と判断した場合に、残りのタイムステップの処理を早期に打ち切る（Cutoff）動的なメカニズムです。\[58\]と\[16\]は、Top-K Cutoffと呼ばれる監視機構と、早期のタイムステップで正しい判断ができるようSNNを最適化する正則化手法を提案しています。

**アクション:** SNN5の推論パイプラインにSNN Cutoff（Top-K Cutoffと専用正則化）を実装します。\[58\]によれば、これによりCIFAR10で**1.76～2.76倍**、イベントベースデータセットで**1.64～1.95倍**のレイテンシ削減（高速化）が、**ほぼゼロの精度低下**で達成されています。これは、SNN5のリアルタイム性を保証するキラー機能となります。

### **6.2. ロードマップ・アクション 6.2: ニューロモーフィック・ハードウェア協調設計**

**未実装機能:** GPU以外の特定ハードウェア（ニューロモーフィック・チップ）へのデプロイメント。

**分析:** SNNの最終的な目標は、GPUよりも桁違いに高いエネルギー効率を持つ、専用のニューロモーフィック・ハードウェア上での動作です。matsushibadenkiがエッジデバイスへの実装を目指す以上、これらのチップへの対応は必須です。

**SOTAハードウェア:**

1. **Intel Loihi 2:** 59 Intelの最新の研究用チップ。63, \[96\], \[68\]によれば、Loihi 2は非同期・並列処理を前提としており、プログラミングはオープンソースの「Lava」フレームワーク 61 を通じて行われます。Lavaは、SNNのアルゴリズムをハードウェア非依存で記述するための抽象化レイヤーを提供します 63。  
2. **SpiNNaker:** 59 大規模シミュレーションに強みを持つ、マンチェスター大学のハードウェア。\[69\], 66, \[97\]によれば、プログラミングはPythonベースの「sPyNNaker」 65 を使用し、標準的なPyNN APIを通じてホスト（Python）とSpiNNakerボード（C言語）が連携します 66。  
3. **IBM TrueNorth / NorthPole:** 59 超低消費電力（70mW 67）の先駆的なチップ。

**アクション:** SNN5のソフトウェア・スタックが、Lava（Loihi 2用）およびsPyNNaker（SpiNNaker用）の両フレームワークに出力可能になるよう、抽象化レイヤーを設計します。特に、フェーズ3で実装した低ビット量子化（QUEST 14）や、フェーズ2Bの純粋なイベント駆動アーキテクチャ（Spikingformer 9）が、これらのハードウェアの制約（例: スパイク演算のみ、オンデバイス学習）と適合することを早期に検証します。

### **6.3. 提案テーブル： ニューロモーフィック・ハードウェア・デプロイメントガイド**

以下の表4は、SNN5のデプロイ先となる主要なニューロモーフィック・ハードウェアの特性と、matsushibadenkiのユースケースとの適合性を示します。

**表4：ニューロモーフィック・ハードウェア・デプロイメントガイド**

| ハードウェア | 開発元 | プログラミング・フレームワーク | アーキテクチャ特性 | 最適なSNN5ユースケース |
| :---- | :---- | :---- | :---- | :---- |
| **Loihi 2** | Intel | **Lava** (オープンソース) \[61\] | 非同期, 並列スパース計算, オンチップ学習 (3-Factor Learning) \[68\]。 | ロボットの適応制御, リアルタイム・センサーフュージョン, オンデバイス学習。 |
| **SpiNNaker** | U. Manchester | **sPyNNaker** (PyNN API) \[69\] | デジタル・マルチコア, パケットベース通信 \[59\], 大規模シミュレーション。 | 大規模ネットワーク（例: 脳シミュレーション）のリアルタイム実行, 複雑なSNNモデルの検証。 |
| **TrueNorth** | IBM | (独自) | 超低電力 (70mW) 67, 100万ニューロン \[64\], リアルタイム・センサー処理。 | ドローン搭載 67, 超低電力のエッジ推論（例: 常時監視センサー）。 |

---

## **VII. 統合的ベンチマーキングとロードマップ総括 (フェーズ5)**

開発したSNN5の性能を客観的に評価し、ロードマップ全体を管理するための基盤を構築します。

### **7.1. ロードマップ・アクション 7.1: 評価基盤とメトリクスの確立**

**欠陥の特定:** おそらく、現在のSNN5の評価指標は「精度（Accuracy）」のみとなっていると推定されます。

**SOTAインサイト:** \[17\]が指摘するように、SNNの評価は「精度」だけでは不十分です。SNNの真の効率を測るには、エネルギー消費に直結する以下のメトリクスが不可欠です。

1. **Synaptic Operations (SynOps):** スパイクが発生した際に行われる加算演算の総数。これはSNNにおける実質的な計算コスト（エネルギー消費）を最もよく表す指標です 70。snnmetrics 18 などのライブラリで計算可能です。  
2. **スパイクレート (Spike Rate):** ネットワーク全体または層ごとの平均スパイク頻度。アクション4.1で見たように、これが省エネ性能を決定します 19。

**SOTA基盤:**

1. **データセット:** 標準的な静的データセット（CIFAR, ImageNet）に加え、ニューロモーフィック・データセット（N-MNIST 72, CIFAR10-DVS 74, DVS128 Gesture 74, N-Caltech101 72）を標準ベンチマークとして導入します。  
2. **フレームワーク:** SpikingJelly 24 は、これらのニューロモーフィック・データセットの処理から、モデル構築、最適化、デプロイまでをサポートする、現在最も強力なSNN開発ツールキットの一つです。  
3. **ベンチマークスイート:** SNNBench 79 は、学習と推論の両方を含む「エンドツーエンド」のアプローチを提供し、AI指向のワークロード（画像分類、音声認識）を評価するための標準化された手法を提供します 79。

**アクション:** SNN5の公式開発基盤としてSpikingJellyを採用します。そして、SNNBenchのワークロードとメトリクス（特に精度, SynOps, スパイクレート）をSNN5のCI/CDパイプラインに統合し、すべての変更が「精度」と「省エネ」の両方の観点で評価される体制を構築します。

### **7.2. ロードマップ・アクション 7.2: 大規模化への展望**

**分析:** matsushibadenkiが将来的にLLM（大規模言語モデル）のような大規模モデル 81 の領域にSNNを適応させる場合、スケーラビリティが最大の課題となります 82。

**SOTAインサイト:** \[98\]では、人間の大脳皮質をモデル化した350万ニューロン、420億シナプスの超大規模SNNが報告されており、SNNのスケーラビリティ研究は着実に進んでいます。\[99\], \[100\], \[101\]は、2025年のAIトレンドが「エッジAI」「オンデバイス学習」「リアルタイムML」であり、これらはまさにSNNが（本ロードマップの実行によって）ANNに対して優位性を持つ分野であることを示しています。

**アクション:** 本ロードマップ（フェーズ1-4）の実行により、SNN5はこれらの2025年のAIトレンドを牽引する、スケーラブルかつ効率的な基盤となります。

### **7.3. 提案テーブル： SNN5 段階的実装ロードマップ（総括）**

以下の表5は、本レポートの総括として、matsushibadenkiが実行すべき「SNN5高度化ロードマップ」を時系列のフェーズにマッピングしたものです。

**表5：SNN5 段階的実装ロードマップ（総括）**

| フェーズ | 主要アクション | 参照 | 目的（解消する欠陥） | 主要メトリクス | 期待される成果 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Phase 1** | **学習基盤のSOTA化** ・STDP $\\rightarrow$ 代理勾配(SG)法 23 ・SG形状の最適化 25 | 1.2, 1.3 | **ダミー実装 (STDP)** 勾配ベースで学習できない欠陥。 | 精度 (Accuracy) | SOTAのSNN開発基盤の確立。 |
| **Phase 2A** (GPU Track) | **高精度アーキテクチャ(1)** ・SEW-ResNet 実装 9 ・時空間アテンション (DTA) \[10\] | 2.1, 2.2 | **未実装 (深層化)** 浅いネットワークによる低精度。 | 精度 (SOTA ANN比) | GPU上でANNに匹敵する高精度モデルの獲得。 |
| **Phase 2B** (HW Track) | **高精度アーキテクチャ(2)** ・**Spikingformer** 実装 9 ・(代替) HsVT 実装 \[54\] | 5.1, 5.2 | **SEW-ResNetの欠陥** 「非スパイク計算」 9 によるHW非互換性。 | 精度, **ハードウェア互換性** | ニューロモーフィック・ハードウェア対応の本命アーキテクチャ獲得。 |
| **Phase 2C** | **ニューロン・ダイナミクス** ・LIF $\\rightarrow$ **PLIF / GLIF** \[7, 8\] ・時間コーディング (TTFS) \[84\] | 3.1, 3.2 | **ダミー実装 (LIF)** ニューロンの表現力不足。 | 精度, タイムステップ($T$) | より少ないタイムステップでの高精度化。 |
| **Phase 3** | **省エネ化の徹底追求** ・スパース性正則化 12 ・QAT (**QUEST**) 14 ・プルーニング (**SBC**) 15 | 4.1-4.4 | **未実装 (省エネ)** 高スパイクレートと高レイテンシ。 | **SynOps**, **スパイクレート**, タイムステップ($T$) | $T \\le 6$ と高スパース性を両立する、真の省エネモデルの実現。 |
| **Phase 4** | **デプロイメント戦略** ・動的推論 (**SNN Cutoff**) \[16\] ・HW協調設計 (Lava, sPyNNaker) | 6.1, 6.2 | **未実装 (リアルタイム性)** 固定レイテンシとGPU依存。 | **レイテンシ (ms)**, 消費電力(mW) | ロボティクスに必要な低レイテンシとエッジデバイスへの実装。 |
| **Phase 5** | **ベンチマーク基盤** ・SpikingJelly 74 導入 ・SNNBench \[79\] 統合 | 7.1 | **ダミー実装 (評価)** 精度のみの不十分な評価。 | 精度, SynOps, $T$, レイテンシ | 開発サイクルの高速化と客観的な性能評価体制の確立。 |

#### **引用文献**

1. matsushibadenki \- GitHub, 11月 1, 2025にアクセス、 [https://github.com/matsushibadenki](https://github.com/matsushibadenki)  
2. 1月 1, 1970にアクセス、 [https://github.com/matsushibadenki/SNN5](https://github.com/matsushibadenki/SNN5)  
3. Direct training high-performance deep spiking neural networks: a review of theories and methods \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/)  
4. Backpropagation with biologically plausible spatiotemporal adjustment for training deep spiking neural networks \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC9214320/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9214320/)  
5. A Unified Platform to Evaluate STDP Learning Rule and Synapse Model using Pattern Recognition in a Spiking Neural Network \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2506.19377v1](https://arxiv.org/html/2506.19377v1)  
6. Bi-sigmoid spike-timing dependent plasticity learning rule for magnetic tunnel junction-based SNN \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1387339/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1387339/full)  
7. Incorporating Learnable Membrane Time Constant To Enhance Learning of Spiking Neural Networks \- CVF Open Access, 11月 1, 2025にアクセス、 [https://openaccess.thecvf.com/content/ICCV2021/papers/Fang\_Incorporating\_Learnable\_Membrane\_Time\_Constant\_To\_Enhance\_Learning\_of\_Spiking\_ICCV\_2021\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf)  
8. GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2210.13768](https://arxiv.org/pdf/2210.13768)  
9. \[2304.11954\] Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2304.11954](https://arxiv.org/abs/2304.11954)  
10. DTA: Dual Temporal-channel-wise Attention for Spiking Neural ..., 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2503.10052](https://arxiv.org/pdf/2503.10052)  
11. arXiv:2503.02689v1 \[cs.CV\] 4 Mar 2025, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2503.02689](https://arxiv.org/pdf/2503.02689)  
12. arXiv:2409.08290v1 \[cs.NE\] 29 Aug 2024, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2409.08290?](https://arxiv.org/pdf/2409.08290)  
13. Backpropagation With Sparsity Regularization for Spiking Neural Network Learning \- PMC, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC9047717/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9047717/)  
14. QUEST: A Quantized Energy-Aware SNN Training Framework for Multi-State Neuromorphic Devices \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2504.00679v1](https://arxiv.org/html/2504.00679v1)  
15. \[2506.03996\] Spiking Brain Compression: Exploring One-Shot Post-Training Pruning and Quantization for Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2506.03996](https://arxiv.org/abs/2506.03996)  
16. Optimizing event-driven spiking neural network with regularization and cutoff \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11880274/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11880274/)  
17. open-neuromorphic/snnmetrics: Metrics for spiking neural networks based on torchmetrics, 11月 1, 2025にアクセス、 [https://github.com/open-neuromorphic/snnmetrics](https://github.com/open-neuromorphic/snnmetrics)  
18. Reconsidering the energy efficiency of spiking neural networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2409.08290v1](https://arxiv.org/html/2409.08290v1)  
19. Surrogate gradient learning in spiking networks trained on event-based cytometry dataset \- Optica Publishing Group, 11月 1, 2025にアクセス、 [https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-9-16260](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-9-16260)  
20. Gradient Descent for Spiking Neural Networks, 11月 1, 2025にアクセス、 [https://proceedings.neurips.cc/paper/7417-gradient-descent-for-spiking-neural-networks.pdf](https://proceedings.neurips.cc/paper/7417-gradient-descent-for-spiking-neural-networks.pdf)  
21. surrogate gradients – Zenke Lab, 11月 1, 2025にアクセス、 [https://zenkelab.org/tag/surrogate-gradients/](https://zenkelab.org/tag/surrogate-gradients/)  
22. Tutorial 6 \- Surrogate Gradient Descent in a Convolutional SNN \- snnTorch \- Read the Docs, 11月 1, 2025にアクセス、 [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_6.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)  
23. SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/)  
24. Directly Training Temporal Spiking Neural Network with Sparse Surrogate Gradient \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2406.19645v1](https://arxiv.org/html/2406.19645v1)  
25. Comparison of the training loss, training accuracy and test accuracy on ImageNet. \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/figure/Comparison-of-the-training-loss-training-accuracy-and-test-accuracy-on-ImageNet\_fig2\_355202636](https://www.researchgate.net/figure/Comparison-of-the-training-loss-training-accuracy-and-test-accuracy-on-ImageNet_fig2_355202636)  
26. Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2304.11954](https://arxiv.org/pdf/2304.11954)  
27. GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks, 11月 1, 2025にアクセス、 [https://papers.neurips.cc/paper\_files/paper/2022/file/cfa8440d500a6a6867157dfd4eaff66e-Paper-Conference.pdf](https://papers.neurips.cc/paper_files/paper/2022/file/cfa8440d500a6a6867157dfd4eaff66e-Paper-Conference.pdf)  
28. The role of membrane time constant in the training of spiking neural networks, 11月 1, 2025にアクセス、 [https://resolver.tudelft.nl/uuid:135b562c-d077-453c-a5d6-1a707da0659b](https://resolver.tudelft.nl/uuid:135b562c-d077-453c-a5d6-1a707da0659b)  
29. Direct training high-performance deep spiking neural networks: a review of theories and methods \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full)  
30. All in one timestep: Enhancing Sparsity and Energy efficiency in Multi-level Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2510.24637v1](https://arxiv.org/html/2510.24637v1)  
31. Reconsidering the energy efficiency of spiking neural networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2409.08290](https://arxiv.org/pdf/2409.08290)  
32. Improving the Sparse Structure Learning of Spiking Neural Networks from the View of Compression Efficiency \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2502.13572v1](https://arxiv.org/html/2502.13572v1)  
33. Heterogeneous Regularization for Fast Rendering Using Deep Spike Neural Network | Vietnam Journal of Computer Science \- World Scientific Publishing, 11月 1, 2025にアクセス、 [https://www.worldscientific.com/doi/10.1142/S2196888824400049](https://www.worldscientific.com/doi/10.1142/S2196888824400049)  
34. Sparse Autoencoders using L1 Regularization with PyTorch \- DebuggerCafe, 11月 1, 2025にアクセス、 [https://debuggercafe.com/sparse-autoencoders-using-l1-regularization-with-pytorch/](https://debuggercafe.com/sparse-autoencoders-using-l1-regularization-with-pytorch/)  
35. L1/L2 Regularization in PyTorch \- GeeksforGeeks, 11月 1, 2025にアクセス、 [https://www.geeksforgeeks.org/machine-learning/l1l2-regularization-in-pytorch/](https://www.geeksforgeeks.org/machine-learning/l1l2-regularization-in-pytorch/)  
36. Energy-Efficient and Dequantization-Free Q-LLMs: A Spiking Neural Network Approach to Salient Value Mitigation \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2510.19498v1](https://arxiv.org/html/2510.19498v1)  
37. ZeroQAT: Your Quantization-aware Training but Efficient \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2509.00031v1](https://arxiv.org/html/2509.00031v1)  
38. (PDF) QUEST: A Quantized Energy-Aware SNN Training Framework for Multi-State Neuromorphic Devices \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/publication/390405725\_QUEST\_A\_Quantized\_Energy-Aware\_SNN\_Training\_Framework\_for\_Multi-State\_Neuromorphic\_Devices](https://www.researchgate.net/publication/390405725_QUEST_A_Quantized_Energy-Aware_SNN_Training_Framework_for_Multi-State_Neuromorphic_Devices)  
39. \[2504.00679\] QUEST: A Quantized Energy-Aware SNN Training Framework for Multi-State Neuromorphic Devices \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2504.00679](https://arxiv.org/abs/2504.00679)  
40. Spiking Brain Compression: Exploring One-Shot Post-Training Pruning and Quantization for Spiking Neural Networks \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2506.03996](https://arxiv.org/pdf/2506.03996)  
41. Post-Training Second-order Compression for Spiking Neural Networks | OpenReview, 11月 1, 2025にアクセス、 [https://openreview.net/forum?id=vHQ1QJ5TIS](https://openreview.net/forum?id=vHQ1QJ5TIS)  
42. A Little Energy Goes a Long Way: Build an Energy-Efficient, Accurate Spiking Neural Network From Convolutional Neural Network \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.759900/pdf](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.759900/pdf)  
43. arXiv:2412.16219v1 \[cs.CV\] 18 Dec 2024, 11月 1, 2025にアクセス、 [https://www.arxiv.org/pdf/2412.16219](https://www.arxiv.org/pdf/2412.16219)  
44. One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2510.23383v1](https://arxiv.org/html/2510.23383v1)  
45. A Fast and Accurate ANN-SNN Conversion Algorithm with Negative Spikes \- IJCAI, 11月 1, 2025にアクセス、 [https://www.ijcai.org/proceedings/2025/0719.pdf](https://www.ijcai.org/proceedings/2025/0719.pdf)  
46. Efficient ANN-SNN Conversion with Error Compensation Learning \- ICML 2025, 11月 1, 2025にアクセス、 [https://icml.cc/virtual/2025/poster/46208](https://icml.cc/virtual/2025/poster/46208)  
47. Temporal Misalignment in ANN-SNN Conversion and its Mitigation via Probabilistic Spiking Neurons \- ICML 2025, 11月 1, 2025にアクセス、 [https://icml.cc/virtual/2025/poster/45627](https://icml.cc/virtual/2025/poster/45627)  
48. Efficient ANN-SNN Conversion with Error Compensation Learning \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2506.01968v1](https://arxiv.org/html/2506.01968v1)  
49. pspikessm: harnessing probabilistic spiking state space models for long-range depen \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2406.02923](https://arxiv.org/pdf/2406.02923)  
50. Spiking Point Transformer for Point Cloud Classification \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2502.15811v1](https://arxiv.org/html/2502.15811v1)  
51. Quantized Spike-driven Transformer \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2501.13492v3](https://arxiv.org/html/2501.13492v3)  
52. Hybrid Spiking Vision Transformer for Object Detection with Event Cameras \- Semantic Scholar, 11月 1, 2025にアクセス、 [https://www.semanticscholar.org/paper/Hybrid-Spiking-Vision-Transformer-for-Object-with-Xu-Deng/55e3c5367ae78ac1c6482e92c4d82ffddf7e6540](https://www.semanticscholar.org/paper/Hybrid-Spiking-Vision-Transformer-for-Object-with-Xu-Deng/55e3c5367ae78ac1c6482e92c4d82ffddf7e6540)  
53. Hybrid Spiking Vision Transformer for Object Detection with Event Cameras (ICML 2025), 11月 1, 2025にアクセス、 [https://arxiv.org/html/2505.07715v1](https://arxiv.org/html/2505.07715v1)  
54. (PDF) Hybrid Spiking Vision Transformer for Event-Based Object Detection \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/publication/379798451\_Hybrid\_Spiking\_Vision\_Transformer\_for\_Event-Based\_Object\_Detection](https://www.researchgate.net/publication/379798451_Hybrid_Spiking_Vision_Transformer_for_Event-Based_Object_Detection)  
55. Hybrid Spiking Vision Transformer for Object Detection with Event Cameras \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2505.07715](https://arxiv.org/abs/2505.07715)  
56. Hybrid Spiking Vision Transformer for Object Detection with Event Cameras \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/2505.07715](https://arxiv.org/pdf/2505.07715)  
57. Optimizing event-driven spiking neural network with regularization and cutoff \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1522788/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1522788/full)  
58. What Is Neuromorphic Computing? | IBM, 11月 1, 2025にアクセス、 [https://www.ibm.com/think/topics/neuromorphic-computing](https://www.ibm.com/think/topics/neuromorphic-computing)  
59. Neuromorphic Computing for Embodied Intelligence in Autonomous Systems: Current Trends, Challenges, and Future Directions \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2507.18139v1](https://arxiv.org/html/2507.18139v1)  
60. Getting Started with Lava — Lava documentation, 11月 1, 2025にアクセス、 [https://lava-nc.org/getting\_started\_with\_lava.html](https://lava-nc.org/getting_started_with_lava.html)  
61. Lava Software Framework — Lava documentation, 11月 1, 2025にアクセス、 [https://lava-nc.org/](https://lava-nc.org/)  
62. Taking Neuromorphic Computing with Loihi 2 to the Next Level Technology Brief \- Intel, 11月 1, 2025にアクセス、 [https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf](https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf)  
63. Neuromorphic Hardware Frameworks \- Meegle, 11月 1, 2025にアクセス、 [https://www.meegle.com/en\_us/topics/neuromorphic-engineering/neuromorphic-hardware-frameworks](https://www.meegle.com/en_us/topics/neuromorphic-engineering/neuromorphic-hardware-frameworks)  
64. Simple​​Data​​Input​​and​​Output​​with​​Spinnaker​​- Lab​​Manual, 11月 1, 2025にアクセス、 [https://spinnakermanchester.github.io/spynnaker/4.0.0/SimpleIO-LabManual.pdf](https://spinnakermanchester.github.io/spynnaker/4.0.0/SimpleIO-LabManual.pdf)  
65. sPyNNaker: A Software Package for Running PyNN Simulations on SpiNNaker \- PMC \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC6257411/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6257411/)  
66. The Rise of Neuromorphic Computing: How Brain-Inspired AI is Shaping the Future in 2025, 11月 1, 2025にアクセス、 [https://www.ainewshub.org/post/the-rise-of-neuromorphic-computing-how-brain-inspired-ai-is-shaping-the-future-in-2025](https://www.ainewshub.org/post/the-rise-of-neuromorphic-computing-how-brain-inspired-ai-is-shaping-the-future-in-2025)  
67. Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications \- Frontiers, 11月 1, 2025にアクセス、 [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00662/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00662/full)  
68. Comparison of SNN accuracy, latency, and energy consumption (MOps),... \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/figure/Comparison-of-SNN-accuracy-latency-and-energy-consumption-MOps-between-direct\_tbl1\_360892514](https://www.researchgate.net/figure/Comparison-of-SNN-accuracy-latency-and-energy-consumption-MOps-between-direct_tbl1_360892514)  
69. List of Neuromorphic Datasets \- SimonWenkel.com, 11月 1, 2025にアクセス、 [https://www.simonwenkel.com/lists/datasets/list-of-neuromorphic-datasets.html](https://www.simonwenkel.com/lists/datasets/list-of-neuromorphic-datasets.html)  
70. Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades \- NIH, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC4644806/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4644806/)  
71. Neuromorphic Datasets Processing — spikingjelly alpha 文档, 11月 1, 2025にアクセス、 [https://spikingjelly.readthedocs.io/zh\_CN/0.0.0.0.8/clock\_driven\_en/13\_neuromorphic\_datasets.html](https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.8/clock_driven_en/13_neuromorphic_datasets.html)  
72. N-Caltech101 \- Garrick Orchard, 11月 1, 2025にアクセス、 [https://www.garrickorchard.com/datasets/n-caltech101](https://www.garrickorchard.com/datasets/n-caltech101)  
73. Classifying neuromorphic data using a deep learning framework for image classification \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/pdf/1807.00578](https://arxiv.org/pdf/1807.00578)  
74. SNNtrainer3D: Training Spiking Neural Networks Using a User-Friendly Application with 3D Architecture Visualization Capabilities \- MDPI, 11月 1, 2025にアクセス、 [https://www.mdpi.com/2076-3417/14/13/5752](https://www.mdpi.com/2076-3417/14/13/5752)  
75. A Unified Evaluation Framework for Spiking Neural Network Hardware Accelerators Based on Emerging Non-Volatile Memory Devices \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/html/2402.19139v1](https://arxiv.org/html/2402.19139v1)  
76. BenchCouncil/SNNBench: Spiking Neural Network Benchmarking \- GitHub, 11月 1, 2025にアクセス、 [https://github.com/BenchCouncil/SNNBench](https://github.com/BenchCouncil/SNNBench)  
77. SNNBench: End-to-end AI-oriented spiking neural network benchmarking \- ResearchGate, 11月 1, 2025にアクセス、 [https://www.researchgate.net/publication/370749159\_SNNBench\_End-to-end\_AI-oriented\_spiking\_neural\_network\_benchmarking](https://www.researchgate.net/publication/370749159_SNNBench_End-to-end_AI-oriented_spiking_neural_network_benchmarking)  
78. \[2409.02111\] Toward Large-scale Spiking Neural Networks: A Comprehensive Survey and Future Directions \- arXiv, 11月 1, 2025にアクセス、 [https://arxiv.org/abs/2409.02111](https://arxiv.org/abs/2409.02111)  
79. Neuromorphic Computing with Large Scale Spiking Neural Networks\[v1\] | Preprints.org, 11月 1, 2025にアクセス、 [https://www.preprints.org/manuscript/202503.1505/v1](https://www.preprints.org/manuscript/202503.1505/v1)  
80. Spiking Neural Networks for Multimodal Neuroimaging: A Comprehensive Review of Current Trends and the NeuCube Brain-Inspired Architecture \- PubMed Central, 11月 1, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC12189790/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12189790/)
