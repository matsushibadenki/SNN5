```mermaid
flowchart TB

  %% --- 感覚入力層 ---
  subgraph SensoryLayer [感覚入力層]
    SENSORY["感覚入力\n(光・音・化学・機械)"]
    RECEPTOR["受容器応答\n(アナログ変換)"]
    ENCODING["スパイク符号化\nRate / Temporal / Latency"]
    SENSORY --> RECEPTOR --> ENCODING
  end

  %% --- 単一ニューロン層（LIFモデル） ---
  subgraph Neuron [単一ニューロン（LIFモデル基準）]
    direction TB
    DENDRITES["樹状突起\n多数のシナプス入力\n(加重和・時定数 τ)"]
    SOMA["細胞体\n統合点"]
    MEMBRANE["膜電位 Vm(t)\n静止: -70 mV\n漏れ統合"]
    THRESH["閾値判定\nVth ≈ -55 mV"]
    SPIKE["活動電位\n(スパイク発火)"]
    REFRAC["不応期\n(2-5 ms)\nVm ← Vreset (-65 mV)"]
    AXON["軸索伝導\n(有髄化 / 伝導速度)"]
    TERMINAL["軸索末端 (Presynaptic)"]

    DENDRITES --> SOMA
    SOMA --> MEMBRANE
    MEMBRANE -->|Vm ≥ Vth| THRESH
    THRESH --> SPIKE
    SPIKE --> REFRAC
    REFRAC -->|Vm ← Vreset| MEMBRANE
    SPIKE --> AXON
    AXON --> TERMINAL
  end

  %% --- シナプス層（興奮性・抑制性） ---
  subgraph Synapse [シナプス（興奮性 / 抑制性）]
    direction LR
    DELAY["シナプス遅延\n(0.5–2 ms)"]
    NT_E["興奮性伝達物質\nGlutamate\n(AMPA / NMDA)"]
    NT_I["抑制性伝達物質\nGABA\n(Cl⁻流入)"]
    POSTSYN["シナプス後電位\n(EPSP / IPSP)\n→ ΔVm(t)"]

    TERMINAL --> DELAY
    DELAY --> NT_E
    DELAY --> NT_I
    NT_E --> POSTSYN
    NT_I --> POSTSYN
    POSTSYN -->|ΔVm| MEMBRANE
  end

  %% --- 可塑性・学習機構 ---
  subgraph Plasticity [可塑性・学習機構]
    STDP["STDP\n(Spike-Timing-Dependent Plasticity)"]
    LTP["LTP\n(長期増強)"]
    LTD["LTD\n(長期抑圧)"]
    WEIGHT["シナプス重み w\n(動的更新)"]
    NEUROMOD["神経修飾\n(Dopamine / 報酬信号)"]

    SPIKE --> STDP
    STDP -->|Δt > 0| LTP
    STDP -->|Δt < 0| LTD
    LTP --> WEIGHT
    LTD --> WEIGHT
    NEUROMOD --> STDP
    WEIGHT -->|伝達効率変化| NT_E
  end

  %% --- ネットワーク構造 ---
  subgraph Network [ネットワーク構造]
    direction TB
    FF["フィードフォワード結合\n(層間伝播)"]
    REC["リカレント結合\n(再帰・短期記憶)"]
    EI["E/Iバランス\n(興奮 ≒ 80% / 抑制 ≒ 20%)"]
    OSC["同期振動\nGamma (30–100 Hz)\nBeta (12–30 Hz)\nAlpha (8–12 Hz)\nTheta (4–8 Hz)"]
    LONG["長距離結合\n(皮質間・遅延含む)"]

    ENCODING --> FF
    FF --> REC
    REC --> FF
    POSTSYN --> EI
    EI --> OSC
    FF --> LONG
    REC --> LONG
  end

  %% --- 情報表現層 ---
  subgraph Info [情報表現・符号化]
    RATE["Rate Code\n(発火頻度符号化)"]
    TEMP["Temporal Code\n(スパイクタイミング / 位相)"]
    POPULATION["Population Code\n(集団符号化)"]
    SPARSE["Sparse Coding\n(省エネ・高識別性)"]

    SPIKE --> RATE
    SPIKE --> TEMP
    RATE --> POPULATION
    TEMP --> POPULATION
    POPULATION --> SPARSE
  end

  %% --- 出力・復号化層 ---
  subgraph Output [出力・復号化層]
    DECODE["スパイク復号化\n(積分 / 投票 / フィルタ)"]
    PERCEPT["知覚・判断"]
    MOTOR["運動出力"]
    FEEDBACK["フィードバック\n(誤差信号 / 感覚再入力)"]

    LONG --> DECODE
    DECODE --> PERCEPT
    DECODE --> MOTOR
    PERCEPT --> FEEDBACK
    FEEDBACK --> NEUROMOD
  end

  %% --- 生物学的制約 ---
  subgraph Meta [生物学的制約（SNNでは通常省略）]
    NOISE["ノイズ\n(熱雑音・確率的放出)"]
    ENERGY["代謝コスト\n(ATP消費)"]
    GLIA["グリア細胞\n(恒常性維持・ATP供給・イオン調整)"]

    NOISE -.->|膜電位変動| MEMBRANE
    ENERGY -.->|発火制約| SPIKE
    GLIA -.->|サポート| POSTSYN
    GLIA -.->|ATP供給| ENERGY
  end

  %% --- フィードバックループ ---
  FEEDBACK -.->|学習信号| STDP
  MOTOR -.->|感覚フィードバック| RECEPTOR

  %% --- スタイル定義 ---
  classDef core fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
  classDef learning fill:#fff3e0,stroke:#f57c00,stroke-width:2px
  classDef optional fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 5 5

  class MEMBRANE,SPIKE,POSTSYN,WEIGHT core
  class STDP,LTP,LTD,NEUROMOD learning
  class NOISE,ENERGY,GLIA optional
  ```
  
  
  # SNN構造への導入アイデア

このドキュメントでは、Mermaid図の各構造要素をどのように**Spiking Neural Network (SNN)** に採用できるかを、実装レベルで整理します。

---

## 1. 生物構造とSNNモデルの対応表

| 図の構成ブロック | SNNモデルでの実装対応 | 実装ライブラリ例 / 備考 |
|------------------|------------------------|-------------------------|
| 感覚入力層 (SENSORY / ENCODING) | スパイク符号化器（`RateCodeEncoder`, `TemporalEncoder`） | `snnTorch`, `BindsNET`, `Nengo` など |
| 単一ニューロン層 (MEMBRANE / THRESH / REFRACT) | LIFニューロン or AdExモデル | `snn.LIFCell()` またはカスタムLIF式 |
| シナプス (NT_E / NT_I / POSTSYN) | 重み付きスパイク伝達（E/Iバランス付き） | Excitatory:Inhibitory = 80:20 |
| 可塑性 (STDP / LTP / LTD / WEIGHT) | Spike-Timing-Dependent Plasticity (STDP) ルール | \( \Delta w = A^+ e^{-\Delta t/\tau^+} - A^- e^{\Delta t/\tau^-} \) |
| ネットワーク構造 (FF / REC / EI / OSC) | フィードフォワード + リカレント構造 | CNN + Recurrent SNN / Liquid State Machine |
| 情報表現層 (RATE / TEMP / POPULATION) | 出力スパイク統計解析 | PSTH, ISI統計, 同期解析 |
| 出力層 (DECODE / MOTOR) | スパイク積分または投票復号化 | `SpikeIntegrator`, `RateDecoder` |
| フィードバック / 報酬信号 (NEUROMOD) | 報酬STDP (Dopamine-like RPE) | 強化学習型SNN |
| Meta層 (NOISE / ENERGY / GLIA) | ノイズ注入・エネルギー正則化 | 生物的安定性の模倣 |

---

## 2. 構築手順（フェーズ別）

### 🔹 Phase 1：符号化層（Encoding）
感覚入力をスパイク列に変換。

```python
# Poisson-like rate encoding
rate = x / x.max()
spike_train = torch.rand_like(rate) < rate
```

Temporal符号化では「入力が大きいほど早くスパイク」を採用。

---

### 🔹 Phase 2：ニューロンモデル（Neuron）
LIFモデルで膜電位動態を実装。

\[
\tau_m \frac{dV_m}{dt} = - (V_m - V_{rest}) + R_m I_{syn}
\]

発火条件：
\[
V_m \ge V_{th} \Rightarrow V_m \leftarrow V_{reset}
\]

```python
import snntorch as snn
lif = snn.Leaky(beta=0.9, threshold=1.0, reset_mechanism="subtract")
```

---

### 🔹 Phase 3：シナプスと学習（Synapse & Plasticity）

STDPに基づき、スパイク時刻差で重み更新。

```python
def stdp_update(pre_t, post_t, w, A_plus, A_minus, tau_plus, tau_minus):
    dt = post_t - pre_t
    dw = torch.where(dt > 0,
                     A_plus * torch.exp(-dt / tau_plus),
                     -A_minus * torch.exp(dt / tau_minus))
    w += dw
    w.clamp_(0, 1)
```

NEUROMODノードはドーパミン報酬信号としてSTDPに乗算可能。

\[ \Delta w = \delta(t) \cdot STDP(\Delta t) \]

---

### 🔹 Phase 4：ネットワーク構造（Network）

| 要素 | SNN構成例 |
|------|-------------|
| FF | 畳み込み型SNN (`snn.Conv2d`) |
| REC | リカレントスパイクネット (`snn.RLeaky`) |
| EI | E/I分離層（重み行列をExcit/Inhibに分離） |
| OSC | 同期振動を解析対象に（FFT, コヒーレンス） |

---

### 🔹 Phase 5：出力復号（Decoding）

```python
out = spike_rec.sum(dim=0)
pred = out.argmax(dim=1)
```

> `DECODE → PERCEPT / MOTOR` に対応。

---

### 🔹 Phase 6：フィードバック学習

誤差信号を報酬STDPに反映し、自己教師的適応を実現。

\[ \Delta w = r(t) \cdot STDP(\Delta t) \]

---

## 3. 推奨ディレクトリ構成

```
SNN_Model/
│
├── encoders/
│   ├── rate_encoder.py
│   └── temporal_encoder.py
│
├── neurons/
│   ├── lif_cell.py
│   └── adex_cell.py
│
├── synapses/
│   ├── stdp.py
│   ├── dopamine_mod.py
│   └── ei_balance.py
│
├── network/
│   ├── feedforward.py
│   ├── recurrent.py
│   └── oscillation_monitor.py
│
├── decoders/
│   └── rate_decoder.py
│
└── meta/
    ├── noise_injector.py
    ├── energy_constraint.py
    └── glia_support.py
```

> 各ファイルがMermaid図のノードに1対1で対応。

---

## 4. 高度化アイデア

1. **時間依存ネットワーク解析**：FFTで同期リズム（Gamma/Theta）を観測。
2. **多モーダル入力**：視覚＝Rate符号、聴覚＝Temporal符号。
3. **Homeostatic Plasticity**：発火率を一定に保つ可塑性ループ。
4. **エネルギー制約学習**：
   \[ L = L_{task} + \lambda \sum_i r_i^2 \]

---

## 5. 導入方針まとめ

| フェーズ | 実装方向 | 効果 |
|-----------|------------|------|
| 感覚入力 | Rate/Temporal符号化器 | 生理的入力表現 |
| ニューロン | LIFセル | 時系列スパイク処理 |
| シナプス | STDP + Dopamine報酬 | 自己組織的学習 |
| ネットワーク | FF + REC + E/I構造 | 安定した動的表現 |
| 出力 | Spike Integration Decoder | 認知・運動表現化 |
| Meta層 | ノイズ・エネルギー制約 | 生物的頑健性再現 |

---

この構成を採用すれば、Mermaid図の概念構造をそのまま**可視的・機能的SNNアーキテクチャ**として実装可能になります。

