```mermaid
flowchart TB
  %% 感覚入力からネットワーク出力まで

  subgraph SensoryLayer [感覚入力層]
    SENSORY["感覚入力\n(光・音・化学・機械)"]
    RECEPTOR["受容器応答\n(アナログ変換)"]
    ENCODING["スパイク符号化\nRate/Temporal/Latency"]
    SENSORY --> RECEPTOR --> ENCODING
  end

  subgraph Neuron [単一ニューロン（LIFモデル基準）]
    direction TB
    DENDRITES["樹状突起\n多数のシナプス入力\n(加重和・時定数 τ)"]
    SOMA["細胞体\n統合点"]
    MEMBRANE["膜電位 Vm(t)\n静止: -70 mV\n漏れ統合"]
    THRESH["閾値判定\nVth ≈ -55 mV"]
    SPIKE["活動電位\n(スパイク発火)"]
    REFRAC["不応期\n(2-5 ms)"]
    AXON["軸索伝導\n(有髄化 / 伝導速度)"]
    TERMINAL["軸索末端"]
    
    DENDRITES --> SOMA
    SOMA --> MEMBRANE
    MEMBRANE -->|Vm ≥ Vth| THRESH
    THRESH --> SPIKE
    SPIKE --> REFRAC
    REFRAC --> MEMBRANE
    SPIKE --> AXON
    AXON --> TERMINAL
  end

  subgraph Synapse [シナプス（興奮性 / 抑制性）]
    direction LR
    NT_E["興奮性\nGlutamate\n(AMPA / NMDA)"]
    NT_I["抑制性\nGABA\n(Cl^- 流入)"]
    DELAY["シナプス遅延\n(0.5-2 ms)"]
    POSTSYN["シナプス後電位\n(EPSP / IPSP)"]
    
    TERMINAL --> DELAY
    DELAY --> NT_E
    DELAY --> NT_I
    NT_E --> POSTSYN
    NT_I --> POSTSYN
    POSTSYN --> DENDRITES
  end

  subgraph Plasticity [可塑性・学習機構]
    STDP["STDP\n(Spike-Timing-Dependent\nPlasticity)"]
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
    WEIGHT --> POSTSYN
  end

  subgraph Network [ネットワーク構造]
    direction TB
    FF["フィードフォワード結合\n(層間伝播)"]
    REC["リカレント結合\n(再帰・短期記憶)"]
    EI["E/Iバランス\n(興奮 ≒ 80% / 抑制 ≒ 20%)"]
    OSC["同期振動\n(Gamma 30-100 Hz\nTheta 4-8 Hz)"]
    LONG["長距離結合\n(大きな遅延含む)"]
    
    ENCODING --> FF
    FF --> REC
    REC --> FF
    POSTSYN --> EI
    EI --> OSC
    FF --> LONG
    REC --> LONG
  end

  subgraph Info [情報表現・符号化]
    RATE["Rate Code\n(発火頻度)"]
    TEMP["Temporal Code\n(スパイクタイミング / Latency)"]
    POPULATION["Population Code\n(集団符号化)"]
    SPARSE["Sparse Coding\n(省エネ・高識別性)"]
    
    SPIKE --> RATE
    SPIKE --> TEMP
    RATE --> POPULATION
    TEMP --> POPULATION
    POPULATION --> SPARSE
  end

  subgraph Output [出力・復号化層]
    DECODE["スパイク復号化\n(積分 / 投票)"]
    PERCEPT["知覚・判断"]
    MOTOR["運動出力"]
    FEEDBACK["フィードバック\n(誤差信号)"]
    
    LONG --> DECODE
    DECODE --> PERCEPT
    DECODE --> MOTOR
    PERCEPT --> FEEDBACK
    FEEDBACK --> NEUROMOD
  end

  subgraph Meta [生物学的制約（SNNでは通常省略）]
    NOISE["ノイズ\n(熱雑音・確率的放出)"]
    ENERGY["代謝コスト\n(ATP消費)"]
    GLIA["グリア細胞\n(恒常性維持)"]
    
    NOISE -.->|影響| MEMBRANE
    ENERGY -.->|制約| SPIKE
    GLIA -.->|サポート| POSTSYN
  end

  %% フィードバックループ
  FEEDBACK -.->|学習信号| STDP
  MOTOR -.->|感覚フィードバック| RECEPTOR

  %% スタイル
  classDef core fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
  classDef learning fill:#fff3e0,stroke:#f57c00,stroke-width:2px
  classDef optional fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 5 5

  class MEMBRANE,SPIKE,POSTSYN,WEIGHT core
  class STDP,LTP,LTD learning
  class NOISE,ENERGY,GLIA optional
  ```