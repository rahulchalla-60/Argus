# Argus — Real-Time Mule Account Detection

## Master Task Checklist

> Tick `[x]` as you complete each item. `[/]` = in progress.

---

## Phase 0 — Project Setup

- [ ] Create project folder structure
  - [ ] `/data` — raw datasets + synthetic generators
  - [ ] `/src` — all engine modules
  - [ ] `/tests` — unit tests
  - [ ] `/models` — saved ML models
  - [ ] `/notebooks` — exploration notebooks (optional)
- [ ] Initialize `requirements.txt` with core dependencies
  - [ ] pandas
  - [ ] networkx
  - [ ] xgboost / lightgbm
  - [ ] scikit-learn
  - [ ] plotly
  - [ ] streamlit
- [ ] Create `README.md` with problem statement placeholder
- [ ] Verify Python environment works (python --version, install deps)

---

## Phase 1 — Data + Graph Foundation (Week 1)

### 1.1 Dataset Acquisition
- [ ] Download PaySim dataset OR IBM AML dataset
- [ ] Inspect raw data — identify columns: sender, receiver, amount, timestamp
- [ ] Document data schema (column names, types, ranges)
- [ ] Check for missing values, duplicates, anomalies
- [ ] Write a data summary (row count, unique accounts, date range, amount distribution)

### 1.2 Transaction Stream Simulator
- [ ] Create `stream_simulator.py`
- [ ] Build generator function that yields one transaction at a time
- [ ] Add configurable speed (real-time, 2x, 10x, max)
- [ ] Add `asyncio.Queue`-based version for async consumers
- [ ] Test: stream 100 transactions, print each one
- [ ] Add timestamp ordering — ensure chronological order

### 1.3 Graph Engine
- [x] Create `graph_engine.py`
- [x] Initialize `networkx.MultiDiGraph()`
- [x] Implement `add_transaction(sender, receiver, amount, timestamp)`
  - [x] Add sender node if not exists
  - [x] Add receiver node if not exists
  - [x] Add directed edge with `{amount, timestamp}` attributes
- [x] Implement `get_node_count()` and `get_edge_count()`
- [x] Implement `get_neighbors(account_id)` — returns in/out neighbors
- [x] Implement `get_transactions(account_id)` — returns all edges for an account
- [x] Test: add 10 transactions, verify graph has correct nodes/edges
- [x] Implement `get_subgraph(account_id, k_hops)` — extract local neighborhood

### 1.4 Live Graph Visualization (Proof of Concept)
- [x] Visualize small graph (5-10 nodes) with matplotlib or plotly
- [x] Show A -> B, A -> C, B -> D structure
- [x] Label edges with amounts
- [x] Verify graph updates when new transaction is added

---

## Phase 2 — Temporal AML Features (Week 2)

### 2.1 Feature Engine Setup
- [x] Create `feature_engine.py`
- [x] Define `AccountFeatures` dataclass/dict structure
- [x] Create per-account feature store (dict of account_id -> features)

### 2.2 Forwarding Delay
- [x] For each account, track `last_received_time` and `last_sent_time`
- [x] Compute: `forwarding_delay = last_sent_time - last_received_time`
- [x] Handle edge cases: account only sends, account only receives
- [x] Test: receive at 10:00:00, send at 10:00:15 -> delay = 15 sec

### 2.3 Pass-Through Ratio
- [x] Track `total_received` and `total_forwarded` per account (rolling window)
- [x] Compute: `pass_through_ratio = total_forwarded / total_received`
- [x] Handle divide-by-zero (account never received)
- [x] Test: received=10000, forwarded=9800 -> ratio=0.98

### 2.4 Fan-In / Fan-Out
- [x] Fan-in: count unique senders to this account (in last N hours)
- [x] Fan-out: count unique receivers from this account (in last N hours)
- [x] Use time-windowed counting (configurable window: 1h, 24h)
- [x] Test: 5 different accounts send to B -> fan_in(B) = 5

### 2.5 Velocity
- [x] Count transactions per minute for each account
- [x] Use sliding window (e.g., last 60 seconds)
- [x] Test: 10 transactions in 1 minute -> velocity = 10

### 2.6 Distinct Counterparties
- [x] Track `unique_senders_last_hour` per account
- [x] Track `unique_receivers_last_hour` per account
- [x] Use time-windowed sets
- [x] Test: account B receives from A, C, D in last hour -> unique_senders = 3

### 2.7 Feature Computation Integration
- [x] Hook feature engine into graph engine — recompute on every new transaction
- [x] Implement `update_features(account_id, transaction)` — incremental update
- [x] Implement `get_features(account_id)` — returns all features as dict
- [x] Test: stream 50 transactions, print features for busiest account
- [x] Measure feature computation time per transaction (target: < 5ms)

---

## Phase 3 — Fraud Scoring Engine (Week 3)

### 3.1 Rule Engine
- [x] Create `rule_engine.py`
- [x] Implement configurable rules:
  - [x] `pass_through_ratio > 0.95` -> HIGH
  - [x] `forwarding_delay < 30 sec` -> MEDIUM
  - [x] `fan_in > 20` -> MEDIUM
  - [x] `velocity > 5 txns/min` -> MEDIUM
- [x] Combine rules: if 2+ MEDIUM triggers -> HIGH
- [x] Return rule-based risk label: LOW / MEDIUM / HIGH
- [x] Test: account with ratio=0.98, delay=8s -> HIGH

### 3.2 Training Data Preparation
- [ ] Label accounts as mule / not-mule (use dataset labels or synthetic labels)
- [ ] For each labeled account, compute all 6 features
- [ ] Create feature matrix X and label vector y
- [ ] Train/test split (80/20)
- [ ] Save training data as CSV for reproducibility

### 3.3 ML Model Training
- [ ] Create `risk_engine.py`
- [ ] Train XGBoost (or LightGBM or RandomForest) classifier
- [ ] Input: forwarding_delay, velocity, fan_in, fan_out, pass_through_ratio, counterparties
- [ ] Output: risk score 0.0 -> 1.0 (use `predict_proba`)
- [ ] Evaluate: accuracy, precision, recall, F1, AUC-ROC
- [ ] Print confusion matrix
- [ ] Save trained model to `/models/risk_model.pkl`

### 3.4 Real-Time Scoring
- [ ] Load saved model in risk_engine
- [ ] Implement `score_account(account_id)` -> returns float 0.0-1.0
- [ ] Hook into stream: every N transactions, re-score affected accounts
- [ ] Test: score account with known mule features -> expect > 0.8
- [ ] Measure scoring latency (target: < 10ms per account)

### 3.5 Alert Generation
- [ ] Define alert threshold (e.g., risk > 0.8)
- [ ] Create alert object: `{account_id, risk_score, timestamp, reasons}`
- [ ] Store alerts in a list/queue
- [ ] Test: inject mule-like transactions -> verify alert is generated

---

## Phase 4 — Risk Propagation (Week 4)

### 4.1 Basic Propagation
- [ ] Implement `propagate_risk(graph, flagged_accounts)`
- [ ] Algorithm: BFS from flagged nodes with decay factor
  - [ ] A = 1.0, B (1 hop) = 0.5, C (2 hops) = 0.25
- [ ] Configurable: decay_factor (default 0.5), max_hops (default 3)
- [ ] Test: flag account A, verify B gets 0.5, C gets 0.25

### 4.2 Neighbor Risk as Feature
- [ ] Compute `neighbor_risk` = max risk among 1-hop neighbors
- [ ] Add `neighbor_risk` to feature vector
- [ ] Re-train model with this new feature
- [ ] Compare performance: with vs without neighbor_risk

### 4.3 Propagation Visualization
- [ ] Color nodes by propagated risk: red (>0.8), orange (0.5-0.8), yellow (0.3-0.5), green (<0.3)
- [ ] Test: visualize a chain A -> B -> C with decaying risk colors

---

## Phase 5 — Device/IP Layer (Week 5)

### 5.1 Synthetic Device/IP Data
- [ ] Generate synthetic `device_id` for each account (some share devices)
- [ ] Generate synthetic `ip_id` for each account (some share IPs)
- [ ] Create realistic clusters: 3-5 accounts per shared device
- [ ] Save as supplementary data file

### 5.2 Heterogeneous Graph
- [ ] Add `Device` and `IP` node types to graph
- [ ] Add edges: Account -> Device (uses), Account -> IP (connects_from)
- [ ] Implement `get_shared_device_accounts(device_id)`
- [ ] Implement `get_shared_ip_accounts(ip_id)`
- [ ] Test: accounts A, B, C all on device X -> returns [A, B, C]

### 5.3 Device/IP Features
- [ ] Compute `shared_device_count` — how many accounts share my device
- [ ] Compute `shared_ip_count` — how many accounts share my IP
- [ ] Compute `risky_device_neighbors` — how many flagged accounts share my device
- [ ] Add to feature vector
- [ ] Re-train model with device/IP features

### 5.4 Device/IP Visualization
- [ ] Visualize bipartite graph: accounts <-> devices
- [ ] Show: 3 accounts connected to 1 device
- [ ] Color by risk

---

## Phase 6 — Explainability (Week 6)

### 6.1 Explainability Engine
- [ ] Create `explainability.py`
- [ ] For each alert, generate human-readable reasons list
- [ ] Use SHAP values or feature importance to rank contributing factors
- [ ] Format output as JSON with risk score + reasons array
- [ ] Test: generate explanation for a known mule account

### 6.2 Subgraph Extraction for Explanation
- [ ] Extract suspicious subgraph around flagged account (2-hop)
- [ ] Identify the flow: Payer -> Mule -> Cashout
- [ ] Return subgraph data for frontend visualization

### 6.3 Cluster Detection
- [ ] Implement community detection (Louvain algorithm via networkx)
- [ ] Identify clusters of tightly connected accounts
- [ ] Compute cluster-level stats: size, total volume, avg risk, shared devices
- [ ] Label clusters: "Mule Ring", "Betting Ring", etc. (heuristics)
- [ ] Test: inject a synthetic mule ring, verify it is detected as a cluster

---

## Phase 7 — Frontend / Dashboard (After Backend)

### 7.1 Streamlit App Setup
- [ ] Create `dashboard.py` (Streamlit entry point)
- [ ] Set up multi-page navigation with sidebar
- [ ] Configure page layout, theme, title ("Argus — AML Detection Platform")

### 7.2 Screen 1 — Executive Dashboard
- [ ] KPI cards: Transactions, Accounts, High Risk, Alerts, Latency, Graph Size
- [ ] Chart: Transactions per minute (line chart, Plotly)
- [ ] Chart: Risk score distribution (histogram)
- [ ] Chart: Alerts over time (bar chart)
- [ ] Auto-refresh every N seconds

### 7.3 Screen 2 — Live Transaction Stream
- [ ] Scrolling table: Time | From | To | Amount | Risk
- [ ] Color-code rows by risk (green/yellow/red)
- [ ] Pause/Resume stream button
- [ ] Search by account ID
- [ ] Filter by amount range
- [ ] Filter by risk level

### 7.4 Screen 3 — Alerts Center
- [ ] Table: Account | Risk Score | Reason | Time
- [ ] Filter: Risk > threshold slider
- [ ] Filter: Time range picker
- [ ] Filter: Alert type dropdown
- [ ] Click row -> navigate to Investigation Workbench

### 7.5 Screen 4 — Investigation Workbench
- [ ] Input: account ID (from alerts or manual entry)
- [ ] Display: Risk score (big number, color-coded)
- [ ] Display: Feature values table
- [ ] Display: Transaction history table for this account
- [ ] Display: Connected accounts list
- [ ] Display: Risk reasons (from explainability engine)
- [ ] Links to: Graph Explorer, Explainability Center

### 7.6 Screen 5 — Graph Explorer
- [ ] View A — Local Neighborhood: 1-hop neighbors
- [ ] View B — k-Hop Expansion: click to expand, configurable k (1-4)
- [ ] View C — Time Window: slider for time period (5 min, 1h, 24h, 7d)
- [ ] View D — Risk Filter: slider to hide nodes below threshold
- [ ] View E — Cluster View: detected communities as grouped nodes
- [ ] View F — Device/IP View: bipartite account <-> device/IP graph
- [ ] Interactive: click nodes for details, hover for tooltips
- [ ] Color nodes by risk level
- [ ] Size nodes by transaction volume

### 7.7 Screen 6 — Device Intelligence Center
- [ ] Table: Top Shared Devices (device_id | account_count)
- [ ] Table: Top Shared IPs (ip_id | account_count)
- [ ] Suspicious device rings visualization
- [ ] Click device -> show connected accounts

### 7.8 Screen 7 — Explainability Center
- [ ] Input: account ID
- [ ] Display: Risk score
- [ ] Display: Contributing factors with importance bars (waterfall chart)
- [ ] Display: Feature values vs population average
- [ ] Display: Top reasons in plain English

### 7.9 Screen 8 — Cluster Analysis
- [ ] Table: Cluster ID | Size | Avg Risk | Shared Devices | Total Volume
- [ ] Click cluster -> show member accounts
- [ ] Visualize cluster as subgraph
- [ ] Filter: high-risk clusters only

### 7.10 Screen 9 — Transaction Replay
- [ ] Date range picker (start, end)
- [ ] Play / Pause / Speed controls (1x, 2x, 5x, 10x)
- [ ] Graph evolves as transactions replay
- [ ] Show transaction counter and current timestamp
- [ ] Highlight newly added edges/nodes

### 7.11 Screen 10 — System Metrics
- [ ] Throughput: transactions/sec (live counter)
- [ ] Latency breakdown: graph update, feature computation, scoring
- [ ] Graph size: node count, edge count (live)
- [ ] Memory usage
- [ ] Charts: latency over time, throughput over time

---

## Phase 8 — Polish & Interview Prep

### 8.1 README
- [ ] Problem statement
- [ ] Architecture diagram (Mermaid or image)
- [ ] AML concepts explained (mule accounts, layering, structuring)
- [ ] Feature explanations with examples
- [ ] Latency measurements
- [ ] Example mule ring detection screenshot
- [ ] Screenshots of all 10 screens
- [ ] How to run instructions

### 8.2 Demo Script
- [ ] Write 2-minute interview demo flow
- [ ] Test end-to-end: transaction arrives -> alert generated
- [ ] Prepare talking points for each screen
- [ ] Prepare answers for expected questions:
  - [ ] "Why not deep learning?"
  - [ ] "How would you scale this?"
  - [ ] "What is the false positive rate?"
  - [ ] "How does risk propagation work?"
  - [ ] "What graph algorithms did you use?"

### 8.3 Final Testing
- [ ] End-to-end test: stream 1000 transactions, verify alerts
- [ ] Performance test: measure latency at 10k, 100k, 1M transactions
- [ ] Edge case test: empty graph, single node, disconnected components
- [ ] Verify all 10 screens render correctly
