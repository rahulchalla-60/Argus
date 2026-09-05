# PaySim Dataset Schema & Summary

| Field | Type | Role | Notes |
|---|---|---|---|
| `step` | int | Timestamp | 1 step = 1 hour (1 to 743, ~31 days) |
| `type` | str | Txn Type | `PAYMENT`, `TRANSFER`, `CASH_OUT`, `CASH_IN`, `DEBIT` |
| `amount` | float | Amount | Min: 0.0, Median: 74,871.94, Max: 92,445,516.64 |
| `nameOrig` | str | Sender ID | 6,353,307 unique accounts |
| `nameDest` | str | Receiver ID | 2,722,362 unique accounts |
| `isFraud` | int | Label | 8,213 cases (0.129% prevalence) |
| `isFlaggedFraud` | int | Rule Baseline | System heuristic flag |

### Quality Check
- Missing values: 0 across all columns
- Duplicates: 0
- Fraud occurs primarily in `TRANSFER` and `CASH_OUT` flows (classic mule pattern: Transfer In → Cash Out).
