# Example Method Decisions

These choices describe the synthetic demonstration only. They are not operating limits, recommendations, or conclusions about a real system.

| Area | Example choice | Boundary |
|---|---|---|
| Input schema | Anonymous timestamped channels | No source-system identifiers are used |
| Data split | Time-ordered training and evaluation windows | Prevents future observations entering past features |
| Missing values | Short gaps handled within contiguous segments | Long gaps remain missing |
| Features | Rolling summaries, variation, slope, and change | Uses historical observations only |
| Baseline | A fixed synthetic reference interval | Not a verified healthy period |
| Evaluation | Lead-time and false-alarm summaries | Illustrative on generated data only |
| Publication | Generic channels and synthetic values | No operational measurements or industry claims |
