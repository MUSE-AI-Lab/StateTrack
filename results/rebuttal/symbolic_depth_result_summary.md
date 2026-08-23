# Gemini-Excluded Symbolic-Depth Audit

- Experiment ID: `p1_rebuttal_20260709_symbolic_depth`
- Valid outputs: 216
- Incremental logged cost: $1.8002
- Gemini excluded: True

## Aggregate by depth

| Depth | n | Constraint | Recall | SIC+ | SIC- | Memory failure |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 72 | 94.4 | 87.5 | 4.2 | 11.1 | 1.4 |
| 30 | 72 | 95.8 | 90.3 | 4.2 | 9.7 | 0.0 |
| 50 | 72 | 90.3 | 97.2 | 9.7 | 2.8 | 0.0 |

## Model-level depth trend

| Model | n | C@12 | C@30 | C@50 | Δ 12→50 | SIC+@50 |
|---|---:|---:|---:|---:|---:|---:|
| gpt-5.4 | 36 | 91.7 | 100.0 | 91.7 | 0.0 | 8.3 |
| gpt-5.4-mini | 36 | 91.7 | 83.3 | 91.7 | 0.0 | 8.3 |
| claude-sonnet-4-6 | 36 | 91.7 | 100.0 | 91.7 | 0.0 | 8.3 |
| deepseek-v4-pro | 36 | 100.0 | 91.7 | 83.3 | -16.7 | 16.7 |
| deepseek-v4-flash | 36 | 91.7 | 100.0 | 83.3 | -8.4 | 16.7 |
| qwen3.6-plus | 36 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 |

## Scope

This targeted analysis isolates symbolic constraints at longer dialogue depths. It should be cited as Gemini-excluded evidence on symbolic constraints at deeper turns.
