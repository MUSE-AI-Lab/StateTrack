# Gemini-Excluded Symbolic Execution-Bottleneck Audit

Post-hoc descriptive mechanism audit over the completed 216-probe symbolic-depth experiment; no new model calls.

- Depth: 50
- Valid paired probes: 72
- Models with at least one SIC+: 5/6

| Model | n | SIC+ events | SIC+ rate |
|---|---:|---:|---:|
| claude-sonnet-4-6 | 12 | 1 | 8.3% |
| deepseek-v4-flash | 12 | 2 | 16.7% |
| deepseek-v4-pro | 12 | 2 | 16.7% |
| gpt-5.4 | 12 | 1 | 8.3% |
| gpt-5.4-mini | 12 | 1 | 8.3% |
| qwen3.6-plus | 12 | 0 | 0.0% |

| Category | n | Compliance | Recall | SIC+ |
|---|---:|---:|---:|---:|
| Numerical | 24 | 79.2% | 100.0% | 20.8% |
| Format | 24 | 95.8% | 95.8% | 4.2% |
| Logical | 24 | 95.8% | 95.8% | 4.2% |

## Concentration of depth-50 numerical SIC+ events

- `num_twelve_words`: 4
- `num_one_integer`: 1

Exploratory numerical-vs-other-symbolic Fisher exact test: odds ratio=6.053, one-sided p=0.0372.

Interpretation: the completed audit is consistent with an execution/self-verification bottleneck rather than simple loss of the rule representation. This is a descriptive mechanism audit, not a causal proof.
