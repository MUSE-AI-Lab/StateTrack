# StateTrack-Bench

This repository contains the public benchmark scenarios, cached evaluation outputs, and reproduction scripts for **When LLMs Forget What They Believe: Characterizing State Integrity Collapse in Long-Horizon Dialogues** (EMNLP 2026, Main Conference).

The repository is designed to make the reported tables and released audits inspectable without external services. Cached outputs are included for offline reproduction; rerunning model generation is provider-dependent and requires credentials configured by the user. No credentials, cookies, or private endpoints are included.

## Reproduce the reported tables

The following commands use only the Python standard library:

```bash
python tools/reproduce_reported_metrics.py
python -m unittest discover -s tests -v
python tools/verify_checksums.py
```

`reproduce_reported_metrics.py` reconstructs the seven-model retention table, the dynamic-state, symbolic-depth, execution-bottleneck, and two-constraint results, and the 70-row checker-calibration summary from the stored JSON files. The test suite checks the reported counts and rates, and the final command verifies the repository checksum manifest.

## Submitted-paper tables

For the submitted benchmark, `results/core/collapse_curves.json` stores the aggregate values reported in Tables 2 and 3. The offline reproduction script reads this file directly and prints the model-level retention table.

## Additional analyses

`results/rebuttal/` contains the released follow-up analyses:

- a 70-output audit separating immediate-task success, constraint compliance, recall, SIC+, and SIC-;
- a 105-output update and revocation experiment;
- a 216-probe symbolic-depth analysis over six models;
- a 34-output two-constraint analysis;
- the response-internal execution audit used to interpret the depth-50 numerical results.

The corresponding runners are under `scripts/paper1_state_collapse/`.

## Blind checker calibration

`results/checker_calibration/` contains two independently worded temperature-zero model-assisted passes over all 70 cached forked-audit rows, disagreement-only adjudication, the final row-level labels, and the aggregate agreement report. The audit calibrates task, persistent-constraint, and recall decisions separately; it is not a human inter-annotator study. The cached metrics reproduce offline, while re-running the judge calls requires local provider configuration.

## SGD dialogue pilot

`results/ecological_pilot/` contains a separate pilot on public human-authored dialogues from the Schema-Guided Dialogue dataset. It includes 30 cases: 15 carryover, 10 overwrite, and 5 revocation or relaxation cases. Seven models produced 210 model-case records, each with an independent behavior response and recall response.

This pilot is separate from the submitted benchmark tables. Its overall active-constraint compliance is 187/210 (89.0%), strict joint success is 140/210 (66.7%), and SIC+ occurs in 9/210 records (4.3%). The selected cases, raw responses, judged records, source manifest, and aggregate statistics are included in the directory. See `results/ecological_pilot/README.md` for file details and reproduction commands.

## Repository layout

- `results/core/`: aggregate cache for the submitted retention tables.
- `results/rebuttal/`: raw outputs and summaries for the targeted analyses.
- `results/ecological_pilot/`: SGD pilot cases, outputs, labels, and summary.
- `results/checker_calibration/`: blind A/B checker labels, one adjudication, and aggregate agreement metrics.
- `scripts/`: experiment prompts, scoring logic, and analysis runners.
- `metadata/snapshot_manifest.csv`: model endpoint and snapshot identifiers recorded for the experiments.
- `tests/`: consistency checks over the stored results.
- `SHA256SUMS.txt`: repository integrity manifest.

Offline table reproduction does not call external APIs. Re-running model generation requires local provider configuration, and results may vary when hosted model endpoints change.
