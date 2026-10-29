# Fusion ablation results

Addition, concatenation and gated fusion were trained with seeds 42, 43 and 44.
All runs used the same 29 training maps, six validation maps, train-only
normalization, AdamW schedule, 25-epoch limit and early-stopping rule. Test data
was not loaded during training or candidate selection.

## Validation

| Fusion | Mean validation error across seeds |
|---|---:|
| Addition | **38.02 m** |
| Concatenation MLP | 38.77 m |
| Sigmoid gate | 39.50 m |

Concatenation was selected as the better learned alternative before the test data
was opened. Addition retained the best validation result overall.

## Test

| Fusion | Mean | Median | P90 | P95 |
|---|---:|---:|---:|---:|
| Addition | 36.16 ± 1.14 m | 22.09 m | **82.47 m** | **117.44 m** |
| Concatenation MLP | **33.87 ± 1.22 m** | **18.27 m** | 82.93 m | 120.27 m |

Concatenation reduced mean error by 6.32% and median error by 17.3%. It also used
5.8% more parameters and slightly increased both tail metrics. The validation and
test rankings differ, so the result supports an accuracy–complexity trade-off
rather than replacing addition as the default.

The comparison uses three seeds on one physical-map split. Seed variation is
reported as sample standard deviation, but it is not a statistical significance
claim. The same 12,288 test users are evaluated by every seed.

Run one condition with:

```powershell
.\.venv\Scripts\python.exe tools\run_fusion_ablation.py --fusion add --seed 42
```

After all nine runs, reproduce validation selection and test evaluation with:

```powershell
.\.venv\Scripts\python.exe tools\compare_fusion_ablation.py
```
