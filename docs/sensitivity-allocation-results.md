# Matched-storage sensitivity allocation result

The frozen CPU study completed on 2026-10-14 using source `f5ba3a9`, five
calibration seeds and one shared 32-paragraph, 6,654-target evaluation panel.
Every mixed method stored exactly 143,409,728 tensor bytes and passed checkpoint
tensor, exact CPU-logit, shifted-loss and target-count checks.

The 25 declared seed/method cells contained only nine unique precision policies.
Magnitude and each of the three fixed shuffles produced identical assignments in
all five calibration seeds; only the direct-sensitivity assignments changed.
Identical deterministic policies were evaluated once and reused by exact
assignment hash, with the source cell, report hash and checkpoint tensor hash
retained. These repeated comparator cells are not independent model trials.

| Calibration seed | Sensitivity PPL | NLL gain over magnitude | NLL gain over median shuffle | PPL / FP32 | Score pass | Practical pass |
| ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 42 | 76.118991 | +0.004599 | +0.086439 | 1.077402 | no | no |
| 142 | 78.264916 | -0.023202 | +0.058637 | 1.107776 | no | no |
| 242 | 76.290658 | +0.002347 | +0.084186 | 1.079832 | no | no |
| 342 | 76.576063 | -0.001387 | +0.080452 | 1.083872 | no | no |
| 442 | 76.855658 | -0.005032 | +0.076808 | 1.087829 | no | no |

Direct sensitivity beat the median fixed shuffle in every seed, but it never met
the preregistered 0.01-nat improvement over magnitude. Seed 142 also exceeded the
allowed 0.02-nat regression. None met the separate 1.05×-FP32 practical-quality
limit. Both the score gate and practical gate therefore failed. This negative
result is not repaired by the separate INT8 embedding result.

The compact public [result summary](../benchmarks/results/pythia70m-sensitivity-allocation-cpu-summary.json)
records the gate inputs and hashes. The complete recovery report has SHA-256
`b89f5abc4ad58c7ff776a8a84a4ecab675f18f97ac0bbeb9a86bb598ff4d2c03`;
independent replay (`7df1d2742e3dc86e73bf7a5f5b23f3b88e3ea7a3eacf2d1ca0b086a7b96d7ab5`)
verified all 25 cells, nine unique policies and raw arithmetic. The earlier MPS
run remains separately incomplete because its frozen CPU/MPS logit-parity gate
failed; no MPS measurements were substituted into this CPU study.

Five calibration seeds sharing one evaluation panel do not establish corpus or
model generalization. CASA and MixQuant provide primary-source context for why
isolated scalar sensitivity may miss cross-layer interactions; their results are
not evidence for this workbench.
