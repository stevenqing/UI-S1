# E3 Oracle and aggregation table

All oracle values are descriptive upper bounds. Deployable subsets require parse failure rate < 5% and step-micro > 30%.

## androidcontrol/low

| Model | Parse failure | Step micro | Episode macro |
|---|---:|---:|---:|
| ui-agile-3b | 0.00% | 79.11% | 80.00% |
| ui-agile-7b | 0.00% | 77.57% | 78.48% |
| gui-r1-7b | 0.00% | 58.13% | 59.14% |
| gui-r1-3b | 0.05% | 56.80% | 57.97% |
| ui-r1-e-3b | 0.00% | 48.88% | 50.63% |

| Oracle scope | Models | Step micro | Episode macro |
|---|---:|---:|---:|
| Full | 5 | 87.54% | 88.18% |
| Deployable | 5 | 87.54% | 88.18% |

Deployable models: gui-r1-3b, gui-r1-7b, ui-agile-3b, ui-agile-7b, ui-r1-e-3b
Minimum greedy subset reaching 95% of full-oracle successes: 2

## androidcontrol/high

| Model | Parse failure | Step micro | Episode macro |
|---|---:|---:|---:|
| ui-agile-7b | 0.00% | 60.76% | 61.61% |
| ui-agile-3b | 0.00% | 58.84% | 59.40% |
| gui-r1-7b | 0.04% | 45.22% | 47.01% |
| gui-r1-3b | 0.17% | 38.64% | 39.30% |
| ui-r1-e-3b | 0.00% | 23.06% | 25.72% |

| Oracle scope | Models | Step micro | Episode macro |
|---|---:|---:|---:|
| Full | 5 | 77.53% | 77.98% |
| Deployable | 4 | 76.27% | 76.66% |

Deployable models: gui-r1-3b, gui-r1-7b, ui-agile-3b, ui-agile-7b
Minimum greedy subset reaching 95% of full-oracle successes: 3

## mind2web/visual

| Model | Parse failure | Step micro | Episode macro |
|---|---:|---:|---:|
| tongui-7b | 0.05% | 52.93% | 55.61% |
| tongui-32b | 0.05% | 52.02% | 54.33% |
| cogagent-18b | 5.87% | 50.14% | 56.04% |
| tongui-3b | 0.00% | 48.99% | 51.38% |
| ui-tars-72b | 5.67% | 39.95% | 43.41% |
| ui-tars-7b | 1.49% | 33.65% | 37.87% |
| seeclick-9.6b | 0.00% | 21.92% | 25.98% |
| ui-tars-2b | 22.12% | 18.85% | 22.66% |
| showui-2b | 0.00% | 18.12% | 19.96% |
| qwen2.5-vl-7b | 11.30% | 5.48% | 5.90% |
| qwen2.5-vl-3b | 22.07% | 0.91% | 0.79% |

| Oracle scope | Models | Step micro | Episode macro |
|---|---:|---:|---:|
| Full | 11 | 81.78% | 83.97% |
| Deployable | 4 | 70.91% | 72.86% |

Deployable models: tongui-32b, tongui-3b, tongui-7b, ui-tars-7b
Minimum greedy subset reaching 95% of full-oracle successes: 5
