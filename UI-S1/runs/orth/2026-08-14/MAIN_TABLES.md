# ORTH Scoping Tables

## Arm 0: CEIL analysis units

| Benchmark | Point | Group 99% CI | Base-row 99% CI | IID pair 99% CI | Sample keys | Base rows |
| --- | ---: | --- | --- | --- | ---: | ---: |
| mind2web | 0.688 | [0.665, 0.709] | [0.663, 0.712] | [0.683, 0.693] | 2021 | 891 |
| screenspot_pro | 0.540 | [0.501, 0.583] | [0.502, 0.578] | [0.532, 0.548] | 968 | 430 |

## Arm 1: OCR matcher-family ranges

| Engine | Matcher | Match rate | Selected-correct match | Recoverable match | Zero-coverage match | Text accuracy | Icon accuracy | Error kappa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| easyocr | exact | [21.06, 50.03]% | [22.14, 52.63]% | [22.45, 48.57]% | [16.72, 43.16]% | [3.28, 4.61]% | [0.17, 0.33]% | [0.02, 0.03] |
| easyocr | normalized | [55.79, 92.66]% | [61.57, 93.25]% | [51.43, 91.84]% | [41.34, 91.49]% | [24.26, 28.05]% | [0.66, 1.32]% | [0.15, 0.18] |
| easyocr | edit | [55.15, 87.73]% | [61.07, 90.47]% | [51.02, 83.67]% | [40.12, 82.37]% | [25.28, 30.81]% | [0.66, 0.99]% | [0.16, 0.20] |
| rapidocr | exact | [6.33, 32.89]% | [7.45, 33.57]% | [5.71, 29.80]% | [3.34, 33.13]% | [1.64, 2.76]% | [0.00, 0.17]% | [0.01, 0.01] |
| rapidocr | normalized | [34.85, 81.47]% | [40.22, 84.11]% | [28.16, 82.04]% | [23.40, 72.95]% | [17.30, 20.68]% | [0.33, 0.83]% | [0.10, 0.12] |
| rapidocr | edit | [35.67, 75.21]% | [41.71, 80.93]% | [27.76, 69.39]% | [23.10, 62.01]% | [19.14, 26.92]% | [0.33, 0.83]% | [0.12, 0.16] |

## Arm 2: DOM/AX availability

`FULL_DOM_AX_UNAVAILABLE_HISTORICAL_DATA_CURRENTLY_MISSING`. Historical complete HTML data was audited, but the local official dataset and scores are missing. The current XFER lane retains only label-selected positive snippets, which are not valid predictor input.

## Arm 3: Identifiable headroom

110 accuracy/kappa cells were evaluated; 23 requested cells required projection to a feasible joint error table. Bayes-fused grounding accuracy is not identified from marginal accuracy and error kappa alone.
