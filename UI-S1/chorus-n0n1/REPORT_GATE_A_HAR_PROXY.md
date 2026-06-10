# GATE A Report - HAR GUI-Odyssey Proxy

**Status:** IN_PROGRESS

## Scope

- Benchmark: GUI-Odyssey random split test (`datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl`)
- Model: `HAR-GUI-3B-GUI-Odyssey`
- Evaluation setting: HAR native prompts, Act2Sum history, all steps, no first-error stop
- SOTA status: `sota_proxy=true` because HiconAgent checkpoint is unavailable locally
- HiconAgent status: `checkpoint_unavailable`
- Resumable results: `related_work/har/outputs/gui_odyssey_paper/full_har_gui_odyssey_20260610.jsonl`

## Progress

- Episodes completed: 4 / 1666
- Steps evaluated: 62 / 62
- Correct steps: 42
- Step SR: 67.74%
- Task SR: 0.00%

## Paper Comparison

- Paper GUI-Odyssey Overall SSR: 62.31%
- Current Overall SSR: 67.74%
- Delta: +5.43 points

## Category Breakdown

| Category | Paper column | Episodes | Steps | Step SR | Task SR |
| --- | --- | ---: | ---: | ---: | ---: |
| General_Tool | Tool | 3 | 46 | 63.04% | 0.00% |
| Social_Sharing | Social | 1 | 16 | 81.25% | 0.00% |

## Truncation Summary

- Generations: 124
- Truncated generations: 0
- Truncated generation rate: 0.00%
- Action generations: 62
- Truncated action generations: 0
- Truncated action rate: 0.00%
- Gate rule: >1% truncated generations invalidates the run

## Qualitative Step Records

```json
{
  "episode_id": "1488142680821851",
  "category": "General_Tool",
  "goal": "Open Chrome and navigate to Spotify. Search for the song 'Think of Me' on Spotify, then listen to it. Once you've listened to the first line of the lyrics, translate that line into Hebrew.",
  "num_steps": 19,
  "steps_evaluated": 19,
  "correct_steps": 13,
  "task_success": false,
  "first_error_step": 7,
  "steps": [
    {
      "step_idx": 0,
      "extract_match": true,
      "type_match": true,
      "gt_action": {
        "action": "click",
        "coordinate": [
          141,
          254
        ],
        "candidate_bbox": [
          [
            128,
            241,
            152,
            268
          ]
        ]
      },
      "pred_action": {
        "action": "click",
        "coordinate": [
          130,
          160
        ]
      },
      "answer": "CLICK:(130,160)",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    },
    {
      "step_idx": 1,
      "extract_match": true,
      "type_match": true,
      "gt_action": {
        "action": "click",
        "coordinate": [
          75,
          146
        ],
        "candidate_bbox": [
          [
            35,
            140,
            68,
            155
          ]
        ]
      },
      "pred_action": {
        "action": "click",
        "coordinate": [
          79,
          100
        ]
      },
      "answer": "CLICK:(79,100)",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    },
    {
      "step_idx": 2,
      "extract_match": true,
      "type_match": true,
      "gt_action": {
        "action": "click",
        "coordinate": [
          456,
          168
        ],
        "candidate_bbox": [
          [
            374,
            160,
            875,
            220
          ]
        ]
      },
      "pred_action": {
        "action": "click",
        "coordinate": [
          577,
          117
        ]
      },
      "answer": "CLICK:(577,117)",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    }
  ]
}
```
```json
{
  "episode_id": "2472499926050391",
  "category": "Social_Sharing",
  "goal": "Use DuckDuckGo to search for an introduction to Notre-Dame Cathedral in Paris, then share the link of the webpage on Facebook with moments.",
  "num_steps": 16,
  "steps_evaluated": 16,
  "correct_steps": 13,
  "task_success": false,
  "first_error_step": 4,
  "steps": [
    {
      "step_idx": 0,
      "extract_match": true,
      "type_match": true,
      "gt_action": {
        "action": "click",
        "coordinate": [
          187,
          488
        ],
        "candidate_bbox": [
          [
            186,
            459,
            228,
            525
          ]
        ]
      },
      "pred_action": {
        "action": "click",
        "coordinate": [
          165,
          343
        ]
      },
      "answer": "CLICK:(165,343)",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    },
    {
      "step_idx": 1,
      "extract_match": true,
      "type_match": true,
      "gt_action": {
        "action": "click",
        "coordinate": [
          561,
          528
        ],
        "candidate_bbox": [
          [
            273,
            496,
            726,
            654
          ]
        ]
      },
      "pred_action": {
        "action": "click",
        "coordinate": [
          471,
          431
        ]
      },
      "answer": "CLICK:(471,431)",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    },
    {
      "step_idx": 2,
      "extract_match": true,
      "type_match": true,
      "gt_action": {
        "action": "click",
        "coordinate": [
          604,
          635
        ],
        "candidate_bbox": [
          [
            165,
            575,
            823,
            642
          ]
        ]
      },
      "pred_action": {
        "action": "click",
        "coordinate": [
          357,
          421
        ]
      },
      "answer": "CLICK:(357,421)",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    }
  ]
}
```
```json
{
  "episode_id": "2545035427384715",
  "category": "General_Tool",
  "goal": "Switch to dark mode using the Setting app and then open the Amazon Kindle app for reading.",
  "num_steps": 10,
  "steps_evaluated": 10,
  "correct_steps": 6,
  "task_success": false,
  "first_error_step": 1,
  "steps": [
    {
      "step_idx": 0,
      "extract_match": false,
      "type_match": true,
      "gt_action": {
        "action": "swipe",
        "coordinate": [
          352,
          550
        ],
        "coordinate2": [
          654,
          553
        ],
        "candidate_bbox": []
      },
      "pred_action": {
        "action": "swipe",
        "coordinate": [
          700,
          368
        ],
        "coordinate2": [
          340,
          368
        ]
      },
      "answer": "SCROLL:LEFT",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    },
    {
      "step_idx": 1,
      "extract_match": true,
      "type_match": true,
      "gt_action": {
        "action": "click",
        "coordinate": [
          879,
          623
        ],
        "candidate_bbox": [
          [
            825,
            587,
            906,
            637
          ]
        ]
      },
      "pred_action": {
        "action": "click",
        "coordinate": [
          501,
          633
        ]
      },
      "answer": "CLICK:(501,633)",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    },
    {
      "step_idx": 2,
      "extract_match": true,
      "type_match": true,
      "gt_action": {
        "action": "swipe",
        "coordinate": [
          508,
          928
        ],
        "coordinate2": [
          480,
          116
        ],
        "candidate_bbox": []
      },
      "pred_action": {
        "action": "swipe",
        "coordinate": [
          520,
          500
        ],
        "coordinate2": [
          520,
          250
        ]
      },
      "answer": "SCROLL:UP",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    }
  ]
}
```
```json
{
  "episode_id": "7014156452611514",
  "category": "General_Tool",
  "goal": "Participate in a Russian language lesson and establish a learning plan using 'Microsoft To Do' and 'Rosetta Stone: Learn, Practice'.",
  "num_steps": 17,
  "steps_evaluated": 17,
  "correct_steps": 10,
  "task_success": false,
  "first_error_step": 2,
  "steps": [
    {
      "step_idx": 0,
      "extract_match": true,
      "type_match": true,
      "gt_action": {
        "action": "click",
        "coordinate": [
          366,
          510
        ],
        "candidate_bbox": [
          [
            347,
            476,
            423,
            514
          ]
        ]
      },
      "pred_action": {
        "action": "click",
        "coordinate": [
          221,
          516
        ]
      },
      "answer": "CLICK:(221,516)",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    },
    {
      "step_idx": 1,
      "extract_match": false,
      "type_match": false,
      "gt_action": {
        "action": "system_button",
        "button": "Home",
        "candidate_bbox": []
      },
      "pred_action": {
        "action": "click",
        "coordinate": [
          199,
          829
        ]
      },
      "answer": "CLICK:(199,829)",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    },
    {
      "step_idx": 2,
      "extract_match": true,
      "type_match": true,
      "gt_action": {
        "action": "click",
        "coordinate": [
          402,
          614
        ],
        "candidate_bbox": [
          [
            351,
            581,
            415,
            636
          ]
        ]
      },
      "pred_action": {
        "action": "click",
        "coordinate": [
          223,
          627
        ]
      },
      "answer": "CLICK:(223,627)",
      "finish_reason": "stop",
      "truncated": false,
      "error": ""
    }
  ]
}
```

## Gate Decision

STOP. Full Phase A reproduction is still running; do not proceed to N0/N1 from this partial proxy result.
