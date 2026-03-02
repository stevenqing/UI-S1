# Pass@K Evaluation Results Comparison

**Date**: 2026-02-02
**Dataset**: Android Control Evaluation (1543 tasks)
**Configuration**: K=8, Temperature=0.7

---

## Overall Rankings

| Rank | Model | Pass@8 | Pass@1 | Sampling Lift | Tasks Evaluated |
|------|-------|--------|--------|---------------|-----------------|
| 1 | **UI-TARS-7B** | **20.61%** | 13.48% | +52.9% | 1543 |
| 2 | **UI-S1-7B** | **18.18%** | 17.92% | +1.4% | 385* |
| 3 | **Qwen2.5-VL-7B** | **16.05%** | 6.67% | +142.6% | 1514 |
| 4 | OS-Atlas-7B | 9.27% | 8.62% | +7.5% | 1543 |
| 5 | OS-Genesis-7B | 1.10% | 1.10% | +0.0% | 1543 |
| 6 | OS-Genesis-7B-hist | 1.04% | 1.04% | +0.0% | 1543 |

*UI-S1-7B only completed 385/1543 tasks due to token length errors (prompts exceeding 32768 token limit)

---

## Computation Time

| Model | Tasks | Duration | Tasks/min | Sec/Task | Total Inferences |
|-------|-------|----------|-----------|----------|------------------|
| OS-Atlas-7B | 1543 | 0h 48m | 31.9 | 1.88s | 12,344 |
| OS-Genesis-7B | 1543 | 1h 24m | 18.3 | 3.28s | 12,344 |
| OS-Genesis-7B-hist | 1543 | 1h 25m | 18.0 | 3.33s | 12,344 |
| Qwen2.5-VL-7B | 1514 | 1h 42m | 14.8 | 4.06s | 12,112 |
| UI-TARS-7B | 1543 | 1h 27m | 17.6 | 3.41s | 12,344 |
| UI-S1-7B | 431* | 2h 20m+ | 3.1 | 19.5s | 3,448+ |

**Notes:**
- Each task requires K=8 samples, so Total Inferences = Tasks × 8
- All jobs ran on 4× GPUs with tensor parallelism
- OS-Atlas-7B is fastest due to simpler prompt format (single image, no history)
- UI-S1-7B is significantly slower due to:
  - Complex multi-turn conversation format
  - 165+ token length errors causing retries (prompts > 32768 tokens)
  - Each failed request retries up to 5× with 5s delays

---

## Detailed Metrics

### UI-TARS-7B
- Pass@8: 318/1543 = 20.61%
- Pass@1: 208/1543 = 13.48%
- Avg successes/task: 0.206
- Tasks with progress: 733/1543 = 47.5%

### UI-S1-7B
- Pass@8: 70/385 = 18.18%
- Pass@1: 69/385 = 17.92%
- Avg successes/task: 0.182
- Tasks with progress: 235/385 = 61.0%

### Qwen2.5-VL-7B
- Pass@8: 243/1514 = 16.05%
- Pass@1: 101/1514 = 6.67%
- Avg successes/task: 0.161
- Tasks with progress: 761/1514 = 50.3%

### OS-Atlas-7B
- Pass@8: 143/1543 = 9.27%
- Pass@1: 133/1543 = 8.62%
- Normal (greedy): 131/1543 = 8.49%
- Pass@8 vs Normal: +0.78%
- Avg successes/task: 0.093
- Tasks with progress: 394/1543 = 25.5%

### OS-Genesis-7B
- Pass@8: 17/1543 = 1.10%
- Pass@1: 17/1543 = 1.10%
- Normal (greedy): 17/1542 = 1.10%
- Pass@8 vs Normal: +0.00%
- Avg successes/task: 0.011
- Tasks with progress: 442/1543 = 28.6%

### OS-Genesis-7B-hist
- Pass@8: 16/1543 = 1.04%
- Pass@1: 16/1543 = 1.04%
- Avg successes/task: 0.010
- Tasks with progress: 440/1543 = 28.5%

---

## Performance by Task Complexity

### Pass@8 Accuracy by Number of Steps

| Model | Easy (1-3) | Medium (4-6) | Hard (7-10) | Very Hard (11-20) |
|-------|------------|--------------|-------------|-------------------|
| UI-TARS-7B | 40.64% | 18.31% | 6.05% | 0.00% |
| UI-S1-7B | 41.07% | 13.07% | 2.53% | 0.00% |
| Qwen2.5-VL-7B | 38.13% | 10.99% | 2.04% | 0.00% |
| OS-Atlas-7B | 20.78% | 7.08% | 1.73% | 0.00% |
| OS-Genesis-7B | 3.88% | 0.00% | 0.00% | 0.00% |
| OS-Genesis-7B-hist | 3.65% | 0.00% | 0.00% | 0.00% |

### Pass@1 Accuracy by Number of Steps

| Model | Easy (1-3) | Medium (4-6) | Hard (7-10) | Very Hard (11-20) |
|-------|------------|--------------|-------------|-------------------|
| UI-TARS-7B | 29.45% | 10.31% | 3.46% | 0.00% |
| UI-S1-7B | 41.07% | 12.50% | 2.53% | 0.00% |
| Qwen2.5-VL-7B | 18.49% | 3.10% | 0.00% | 0.00% |
| OS-Atlas-7B | 20.09% | 6.00% | 1.73% | 0.00% |
| OS-Genesis-7B | 3.88% | 0.00% | 0.00% | 0.00% |
| OS-Genesis-7B-hist | 3.65% | 0.00% | 0.00% | 0.00% |

### Task Distribution
- Easy (1-3 steps): 438 tasks (28.4%)
- Medium (4-6 steps): 650 tasks (42.1%)
- Hard (7-10 steps): 347 tasks (22.5%)
- Very Hard (11-20 steps): 104 tasks (6.7%)

### Task Examples by Difficulty

#### Easy (1-3 steps)
| Steps | Example Task |
|-------|--------------|
| 1 | View today's (20th December) moon phase on the Phases of the Moon app. |
| 2 | Open the radio app and browse through the KRRO FM 103.7 radio stations that are available. |
| 3 | Open the Artsy app and Learn more about the art "Crucifixion (Corpus Hypercubus) 1954". |

#### Medium (4-6 steps)
| Steps | Example Task |
|-------|--------------|
| 4 | Delete the event called dinner at Carlos House on September 25, 2023. |
| 5 | I want to listen to Sleep Meditation for Deep Sleep on the Balance app so I can sleep soundly tonight. |
| 6 | Open the Cx file Explorer and rename the Flowers folder to Flora. |

#### Hard (7-10 steps)
| Steps | Example Task |
|-------|--------------|
| 7 | Open the CNN News app and share the article "Trump pleads not guilty to 4 felonies in 2020 election case" with dbwscratch.test.id3@gmail.com through Gmail. |
| 8 | Open the Adidas app, Add DROPSET 2 TRAINER shoes of size 10 to cart for mom. |
| 9 | In commemoration of my nephew's fifth birthday, seek a 2-piece sweater in the kids category on the Zara app. |

#### Very Hard (11-20 steps)
| Steps | Example Task |
|-------|--------------|
| 11 | Open All Trails app, search for trails near 98110, make sure to sort by distance up to 10 miles and set the difficulty filter to Easy. |
| 11 | I would like to view news in sports and science categories on The Washington Post app to keep myself informed. |
| 17 | Open the Zoho Meeting app and Schedule a meet for July 23rd from 1:30 PM to 2:00 PM with the topic name as XYZ. |

### Step Distribution Within Categories
```
Easy (1-3):      1 step: 116    2 steps: 122    3 steps: 200
Medium (4-6):    4 steps: 221   5 steps: 256    6 steps: 173
Hard (7-10):     7 steps: 138   8 steps: 102    9 steps: 65    10 steps: 42
Very Hard (11+): 11 steps: 30   12 steps: 18    13 steps: 18   14-20 steps: 38
```

### Office/Productivity App Examples by Difficulty (42 tasks total)

#### Easy (1-3 steps) - 9 tasks
| Steps | Example Task |
|-------|--------------|
| 1 | In the Polaris Office app, I would like to open the New word.docx. |
| 2 | I want tableau_blueprint.pdf to access offline, access the tableau_blueprint.pdf in the Drive app. |
| 3 | I want to add a title "DIY PROJECTS" on this slide in the Slides app. |
| 3 | View the "Copy of the Queen's Gambit Book" pdf file for me on the Drive app. |

#### Medium (4-6 steps) - 18 tasks
| Steps | Example Task |
|-------|--------------|
| 4 | Open the Xodo app and highlight the significance text in the welcome pdf. |
| 4 | Open the existing Blank template word file in the WORD OFFICE app. |
| 5 | To make seeing the agents.txt file easier for me in the future, upload it to the OneDrive app. |
| 5 | Open the Google Docs app and edit the Crash document then change the text. |

#### Hard (7-10 steps) - 14 tasks
| Steps | Example Task |
|-------|--------------|
| 7 | Open the PDF Reader Pro app and add a drawing to the dummy pdf file. |
| 8 | I want to save the Document 3 with the name Yoga in Microsoft word app. |
| 9 | Open the Keep Notes app and share the swimming class note to dbwscratch.test.id3@gmail.com through Gmail. |
| 10 | Send the Keep notes app's Places to Visit notes via gmail at Thomas123@gmail.com. |

#### Very Hard (11+ steps) - 1 task
| Steps | Example Task |
|-------|--------------|
| 12 | In the DeftPDF app, Share test pdf to dbwscratch.test.id2@gmail.com via gmail. |

**App Distribution:**
- PDF-related: 15 tasks (Xodo, PDF Reader Pro, DeftPDF, Drive)
- Word/Docs: 14 tasks (Polaris Office, Google Docs, Word Office, Microsoft Word)
- Notes: 3 tasks (Keep Notes)
- Slides/PowerPoint: 4 tasks (Google Slides, Slides app, PowerPoint)
- Cloud Storage: 6 tasks (Drive, OneDrive)

### Deep Research Task Examples by Difficulty (297 tasks total)

Tasks involving information retrieval, learning, news reading, and complex multi-step searches with filters/sorting.

#### Easy (1-3 steps) - 147 tasks
| Steps | Type | Example Task |
|-------|------|--------------|
| 2 | Browse | Open the radio app and browse through the KRRO FM 103.7 radio stations. |
| 3 | Learn | Open the Artsy app and Learn more about the art "Crucifixion (Corpus Hypercubus) 1954". |
| 3 | News | Open The Guardian news app and read the news article about Donald Trump. |

#### Medium (4-6 steps) - 306 tasks
| Steps | Type | Example Task |
|-------|------|--------------|
| 4 | Search | Open the smart news App search for covid 19 in the search bar. |
| 5 | Research | Open the ArtStation app and find an artwork inspired by sonya agafonova. |
| 6 | Browse | Show me some of the sustainability art pieces on the Pinterest app for my research on sustainable energy. |

#### Hard (7-10 steps) - 156 tasks
| Steps | Type | Example Task |
|-------|------|--------------|
| 7 | Learn | Open the Stellarium app and learn about stars. |
| 7 | Learn | I want to learn about ratio patterns. Open the ratio patterns chapter in the Math Tests app. |
| 10 | News | In the Flipboard news app, read the news article on Phoenix's record heat is killing off cactuses. |
| 10 | News | View the news articles in the Business category on The Hindu News app. |

#### Very Hard (11-20+ steps) - 49 tasks
| Steps | Type | Example Task |
|-------|------|--------------|
| 11 | News | View the news articles in the Business category on The CNN News app. |
| 13 | News | In the BBC news app, read the news article on Google alert failed on Turkey quake. |
| 16 | Learn | I want to learn about the lunar calendar on the Moonly app. |
| 18 | News | Read the news article in the entertainment category on the Google News app. |
| 19 | Search | Open the Art & Culture app and search for an article about "A Modern Painting of Ancient Myths". |
| 19 | Flight | In the Momondo app, find a flight from Scotland to Canada departing August 10, returning August 15, then select a flight between 11:30 a.m. to 12 p.m. |
| 21 | Flight | Look for a flight from Detroit to Las Vegas in business class for 4 passengers on Expedia. |
| 24 | Learn | Open the infinite painter app and click on the Gradients to learn about gradient techniques to make digital art. |

**Research Task Categories:**
- News/Article Reading: ~80 tasks
- Learning/Studying: ~45 tasks
- Complex Search with Filters: ~60 tasks
- Product Research/Comparison: ~50 tasks
- Art/Culture Exploration: ~40 tasks
- Travel/Flight Search: ~22 tasks

---

## Sampling Benefit Analysis

### Pass@8 vs Pass@1 Lift

| Model | Pass@1 | Pass@8 | Absolute Lift | Relative Lift |
|-------|--------|--------|---------------|---------------|
| Qwen2.5-VL-7B | 6.67% | 16.05% | +9.38% | **+142.6%** |
| UI-TARS-7B | 13.48% | 20.61% | +7.13% | **+52.9%** |
| OS-Atlas-7B | 8.62% | 9.27% | +0.65% | +7.5% |
| UI-S1-7B | 17.92% | 18.18% | +0.26% | +1.4% |
| OS-Genesis-7B | 1.10% | 1.10% | +0.00% | +0.0% |
| OS-Genesis-7B-hist | 1.04% | 1.04% | +0.00% | +0.0% |

### Interpretation
- **Qwen2.5-VL-7B**: Benefits most from sampling - high variance in outputs
- **UI-TARS-7B**: Strong benefit from sampling - moderate variance
- **UI-S1-7B**: Minimal benefit - already consistent at Pass@1
- **OS-Genesis**: No benefit - completely deterministic failures

---

## Consistency Analysis

How often do all 8 samples produce the same result?

| Model | All 8 Succeed | All 8 Fail | Mixed (1-7) |
|-------|---------------|------------|-------------|
| UI-TARS-7B | 0 (0.0%) | 1225 (79.4%) | 318 (20.6%) |
| UI-S1-7B | 0 (0.0%) | 320 (83.1%) | 65 (16.9%) |
| Qwen2.5-VL-7B | 0 (0.0%) | 1271 (84.0%) | 243 (16.0%) |
| OS-Atlas-7B | 0 (0.0%) | 1400 (90.7%) | 143 (9.3%) |
| OS-Genesis-7B | 0 (0.0%) | 1526 (98.9%) | 17 (1.1%) |
| OS-Genesis-7B-hist | 0 (0.0%) | 1527 (99.0%) | 16 (1.0%) |

### Key Observations
- No model achieved 8/8 success on any task
- OS-Genesis models show 99% consistent failures (systematic issues)
- UI-TARS shows highest variance (20.6% mixed results)

---

## OS-Atlas-7B: Normal vs Pass@K Comparison

### Summary
| Metric | Normal (T=0) | Pass@8 (T=0.7) | Delta |
|--------|--------------|----------------|-------|
| Accuracy | 8.49% | 9.27% | +0.78% |
| Tasks Succeeded | 131/1543 | 143/1543 | +12 tasks |

### Impact Analysis
- **16 tasks** succeeded with Pass@8 that failed with Normal (sampling helped)
- **1 task** succeeded with Normal but failed all 8 samples (sampling hurt)
- **Net gain**: +15 tasks

### Breakdown by Complexity
| Steps | Count | Normal | Pass@8 | Delta |
|-------|-------|--------|--------|-------|
| 1-3 | 438 | 19.63% | 20.78% | +1.14% |
| 4-6 | 650 | 6.00% | 7.08% | +1.08% |
| 7-10 | 347 | 1.73% | 1.73% | +0.00% |
| 11-20 | 104 | 0.00% | 0.00% | +0.00% |

---

## Key Insights

### 1. Top Performers
UI-TARS-7B, UI-S1-7B, and Qwen2.5-VL-7B significantly outperform OS-Atlas and OS-Genesis models, with 2-20x higher accuracy.

### 2. Sampling Benefit Varies Dramatically
- **High benefit**: Qwen2.5-VL (+142.6%), UI-TARS (+52.9%)
- **Low benefit**: OS-Atlas (+7.5%), UI-S1 (+1.4%)
- **No benefit**: OS-Genesis (0%) - failures are systematic, not stochastic

### 3. Multi-Step Task Performance
- All models fail completely on 11+ step tasks
- Only UI-TARS shows meaningful performance (6%) on 7-10 step tasks
- Easy tasks (1-3 steps) see 20-41% success rates for top models

### 4. Model Characteristics
- **UI-S1-7B**: High Pass@1 accuracy, minimal variance - well-calibrated but limited by token length issues
- **Qwen2.5-VL-7B**: Low Pass@1 but high Pass@8 - benefits greatly from multiple attempts
- **OS-Genesis**: Systematic failures indicate fundamental capability gaps

### 5. UI-S1 Evaluation Limitation
UI-S1 only completed 25% of tasks due to prompts exceeding 32768 token limit. This occurs because:
- Complex system prompt with full action space definitions
- Multi-turn conversation history with up to 2 history images
- Verbose format requirements (`<think>`, `<action>` tags)

---

## Recommendations

1. **For production deployment**: UI-TARS-7B offers the best balance of accuracy and sampling benefit
2. **For single-attempt scenarios**: UI-S1-7B shows highest Pass@1 accuracy (on completed tasks)
3. **For cost-sensitive applications**: Qwen2.5-VL benefits most from additional samples
4. **For UI-S1 evaluation**: Consider reducing `n_history_image_limit` or increasing `max_model_len`

---

## Files Reference

### Pass@K Results
- `results/pass_k/OS_Atlas_7b_pass_k8.jsonl`
- `results/pass_k/OS_Genesis_7b_pass_k8.jsonl`
- `results/pass_k/OS_Genesis_7b_with_history_pass_k8.jsonl`
- `results/pass_k/Qwen2.5-VL-7B_pass_k8.jsonl`
- `results/pass_k/UI-S1-7B_pass_k8.jsonl`
- `results/pass_k/ui-tars_7b_pass_k8.jsonl`

### Normal Results
- `results/OS_Atlas_7b.jsonl`
- `results/OS_Genesis_7b.jsonl`
- `results/OS_Genesis_7b_with_history.jsonl`
