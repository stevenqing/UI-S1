# Cooperative LoRA v3 ep2 vs SVD LoRA r=256: Detailed Error Analysis

## 1. Overall Results

| Model | Correct | Total | Rate |
|-------|---------|-------|------|
| SVD LoRA r=256 | 9036 | 19046 | 47.4% |
| Coop v3 ep2 | 8782 | 19046 | 46.1% |
| Coop v3 ep1 | 8447 | 19046 | 44.4% |

## 2. Per-Sample Contingency (Coop ep2 vs SVD)

|  | Coop Correct | Coop Wrong |
|---|---|---|
| **SVD Correct** | 7251 (38.1%) | 1785 (9.4%) |
| **SVD Wrong** | 1531 (8.0%) | 8479 (44.5%) |

- Net gap: 254 samples (1.33%)
- Oracle ensemble (either correct): 10567 (55.5%)

## 3. Response Format Distribution

| Format | Coop All | Coop Fail | SVD All | SVD Fail |
|--------|----------|-----------|---------|----------|
| degenerate_empty_function | 825 | 825 | 747 | 747 |
| degenerate_null_coordinate | 124 | 124 | 139 | 139 |
| empty | 174 | 174 | 104 | 104 |
| malformed_json_in_tool_call | 2099 | 1627 | 149 | 145 |
| no_parseable_action | 24 | 18 | 0 | 0 |
| thought+tool_call | 15798 | 7496 | 0 | 0 |
| tool_call_only | 2 | 0 | 17907 | 8875 |

## 4. Error Type Classification (Failed Samples Only)

### 4a. All Coop Failures

| Error Type | Count | % of Failures |
|------------|-------|---------------|
| wrong_coordinate | 4575 | 44.6% |
| wrong_function | 1819 | 17.7% |
| parse_failure | 1819 | 17.7% |
| empty_function | 825 | 8.0% |
| wrong_arg_text | 395 | 3.8% |
| wrong_coordinate+wrong_other_args | 384 | 3.7% |
| wrong_arg_keys | 96 | 0.9% |
| missing_arg_keys | 61 | 0.6% |
| missing_arg_clear_current_text | 58 | 0.6% |
| missing_arg_text | 53 | 0.5% |
| missing_arg_control_focus | 52 | 0.5% |
| wrong_arg_wheel_dist | 40 | 0.4% |
| wrong_arg_button | 34 | 0.3% |
| wrong_arg_control_focus | 15 | 0.1% |
| wrong_arg_font_size | 8 | 0.1% |
| missing_arg_desktop_x | 6 | 0.1% |
| wrong_arg_font_name | 5 | 0.0% |
| wrong_arg_double | 4 | 0.0% |
| wrong_arg_columns | 4 | 0.0% |
| missing_arg_font_name | 4 | 0.0% |
| wrong_arg_rows | 3 | 0.0% |
| missing_arg_current_slide_only | 2 | 0.0% |
| wrong_arg_start_index | 1 | 0.0% |
| wrong_arg_clear_current_text | 1 | 0.0% |

### 4b. All SVD Failures

| Error Type | Count | % of Failures |
|------------|-------|---------------|
| wrong_coordinate | 5512 | 55.1% |
| wrong_function | 1893 | 18.9% |
| empty_function | 747 | 7.5% |
| wrong_arg_text | 531 | 5.3% |
| wrong_coordinate+wrong_other_args | 516 | 5.2% |
| parse_failure | 249 | 2.5% |
| wrong_arg_keys | 151 | 1.5% |
| missing_arg_keys | 139 | 1.4% |
| missing_arg_clear_current_text | 57 | 0.6% |
| missing_arg_control_focus | 55 | 0.5% |
| wrong_arg_wheel_dist | 36 | 0.4% |
| wrong_arg_button | 32 | 0.3% |
| missing_arg_text | 27 | 0.3% |
| missing_arg_font_name | 14 | 0.1% |
| wrong_arg_rows | 9 | 0.1% |
| wrong_arg_clear_current_text | 6 | 0.1% |
| wrong_arg_start_index | 6 | 0.1% |
| wrong_arg_font_size | 6 | 0.1% |
| wrong_arg_control_focus | 5 | 0.0% |
| missing_arg_desktop_x | 4 | 0.0% |
| missing_arg_pressed | 3 | 0.0% |
| wrong_arg_double | 3 | 0.0% |
| wrong_arg_font_name | 3 | 0.0% |
| wrong_arg_end_index | 3 | 0.0% |
| wrong_arg_columns | 2 | 0.0% |
| wrong_arg_color | 1 | 0.0% |

### 4c. SVD-Unique Wins: Why Coop Failed (SVD right, Coop wrong)

| Error Type | Count | % of 1785 SVD-unique wins |
|------------|-------|------|
| wrong_coordinate | 941 | 52.7% |
| parse_failure | 322 | 18.0% |
| wrong_function | 293 | 16.4% |
| empty_function | 66 | 3.7% |
| wrong_arg_text | 46 | 2.6% |
| wrong_coordinate+wrong_other_args | 35 | 2.0% |
| missing_arg_text | 21 | 1.2% |
| wrong_arg_keys | 18 | 1.0% |
| missing_arg_clear_current_text | 15 | 0.8% |
| missing_arg_control_focus | 11 | 0.6% |
| wrong_arg_wheel_dist | 7 | 0.4% |
| wrong_arg_button | 6 | 0.3% |
| missing_arg_keys | 2 | 0.1% |
| wrong_arg_font_name | 1 | 0.1% |
| wrong_arg_clear_current_text | 1 | 0.1% |

### 4d. Coop-Unique Wins: Why SVD Failed (Coop right, SVD wrong)

| Error Type | Count | % of 1531 Coop-unique wins |
|------------|-------|------|
| wrong_coordinate | 1071 | 70.0% |
| wrong_function | 208 | 13.6% |
| wrong_arg_text | 54 | 3.5% |
| empty_function | 48 | 3.1% |
| wrong_coordinate+wrong_other_args | 39 | 2.5% |
| parse_failure | 24 | 1.6% |
| missing_arg_keys | 20 | 1.3% |
| missing_arg_clear_current_text | 12 | 0.8% |
| missing_arg_control_focus | 12 | 0.8% |
| wrong_arg_keys | 12 | 0.8% |
| missing_arg_font_name | 10 | 0.7% |
| wrong_arg_button | 6 | 0.4% |
| missing_arg_text | 3 | 0.2% |
| wrong_arg_wheel_dist | 3 | 0.2% |
| wrong_arg_rows | 3 | 0.2% |
| missing_arg_pressed | 2 | 0.1% |
| wrong_arg_clear_current_text | 2 | 0.1% |
| wrong_arg_font_size | 2 | 0.1% |

## 5. Per-Domain Breakdown

| Domain | Total | SVD | SVD% | Coop | Coop% | SVD-only | Coop-only | Gap |
|--------|-------|-----|------|------|-------|----------|-----------|-----|
| excel | 5366 | 2351 | 43.8% | 2268 | 42.3% | 528 | 445 | +83 |
| ppt | 5381 | 2892 | 53.7% | 2808 | 52.2% | 475 | 391 | +84 |
| word | 8299 | 3793 | 45.7% | 3706 | 44.7% | 782 | 695 | +87 |

## 6. By Action Function Type

| Function | Total | SVD% | Coop% | SVD-only | Coop-only | Net |
|----------|-------|------|-------|----------|-----------|-----|
| click | 14467 | 53.2% | 51.6% | 1397 | 1170 | +227 |
| type | 3411 | 29.8% | 27.3% | 331 | 244 | +87 |
| select_text | 496 | 44.0% | 49.6% | 23 | 51 | -28 |
| wheel_mouse_input | 311 | 15.1% | 16.1% | 19 | 22 | -3 |
| set_font | 96 | 36.5% | 54.2% | 5 | 22 | -17 |
| insert_table | 69 | 15.9% | 17.4% | 8 | 9 | -1 |
| select_paragraph | 49 | 8.2% | 18.4% | 1 | 6 | -5 |
| summary | 45 | 0.0% | 0.0% | 0 | 0 | +0 |
| select_table | 29 | 17.2% | 37.9% | 1 | 7 | -6 |
| set_background_color | 25 | 0.0% | 0.0% | 0 | 0 | +0 |
| select_table_range | 20 | 0.0% | 0.0% | 0 | 0 | +0 |
| save_as | 13 | 0.0% | 0.0% | 0 | 0 | +0 |
| insert_excel_table | 5 | 0.0% | 0.0% | 0 | 0 | +0 |
| set_focus | 5 | 0.0% | 0.0% | 0 | 0 | +0 |
| table2markdown | 3 | 0.0% | 0.0% | 0 | 0 | +0 |
| run_shell | 1 | 0.0% | 0.0% | 0 | 0 | +0 |
| Spinner | 1 | 0.0% | 0.0% | 0 | 0 | +0 |

## 6b. Structured Operations: Where Coop Excels

Coop significantly outperforms SVD on API-style structured operations (not simple click/type):

| Function | N | SVD% | Coop% | Delta | Coop-only | SVD-only |
|----------|---|------|-------|-------|-----------|----------|
| set_font | 96 | 36.5% | 54.2% | **+17.7%** | 22 | 5 |
| select_table | 29 | 17.2% | 37.9% | **+20.7%** | 7 | 1 |
| select_paragraph | 49 | 8.2% | 18.4% | **+10.2%** | 6 | 1 |
| select_text | 496 | 44.0% | 49.6% | **+5.6%** | 51 | 23 |

**Why thought helps here**: These operations require choosing a non-obvious API function. SVD defaults to `click` or `type` (the dominant actions), while coop's thought reasoning identifies the correct specialized function. Examples:

- `select_table`: Coop thinks "I will use the 'select_table' shortcut API" → correct. SVD clicks on a random UI element.
- `set_font`: Coop reasons "I will increase the font size using the set_font API" → correct. SVD tries to type keyboard shortcuts.
- `select_paragraph`: Coop plans "I will use the 'select_paragraph' shortcut API to select the first paragraph" → correct. SVD tries Delete key.

However, these are rare (670 / 19046 = 3.5%), so their absolute impact is ~86 net coop-unique wins. Meanwhile, `click` alone contributes +227 net to SVD's advantage due to higher volume.

## 7. By Step Position (Early vs Late Steps)

### Coarse buckets

| Position | Total | SVD% | Coop% | Gap |
|----------|-------|------|-------|-----|
| first_step | 2549 | 52.4% | 51.5% | +0.9% |
| early_half | 7446 | 47.9% | 46.9% | +1.0% |
| late_half | 8821 | 46.0% | 44.4% | +1.6% |
| last_step | 191 | 33.5% | 28.3% | +5.2% |
| single_step | 39 | 30.8% | 15.4% | +15.4% |

### Fine-grained step index

| Step Index | N | SVD% | Coop% | Gap |
|------------|---|------|-------|-----|
| step 0 (first) | 2588 | 52.0% | 51.0% | +1.1% |
| step 1 | 2627 | 59.1% | 57.3% | +1.8% |
| step 2-4 | 5583 | 50.7% | 49.3% | +1.4% |
| step 5-9 | 4317 | 42.0% | 40.9% | +1.1% |
| step 10-19 | 3134 | 39.4% | 37.7% | +1.7% |
| step 20+ | 797 | 32.6% | 32.6% | +0.0% |

The gap is stable at ~1-2% across most positions, and **disappears at step 20+** (long action sequences). This suggests coop's thought reasoning may actually help maintain performance in very long sequences where SVD degrades equally.

## 7b. Coordinate Error Distance (When Both Wrong)

When both models predict the wrong coordinate, how close do they get to the target rectangle center?

| Metric | SVD | Coop |
|--------|-----|------|
| Mean distance (px) | 226.6 | 231.2 |
| Median distance (px) | 141.5 | 145.5 |
| Closer to target | 63.9% | **36.1%** |

N = 5898 samples where both models predicted wrong coordinates and both responses were parseable.

**Interpretation**: When coop fails at coordinates, it fails *harder* than SVD (farther from target). This aligns with the thought-misdirection pattern — a wrong thought can send the coordinate prediction to a completely different UI region (e.g., "click the Editing menu" → ribbon area instead of the correct cell), while SVD without thought tends to stay in the visual neighborhood of the target.

However, coop has **130 fewer coordinate errors overall** (from gap decomposition), meaning the thought mechanism helps MORE samples succeed at coordinates than it hurts. The net effect is positive despite the larger error magnitude on failures.

## 8. Thought Quality Analysis (Coop only)

| Category | Count | Mean Words | Median Words |
|----------|-------|------------|--------------|
| correct | 8774 | 32.2 | 33.0 |
| wrong_coord | 4959 | 32.1 | 33.0 |
| wrong_fn | 1819 | 32.5 | 33.0 |
| parse_fail | 2472 | 42.9 | 44.0 |

## 9. Status Prediction Analysis

| Model | Status Match | Total w/ Status | Match Rate |
|-------|-------------|-----------------|------------|
| Coop ep2 | 15864 | 16749 | 94.7% |
| SVD | 17955 | 18793 | 95.5% |

Top status errors (Coop):

- CONTINUE->FINISH: 778
- FINISH->CONTINUE: 107

Top status errors (SVD):

- CONTINUE->FINISH: 714
- FINISH->CONTINUE: 124

## 10. Representative Error Examples

### 10a. SVD-unique wins: Wrong Coordinate (Coop thought led to wrong location)

**word_search_word_4_20199_6**
- Request: Create a Venn diagram in the Word document.
- GT function: `click`, GT rect: {'left': 398, 'top': 220, 'right': 470, 'bottom': 276}
- Coop response: `<thought>To create a Venn diagram, I need to select the 'Relationship' option from the SmartArt Graphic gallery.</thought>
<tool_call>
{
    "function": "click",
    "args": {
        "button": "left",
        "double": false,
        "coordinate": [
            753,
            534
        ]
    },
    "status": "CONTINUE"
}
</tool_call>`
- SVD response: `<tool_call>
{
    "function": "click",
    "args": {
        "button": "left",
        "double": false,
        "coordinate": [
            407,
            245
        ]
    },
    "status": "CONTINUE"
}
</tool_call>`

**excel_search_excel_4_9907_7**
- Request: Trace the antecedents of data in the Excel spreadsheet by checking for any formulas or data connecti
- GT function: `click`, GT rect: {'left': 745, 'top': 273, 'right': 824, 'bottom': 291}
- Coop response: `<thought>Since there are no data connections listed in the 'Navrae en Verbindings' pane, I need to check for any formulas in the cells. I will use the 'Show Formulas' button to display all formulas in the spreadsheet.</thought>
<tool_call>
{
    "function": "click",
    "args": {
        "button": "left",
        "double": false,
        "coordinate": [
            972,
            246
        ]
 `
- SVD response: `<tool_call>
{
    "function": "click",
    "args": {
        "button": "left",
        "double": false,
        "coordinate": [
            807,
            279
        ]
    },
    "status": "CONTINUE"
}
</tool_call>`

**excel_search_excel_4_12018_2**
- Request: Select a specific range of cells in the Excel spreadsheet and use the 'Zoom to Selection' feature to
- GT function: `click`, GT rect: {'left': 354, 'top': 56, 'right': 392, 'bottom': 86}
- Coop response: `<thought>To proceed with the sub-task, I need to click the 'Zoom to Selection' button to zoom into the selected range of cells.</thought>
<tool_call>
{
    "function": "click",
    "args": {
        "button": "left",
        "double": false,
        "coordinate": [
            860,
            710
        ]
    },
    "status": "CONTINUE"
}
</tool_call>`
- SVD response: `<tool_call>
{
    "function": "click",
    "args": {
        "button": "left",
        "double": false,
        "coordinate": [
            372,
            70
        ]
    },
    "status": "CONTINUE"
}
</tool_call>`

### 10b. SVD-unique wins: Wrong Function

**word_search_word_4_13237_1**
- GT: `type`, Coop predicted: `click`
- Request: Add the word 'allegiant' to the Microsoft Word document.

**word_search_word_4_10054_1**
- GT: `select_text`, Coop predicted: `type`
- Request: Remove the highlighted text 'human animal intelligence' in the Word document.

**ppt_search_ppt_4s_883_7**
- GT: `click`, Coop predicted: `wheel_mouse_input`
- Request: Open the Version History for the current PowerPoint presentation to view past versions.

### 10c. SVD-unique wins: Parse/Format Failure

**excel_search_excel_4s_4122_14** (type: parse_failure)
- Coop response: `<thought>Since the dialog indicates 'No cells were found,' there are no formula cells in the current worksheet to review for invalid references. I should now check if there are any other sheets in the workbook that might contain formulas or errors. The next logical step is to switch to the 'in' shee`

**excel_search_excel_4s_17215_1** (type: parse_failure)
- Coop response: `<thought>To create a basic automotive maintenance checklist, I need to add four new columns to the left side of the existing data. The first step is to select an empty area to the left of the existing columns so that the new columns do not overwrite existing data. I will select cell A1 to start inse`

**excel_search_excel_4_5405_7** (type: empty_function)
- Coop response: `<tool_call>
{
    "function": "",
    "args": {
        "coordinate": [
            null,
            null
        ]
    },
    "status": "FINISH"
}
</tool_call>`

**word_search_word_4_6760_10** (type: empty_function)
- Coop response: `<tool_call>
{
    "function": "",
    "args": {
        "coordinate": [
            null,
            null
        ]
    },
    "status": "FINISH"
}
</tool_call>`

### 10d. Coop-unique wins: How Thought Helps

**excel_search_excel_4_18121_2**
- Request: Create a formula in Excel using the COUNTA function to maintain an updated total of entries in the '
- GT: `type`, rect: {'left': 475, 'top': 266, 'right': 538, 'bottom': 286}
- Coop (correct): `<thought>I need to input the formula '=COUNTA(A:A)' in the selected cell H4 to count all non-empty cells in the 'Name' column.</thought>
<tool_call>
{
    "function": "type",
    "args": {
        "text": "=COUNTA(A:A)",
        "coordinate": [
            505,
            273
        ]
    },
    "status": "CONTINUE"
}
</tool_call>`
- SVD (wrong coord): `<tool_call>
{
    "function": "type",
    "args": {
        "text": "=COUNTA(A:A)",
        "coordinate": [
            505,
            214
        ]
    },
    "status": "CONTINUE"
}
</tool_call>`

**excel_search_excel_4_1687_3**
- Request: Remove all hyperlinks from the Excel spreadsheet.
- GT: `click`, rect: {'left': 791, 'top': 88, 'right': 846, 'bottom': 184}
- Coop (correct): `<thought>Since the context menu does not show a 'Remove Hyperlinks' option, I should try clicking the 'Editing' menu item (label 64) in the ribbon, which may reveal more options related to editing and formatting, including hyperlink removal.</thought>
<tool_call>
{
    "function": "click",
    "args": {
        "button": "left",
        "double": false,
        "coordinate": [
            815,
   `
- SVD (wrong coord): `<tool_call>
{
    "function": "click",
    "args": {
        "button": "left",
        "double": false,
        "coordinate": [
            670,
            588
        ]
    },
    "status": "CONTINUE"
}
</tool_call>`

**excel_search_excel_4s_2153_2**
- Request: Highlight the last month of each quarter (March, June, September, December) in the Excel spreadsheet
- GT: `click`, rect: {'left': 180, 'top': 142, 'right': 204, 'bottom': 166}
- Coop (correct): `<thought>To highlight the selected cells, I need to open the Fill Color dropdown menu. This is typically done by clicking the Fill Color button in the Home tab. Once the dropdown is open, I can select a bright color for highlighting.</thought>
<tool_call>
{
    "function": "click",
    "args": {
        "button": "left",
        "double": false,
        "coordinate": [
            191,
           `
- SVD (wrong coord): `<tool_call>
{
    "function": "click",
    "args": {
        "button": "left",
        "double": false,
        "coordinate": [
            497,
            101
        ]
    },
    "status": "CONTINUE"
}
</tool_call>`

## 11. Epoch 1 → Epoch 2 Drift Analysis

- Stable correct (both epochs): 7724
- ep1-only correct (regressed in ep2): 723
- ep2-only correct (improved in ep2): 1058
- Net change: +335

Why ep1 correct samples regressed in ep2:

| Error Type | Count |
|------------|-------|
| wrong_coordinate | 377 |
| parse_failure | 136 |
| wrong_function | 110 |
| empty_function | 31 |
| wrong_coordinate+wrong_other_args | 14 |
| missing_arg_control_focus | 13 |
| missing_arg_text | 9 |
| missing_arg_clear_current_text | 9 |
| wrong_arg_text | 9 |
| wrong_arg_wheel_dist | 4 |
| wrong_arg_button | 3 |
| wrong_arg_keys | 3 |
| missing_arg_keys | 3 |
| wrong_arg_double | 1 |
| missing_arg_font_name | 1 |

## 12. Summary & Key Takeaways

### CORRECTION: `max_new_tokens=1024` Experiment Disproves Truncation Hypothesis

Re-running with `max_new_tokens=1024` (job 3615778) produced **identical results** to `max_new_tokens=512`:

| Shard | 512-tok | 1024-tok |
|-------|---------|----------|
| 0 | 2024 (42.5%) | 2024 (42.5%) |
| 1 | 2422 (50.9%) | 2422 (50.9%) |
| 2 | 2222 (46.7%) | 2222 (46.7%) |
| 3 | 2114 (44.4%) | 2114 (44.4%) |

The 1625 responses that appeared "truncated" (no `</tool_call>` in 500-char saved response) actually had complete, valid JSON — the `</tool_call>` tag was at character position 500-800, cut off only by our **save truncation** (`response[:500]`), not by model token budget.

Re-examining with 1000-char saves confirms: **1568/1625** "truncated" responses have valid `</tool_call>` and well-formed JSON. They fail because the **action itself is wrong** (wrong coordinate, wrong function), not because of format.

### Corrected Gap Decomposition

Total gap: SVD 9036 - Coop 8782 = 254 samples (1.33%)

| Error Type | Net contribution to gap |
|------------|------------------------|
| **wrong_function** | **+171** |
| **wrong_coordinate** | **+42** |
| missing_text | +19 |
| empty_function | +18 |
| wrong_keys | +16 |
| no_parseable_action | +12 |
| wrong_coord+other | +6 |
| wrong_wheel_dist | +4 |
| missing_clear_current_text | +3 |
| wrong_button | +2 |
| wrong_text | +1 |
| wrong_font_name | +1 |
| wrong_clear_current_text | +0 |
| missing_control_focus | -1 |
| wrong_font_size | -1 |
| missing_pressed | -2 |
| wrong_rows | -3 |
| malformed_json | -6 |
| missing_font_name | -10 |
| missing_keys | -18 |

### Corrected Coop Failure Classification (with 1000-char responses)

| Error Type | Count | % of Failures |
|------------|-------|---------------|
| wrong_coordinate | 5280 | 51.4% |
| wrong_function | 2362 | 23.0% |
| empty_function | 825 | 8.0% |
| wrong_text | 550 | 5.4% |
| wrong_coord+other | 469 | 4.6% |
| no_parseable_action | 234 | 2.3% |
| others | 524 | 5.1% |

### The Real Story: Thought Helps Localization but Hurts Function Selection

The gap is **not** a format problem. It is a genuine reasoning problem with two components:

1. **wrong_function (+171)**: The dominant gap source. Coop's thought reasoning sometimes selects the wrong API function. The thought narrates a plausible but incorrect plan (e.g., "I will use set_font" when the GT is "click"). SVD, without thought, relies on the visual pattern alone and picks the right function more often for common operations (click, type).

2. **wrong_coordinate (+42)**: Coop also has slightly more coordinate errors on net. The thought can misdirect coordinate prediction to a completely different UI region (see Section 7b: when both models are wrong, SVD is closer to target 63.9% of the time).

3. **Thought still helps on structured operations**: Section 6b shows Coop outperforms SVD on set_font (+17.7pp), select_table (+20.7pp), select_paragraph (+10.2pp). But these are rare (3.5% of samples) and don't offset the function selection errors on click/type (96.5% of samples).

### Root Cause: Thought-Induced Function Distribution Shift

Analysis of 2362 wrong_function cases by thought content:

| Category | Count | % |
|----------|-------|---|
| thought_no_function_mention | 1771 | 75.0% |
| thought_caused_error | 547 | 23.2% |
| thought_correct_model_ignored | 26 | 1.1% |
| thought_ambiguous_both | 18 | 0.8% |

**75% of wrong_function errors occur when the thought doesn't mention any function name.** The thought describes intent ("to remove the header content") but not function choice. The model makes a wrong function selection independently.

**Function confusion matrix reveals systematic bias — Coop over-predicts structured APIs:**

| GT → Pred | Coop | SVD | Diff | Interpretation |
|-----------|------|-----|------|----------------|
| click → select_text | 112 | 33 | **+79** | Over-use of structured API |
| click → wheel_mouse_input | 95 | 24 | **+71** | Over-use of structured API |
| click → select_table_range | 71 | 25 | **+46** | Over-use of structured API |
| type → select_text | 61 | 17 | **+44** | Over-use of structured API |
| click → drag | 43 | 5 | **+38** | Over-use of structured API |
| type → click | 580 | 492 | +88 | Bidirectional confusion |
| click → type | 749 | 673 | +76 | Bidirectional confusion |
| select_text → click | 81 | 94 | -13 | SVD over-predicts click |
| wheel → click | 127 | 157 | -30 | SVD over-predicts click |

**Diagnosis**: Coop's training data (with thought augmentation) contains structured API calls (select_text, select_table, set_font) at higher prominence than the eval distribution. This shifts LoRA_A's function prior — the model over-calls structured APIs, improving recall on those rare functions (§6b) but hurting precision on the dominant click/type operations.

This is a classic **precision-recall tradeoff**: Coop's select_text recall improves from 44.0% to 49.6%, but at the cost of 79 false positives (click→select_text). The net effect is negative because click is 28× more frequent than select_text (14467 vs 496).

### Key Findings

1. **The gap is a reasoning problem, not format**: wrong_function (+171) and wrong_coordinate (+42) are the true gap sources. Truncation was a measurement artifact.
2. **Thought is a double-edged sword**: Helps on 8.0% of samples (coop-unique wins), hurts on 9.4% (SVD-unique wins). Net negative by 1.3%.
3. **Thought helps structured operations**: set_font, select_table, select_paragraph significantly improved. These are cases where reasoning about the correct API is genuinely helpful.
4. **Thought hurts common operations**: For click/type (93.8% of data), thought adds a reasoning step that sometimes misdirects the model. The thought narrates a plan that sounds correct but targets the wrong UI element.
5. **Late steps hurt more**: Gap widens from +0.9% (first step) to +5.2% (last step). Thought-based reasoning may accumulate errors with longer action histories.
6. **Domain uniformity**: The gap is consistent across Word (+87), Excel (+83), PPT (+84).
7. **Complementarity**: Oracle ensemble reaches 55.5% vs SVD 47.4% / Coop 46.1%. The models solve genuinely different subsets.
8. **Degenerate outputs**: 825 empty-function responses (8.0% of failures) represent cases where the model outputs `{"function": "", "status": "FINISH"}`. These are not format errors but model uncertainty manifesting as premature termination.
9. **Status prediction**: Both models ~95% status accuracy. Not a differentiating factor.
10. **Epoch drift**: ep1→ep2 gained 1058, lost 723. Net +335 improvement, but with significant instability.

---

## 13. Cooperative LoRA v4: 3-Agent Architecture (V, T, A)

### 13.1 Motivation

The root cause of v3's 1.3% gap is **thought-induced function distribution shift** (§12): training LoRA_A on `<thought>` tokens causes it to over-predict structured APIs (click→select_text +79, click→wheel_mouse_input +71). 75% of wrong_function cases have thoughts that don't even mention a function name — the contamination is in LoRA_A's learned prior, not from explicit thought misdirection.

**Solution**: Split 2-agent (V, A) into 3-agent (V, T, A) to isolate the thought training signal:
- **LoRA_V**: Image tokens (binding-optimized) — unchanged
- **LoRA_T**: Thought tokens (reasoning-optimized) — NEW, absorbs thought training signal
- **LoRA_A**: Instruction + action tokens (action-optimized) — no longer trained on thought tokens

Communication happens naturally through causal attention: LoRA_A tokens attend to LoRA_T and LoRA_V tokens' keys/values.

### 13.2 Code Locations

| File | Role | Changes for v4 |
|------|------|----------------|
| `verl/models/cooperative/cooperative_lora.py` | Per-layer LoRA linear module | Added `num_agents` param, `lora_A_t`/`lora_B_t` params, 3-way `torch.where` routing with int8 mask (0=A, 1=V, 2=T) |
| `verl/models/cooperative/cooperative_wrapper.py` | Model wrapper: mask creation, save/load, generation | Added `_build_3way_mask()` for thought span detection via bigram patterns, per-batch-element thought state machine for generation, `lora_t.pt` save/load |
| `train_cooperative.py` | Training script | Added `--num_agents` flag (choices=[2,3]), passed to wrapper, saved in experiment config |
| `evaluation/eval_cooperative_batch.py` | Batch evaluation | Auto-detects `num_agents` from `cooperative_config.json` or `lora_t.pt` existence |
| `scripts/exp_cooperative/train_thought_v4_3agent.slurm` | Training launch (full 2 epochs) | 4 nodes × 4 GPUs, `--num_agents 3` |
| `scripts/exp_cooperative/train_thought_v4_3agent_ep2.slurm` | Epoch 2 resume | Resumes from epoch-1 checkpoint |
| `scripts/exp_cooperative/eval_thought_v4_ep1_batch.slurm` | Eval launch | 1 node × 4 GPUs, 4 shards, batch_size=32 |

### 13.3 Key Technical Details

**Thought token detection via bigram**: `<thought>` is NOT a special token in Qwen2.5-VL. It tokenizes as two regular tokens:
- `<thought>` → `(13708, 2450)` — defined as `THOUGHT_OPEN_BIGRAM`
- `</thought>` → `(522, 60565)` — defined as `THOUGHT_CLOSE_BIGRAM`

Training mask (`_build_3way_mask`): scans each batch element for open/close bigrams, marks all tokens inside thought spans as mask=2. Image tokens always mask=1 regardless of thought context.

**Generation thought state machine**: During autoregressive decode, can't scan full sequence each step. Instead, per-batch-element tracking with tensors `_in_thought: [B] bool` and `_last_token: [B] int64`. On each decode step, vectorized bigram check updates each element independently.

**Backward compatibility**: `num_agents=2` preserves exact v3 behavior — bool mask, no thought detection, no LoRA_T parameters.

### 13.4 Training Configuration Comparison

| Config | V3 (2-agent) | V4 (3-agent) |
|--------|-------------|-------------|
| Adapters | LoRA_V + LoRA_A | LoRA_V + LoRA_T + LoRA_A |
| Trainable params | 1,291,845,632 (~1.3B) | 1,937,768,448 (~1.9B) |
| LoRA rank | 256 | 256 |
| LoRA alpha | 512 | 512 |
| Target modules | q/k/v/o_proj + gate/up/down_proj (7) | same |
| Learning rate | 1e-5 | 1e-5 |
| Effective batch size | 16 × 1 × 8 = 128 | 16 × 1 × 8 = 128 |
| Epochs | 2 | 2 |
| Training data | gui360_train_thought.jsonl (97,647) | same |
| Infrastructure | 4 nodes × 4 GPUs (A100-80GB) | same |

### 13.5 Training Loss Trajectory (V4 3-Agent)

**Epoch 1** (job 3616527, completed in ~2h21m before preemption):

| Epoch | Loss | CE Loss | Notes |
|-------|------|---------|-------|
| 0.01 | 10.76 | 1.35 | Initial |
| 0.05 | 3.84 | 0.45 | Warmup ending |
| 0.10 | 3.09 | 0.35 | Rapid descent |
| 0.50 | 2.50 | 0.30 | Steady convergence |
| 1.00 | 2.24 | 0.28 | Epoch 1 end |
| 1.05 | 2.09 | 0.25 | (killed by SIGTERM/preemption) |

Epoch-1 checkpoint saved successfully at `train_GUI_360/llamafactory/output/cooperative_thought_v4_3agent/epoch-1/`.

**Epoch 2** (job 3618089, resumed from epoch-1 checkpoint):

| Epoch | Loss | CE Loss | Notes |
|-------|------|---------|-------|
| 0.0 | ~2.1 | ~0.25 | Start (from ep1 weights) |
| 0.50 | 2.04 | 0.24 | Midpoint |
| 0.66 | 1.96 | 0.24 | Latest (still running) |

### 13.6 Eval: V4 Epoch-1 — Batch Generation Bug

**Bug discovered**: The initial v4 eval (job 3618128) showed catastrophically low success rates (~2% on some shards vs v3's ~45%), with 3× slower inference speed.

| Shard | V3 ep2 rate | V4 ep1 rate (buggy) | V3 speed | V4 speed |
|-------|-------------|---------------------|----------|----------|
| 0 | ~42.5% | 2.3% | 1.4 s/s | 0.4 s/s |
| 1 | ~50.9% | 11.5% | — | 0.5 s/s |
| 2 | ~46.7% | 12.5% | — | 0.5 s/s |
| 3 | ~44.4% | 1.9% | — | 0.4 s/s |

**Root cause**: The thought state machine used **scalar** `_in_thought_block` and `_last_token_id` shared across all batch elements. With batch_size=32, when sample A enters `<thought>` but sample B hasn't yet, B's tokens are incorrectly routed to LoRA_T. This causes massive token misrouting — most action tokens get sent to the wrong adapter.

This bug does NOT affect:
- V3 eval (2-agent uses `mask = (ids == IMAGE_PAD_ID)`, no thought state)
- V4 training (uses `_build_3way_mask` which scans each batch element independently)

**Fix**: Changed thought state to per-batch-element tensors:
- `_in_thought_block` (scalar bool) → `_in_thought` (tensor `[B]` bool)
- `_last_token_id` (scalar int) → `_last_token` (tensor `[B]` int64)
- `_update_thought_state`: vectorized bigram check per batch element
- `_init_thought_state_from_prefill`: independent scan per batch element

Fixed eval resubmitted as job 3628687 (pending).

### 13.7 Expected Outcome

If the 3-agent separation works as designed:
- LoRA_A no longer sees thought tokens during training → its function prior should match SVD's distribution (no select_text over-prediction)
- LoRA_T absorbs the reasoning signal independently
- Net result: thought reasoning benefit without function contamination

| Model | Config | Trainable | Expected AP |
|-------|--------|-----------|-------------|
| SVD LoRA r=256 | 1 adapter | 646M | 47.4% |
| Coop v3 ep2 (2-agent) | 2 adapters | 1.3B | 46.1% |
| Coop v4 (3-agent) | 3 adapters | 1.9B | ≥47.4% (target) |

