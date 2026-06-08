# Subtask Oracle Ablation: What Drives the 35.2% TSR?

## Background

Our SFT-272 model achieves 21.9% TSR on the balanced test set with the standard prompt. When we switch to a "subtask" prompt — which uses **GT thought as oracle step instruction** and **compressed history** — TSR jumps to **35.2%** (+13.3pp, +61% relative).

This document dissects exactly where this improvement comes from.

## The Two Variables

The subtask prompt differs from standard in two ways:

### 1. Oracle Step Instruction (GT Thought)

The standard prompt only tells the model the **overall task goal**:

```
The instruction is:
Create a new worksheet in Excel and set up a budget calculator...
```

The subtask prompt adds a **per-step oracle instruction** from the GT thought field:

```
Overall Task:
Create a new worksheet in Excel and set up a budget calculator...

Current Step Instruction:
To begin setting up the budget calculator, I need to first add a new
worksheet. The 'Add Sheet' button (label 76) is clearly visible...
```

This tells the model **exactly what to do at this step** — which UI element to target, why, and what the expected outcome is. This is oracle information not available at test time.

### 2. Compressed History Format

The standard prompt uses verbose action descriptions:

```
Step 1: click(coordinate=[205, 690], button='left', double=False, pressed=None)
Step 2: type(text='Budget Calculator', clear_current_text=False)
```

The subtask prompt uses brief compressed descriptions:

```
Step 1: click([205.0, 690.0])
Step 2: type('Budget Calculator')
```

## Ablation Design

To isolate each factor, we ran 4 experiments (all with pred history, SFT-272, balanced 1K test set):

| Experiment | Prompt Template | History Format | Oracle Thought |
|---|---|---|---|
| **standard** | Standard | Verbose | No |
| **standard_compressed** | Standard | Compressed | No |
| **subtask_verbose** | Subtask (oracle) | Verbose | Yes |
| **subtask** | Subtask (oracle) | Compressed | Yes |

## Results

### Overall Metrics

| Experiment | TSR | StepSR | Avg Progress |
|---|---|---|---|
| standard | 21.9% | 46.3% | 34.2% |
| standard_compressed | 21.7% | 47.1% | 34.6% |
| subtask_verbose | 34.9% | 66.5% | 50.7% |
| **subtask** | **35.2%** | **66.9%** | **50.9%** |

### By Task Length

| Length | standard | std_compressed | sub_verbose | subtask |
|---|---|---|---|---|
| 1 step | 72.3% | 63.9% | 69.7% | 69.7% |
| 2-3 steps | 33.3% | 31.8% | 51.8% | 51.8% |
| 4-5 steps | 25.3% | 23.7% | 43.9% | 43.4% |
| 6+ steps | 6.6% | 6.6% | 16.0% | 16.8% |

### StepSR by Task Length

| Length | standard | std_compressed | sub_verbose | subtask |
|---|---|---|---|---|
| 1 step | 72.3% | 63.9% | 69.7% | 69.7% |
| 2-3 steps | 56.4% | 57.6% | 77.1% | 76.5% |
| 4-5 steps | 62.1% | 63.1% | 78.7% | 78.0% |
| 6+ steps | 41.2% | 43.6% | 63.8% | 64.4% |

## Factor Attribution

### Compressed History: No Effect

```
standard_compressed (21.7%) - standard (21.9%) = -0.2pp
```

Compressed history alone provides **zero improvement**. The model's accuracy is not bottlenecked by history format verbosity. Whether the history says `click(coordinate=[205, 690], button='left', double=False, pressed=None)` or `click([205.0, 690.0])`, the model performs the same.

### Oracle Thought: The Dominant Driver

```
subtask_verbose (34.9%) - standard (21.9%) = +13.0pp
```

Oracle thought accounts for **97.7% of the total improvement** (13.0 out of 13.3pp). Providing per-step GT instructions fundamentally changes the task from "figure out what to do next in a multi-step plan" to "execute this single described action."

### Interaction Effect: Minimal

```
subtask (35.2%) - subtask_verbose (34.9%) = +0.3pp  (compressed hist on top of oracle)
subtask (35.2%) - standard_compressed (21.7%) = +13.5pp  (oracle on top of compressed hist)
Interaction = 35.2% - 34.9% - 21.7% + 21.9% = +0.5pp
```

The interaction between the two factors is negligible (+0.5pp). They are essentially independent, and compressed history contributes almost nothing in either setting.

## Why Oracle Thought Works So Well

### What GT Thought Contains

Each step's GT thought provides 3 types of oracle information:

1. **Action intent**: What should be done and why
   > *"To rename the worksheet, I need to double-click the 'Sheet1' tab to enable editing"*

2. **UI element identification**: Which element to interact with
   > *"The 'Add Sheet' button (label 76) is clearly visible"*

3. **Sequential context**: Where this step fits in the overall plan
   > *"After clicking the 'Spelling' button, the next expected UI is a suggestion panel"*

### The Core Mechanism

Without oracle thought, the model must:
1. Recall the full task goal
2. Infer what has been done from history
3. Determine what comes next in the plan
4. Identify the correct UI element
5. Choose the right action type and parameters

With oracle thought, steps 2-3 are eliminated — the model only needs to:
1. Read the step instruction
2. Identify the correct UI element
3. Execute the described action

This transforms a **planning + execution** task into a pure **execution** task. The planning burden — which compounds errors over long horizons — is offloaded to the oracle.

### Evidence: Improvement Scales with Task Length

| Task Length | Standard TSR | Subtask TSR | Absolute Gain | Relative Gain |
|---|---|---|---|---|
| 1 step | 72.3% | 69.7% | -2.6pp | -4% |
| 2-3 steps | 33.3% | 51.8% | +18.5pp | +56% |
| 4-5 steps | 25.3% | 43.4% | +18.1pp | +72% |
| 6+ steps | 6.6% | 16.8% | +10.2pp | +155% |

The improvement is largest on multi-step tasks (2-5 steps: +18pp), confirming that oracle thought helps by eliminating planning errors that accumulate over steps. On 1-step tasks (where no planning is needed), there is no improvement.

## Implications

### 1. Planning is the Bottleneck, Not Execution

The model can execute individual actions well (StepSR 66.9% with oracle instructions vs 46.3% without). The gap is in **deciding what to do**, not in **doing it**. This reframes the problem from "improve per-step accuracy" to "improve step-level planning."

### 2. Step-Level RL Failed for the Right Reason

All 6 RL experiments in V19 (GRPO R1-R3, RFT, AS-GRPO v2-v3) tried to improve per-step execution. But the model's execution ability is already near its ceiling — the real bottleneck is planning, which step-level RL cannot address.

### 3. Upper Bound Analysis

| Method | TSR | Gap to Oracle |
|---|---|---|
| Standard (baseline) | 21.9% | 13.3pp below |
| Type Focused (best prompt) | 23.6% | 11.6pp below |
| BoN-5 LogProb (best overall) | 26.6% | 8.6pp below |
| **Subtask Oracle** | **35.2%** | **— (upper bound)** |

Even our best method (BoN-5, 26.6%) is 8.6pp below the oracle upper bound. There is substantial room for improvement through better planning/decomposition.

### 4. Potential Directions

Since the bottleneck is planning (knowing what to do next), promising directions include:

- **Task decomposition models**: Train a separate model to generate step-level instructions from the overall goal + current screenshot, then feed these to the action model
- **Hierarchical agents**: High-level planner generates subtask descriptions, low-level executor takes actions
- **Plan-then-act prompting**: First generate a multi-step plan, then execute step by step (but this requires the plan to adapt to actual screen state)
- **DAgger with reasoning**: The previous DAgger attempt failed because GT responses lacked reasoning. A fixed version could use the GT thought as the reasoning prefix + GT action, training the model to both plan and execute

## Summary

**The 35.2% TSR improvement is almost entirely (97.7%) driven by providing oracle GT thought as step-level instructions.** Compressed history format contributes nothing. This reveals that the fundamental bottleneck in long-horizon GUI navigation is **planning** (deciding what to do next), not **execution** (performing the action). The model already executes well given clear instructions — it just cannot figure out what to do on its own over multi-step trajectories.
