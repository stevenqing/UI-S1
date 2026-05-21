# Plan B: Route Coord Tokens → LoRA_V (AC Data Analysis)

## AC Assistant Response Structure

Each assistant turn has format:
```
<think>... long reasoning ...</think>
<action>{"action": "click", "coordinate": [540.0, 389.8], "bbox": [360, 327, 695, 466]}</action>
```

## Tokenization of Action Region

```
Position  Token ID    Text            Region
─────────────────────────────────────────────────
[255]     27          '<'             action_tag
[256]     1311        'action'        action_tag
[257]     88863       '>{"'           action_json
[258]     1311        'action'        action_json
[259]     788         '":'            action_json
[260]     330         ' "'            action_json
[261]     3678        'click'         action_type → LoRA_A
[262]     497         '",'            action_json
[263]     330         ' "'            action_json
[264]     62526       'coordinate'    COORD_KEY (trigger)
[265]     788         '":'
[266]     508         ' ['            COORD_START
[267]     20          '5'             COORD_DIGIT → LoRA_V
[268]     19          '4'             COORD_DIGIT → LoRA_V
[269]     15          '0'             COORD_DIGIT → LoRA_V
[270]     13          '.'             COORD_DOT   → LoRA_V
[271]     15          '0'             COORD_DIGIT → LoRA_V
[272]     11          ','             COORD_SEP   → LoRA_V
[273]     220         ' '             COORD_SEP   → LoRA_V
[274-290] digits...                   COORD_DIGIT → LoRA_V
[291]     1125        '],'            COORD_END
[292]     330         ' "'
[293]     58456       'bbox'          BBOX_KEY (trigger)
[294-313] digits...                   BBOX_DIGIT  → LoRA_V
[314]     81136       ']}</'          action_end
[315]     1311        'action'        action_tag
[316]     29          '>'             action_tag
```

## Key Token IDs for Mask Construction

| Token | ID | Role |
|-------|-----|------|
| `coordinate` | 62526 | Triggers coord region |
| `bbox` | 58456 | Triggers bbox region |
| `IMAGE_PAD` | 151655 | Image token |
| digits 0-9 | 15-24 | Coord values |
| `.` | 13 | Decimal point |
| `,` | 11 | Separator |
| ` ` | 220 | Space |
| ` [` | 508 | Array start |
| `],` | 1125 | Array end (mid-JSON) |
| `]}` | 81136 | Array end (end-JSON) |

Note: `coordinate2` (for swipe) tokenizes as `coordinate` (62526) + `2` (17)

## Token Budget Analysis

### Current v6.5 (LoRA_V gets 0% CE loss)
```
Total assistant tokens: 1,863,599 (100%)
  Think tokens:         1,570,425 (84.3%)  → LoRA_A
  Action non-coord:       165,274 (8.9%)   → LoRA_A
  Coord tokens:           127,900 (6.9%)   → LoRA_A  ← WRONG

LoRA_V CE loss: 0 tokens (0%)
```

### Plan B: coord + bbox → LoRA_V
```
Total assistant tokens: 1,863,599 (100%)
  Think tokens:         1,570,425 (84.3%)  → LoRA_A
  Action non-coord:       111,432 (6.0%)   → LoRA_A
  Coord tokens:           127,900 (6.9%)   → LoRA_V  ← FIXED
  Bbox tokens:             53,842 (2.9%)   → LoRA_V  ← FIXED

LoRA_V CE loss: 181,742 tokens (9.8% of assistant)
LoRA_A CE loss: 1,681,857 tokens (90.2% of assistant)
```

## Action Type Distribution

| Action type | Count | Has coordinates | Has bbox |
|-------------|-------|----------------|----------|
| click | 3,191 (51.8%) | Yes | Some (2,546) |
| swipe | 721 (11.7%) | Yes (2 coords) | No |
| long_press | 8 (0.1%) | Yes | No |
| terminate | 950 (15.4%) | No | No |
| type | 372 (6.0%) | No | No |
| open | 354 (5.7%) | No | No |
| wait | 352 (5.7%) | No | No |
| system_button | 217 (3.5%) | No | No |

63.6% of actions have coordinates → LoRA_V gets CE loss on majority of action turns.

## Mask Construction Algorithm

```python
def build_coord_aware_mask(input_ids):
    """Build mask where True = LoRA_V (image OR coord/bbox tokens)."""
    mask = (input_ids == IMAGE_PAD_ID)  # base: image tokens

    COORD_KEY = 62526   # 'coordinate'
    BBOX_KEY = 58456    # 'bbox'
    BRACKET_OPEN = 508  # ' ['
    DIGIT_IDS = set(range(15, 25))  # 0-9
    COORD_PUNCT = {13, 11, 220}     # '.', ',', ' '
    BRACKET_CLOSE_TOKENS = {1125, 81136}  # '],', ']}'

    for b in range(input_ids.shape[0]):
        in_coord = False
        for i in range(input_ids.shape[1]):
            tid = input_ids[b, i].item()

            if tid in (COORD_KEY, BBOX_KEY):
                in_coord = True
                continue

            if in_coord:
                if tid == BRACKET_OPEN:
                    continue  # skip the opening bracket
                elif tid in BRACKET_CLOSE_TOKENS:
                    in_coord = False
                    continue
                elif tid in DIGIT_IDS or tid in COORD_PUNCT:
                    mask[b, i] = True  # route to LoRA_V
                else:
                    in_coord = False  # unexpected token, exit

    return mask
```
