import json, glob
from collections import Counter, defaultdict

PROJECT = '/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1'

with open(f'{PROJECT}/v13_gui_360/data/gui360_test_968.jsonl') as f:
    test_eps = [json.loads(l) for l in f]
meta = {}
for ep in test_eps:
    eid = ep['episode_id']
    path = ep['steps'][0]['screenshot']
    parts = path.split('/')
    app = parts[parts.index('image') + 1]
    meta[eid] = {'app': app, 'num_steps': ep['num_steps'], 'goal': ep['goal']}

def find_latest(d):
    files = sorted(glob.glob(f'{d}/eval_results_*.json'))
    return files[-1] if files else None

def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return {ep['episode_id']: ep for ep in data['episodes']}

v13 = load_results(find_latest(f'{PROJECT}/v13_gui_360/outputs/epoch-3'))
grpo = load_results(find_latest(f'{PROJECT}/v12_gui_360/outputs/std_lora_grpo/epoch-3'))

v13_solved = set(eid for eid, ep in v13.items() if ep['tsr'] > 0.99)
grpo_solved = set(eid for eid, ep in grpo.items() if ep['tsr'] > 0.99)
grpo_only = grpo_solved - v13_solved

# ── 1. GRPO-only 27个step1错误: 按错误类型分类 ──
print('=' * 80)
print('GRPO-only V13 step1就错: 错误分类')
print('=' * 80)

type_mismatch = []
coord_wrong = []

for eid in grpo_only:
    if v13[eid]['progress'] > 0:
        continue
    sr = v13[eid]['step_results'][0]
    gt_type = sr.get('gt_type', '?')
    pred_type = sr.get('pred_type', '?')

    if pred_type != gt_type:
        type_mismatch.append(eid)
    else:
        coord_wrong.append(eid)

print(f'  动作类型错误 (V13 pred != GT): {len(type_mismatch)}')
print(f'  坐标错误 (类型对但位置错): {len(coord_wrong)}')

gt_vs_pred = Counter()
for eid in type_mismatch:
    sr = v13[eid]['step_results'][0]
    gt_vs_pred[(sr.get('gt_type','?'), sr.get('pred_type','?'))] += 1
print(f'\n  类型错误明细 (GT -> V13_pred):')
for (gt, pred), cnt in gt_vs_pred.most_common():
    print(f'    {gt} -> {pred}: {cnt}')

# ── 2. GRPO 在这些 type_mismatch 案例中的预测 ──
print()
print('=' * 80)
print('GRPO 在 type_mismatch 案例中的预测')
print('=' * 80)

for eid in type_mismatch:
    sr_v13 = v13[eid]['step_results'][0]
    sr_grpo = grpo[eid]['step_results'][0]
    gt_type = sr_v13.get('gt_type')
    m = meta[eid]
    print(f'  [{m["num_steps"]}s|{m["app"]}] GT={gt_type}, V13={sr_v13.get("pred_type")}, GRPO={sr_grpo.get("pred_type")}, GRPO_ok={sr_grpo["success"]}')

# ── 3. coord_wrong: V13 类型对了但坐标错 ──
print()
print('=' * 80)
print(f'坐标错误 ({len(coord_wrong)}): V13 类型对但位置错, GRPO做对')
print('=' * 80)

for eid in coord_wrong:
    sr_v13 = v13[eid]['step_results'][0]
    sr_grpo = grpo[eid]['step_results'][0]
    m = meta[eid]
    v13_coord = sr_v13.get('pred_action', {}).get('coordinate', '?')
    grpo_coord = sr_grpo.get('pred_action', {}).get('coordinate', '?')
    # episode_id might not match list index
    ep_data = next((e for e in test_eps if e['episode_id'] == eid), None)
    gt_action = ep_data['steps'][0]['action'] if ep_data else {}
    gt_coord = gt_action.get('coordinate', '?')

    print(f'  [{m["num_steps"]}s|{m["app"]}] {m["goal"][:65]}')
    print(f'    GT={gt_coord}, V13={v13_coord}(r={sr_v13.get("content_reward",0):.2f}), GRPO={grpo_coord}(r={sr_grpo.get("content_reward",0):.2f})')

# ── 4. 14个 partial progress 的分析 ──
print()
print('=' * 80)
print('GRPO-only 中 V13 有 partial progress 的 (14个)')
print('=' * 80)

for eid in sorted(grpo_only, key=lambda x: meta[x]['num_steps']):
    if v13[eid]['progress'] == 0:
        continue
    v13_ep = v13[eid]
    m = meta[eid]

    first_err = None
    err_type = '?'
    for sr in v13_ep['step_results']:
        if not sr['success']:
            first_err = sr['step_idx']
            gt_t = sr.get('gt_type', '?')
            pred_t = sr.get('pred_type', '?')
            if pred_t != gt_t:
                err_type = f'type({gt_t}->{pred_t})'
            else:
                err_type = f'coord(cr={sr.get("content_reward", 0):.2f})'
            break

    print(f'  [{m["num_steps"]}s|{m["app"]}] prog={v13_ep["progress"]:.0%}, err_step={first_err+1}/{m["num_steps"]}: {err_type}')
    print(f'    {m["goal"][:80]}')

# ── 5. 全局 type/swipe 能力对比 ──
print()
print('=' * 80)
print('全局 type/swipe 成功率: V13 vs StdLoRA+GRPO')
print('=' * 80)

all_ids = set(v13.keys()) & set(grpo.keys())
for label, ep_dict in [('V13+SP ep3', v13), ('StdLoRA+GRPO ep3', grpo)]:
    stats = defaultdict(lambda: [0,0])
    for eid in all_ids:
        for sr in ep_dict[eid]['step_results']:
            gt = sr.get('gt_type', '?')
            stats[gt][1] += 1
            if sr['success']:
                stats[gt][0] += 1
    print(f'\n{label}:')
    for t in ['click', 'type', 'swipe']:
        c, tot = stats[t]
        if tot > 0:
            print(f'  {t:>6}: {c:>4}/{tot:>4} = {c/tot*100:.1f}%')

# ── 6. type/swipe 能力随 epoch 变化 ──
print()
print('=' * 80)
print('Type/Swipe 能力随 epoch 变化')
print('=' * 80)

for method_name, method_dir in [
    ('V13+SP', '{PROJECT}/v13_gui_360/outputs/epoch-{ep}'),
    ('StdLoRA+GRPO', '{PROJECT}/v12_gui_360/outputs/std_lora_grpo/epoch-{ep}'),
    ('StdLoRA+SP', '{PROJECT}/v12_gui_360/outputs/std_lora_sp/epoch-{ep}'),
]:
    print(f'\n{method_name}:')
    for ep_idx in range(4):
        d = method_dir.format(PROJECT=PROJECT, ep=ep_idx)
        p = find_latest(d)
        if not p:
            continue
        ep_dict = load_results(p)
        stats = defaultdict(lambda: [0,0])
        for eid, ep in ep_dict.items():
            for sr in ep['step_results']:
                gt = sr.get('gt_type', '?')
                stats[gt][1] += 1
                if sr['success']:
                    stats[gt][0] += 1
        parts = []
        for t in ['click', 'type', 'swipe']:
            c, tot = stats[t]
            if tot > 0:
                parts.append(f'{t}={c}/{tot}({c/tot*100:.1f}%)')
            else:
                parts.append(f'{t}=N/A')
        print(f'  ep{ep_idx}: {"  ".join(parts)}')
