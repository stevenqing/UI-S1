#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/test_model_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/test_model_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CKPT_STEP="${CKPT_STEP:-80}"
MODEL_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/ac_full_sft/checkpoint-${CKPT_STEP}"
VLLM_PORT=8000

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

echo "Starting vLLM server for checkpoint-${CKPT_STEP}..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 &
VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

cd "$PROJECT_DIR"

echo ""
echo "========== Testing Model Output =========="
echo ""

python3 -c "
import json, sys, os, copy
sys.path.insert(0, '$PROJECT_DIR')
from x.data.agent.json import JsonFormat
from x.data.agent.space.std_space import RAW_SPACE
from evaluation.qwenvl_utils import call_mobile_agent_vllm, find_last_image_ele
from x.qwen.data_format import slim_messages

# Test both with and without thought
for thought_mode, add_thought in [('no_thought', False), ('with_thought', True)]:
    print(f'\n===== Mode: {thought_mode} =====')
    if add_thought:
        fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)
    else:
        fm = JsonFormat(RAW_SPACE, add_thought=False)

    with open('evaluation/dataset/android_control_evaluation_fixed.jsonl') as f:
        line = json.loads(f.readline())

    # Fix line
    for step in line['steps']:
        check_options = copy.deepcopy(step['action_content'])
        if 'bbox' in step:
            check_options['candidate_bbox'] = step['bbox']
        else:
            check_options['candidate_bbox'] = []
        step['check_options'] = check_options

    state = fm.gen_next_round(line, None, previous_model_response=None)
    if state is None:
        print('gen_next_round returned None')
        continue

    messages = state['messages']
    messages = slim_messages(messages=messages, num_image_limit=2)

    # Show system prompt snippet
    for m in messages:
        if m['role'] == 'system':
            content = m.get('content', '')
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        print(f'System prompt snippet: {item[\"text\"][:200]}...')
            elif isinstance(content, str):
                print(f'System prompt snippet: {content[:200]}...')
            break

    print('Calling vLLM...')
    model_response = call_mobile_agent_vllm(messages=messages, model_name='$MODEL_PATH')
    print(f'Model response (repr): {repr(model_response[:500])}')

    try:
        result = fm.parse_response(model_response)
        print(f'Parsed OK: action_content={json.dumps(result.get(\"action_content\"), default=str)}')
    except Exception as e:
        print(f'Parse ERROR: {e}')
"

echo ""
echo "========== Test Complete =========="

kill $VLLM_PID 2>/dev/null || true
