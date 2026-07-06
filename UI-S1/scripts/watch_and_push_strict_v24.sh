#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source .venv-qwen3-vllm/bin/activate

OUT="${OUT:-outputs/critstep_verifier_v2/strict}"
V24="${V24:-v24}"
REPO_ID="${REPO_ID:-Stevenshuqing/gui360-verifier-v24}"
LOG="${OUT}/logs/watch_and_push_v24.log"
mkdir -p "${OUT}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] waiting for strict summary: ${OUT}/combine/strict_summary.json"
while [[ ! -s "${OUT}/combine/strict_summary.json" || ! -s "${OUT}/combine/strict_eval.md" ]]; do
  if ! pgrep -af 'run_strict_verifier_overnight|score_critstep_verifier_stage2_comparative.py.*strict/stage2_test|combine_critstep_verifier_v2_strict.py' >/dev/null; then
    echo "No strict process is running and strict summary is missing." >&2
    exit 1
  fi
  date -u '+[%Y-%m-%d %H:%M:%S UTC] still waiting...'
  sleep 300
done

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] strict summary detected; syncing into ${V24}"
python - <<'PY'
import json, shutil
from pathlib import Path
from datetime import datetime, timezone
root=Path.cwd()
out=Path('outputs/critstep_verifier_v2/strict')
v24=Path('v24')
strict_dst=v24/'results/strict'
if strict_dst.exists():
    shutil.rmtree(strict_dst)
shutil.copytree(out, strict_dst, ignore=shutil.ignore_patterns('logs/*.log'))
for rel in [
    'scripts/build_strict_verifier_pools.py',
    'scripts/combine_critstep_verifier_v2_strict.py',
    'scripts/run_strict_verifier_overnight.sh',
    'scripts/watch_and_push_strict_v24.sh',
]:
    src=root/rel
    if src.exists():
        dst=v24/'repo_overlay'/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
manifest_path=v24/'manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
summary=json.loads((out/'combine/strict_summary.json').read_text(encoding='utf-8'))
manifest['strict_final_result']={
    'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    'path': 'results/strict',
    'gate': summary['gate'],
    'paper_method': summary['paper_method'],
    'paper_accuracy': summary['selection_accuracy'][summary['paper_method']],
    'best_method': summary['best_method'],
    'best_accuracy': summary['selection_accuracy'][summary['best_method']],
    'stage1_test_accuracy': summary['selection_accuracy']['stage1_cot_vote_k8'],
    'stage2_test_accuracy': summary['selection_accuracy']['stage2_tournament'],
    'train_steps': summary['n_train_steps'],
    'test_steps': summary['n_test_steps'],
    'episode_intersection_count': summary['episode_intersection_count'],
}
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps(manifest['strict_final_result'], indent=2, ensure_ascii=False))
PY

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] validating ${V24}"
./v24/validate_package.sh | sed -n '1,140p'

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] uploading ${V24} to ${REPO_ID}"
hf upload "${REPO_ID}" "${V24}" . --repo-type dataset --commit-message "Add strict train-test verifier aggregation results"

python - <<'PY'
from huggingface_hub import HfApi
repo_id='Stevenshuqing/gui360-verifier-v24'
api=HfApi()
info=api.repo_info(repo_id=repo_id, repo_type='dataset')
files=api.list_repo_files(repo_id=repo_id, repo_type='dataset')
print('repo', repo_id)
print('sha', info.sha)
print('files', len(files))
print('has strict_eval', 'results/strict/combine/strict_eval.md' in files)
PY

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] strict v24 push complete"