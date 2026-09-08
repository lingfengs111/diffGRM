#!/usr/bin/env bash
set -euo pipefail
cd /home/lingfengs111/codes/GR_variant/DiffGRM
/home/lingfengs111/miniconda3/envs/diffgrm/bin/python - <<'PY'
import json, os, subprocess
from pathlib import Path
import numpy as np
root = Path('runs/history_interest_20260907/v1').resolve()
source = root / 'smoke/science23/interest2'
output = root / 'smoke_test/science23/interest2'
output.mkdir(parents=True, exist_ok=False)
command = json.loads((source / 'command.json').read_text())['argv']
command.remove('--validation-only')
command[command.index('--output-dir') + 1] = str(output)
command += ['--epochs', '0', '--init-trained-checkpoint', str(source / 'best.pt')]
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
(output / 'command.json').write_text(json.dumps(command, indent=2))
with (output / 'train.log').open('w') as log:
    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
result = json.loads((output / 'result.json').read_text())
alpha = result['selected_fusion_alpha']
ranks = np.load(output / 'test_ranks.npz')['fused_rank']
errors = {}
for k in (5, 10):
    for name, values in [('recall', (ranks <= k).astype(float)),
                         ('ndcg', np.where(ranks <= k, 1 / np.log2(ranks + 1), 0))]:
        key = f'fused_a{alpha:g}_{name}@{k}'.replace('.', 'p')
        errors[key] = abs(float(values.mean()) - result['test'][key])
assert max(errors.values()) < 1e-7, errors
(output / 'rank_metric_check.json').write_text(json.dumps(errors, indent=2))
print('Test-path checkpoint reload and NPZ metric reconstruction passed:', errors)
PY
