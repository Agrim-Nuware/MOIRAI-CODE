import time

import torch

from uni2ts.loss.packed import PackedNLLLoss
from uni2ts.model.moirai_moe import MoiraiMoEModule

torch.manual_seed(0)

NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)

module = MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-base")
module.train()

print("patch_sizes:", module.patch_sizes, flush=True)
print("num params:", sum(p.numel() for p in module.parameters()), flush=True)

BATCH_SIZE = 16
NUM_VARIATES = 5  # Open, High, Low, Close, Volume
CONTEXT_LEN = 300
PRED_LEN = 20
PATCH_SIZE = 16
MAX_PATCH = max(module.patch_sizes)

num_patches_per_variate = (CONTEXT_LEN + PRED_LEN) // PATCH_SIZE
seq_len = NUM_VARIATES * num_patches_per_variate

print(f"threads={NUM_THREADS} batch_size={BATCH_SIZE}, num_variates={NUM_VARIATES}, seq_len={seq_len}, max_patch={MAX_PATCH}", flush=True)

target = torch.randn(BATCH_SIZE, seq_len, MAX_PATCH)
observed_mask = torch.ones(BATCH_SIZE, seq_len, MAX_PATCH, dtype=torch.bool)
sample_id = torch.ones(BATCH_SIZE, seq_len, dtype=torch.long)
time_id = torch.arange(num_patches_per_variate).repeat(NUM_VARIATES).unsqueeze(0).repeat(BATCH_SIZE, 1)
variate_id = torch.arange(NUM_VARIATES).repeat_interleave(num_patches_per_variate).unsqueeze(0).repeat(BATCH_SIZE, 1)
prediction_mask = torch.zeros(BATCH_SIZE, seq_len, dtype=torch.bool)
n_pred_patches = PRED_LEN // PATCH_SIZE
for v in range(NUM_VARIATES):
    start = v * num_patches_per_variate + (num_patches_per_variate - n_pred_patches)
    prediction_mask[:, start : start + n_pred_patches] = True
patch_size_t = torch.full((BATCH_SIZE, seq_len), PATCH_SIZE, dtype=torch.long)

loss_func = PackedNLLLoss()
optimizer = torch.optim.AdamW(module.parameters(), lr=1e-5)

def run_step():
    optimizer.zero_grad()
    distr = module(
        target=target,
        observed_mask=observed_mask,
        sample_id=sample_id,
        time_id=time_id,
        variate_id=variate_id,
        prediction_mask=prediction_mask,
        patch_size=patch_size_t,
    )
    loss = loss_func(
        pred=distr,
        target=target,
        prediction_mask=prediction_mask,
        observed_mask=observed_mask,
        sample_id=sample_id,
        variate_id=variate_id,
    )
    loss.backward()
    optimizer.step()
    return loss.item()

print("\nwarming up...", flush=True)
for i in range(2):
    t0 = time.time()
    loss_val = run_step()
    print(f"warmup step {i}: loss={loss_val:.4f} time={time.time()-t0:.2f}s", flush=True)

print("\ntimed steps...", flush=True)
times = []
for i in range(5):
    t0 = time.time()
    loss_val = run_step()
    dt = time.time() - t0
    times.append(dt)
    print(f"step {i}: loss={loss_val:.4f} time={dt:.2f}s", flush=True)

avg = sum(times) / len(times)
print(f"\naverage time per step (batch_size={BATCH_SIZE}, threads={NUM_THREADS}): {avg:.2f}s", flush=True)
for n_steps in [200, 500, 1000, 2000, 5000]:
    total_sec = avg * n_steps
    print(f"  {n_steps} steps -> {total_sec/60:.1f} min ({total_sec/3600:.2f} hours)", flush=True)
