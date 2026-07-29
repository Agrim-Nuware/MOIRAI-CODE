import sys
import torch

from uni2ts.loss.packed import PackedNLLLoss
from uni2ts.model.moirai_moe import MoiraiMoEModule

torch.manual_seed(0)
torch.set_num_threads(1)

module = MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-base")
module.train()
print("model loaded", flush=True)

BATCH_SIZE = 16
NUM_VARIATES = 5
CONTEXT_LEN = 300
PRED_LEN = 20
PATCH_SIZE = 16
MAX_PATCH = max(module.patch_sizes)

num_patches_per_variate = (CONTEXT_LEN + PRED_LEN) // PATCH_SIZE
seq_len = NUM_VARIATES * num_patches_per_variate

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

print("STEP 1: forward pass with grad disabled (no_grad)", flush=True)
with torch.no_grad():
    distr = module(
        target=target, observed_mask=observed_mask, sample_id=sample_id,
        time_id=time_id, variate_id=variate_id, prediction_mask=prediction_mask,
        patch_size=patch_size_t,
    )
print("STEP 1 OK", flush=True)

print("STEP 2: forward pass WITH grad enabled", flush=True)
distr = module(
    target=target, observed_mask=observed_mask, sample_id=sample_id,
    time_id=time_id, variate_id=variate_id, prediction_mask=prediction_mask,
    patch_size=patch_size_t,
)
print("STEP 2 OK", flush=True)

print("STEP 3: compute loss", flush=True)
loss_func = PackedNLLLoss()
loss = loss_func(
    pred=distr, target=target, prediction_mask=prediction_mask,
    observed_mask=observed_mask, sample_id=sample_id, variate_id=variate_id,
)
print(f"STEP 3 OK, loss={loss.item()}", flush=True)

print("STEP 4: backward pass", flush=True)
loss.backward()
print("STEP 4 OK", flush=True)

print("ALL STEPS PASSED", flush=True)
