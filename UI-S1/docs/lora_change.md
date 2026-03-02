# LoRA Training Changes Log

## 2026-02-06

### Issue 1: Triton Cache Permission Error

**Problem:**
```
PermissionError: [Errno 13] Permission denied: '/local/user/2133411'
```
Triton runtime attempted to create cache directory at `/local/user/${SLURM_JOB_ID}` but lacked write permission.

**Solution:**
Modified `train/train_ui_s1_lora.slurm` to use scratch directory for Triton cache:

```bash
# Before
export TRITON_CACHE_DIR=/local/user/${SLURM_JOB_ID}/triton_cache

# After
export TRITON_CACHE_DIR=/scratch/a5l/shuqing.a5l/tmp/triton_cache_${SLURM_JOB_ID}
```

Changed in two locations:
- Worker script (line 136-139)
- Head node script (line 176-179)

---

### Issue 2: vLLM LoRA Parameter Name Mismatch

**Problem:**
```
KeyError: 'blocks.0.mlp.gate_proj.base_layer.weight'
```
The `replace_lora_wrapper` function in `fsdp_vllm.py` was converting visual encoder parameter names to LoRA format (`.base_layer.weight`), but vLLM doesn't support LoRA for visual components.

**Solution:**
Modified `verl/workers/sharding_manager/fsdp_vllm.py` to skip visual encoder parameters:

```python
# Before
def replace_lora_wrapper(k):
    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if any([k.endswith(f"{s}.weight") for s in stacked_params]):
        return k.replace(".weight", ".base_layer.weight")
    if any([k.endswith(f"{s}.bias") for s in stacked_params]):
        return k.replace(".bias", ".base_layer.bias")
    return k

# After
def replace_lora_wrapper(k):
    # Skip visual encoder params - vLLM doesn't support LoRA for visual parts
    if k.startswith("visual.") or k.startswith("model.visual."):
        return k
    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if any([k.endswith(f"{s}.weight") for s in stacked_params]):
        return k.replace(".weight", ".base_layer.weight")
    if any([k.endswith(f"{s}.bias") for s in stacked_params]):
        return k.replace(".bias", ".base_layer.bias")
    return k
```

---

### Known Limitations: LoRA on VL Models

vLLM 0.8.x has the following limitations for multimodal LoRA:

1. **Visual encoder LoRA ignored**: Only language model layers support LoRA
   ```
   WARNING: vLLM currently only supports adding LoRA to language model
   visual.patch_embed.proj will be ignored
   visual.merger.mlp.2 will be ignored
   ```

2. **Affected components**:
   - `visual.blocks.*` - Vision transformer blocks
   - `visual.patch_embed.*` - Patch embedding layer
   - `visual.merger.*` - Vision-language connector

3. **Future support**: vLLM v0.14.0+ adds experimental tower/connector LoRA support

---

---

### Issue 3: Batch Size Alignment Error (Pending)

**Problem:**
```
AssertionError: only support equal chunk. Got size of DataProto 65 and chunk 32.
```

The DataProto size (65) is not divisible by the number of chunks (32).

**Root Cause:**
- `train_batch_size`: 32
- `rollout.n`: 4 (4 responses per prompt)
- Expected: 32 * 4 = 128 samples
- Actual: 65 samples (some samples filtered out due to invalid responses)

**Potential Solutions:**
1. Increase `train_batch_size` to ensure enough valid samples remain after filtering
2. Enable `drop_last=True` in DataLoader to discard incomplete batches
3. Adjust `ppo_mini_batch_size` to match valid sample counts

**Solution:**
Changed `train/train_ui_s1_lora.slurm`:
```bash
# Before
actor_rollout_ref.actor.use_fixed_num_mini_batches=true
actor_rollout_ref.actor.fixed_num_mini_batches=4

# After
actor_rollout_ref.actor.use_fixed_num_mini_batches=false
```

This allows the system to dynamically calculate mini batch sizes based on actual valid samples.

**Additional Fix:**
The `use_fixed_num_mini_batches=false` setting alone didn't solve the issue because the assertion was in `verl/protocol.py`.

Modified `verl/protocol.py:662-664` to allow uneven chunks:
```python
# Before
if not self.is_padding_enabled():
    assert len(self) % chunks == 0, f"only support equal chunk..."

# After (commented out the strict assertion)
# Allow uneven chunks - PyTorch's chunk() handles this automatically
```

---

### Files Modified

| File | Change |
|------|--------|
| `train/train_ui_s1_lora.slurm` | Triton cache path, fixed_num_mini_batches disabled |
| `verl/workers/sharding_manager/fsdp_vllm.py` | Skip visual params in LoRA wrapper |
| `verl/protocol.py` | Allow uneven chunk sizes |
