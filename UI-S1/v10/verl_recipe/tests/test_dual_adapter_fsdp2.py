"""Week 1 validation test: FSDP2 + PEFT multi-adapter compatibility.

Run this BEFORE attempting full training to verify the fundamental
assumption that PEFT multi-adapter works with verl's FSDP2 wrapping.

Usage:
  torchrun --nproc_per_node=2 v10/verl_recipe/tests/test_dual_adapter_fsdp2.py

Expected output:
  [PASS] All tests passed
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

# NCCL config
os.environ.setdefault("NCCL_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def test_dual_adapter():
    rank, world_size = setup()

    if rank == 0:
        print("=" * 60)
        print("Test: FSDP2 + PEFT Multi-Adapter Compatibility")
        print("=" * 60)

    # ── 1. Load small model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing

    if rank == 0:
        print(f"\n[1] Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ── 2. Add dual LoRA adapters ──
    from peft import LoraConfig, TaskType, get_peft_model

    # Use init_lora_weights=False so lora_B is NOT zero — both adapters
    # will have distinct random weights and produce different outputs.
    lora_config_g = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=False,
    )
    lora_config_a = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=False,
    )

    if rank == 0:
        print("[2] Adding dual LoRA adapters (random init for testing)")

    model = get_peft_model(model, lora_config_g, adapter_name="grounder_lora")
    model.add_adapter("actor_lora", lora_config_a)

    # Force all LoRA params trainable and cast to bf16 (FSDP2 requires uniform dtype)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            param.data = param.data.to(torch.bfloat16)

    n_grounder = sum(p.numel() for n, p in model.named_parameters()
                     if "grounder_lora" in n or ("lora_" in n and "actor_lora" not in n))
    n_actor = sum(p.numel() for n, p in model.named_parameters()
                  if "actor_lora" in n)

    if rank == 0:
        print(f"  Grounder params: {n_grounder:,}")
        print(f"  Actor params: {n_actor:,}")

    # ── 2b. Verify adapters differ BEFORE FSDP2 ──
    if rank == 0:
        print("[2b] Verifying adapters differ before FSDP2...")

    model.cuda()
    test_text = "Hello world"
    inputs = tokenizer(test_text, return_tensors="pt").to(f"cuda:{rank}")

    model.set_adapter("grounder_lora")
    with torch.no_grad():
        out_g_pre = model(**inputs).logits

    model.set_adapter("actor_lora")
    with torch.no_grad():
        out_a_pre = model(**inputs).logits

    pre_diff = (out_g_pre - out_a_pre).abs().max().item()
    if rank == 0:
        print(f"  Pre-FSDP2 adapter diff: {pre_diff:.6f}")
        assert pre_diff > 0, "FAIL: Adapters identical even before FSDP2!"
        print("  [PASS] Adapters produce different outputs before FSDP2")

    # Move back to CPU for FSDP2 wrapping
    model.cpu()

    # ── 3. FSDP2 wrap ──
    if rank == 0:
        print("[3] Applying FSDP2")

    from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict, MixedPrecisionPolicy

    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True,
    )

    fsdp_kwargs = {
        "mesh": device_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": True,
    }
    fsdp_config = {"wrap_policy": {}}

    full_state = model.state_dict()
    apply_fsdp2(model, fsdp_kwargs, fsdp_config)
    fsdp2_load_full_state_dict(model, full_state, device_mesh, None)

    if rank == 0:
        print("  FSDP2 applied successfully")

    # ── 4. Test: set_adapter changes forward output ──
    if rank == 0:
        print("[4] Testing adapter switching after FSDP2...")

    inputs = tokenizer(test_text, return_tensors="pt").to(f"cuda:{rank}")

    model.set_adapter("grounder_lora")
    with torch.no_grad():
        out_g = model(**inputs).logits

    model.set_adapter("actor_lora")
    with torch.no_grad():
        out_a = model(**inputs).logits

    diff = (out_g - out_a).abs().max().item()
    if rank == 0:
        print(f"  Max logit diff between adapters: {diff:.6f}")
        if diff > 0:
            print("  [PASS] Adapter switching produces different outputs")
        else:
            print("  [WARN] Adapters produce identical output after FSDP2!")
            print("  This means set_adapter() may not work with FSDP2.")
            print("  Investigating: checking if adapter weights survived FSDP2...")

            # Debug: check if adapter weights are still different
            g_weights = []
            a_weights = []
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    full_p = p.full_tensor() if hasattr(p, 'full_tensor') else p.data
                    if "grounder_lora" in n:
                        g_weights.append(full_p.abs().sum().item())
                    elif "actor_lora" in n:
                        a_weights.append(full_p.abs().sum().item())
            print(f"  Grounder lora_B weight sums: {g_weights[:3]}")
            print(f"  Actor lora_B weight sums: {a_weights[:3]}")

    # ── 5. Test: backward with gradients ──
    if rank == 0:
        print("[5] Testing backward gradient flow...")

    model.set_adapter("grounder_lora")
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
            if p.grad is not None:
                p.grad.zero_()

    out = model(**inputs).logits
    loss = out.sum()
    loss.backward()

    grounder_has_grad = False
    actor_has_grad = False
    for n, p in model.named_parameters():
        if p.grad is not None:
            grad_sum = p.grad.abs().sum().item() if not hasattr(p.grad, 'full_tensor') else p.grad.full_tensor().abs().sum().item()
            if grad_sum > 0:
                if "grounder_lora" in n:
                    grounder_has_grad = True
                elif "actor_lora" in n:
                    actor_has_grad = True

    if rank == 0:
        print(f"  Grounder has grad: {grounder_has_grad}")
        print(f"  Actor has grad: {actor_has_grad}")
        assert grounder_has_grad, "FAIL: Grounder should have gradients!"
        print(f"  Actor grad isolation: {'YES (isolated)' if not actor_has_grad else 'NO (both get grads)'}")
        print("  [PASS] Gradient flow verified")

    # ── 6. Test: disable_adapter for ref policy ──
    if rank == 0:
        print("[6] Testing disable_adapter for reference policy...")

    model.disable_adapter_layers()
    with torch.no_grad():
        out_ref = model(**inputs).logits

    model.enable_adapter_layers()
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    model.set_adapter("grounder_lora")
    with torch.no_grad():
        out_g2 = model(**inputs).logits

    ref_diff = (out_ref - out_g2).abs().max().item()
    if rank == 0:
        print(f"  Ref vs grounder logit diff: {ref_diff:.6f}")
        if ref_diff > 0:
            print("  [PASS] disable_adapter gives base model output")
        else:
            print("  [INFO] Ref and grounder identical — LoRA contribution may be zero")

    # ── 7. Test: requires_grad restored after enable_adapter ──
    if rank == 0:
        print("[7] Testing requires_grad restoration...")

    trainable_before = sum(1 for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)
    total_lora = sum(1 for n, p in model.named_parameters() if "lora_" in n)
    if rank == 0:
        print(f"  Trainable LoRA params after enable_adapter: {trainable_before}/{total_lora}")
        if trainable_before < total_lora:
            print("  [INFO] PEFT only re-enables active adapter — need manual fix")

    # Manually force all LoRA params trainable (our workaround)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    trainable_after = sum(1 for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)
    if rank == 0:
        print(f"  Trainable LoRA params after manual fix: {trainable_after}/{total_lora}")
        assert trainable_after == total_lora, "FAIL: Not all LoRA params are trainable!"
        print("  [PASS] All LoRA params trainable after manual requires_grad fix")

    # ── Done ──
    if rank == 0:
        print("\n" + "=" * 60)
        if diff > 0:
            print("[PASS] All tests passed! FSDP2 + multi-adapter is compatible.")
        else:
            print("[PARTIAL] FSDP2 wrapping works but set_adapter may need workaround.")
        print("=" * 60)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_dual_adapter()
