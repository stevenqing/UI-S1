# Multi-Node Debug Change Log

## Session Started: 2026-01-31

This file tracks all modifications made during the debugging session.

---

## 2026-01-31 01:26 - Job 2081845 Test

**Status**: Tested NCCL_SOCKET_IFNAME=hsn fix

**Observations**:
- Ranks 0-3 (head node) initialized successfully
- Ranks 4-7 (remote node) stuck at `init_process_group`
- Same hang pattern as before

**Key Finding**: Network interfaces are named `hsn0`, `hsn1`, `hsn2`, `hsn3`, NOT just `hsn`

---

## 2026-01-31 01:32 - Updated NCCL Configuration

**File Modified**: `verl/workers/fsdp_workers.py` (lines 117-132)

**Changes**:
1. Changed `NCCL_SOCKET_IFNAME` from `hsn` to `hsn0` (the specific interface name)
2. Added `NCCL_DEBUG=INFO` to get detailed NCCL diagnostics
3. Added `NCCL_DEBUG_SUBSYS=INIT,NET` to focus on initialization and networking
4. Added `NCCL_IB_TIMEOUT=50` to increase timeout

**Job Submitted**: 2081848

---

## 2026-01-31 01:35-02:00 - Multiple Iterations

### Job 2081849 - hsn0 with NCCL_IB_DISABLE

**Changes**:
- Changed `NCCL_SOCKET_IFNAME` to `hsn0` (specific interface)
- Added `NCCL_IB_DISABLE=1` to force socket transport
- Added `NCCL_NET=Socket`

**Observations**:
- Ranks 4-7 (remote node) initialized successfully
- Ranks 0-3 (head node) stuck - REVERSED from before!
- Same pattern in subsequent jobs

### Root Cause Analysis

The issue appears to be an NCCL initialization hang where some ranks complete `init_process_group` but others don't. Key observations:

1. **TCP connectivity is working**: All ranks can connect to MASTER_ADDR:MASTER_PORT
2. **TCPStore creation works**: All 8 ranks successfully create/connect to TCPStore
3. **NCCL init hangs**: The actual NCCL communicator setup hangs

### Error Message (from Job 2081855)
```
[5] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: wait timeout after 300000ms
```

This indicates that ranks are waiting for the ncclUniqueId from rank 0, but it never arrives.

---

## Current State (2026-01-31 02:10)

**File Modified**: `verl/workers/fsdp_workers.py`

**Current Configuration**:
```python
os.environ["NCCL_SOCKET_IFNAME"] = "hsn"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
os.environ["NCCL_IB_TIMEOUT"] = "50"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_NET"] = "Socket"
```

**Changes Made**:
1. Added explicit TCPStore creation instead of `env://`
2. Added connectivity test before init
3. Added hostname and timeout to diagnostics
4. Added device_id parameter to init_process_group

**Status**: Still hanging at NCCL init_process_group

---

## Next Steps to Try

1. **Check NCCL logs more carefully** - Enable NCCL_DEBUG=TRACE
2. **Try different NCCL settings**:
   - `NCCL_P2P_LEVEL=LOC` or `NCCL_P2P_DISABLE=1`
   - `NCCL_SHM_DISABLE=1`
3. **Use file-based rendezvous** - Ensure VERL_RENDEZVOUS_FILE is propagated to workers
4. **Check if it's a PyTorch/NCCL version issue** - Try with different versions
5. **Create minimal reproduction script** - Test NCCL init outside of verl/Ray

---

