# Correction 007: Empty AndroidControl Instruction

Date: 2026-08-12

Timing: after finite-coordinate correction commit `99bc7640cfe1cfc36bf4f18d1b97ccc4d4a9c5d2` and smoke authorization commit `d289c1e`, before any real private-label fold was opened, smoke result was written, or training began.

The second authorized smoke consumed its nonce and passed coordinate validation, then failed during public-only metadata validation for `androidcontrol/low/ac_1874`. Its frozen instruction is the empty string.

The assembly audit had incorrectly required every string metadata field to be nonempty. Public identity and file provenance require nonempty sample key, row ID, group, and image path. Task instruction is content, not identity, and the frozen AndroidControl bank contains three empty instructions. VUS histories also legitimately contain empty lists.

The correction requires instruction to be a string but permits it to be empty. It preserves nonempty identity/path checks, exact history types, SHA-256 validation, and exact row schemas. A synthetic Android public-row test now covers empty instruction acceptance.

The failed smoke opened no private-label fold, produced no result, and computed no private metric. Its authorization receipt is retained and cannot be replayed. PID 2274 was not altered.