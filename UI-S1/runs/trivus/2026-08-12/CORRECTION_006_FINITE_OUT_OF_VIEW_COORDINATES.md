# Correction 006: Finite Out-of-View Coordinates

Date: 2026-08-12

Timing: after metric-free smoke implementation commit `b49b0e3974b441e27291f6380396c6e9c39682d7` and authorization commit `5cd77632b4053ea5c4908859152dd06626eee348`, before any real private-label fold was opened, smoke result was written, optimizer was constructed, or model training began.

The first authorized real-data smoke consumed its nonce, validated committed inputs, and then failed during public-only candidate validation. Frozen ScreenSpot-Pro sample `screenspot_pro/C_uni/blender_windows_31` contains candidate coordinate `[1.048046875, 0.24375]`.

The adapter had incorrectly imposed a `[0,1]` viewport range. The frozen VUS public producer does not clip ScreenSpot points: it divides the original pixel point by image dimensions and requires only finite values. CEV likewise retains the original pixel geometry. Clipping or rejecting finite out-of-view points would change pair distances and threshold equivalence.

A public-only structural audit found:

- Mind2Web: 186,138 coordinate scalars, all finite and within `[0,1]`;
- ScreenSpot-Pro: 151,776 scalars, all finite, 17 outside `[0,1]`, range `[0.0,1.05]`;
- AndroidControl: 16,780 scalars, all finite and within `[0,1]`.

The correction restores the exact frozen finite-only coordinate contract. Boolean, string, and non-finite coordinate scalars remain rejected. No clipping is introduced. A regression test accepts finite `1.05/-0.01` values and rejects infinity.

The failed smoke opened no private-label fold, produced no result, and computed no private metric. Its authorization receipt is retained and the nonce cannot be replayed. PID 2274 was not altered.