# Correction 005: ScreenSpot Pixel Round Trip

Date: 2026-08-12

Timing: after private-scale seal commit `07130de20e81989946359989d6911fb78d682e13` and fallback-context authorization commit `bae078cfd0f9c5e9d2c450046bcea08007f7f086`, before any fallback-context output directory, performance metric, feature assembly, or model fit.

Formal context generation consumed its one-time authorization and then stopped at the frozen final-index anchor for outer fold 2, sample `screenspot_pro/C_self/powerpoint_windows_24`. The reconstructed fallback index differed from the previously audited VUS-SR index.

The public VUS bank stores ScreenSpot points normalized by image width/height, while CEV uses the original pixel coordinate and an exact 14-pixel threshold. Multiplying the serialized normalized values by image size reconstructed values such as `838.0000000000001` rather than the original `838.0`. This floating-point round trip can change a complete-link equivalence exactly at the hard threshold.

A result-free structural audit verified all 75,888 public ScreenSpot candidate slots against the frozen raw slot bank after integer rounding. Every slot matched exactly. All 151,776 scalar coordinates lay within `1e-9` of the integer pixel grid; the maximum residual was `4.547473508864641e-13`.

The correction restores ScreenSpot public coordinates as `round(normalized * image_size)` before CEV selection. Mind2Web continuous normalized coordinates are unchanged. The 14,644 final VUS index anchors remain the full-run execution gate.

The failed transaction published no context directory. Its authorization receipt was created and the nonce cannot be replayed. PID 2274 was not altered.