# Amendment 001: G2 P1 N8 Fallback

Date: 2026-08-02

Status: frozen after 1,176 label-free GTA1-72B proposal rows and before any G2 region-scoring inference or G2 accuracy result.

The preregistered P1 control has a primary budget of 12 and an allowed fallback budget of 8. On identity `stata_windows_27`, the official GTA1-72B attention proposer produced only seven unique crops from the top-100 attention positions. Thus P1 N12 is structurally undefined on the complete benchmark without candidate duplication, while P1 N8 remains exactly defined as the full image plus all seven unique crops.

This amendment activates the global P1 N8 fallback for every row. P2 remains three lineages by four shared regions at 12 forwards. The three proposal-perturbation MDE cells remain 12 forwards and require only three crops. No target field, correctness label, model output from region scoring, or aggregate accuracy was available when this decision was made.

The 1,176 existing label-free proposal rows are retained. Their derived `required_region_indices_by_model` fields are deterministically normalized before resume: GTA1 requires region indices 0-7 plus the cross-seed perturbation union; UI-Venus and Qwen3.5 require indices 0-3 plus the same union. Region coordinates, coverage values, official ranks, perturbation selections, and region hashes are unchanged.

The paper must label P1 as an 8-forward fallback and must not describe P2-P1 as an equal-budget comparison. The primary absolute 73.1 test and P2's 12-forward budget are unchanged.
