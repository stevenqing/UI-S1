# OTEXT Amendment 002: Stage-1 control semantics

Date: 2026-08-14
Timing: before OCR completion, Stage-0 results, selected parameters, or any outer-test access.

O-K3 and O-K4 are evaluated under both mandatory baseline origins. Arm O minus random-gate and Arm O minus blind-match must each have a strictly positive 99% CI lower bound for both majority-origin and dev-selection-origin outputs. If either baseline-origin comparison includes zero, the corresponding kill condition triggers.

Bootstrap seed offsets from base `20260814` are frozen as: O-P1 majority 101, O-P1 dev-selection 102; O-P2 density-vs-majority/dev-selection 201/202; O-P2 availability-majority-vs-majority/dev-selection 203/204; random controls 301/302; blind controls 401/402. All use identical fold/application grouped paired sampling code.

RapidOCR reports the same quantities with offset +1000 but cannot alter EasyOCR primary conditions.
