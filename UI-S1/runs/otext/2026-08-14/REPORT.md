# OTEXT OCR Validation Report

Date: 2026-08-14

Status: `OTEXT_STOPPED_O_K1_STAGE0`

## Evidence status

OTEXT is **post-selection validation**, not confirmatory evidence. ORTH used all 1,581 ScreenSpot-Pro labels to select the OCR/text direction. OTEXT preregisters and nests tuning, regenerates both OCR engines, and uses held-out folds, but a paper method claim still requires new untouched data.

## Stage 0

EasyOCR is the sole primary engine. Its weighted nested validation minimum gain across majority and nested dev-selection is +0.06 pp against the 0.70 pp gate. RapidOCR is replication only. Selected parameters and full inner-validation curves are retained in `SELECTED_PARAMETERS.json` and `STAGE0.json`.

Stage 1 was not run because EasyOCR failed O-G1.

## Boundaries

No `ui_type`, row class, GT overlap, or label-dependent statistic enters the runtime gate. Text/icon, gate-conditional accuracy, and conditional correctness remain evaluation-side. No existing project status changes, and failed settings cannot be rescued by retuning inside this round.
