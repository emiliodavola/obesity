# Archive: 2026-06-23-04_ultimate_lean_model

## Overview
This archive contains the documentation for the successful implementation of the `04_ultimate_lean_model`, which identified that feature parsimony is critical for TabPFN calibration in the obesity dataset.

## Artifacts Archived
- Design: `openspec/designs/04_ultimate_lean_model.md`
- Specification: `openspec/specs/04_ultimate_lean_model.md`
- Implementation: `openspec/apply-progress/04_ultimate_lean_model.md`

## Experimental Summary
The project successfully transitioned from a high-complexity baseline to a minimalist, high-performance model by removing "density noise" (medical complexity features) while retaining demographic and structural care signals.

| Metric | Baseline (01) | Final (04) | Delta |
| :--- | :--- | :--- | :--- |
| ROC-AUC | 0.7604 | 0.9260 | +0.1656 |
| Logloss | 0.5978 | 0.3764 | -0.2214 |

## Historical Note (Integrity Correction)
During the transition from `02_feature_selection` to `03_medical_complexity`, a mismatch occurred where the `sdd-spec` phase erroneously targeted the previous change (`03`) while the `sdd-apply` phase correctly implemented `04`. This was resolved via manual re-specification and verification in the subsequent phase, restoring the SDD chain integrity.
