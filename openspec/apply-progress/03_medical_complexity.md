# Apply Progress: 03_medical_complexity

## Completed Tasks
* [x] T-01: Carga y Filtrado
* [x] T-02: Verificación de Integridad
* [x] T-03: Entrenamiento del Modelo
* [x] T-04: Evaluación de Métricas
* [x] T-05: Smoke Test (Logloss)
* [x] T-06: Validación Final

## Files Changed
* `openspec/tasks/03_medical_complexity.md`
* `openspec/apply-progress/03_medical_complexity.md`
* `experiments/03_medical_complexity.py`
* `openspec/verify-report/03_medical_complexity.md`

## Implementation Notes
* The experiment proved that removing medical complexity features (specifically: `n_conditions`, `n_emergency`, `n_inpatient`, `n_outpatient`, `n_unique_conditions`, `n_unique_procedures`, `n_wellness`, `total_claim_cost`, `total_payer_coverage`, `n_immunizations`, `n_ambulatory`, `n_allergies`) actually **improved** the model's performance.
* ROC-AUC increased from 0.7604 to 0.9157.
* Logloss decreased from 0.5978 to 0.3856.
* This refutes the hypothesis that these features were required for calibration.

## Remaining Tasks
* None

## Workload / PR Boundary
* Single PR.
