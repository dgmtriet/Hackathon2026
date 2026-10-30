# Hackathon2026
MATLAB Hackathon 2026 by CCEB ANATA, held on 30/10/2026

Team members:
1. Duong Minh Triet
2. Nguyen Dang Quang
3. Huynh Xuan Lam
4. Vuu Tan Loc
## Site EUI prediction pipeline (branch `lam_workings`)

Run `runAll` from the repo root in MATLAB. It runs three scripts and writes `submission.csv`.
The CSVs are read from `data/`.

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_prepare_data.m` | Cleans the data, engineers features, adds building history, splits years 1-5 / year 6 |
| 2 | `step2_train_models.m` | Compares baselines, Bagged Trees and Boosted Trees, tunes boosting, picks the final approach |
| 3 | `step3_make_submission.m` | Retrains the chosen approach on all six years and writes `submission.csv` |

Helpers: `trainForest.m`, `trainBoost.m`, `predictBoost.m`, `fillSentinel.m`.
`compare_app_models.m` (optional, after step 2) adds Fine Tree, Medium Tree and Neural Network to the comparison.

### Thinking flow

1. **Validate like the test set.** Test is a single future year (7) with no State_6 buildings,
   while State_6 is 67% of train. So we train on years 1-5, validate on year 6, and choose models
   on a "test-like" score that leaves State_6 out. A random split would be too optimistic.
2. **Domain features.** Flag missing Energy Star ratings, treat `year_built = 0` as missing,
   log floor area, summarise monthly temperatures into winter/summer averages and their range.
   Drop the wind and fog columns (88-94% missing in test) and `Year_Factor` (test year is out of range).
3. **Building history (the big win).** There is no building id, but a building keeps the same
   state, class, facility type, floor area and year built every year. Using that as a key, each row
   gets the building's Site EUI from strictly earlier years: `lag_last`, `lag_mean`, `lag_count`,
   `lag_gap`. 93.4% of test rows have history.
4. **Missing values for trees.** MATLAB trees without surrogate splits stop a NaN at the split
   node, so tree models see missing numbers as -999 (`fillSentinel.m`).
5. **Pick from the evidence.** Step 2 keeps whichever of Bagged Trees, tuned Boosted Trees or
   their average scores best on the test-like validation; step 3 retrains it on all years.

### Results (RMSE on year 6; test-like = without State_6)

| Model (Regression Learner name) | All states | Test-like |
|---|---|---|
| Average of Bagged Trees and Boosted Trees (final) | 31.92 | **25.99** |
| Boosted Trees, tuned | 33.60 | 26.35 |
| Bagged Trees (Random Forest) | 31.42 | 26.35 |
| Boosted Trees | 36.92 | 27.15 |
| Medium Tree | 36.72 | 28.79 |
| Fine Tree | 37.65 | 31.01 |
| Linear | 36.10 | 35.83 |
| Neural Network (one layer, 10 units) | 47.80 | 50.60 |
| Mean of training data | 57.21 | 64.88 |

Without the building-history features the best test-like RMSE was 46.84.
Most important predictors (boosted model): `lag_last`, `facility_type`, `lag_mean`, `energy_star_rating`.
