# Version 4 — dashboard.pdf revisions

Implemented from the updated `/home/one/Downloads/dashboard.pdf` dated 2026-09-29. The working project is now **Version 4**.

## Changes and locations

| Tab | Revision |
|---|---|
| Overview | Restored green up/red down indicators beneath all five KPI cards, with QoQ and YoY comparisons. Unchanged values use a neutral dash; missing baselines show N/A. |
| Price Tier Performance | Removed the Welch's ANOVA & Games–Howell expander. The tier overview, grouped bars, progress table, and adoption heatmap remain. |
| 5G Market Penetration | Made the units-over-time chart full width and labeled every quarter in the selected Actual range. Removed the Welch's t-test p-value column from the comparison table; mean values, difference verdicts, and revenue contributions remain. |
| Trends and Forecast → Econometric Forecasting Engine | Restored the reference's quarterly timeline styling, retained dashed forecast values, and added a red dashed Forecast Horizon marker at the first projected quarter. No yellow marker was added. |
| Regional Conditions | Replaced technical r/p/n cell strings with plain-language results such as “Moderate positive link” and “No clear relationship,” with an explanation beneath the table. |
| Action Center / Data Explorer | Kept the Version 3 functions unchanged. |

The KPI headline values retain their existing scope. Indicator changes compare the **latest Actual quarter** against exact calendar-quarter baselines: adoption/market share use percentage points, ASP uses dollars, and revenue/units use percentage change. A missing or zero percentage-change baseline is unavailable, not zero growth.

Regional descriptions remain based on computed Pearson correlations. “Not enough data” covers insufficient or constant observations; “No clear relationship” covers p ≥ .05 or absolute r below .2. Other descriptions use weak (<.4), moderate (<.7), or strong (≥.7) absolute correlation and the sign of r. These are descriptive conventions, not evidence of causation.

## Backups and verification

- Version 1 reference remains `/tmp/Samsung_Dashboard_before_BI_LE1.py`.
- Version 2 code snapshot remains `/tmp/bebe_version2_before_dashboard_pdf.zip`.
- Version 3 project snapshot: `/tmp/bebe_version3_before_v4.zip` (project files, datasets, assets, and tests; excludes Git internals and the virtual environment).
- Version 3 dashboard alone: `/tmp/Samsung_Dashboard_version3_before_v4.py`.

Backups under `/tmp` are temporary storage.

All 12 regression tests passed, including calendar KPI deltas, plain-language correlation classifications, removed UI fields, all quarterly labels, and the red forecast boundary. Action Center and Data Explorer were also compared against the Version 3 snapshot and found unchanged.

Run tests with `.venv/bin/python -m unittest discover -s tests -v`.

Open the running dashboard at **http://127.0.0.1:8501** and refresh the browser.
