# BI_LE1 implementation and feature guide

Implemented on 2026-09-29 against the supplied `BI_LE1.pdf`.

Open http://127.0.0.1:8501 and refresh an existing browser session. Use the seven tabs below the dashboard title; scroll the tab strip horizontally if the final tabs are outside the visible area.

## Changes and where to find them

| Location | What changed / how to inspect it |
|---|---|
| Sidebar on every tab | Region, Price Tier, Year range, and Data Type. Actual is the default. Reset All Filters restores the current view's defaults. |
| Sidebar on tabs 2, 4, 7 only | 5G Capability control. It does not affect the other tabs. |
| 1. Overview | Five Actual-only cards: adoption, blended ASP, revenue growth (QoQ with YoY underneath), market share, and total units. Below them: quarterly units/revenue lines and a 5G/Non-5G unit-share donut. Cards follow region/tier/year filters; adoption includes both capabilities. |
| 2. Price Tier Performance | Performance metric selector changes grouped bars of mean Units, Revenue, or ASP. Below: tier-by-year 5G unit-share heatmap and tier revenue contribution. Open **Welch's ANOVA & Games–Howell comparisons** for ANOVA and nonsignificant post-hoc pairs. |
| 3. 5G Market Penetration | Stacked quarterly units, ranked regional adoption, 5G/Non-5G means with Welch's t-test results, revenue contributions, and a sortable model table containing share of 5G units, units, revenue, blended ASP, and capability. |
| 4. Trends and Forecast | Metric selector (Units, Revenue, Market Share); breakdown selector (Total, Region, Price Tier, Model); QoQ/YoY control; trend lines and growth bars. In the sidebar, choose **Actual + Forecast** to overlay supplied projections. Forecast lines are dashed and forecast bars patterned. The Q3 2026 study-status note appears below the charts. |
| 5. Regional Conditions | Pooled and per-region correlations; indicator selector for the regional scatter; stated preference versus actual adoption. The correlation/causation and shared-time-trend caveat is always displayed. |
| 6. Action Center | Criteria at the top; regional adoption gaps against the same-quarter selected portfolio average; active models with two or more consecutive declining YoY quarters; pricing/marketing/product suggestions. Expand **Suggested-action rules** for the thresholds. No flags is a valid result. |
| 7. Data Explorer | Search any column; expand **Column filters** to choose multiple numeric/categorical filters; sort by clicking table headers; **Download matching records (CSV)** exports the resulting rows. Expand **Data dictionary** for column definitions. Expand **Data preparation — Raw / Cleaned** for the snapshot toggle and observed quality counts. |

## Scope and design

- Retained the Samsung palette, typography, hero header, rounded cards, chart styling, sidebar presentation, and tab navigation. The KPI grid now accommodates five cards.
- Each view has the PDF's question header. Chart counts by tab are 2, 3, 2, 2, 2, 0, 0; additional detail is in tables/expanders. The five expressly required Overview KPI cards are retained separately from chart counts.
- Forecast records appear only on tab 4. Other tabs, including both raw/cleaned previews, use Actual records.
- Removed out-of-scope interactive future-model fitting, product image drilldowns, extra regional charts, and generic portfolio risk/strategy panels from the interface. The main trend chart also serves the required Actual-vs-Forecast view, avoiding a duplicate chart.
- Existing CSV files and image assets were not changed.

## Calculation corrections

- Adoption consistently means 5G units divided by all units in the selected scope.
- Model share of 5G units uses the selected portfolio's total 5G units, rather than a binary 100%/0% capability label.
- QoQ and YoY use exact calendar-quarter baselines. Missing quarters or zero baselines yield unavailable growth instead of a misleading comparison.
- Action Center regional adoption and its benchmark use the same latest Actual quarter and selected portfolio. Model flags require an uninterrupted decline ending in that latest quarter.
- Infrastructure correlations use one observation per region-quarter; repeated model-level infrastructure values are not summed.
- Tests exclude insufficient/constant groups, disclose exclusions, and do not present nonsignificance as equivalence.

## Data preparation limitation

The bundled raw CSV and cleaned v3 CSV are different snapshots (816 versus 1,035 total rows). The explorer shows their actual values and quality counts, but does not falsely claim they constitute a complete reproducible before/after pipeline. Raw preview filtering leaves source values unchanged. Search/column filters apply to the main cleaned table and CSV export; sidebar filters apply to both snapshot previews.

## Validation

Run `.venv/bin/python -m unittest discover -s tests -v`.

Eight regression tests cover unit-weighted adoption, exact-quarter growth, consecutive decline flags, statistical results against independent library implementations, all seven tab/filter boundaries, forecast styling and isolation, records search/column filters, reset, and empty/narrow selections.

All eight tests passed. Real Chrome checks verified tab navigation, Actual + Forecast selection with dashed projections, the retained visual style, and search/download behavior: searching for `Galaxy A56` displayed 30 Actual records, and the downloaded CSV contained exactly those 30 matching rows. Dependency checks and the running server's health check also passed.

The UI uses Streamlit's stateful tabs so changing tabs can update sidebar controls. `requirements.txt` now requires Streamlit 1.64 or newer in the 1.x series. AppTest currently omits tab state when serializing other widget interactions; the tests explicitly preserve it, and browser checks separately verify actual tab behavior.
