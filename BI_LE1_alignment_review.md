# BI_LE1 dashboard alignment review

**Historical review, before implementation.** The gaps below describe the original dashboard. The implementation and current feature locations are documented in [CHANGES_BI_LE1.md](CHANGES_BI_LE1.md).

Reviewed 2026-09-29 against `/home/one/Downloads/BI_LE1.pdf`, using the current `Samsung_Dashboard.py` source. This is a requirements review, not a claim that the supplied sales data is independently verified.

## Conclusion

The dashboard is partially aligned. All seven tab names match, and many required calculations exist, but content placement, filter rules, and several required views do not match. The PDF specifies dashboard behavior and presentation, not a required Python folder architecture; a single-file Streamlit application is not itself a violation.

## Shared requirements

| PDF requirement | Current implementation | Assessment |
|---|---|---|
| Region, Price Tier, Year, Data Type filters; default Actual | All controls exist, but Data Type defaults to Actual + Forecast | Partial |
| Capability filter only on tabs 2, 4, 7 | One global capability filter is always visible | Not aligned |
| One-line question per tab | Question headers exist; some wording differs and Action Center has an additional infrastructure section header | Mostly present |
| Consistent 5G/Non-5G colors | Central capability palette maps aliases to blue/gray | Present in main capability views |
| At most four visuals per tab | Market Penetration renders five Plotly charts, plus product presentation and tables | Not aligned |
| Forecast only in tab 4 | Global filtered data can include forecasts in other tabs; model-generated forecasts are in tab 5 | Not aligned |

## Tab requirements

### 1. Overview

Four Actual-only KPI cards exist: adoption, ASP, revenue growth (QoQ/YoY), and market share. Total units is calculated but is missing as the fifth card requested by the PDF. The cards sit above the tab container and ignore all sidebar filters, not just the adoption card's capability exception. A capability donut exists. The requested quarterly units/revenue trend line is replaced by annual stacked unit bars.

### 2. Price Tier Performance

Welch's ANOVA is in an expander. The chart uses normalized totals and blended ASP with metric checkboxes, rather than a selector for mean Units/Revenue/ASP. The adoption time series is portfolio-wide, not tier-by-year. Revenue contribution is calculated, but a dedicated tier contribution view is elsewhere. No Games-Howell implementation or nonsignificant-pairs output was found. Welch's t-test comparison currently lives here, although the PDF places it in tab 3.

### 3. 5G Market Penetration

Actual-only stacked area exists. Product drilldown, ranking, regional product sales, tier revenue donut, and a model summary table are present. Missing the requested ranked regional adoption view and colocated 5G/Non-5G comparison panel with Welch's t-tests. Model adoption is a binary 100/0 capability mapping rather than a model's share of 5G units. Five chart calls exceed the four-visual limit.

### 4. Trends and Forecast

Metric and QoQ/YoY controls exist. Breakdown supports total, region, and price tier, but not model. The aggregate quarterly chart separates forecast data with dashed styling. A dedicated growth-rate bar chart and the explicit note that Q3 2026 is in progress are missing. Forecast visibility is not restricted to this tab. The separate fitted forecasting module is in tab 5.

### 5. Regional Conditions

Pooled and regional infrastructure correlation table exists, with the required causation/shared-time-trend caveat. Most of the tab is occupied by the unrelated forecasting module. The indicator-selector scatter is in tab 6. A stated-preference versus actual-adoption regional comparison is missing.

### 6. Action Center

Criteria, flagged regions with adoption gaps, models with at least two consecutive negative YoY quarters, and rule-based suggestions exist. This is the closest content match, but the infrastructure scatter and correlation heatmap also render in this tab. The benchmark uses filtered Actual records across time, while regional adoption uses the latest quarter; it is not an unfiltered portfolio benchmark. Adoption is record-based here, versus unit-based in the headline. YoY uses a four-row shift and needs care when quarters are filtered out. Forecast-only filtering leaves no Actual rows and can fail when retrieving the latest period.

### 7. Data Explorer

Contains portfolio health/risk charts, recommendations, and an aggregated model-health table. It does not provide the required cleaned-records explorer with column filters and a matching CSV download. The existing download exports the complete cleaned dataset from the sidebar. The data dictionary is in tab 6. No collapsed Raw/Cleaned comparison control exists.

## Recommended implementation order

1. Correct tab-aware filtering: Actual by default, forecasts only on tab 4, capability controls only on tabs 2/4/7; handle empty Actual subsets.
2. Move existing content into the PDF's intended tabs before adding new charts.
3. Implement the records explorer and raw/cleaned comparison.
4. Add missing quarterly overview trend, tier-by-year adoption, Games-Howell pairs, regional adoption ranking, model breakdown, growth bars, and preference/adoption comparison.
5. Standardize adoption definitions and benchmark scope, then verify the four-visual limit and all filter combinations.

No dashboard feature restructuring was performed as part of this review.
