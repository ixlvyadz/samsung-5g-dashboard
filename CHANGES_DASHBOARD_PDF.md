# Version 3 revisions from dashboard.pdf

Reference: `/home/one/Downloads/dashboard.pdf`, reviewed 2026-09-29.

The current working project is Version 3. Version 2 is the snapshot taken immediately before these dashboard.pdf revisions. Version 1 at `/tmp/Samsung_Dashboard_before_BI_LE1.py` was used as the reference for the tier overview. Before these revisions, the Version 2 application was copied to `/tmp/Samsung_Dashboard_version2_before_dashboard_pdf.py`; a code/documentation snapshot is also at `/tmp/bebe_version2_before_dashboard_pdf.zip`.

## Where to find the changes

| Location | Revision |
|---|---|
| Top of dashboard | The seven-tab navigation now appears above the title/header. Question headers remain on each view. |
| Sidebar | Year slider track and both handles are black. Data Type remains selectable only in Trends and Forecast. |
| Overview | Retains the five Actual KPI cards, quarterly units/revenue lines, and capability donut. Both capabilities remain included; region/tier/year selections still apply. |
| Price Tier Performance | Restored Version 1's Tier Performance Overview: Units Sold, Gross Revenue, and Blended ASP bars across the selected tiers, metric checkboxes, and progress-bar table with highest-value indicators. Each metric is indexed independently to its largest tier; labels and tooltips show actual values. |
| Price Tier Performance, below overview | Retains the tier-by-year adoption heatmap and Welch/Games–Howell expander. Removes the separate revenue contribution donut. Significance badge is calculated from the selected data. |
| 5G Market Penetration | Retains the stacked area and ranked adoption views. Quarter ordering is explicit; all quarters in the selected Actual range are represented, with annual tick labels and exact quarters available on hover. |
| 5G Market Penetration, below comparison | Removes the duplicate open Model Portfolio table. Keeps the model image/card, capability tag, sales, revenue, ASP, market share, lifecycle, and expandable regional/lifecycle tables. |
| Comprehensive Product Portfolio Matrix expander | Sortable columns: model, price tier, 5G, share of 5G units, units sold, revenue, and ASP. Share of 5G units uses the selected portfolio's 5G units as denominator. |
| Portfolio Ranking — Top 12 Products | Restored ranked model bars. Dropdown choices: Units Sold, Total Revenue, Derived ASP, Avg Market Share, Adoption Rate. No tier-breakdown donut. Model adoption is 5G units divided by model units; dedicated hardware therefore yields 100% or 0%, with ties sorted by model name. |
| Trends and Forecast | Existing controls, historical/projection charts, and forecasting expander retained. |
| Regional Conditions | Compact correlation table: one pooled row and one row per region, with coverage/subscribers/speed/preference columns displaying r, p-value, and n. Scatter, preference/adoption chart, and regional commercial matrix retained. Correlation heatmap removed. |
| Action Center | Criteria, flags, suggested-action rules, strategic prescriptions, and playbook retained. Portfolio health classification/regional risk expander removed. |
| Data Explorer | Existing search, filters, downloads, dictionary, and raw/cleaned preview retained. |

Chart margins, legend placement, and model-card label spacing were adjusted for alignment while retaining the Samsung styling. No datasets were modified.

## Validation and running

Nine regression tests pass, including the new three-metric tier overview, all five ranking choices, portfolio columns, compact correlation table, and required removals. Existing statistical, filtering, forecast-isolation, and search tests also pass. Browser checks verified navigation placement, the black slider, restored views, and localhost health.

Open **http://127.0.0.1:8501**. To restart:

```bash
cd "/home/one/Downloads/bebe dashboard"
source .venv/bin/activate
python -m streamlit run Samsung_Dashboard.py
```

No commit or push was performed.
