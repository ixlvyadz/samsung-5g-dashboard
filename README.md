# Decoding Demand: Samsung 5G Dashboard

A Streamlit dashboard using local CSV datasets. Styling, cleaning, and the seven views are in `Samsung_Dashboard.py`; shared statistical and calendar calculations are in `dashboard_analytics.py`. The application prefers `Samsung_5G_Cleaned_Dataset_v3.csv`. It displays supplied Forecast records only in Trends and Forecast.

## Run with the configured local environment

From this project directory:

```bash
source .venv/bin/activate
python -m streamlit run Samsung_Dashboard.py
```

Alternatively, without activation:

```bash
.venv/bin/python -m streamlit run Samsung_Dashboard.py
```

## Set up a new environment on Ubuntu/Debian

Install your Python version's venv package if `python3 -m venv` reports that ensurepip is missing (for this machine, `sudo apt-get install python3.14-venv`). Then:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run Samsung_Dashboard.py
```

Dependencies are isolated in `.venv`; installing them does not add pip or Streamlit to `/usr/bin/python3`.

The current project is **Version 4**. See [CHANGES_VERSION4.md](CHANGES_VERSION4.md) for the latest revisions and their locations. [CHANGES_DASHBOARD_PDF.md](CHANGES_DASHBOARD_PDF.md) records Version 3, and [CHANGES_BI_LE1.md](CHANGES_BI_LE1.md) records the earlier implementation. `BI_LE1_alignment_review.md` preserves the original pre-implementation gap review.

## Verify

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The app requires Streamlit 1.64+ for stateful tabs and tab-specific sidebar controls. Regression comparisons additionally use SciPy's Games–Howell reference implementation (`scipy>=1.16`); the configured environment includes it.
