"""
data_cleaning_v3.py  — Documented cleaning step for Samsung_5G_Cleaned_Dataset_v3.csv

Changes from v2 (Samsung_5G_Cleaned_Dataset.csv):
  1. Remove the 2026-Q2 Galaxy A56 5G / Europe row where Regional 5G Coverage = 102.73
     (physically impossible; 87.73 is retained as the authoritative value).
  2. Restore 'Budget Legacy 4G' as its own price tier instead of merging into 'Budget'.
     The 94 Budget Legacy 4G rows belong to Galaxy A05, Galaxy A06 4G, Galaxy A07 4G.

Rows: 1036 -> 1035 (1 duplicate coverage row removed)
Tiers: 5  -> 6  (Budget Legacy 4G restored)
"""

import pandas as pd
import numpy as np

V2_PATH  = "Samsung_5G_Cleaned_Dataset.csv"
V3_PATH  = "Samsung_5G_Cleaned_Dataset_v3.csv"
RAW_PATH = r"C:\Users\Baberose\Downloads\Samsung_5G_BI_Dataset_RAW-1.csv"

# ── Load v2 ───────────────────────────────────────────────────────────────────
df = pd.read_csv(V2_PATH)
print(f"v2 shape: {df.shape}")
print(f"v2 Price Tier:\n{df['Price Tier'].value_counts()}\n")

# ── CHANGE 1: Remove duplicate impossible-coverage row ────────────────────────
dup_mask = (
    (df['Product Model'] == 'Galaxy A56 5G') &
    (df['Region'] == 'Europe') &
    (df['Period'] == '2026-Q2') &
    (df['Regional 5G Coverage (%)'] == 102.73)
)
n_removed = dup_mask.sum()
assert n_removed == 1, f"Expected exactly 1 duplicate row, found {n_removed}"
df_v3 = df[~dup_mask].copy().reset_index(drop=True)
print(f"CHANGE 1: Removed {n_removed} row (Galaxy A56 5G / Europe / 2026-Q2 / Coverage=102.73)")

# ── CHANGE 2: Restore 'Budget Legacy 4G' from raw source ──────────────────────
raw = pd.read_csv(RAW_PATH)

# Normalize region casing to match v2's canonical form
region_mapping = {
    'asia-pacific': 'Asia-Pacific',
    'europe': 'Europe',
    'middle east & africa': 'Middle East & Africa',
    'north america': 'North America',
    'latin america': 'Latin America',
}
raw['Region_norm'] = raw['Region'].astype(str).str.strip().str.lower().map(region_mapping).fillna(
    raw['Region'].str.strip()
)
raw['Model_norm'] = raw['Product Model'].astype(str).str.strip()
raw['Quarter_norm'] = raw['Quarter'].astype(str).str.strip().str.upper()
raw['Year_norm'] = raw['Year'].fillna(0).astype(int)

# Build lookup: (Model, Year, Quarter, Region) -> Price Tier
raw['_key'] = (raw['Model_norm'] + '|' +
               raw['Year_norm'].astype(str) + '|' +
               raw['Quarter_norm'] + '|' +
               raw['Region_norm'])
tier_lookup = raw.set_index('_key')['Price Tier'].to_dict()

# Apply lookup
df_v3['_key'] = (df_v3['Product Model'].str.strip() + '|' +
                 df_v3['Year'].astype(int).astype(str) + '|' +
                 df_v3['Quarter'].str.strip().str.upper() + '|' +
                 df_v3['Region'].str.strip())
restored = df_v3['_key'].map(tier_lookup)
# Only apply where raw says Budget Legacy 4G (don't overwrite other tiers)
is_legacy = restored == 'Budget Legacy 4G'
df_v3.loc[is_legacy, 'Price Tier'] = 'Budget Legacy 4G'
df_v3 = df_v3.drop(columns=['_key'])

n_legacy = is_legacy.sum()
print(f"CHANGE 2: Restored {n_legacy} rows to 'Budget Legacy 4G' tier via key lookup")

# Fallback: any remaining Budget rows for the known legacy models also get reclassified
LEGACY_MODELS = {'Galaxy A05', 'Galaxy A06 4G', 'Galaxy A07 4G'}
fallback_mask = (df_v3['Price Tier'] == 'Budget') & (df_v3['Product Model'].isin(LEGACY_MODELS))
n_fallback = fallback_mask.sum()
if n_fallback > 0:
    df_v3.loc[fallback_mask, 'Price Tier'] = 'Budget Legacy 4G'
    print(f"CHANGE 2 (fallback): Reclassified {n_fallback} additional rows for known legacy models")

print(f"\nv3 Price Tier distribution:")
print(df_v3['Price Tier'].value_counts().sort_index())
print(f"\nBudget Legacy 4G models: {df_v3[df_v3['Price Tier']=='Budget Legacy 4G']['Product Model'].unique().tolist()}")

# ── Verify no remaining 102.73 coverage ───────────────────────────────────────
assert (df_v3['Regional 5G Coverage (%)'] > 100).sum() == 0, "Still have impossible coverage values!"
print("\nVerification: No Regional 5G Coverage > 100% in v3. OK.")

# ── Actual/Forecast counts ────────────────────────────────────────────────────
actual_n = (df_v3['Data Type'] == 'Actual').sum()
forecast_n = (df_v3['Data Type'] == 'Forecast').sum()
print(f"Actual rows: {actual_n}  |  Forecast rows: {forecast_n}  |  Total: {len(df_v3)}")

# ── Save v3 ───────────────────────────────────────────────────────────────────
df_v3.to_csv(V3_PATH, index=False)
print(f"\nSaved: {V3_PATH}  shape: {df_v3.shape}")
