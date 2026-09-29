"""Calculations shared by the seven BI_LE1 dashboard views."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def adoption_rate(frame):
    total = frame['Units Sold'].sum()
    return frame.loc[frame['5G Capability'].eq('Yes'), 'Units Sold'].sum() / total * 100 if total else np.nan


def adoption_by(frame, columns):
    data = frame.assign(FiveGUnits=frame['Units Sold'].where(frame['5G Capability'].eq('Yes'), 0))
    result = data.groupby(columns, observed=True).agg(
        Units=('Units Sold', 'sum'), FiveGUnits=('FiveGUnits', 'sum')
    ).reset_index()
    result['Adoption (%)'] = result['FiveGUnits'].div(result['Units'].replace(0, np.nan)) * 100
    return result


def quarter_label(index):
    return f'{2019 + int(index) // 4}-Q{int(index) % 4 + 1}'


def kpi_changes(frame):
    """Latest-quarter changes against exact prior-quarter/year baselines."""
    quarterly = frame.assign(FiveGUnits=frame['Units Sold'].where(frame['5G Capability'].eq('Yes'), 0)).groupby('Quarter_Index').agg(
        Units=('Units Sold', 'sum'), Revenue=('Revenue ($)', 'sum'),
        Share=('Market Share (%)', 'mean'), FiveGUnits=('FiveGUnits', 'sum'))
    quarterly['Adoption'] = quarterly['FiveGUnits'].div(quarterly['Units'].replace(0, np.nan)) * 100
    quarterly['ASP'] = quarterly['Revenue'].div(quarterly['Units'].replace(0, np.nan))
    latest = int(quarterly.index.max())
    changes = {}
    for metric in ['Adoption', 'ASP', 'Revenue', 'Share', 'Units']:
        changes[metric] = {}
        for label, lag in [('QoQ', 1), ('YoY', 4)]:
            current = quarterly.at[latest, metric]
            prior = quarterly.at[latest - lag, metric] if latest - lag in quarterly.index else np.nan
            changes[metric][label] = ((current / prior - 1) * 100 if pd.notna(prior) and prior != 0 else np.nan) if metric in ['Revenue', 'Units'] else current - prior
    return quarter_label(latest), changes


def correlation_description(r, p):
    """Plain-language association summary; strength thresholds are descriptive."""
    if pd.isna(r) or pd.isna(p):
        return 'Not enough data'
    if p >= .05 or abs(r) < .2:
        return 'No clear relationship'
    strength = 'Strong' if abs(r) >= .7 else 'Moderate' if abs(r) >= .4 else 'Weak'
    direction = 'positive' if r > 0 else 'negative'
    return f'{strength} {direction} link'


def quarterly_series(frame, metric, breakdown=None):
    """Keep missing quarters as gaps so shifts compare exact calendar quarters."""
    data = frame.copy()
    data['Series'] = data[breakdown] if breakdown else 'Total'
    results = []
    for name, group in data.groupby('Series', observed=True):
        grouped = group.groupby('Quarter_Index')[metric]
        values = grouped.mean() if metric == 'Market Share (%)' else grouped.sum(min_count=1)
        index = pd.RangeIndex(int(values.index.min()), int(values.index.max()) + 1, name='Quarter_Index')
        result = values.reindex(index).rename('Value').to_frame()
        result['Data Type'] = group.groupby('Quarter_Index')['Data Type'].first().reindex(index)
        result['QoQ (%)'] = (result['Value'] / result['Value'].shift(1).replace(0, np.nan) - 1) * 100
        result['YoY (%)'] = (result['Value'] / result['Value'].shift(4).replace(0, np.nan) - 1) * 100
        result['Series'] = name
        result['Period'] = [quarter_label(i) for i in index]
        results.append(result.reset_index())
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def tier_significance(frame):
    """Welch ANOVA and Games–Howell p-values (studentized range, alpha=.05)."""
    anova, pairs = [], []
    for metric in ['Units Sold', 'Revenue ($)', 'ASP']:
        groups = {tier: group[metric].dropna().to_numpy(dtype=float)
                  for tier, group in frame.groupby('Price Tier', observed=True)}
        # Neither Welch nor Games–Howell is defined for n<2 or zero sample variance.
        valid = {tier: values for tier, values in groups.items()
                 if len(values) >= 2 and np.isfinite(values).all() and np.var(values, ddof=1) > 0}
        excluded = ', '.join(sorted(set(groups) - set(valid)))
        if len(valid) < 2:
            anova.append({'Metric': metric, 'F': np.nan, 'p-value': np.nan,
                          'Result': 'Insufficient nonconstant groups', 'Excluded tiers': excluded})
            continue
        values = list(valid.values())
        k = len(values)
        n = np.array([len(v) for v in values], dtype=float)
        means = np.array([v.mean() for v in values])
        variance = np.array([v.var(ddof=1) for v in values])
        weights = n / variance
        weighted_mean = np.sum(weights * means) / weights.sum()
        correction = np.sum((1 - weights / weights.sum()) ** 2 / (n - 1))
        f_value = (np.sum(weights * (means - weighted_mean) ** 2) / (k - 1)
                   / (1 + 2 * (k - 2) * correction / (k * k - 1)))
        df2 = (k * k - 1) / (3 * correction)
        p_value = stats.f.sf(f_value, k - 1, df2)
        anova.append({'Metric': metric, 'F': f_value, 'df1': k - 1, 'df2': df2,
                      'p-value': p_value, 'Result': 'Significant' if p_value < .05 else 'Not significant',
                      'Excluded tiers': excluded})
        for a, b in combinations(valid, 2):
            x, y = valid[a], valid[b]
            vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
            dof = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
            q = abs(x.mean() - y.mean()) / np.sqrt((vx + vy) / 2)
            p = float(stats.studentized_range.sf(q, k, dof))
            pairs.append({'Metric': metric, 'Tier A': a, 'Tier B': b,
                          'Mean difference (A − B)': x.mean() - y.mean(),
                          'p-value': p, 'Significant': p < .05})
    return pd.DataFrame(anova), pd.DataFrame(pairs)


INDICATORS = ['Regional 5G Coverage (%)', '5G Subscribers (millions)',
              'Avg 5G Speed (Mbps)', 'Preference for 5G (%)']


def regional_quarters(frame):
    """One observation per region-quarter; never sum repeated infrastructure values."""
    macro = frame.groupby(['Region', 'Quarter_Index'], observed=True)[INDICATORS].mean()
    sales = adoption_by(frame, ['Region', 'Quarter_Index']).set_index(['Region', 'Quarter_Index'])
    result = macro.join(sales).reset_index()
    result['Period'] = result['Quarter_Index'].map(quarter_label)
    return result


def correlation_table(regional):
    rows = []
    scopes = [('Pooled', regional), *list(regional.groupby('Region', observed=True))]
    for scope, data in scopes:
        for indicator in INDICATORS:
            valid = data[[indicator, 'FiveGUnits']].dropna()
            if len(valid) < 3 or valid[indicator].nunique() < 2 or valid['FiveGUnits'].nunique() < 2:
                r, p = np.nan, np.nan
            else:
                r, p = stats.pearsonr(valid[indicator], valid['FiveGUnits'])
            rows.append({'Scope': scope, 'Indicator': indicator, 'Observations': len(valid),
                         'Pearson r': r, 'p-value': p})
    return pd.DataFrame(rows)


def flagged_models(actual):
    """Require uninterrupted calendar quarters ending in the latest Actual quarter."""
    rows = []
    if actual.empty:
        return pd.DataFrame()
    latest = int(actual['Quarter_Index'].max())
    active = actual.loc[actual['Quarter_Index'].eq(latest), 'Product Model'].unique()
    for model in active:
        data = actual[actual['Product Model'].eq(model)]
        series = quarterly_series(data, 'Units Sold').set_index('Quarter_Index')
        streak, cursor = 0, latest
        while cursor in series.index and pd.notna(series.at[cursor, 'YoY (%)']) and series.at[cursor, 'YoY (%)'] < 0:
            streak += 1
            cursor -= 1
        if streak >= 2:
            tier = data['Price Tier'].mode().iloc[0]
            rows.append({'Model': model, 'Price Tier': tier, 'Consecutive declining quarters': streak,
                         'Latest YoY (%)': series.at[latest, 'YoY (%)'], 'Trigger quarter': quarter_label(latest),
                         'Suggested action': 'Pricing — review price competitiveness' if tier in ['Flagship', 'Premium', 'Premium Foldable']
                         else 'Product — review refresh or retirement'})
    return pd.DataFrame(rows)
