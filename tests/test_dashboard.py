"""Regression coverage for PDF-required calculations and tab/filter boundaries.

Run: .venv/bin/python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway
from streamlit.testing.v1 import AppTest

from dashboard_analytics import (adoption_rate, flagged_models, quarterly_series, tier_significance,
                                 kpi_changes, correlation_description)

ROOT = Path(__file__).resolve().parents[1]


class CalculationTests(unittest.TestCase):
    def test_kpi_changes_use_calendar_baselines_and_correct_units(self):
        frame = pd.DataFrame({'Quarter_Index': [0, 3, 4], 'Units Sold': [100, 80, 120],
                              'Revenue ($)': [1000, 1600, 3000], 'Market Share (%)': [20, 24, 25],
                              '5G Capability': ['No', 'Yes', 'Yes']})
        period, changes = kpi_changes(frame)
        self.assertEqual(period, '2020-Q1')
        self.assertEqual(changes['Adoption']['YoY'], 100)
        self.assertEqual(changes['ASP']['QoQ'], 5)
        self.assertEqual(changes['Share']['QoQ'], 1)
        self.assertAlmostEqual(changes['Units']['QoQ'], 50)
        self.assertTrue(pd.isna(kpi_changes(frame.iloc[:1])[1]['Units']['QoQ']))

    def test_plain_language_correlations(self):
        self.assertEqual(correlation_description(.8, .001), 'Strong positive link')
        self.assertEqual(correlation_description(-.5, .01), 'Moderate negative link')
        self.assertEqual(correlation_description(.6, .2), 'No clear relationship')
        self.assertEqual(correlation_description(np.nan, np.nan), 'Not enough data')

    def test_adoption_weights_units_not_records(self):
        frame = pd.DataFrame({'5G Capability': ['Yes', 'No'], 'Units Sold': [90, 10]})
        self.assertEqual(adoption_rate(frame), 90)

    def test_calendar_growth_does_not_skip_missing_quarters(self):
        frame = pd.DataFrame({'Quarter_Index': [0, 2, 4, 5], 'Units Sold': [100, 200, 50, 40],
                              'Data Type': ['Actual'] * 4})
        series = quarterly_series(frame, 'Units Sold').set_index('Quarter_Index')
        self.assertTrue(pd.isna(series.at[2, 'QoQ (%)']))
        self.assertAlmostEqual(series.at[4, 'YoY (%)'], -50)
        self.assertTrue(pd.isna(series.at[5, 'YoY (%)']))

    def test_decline_flags_require_consecutive_calendar_quarters(self):
        def sample(quarters):
            return pd.DataFrame({'Quarter_Index': quarters, 'Units Sold': [100 if i < 4 else 50 for i in quarters],
                                 'Data Type': 'Actual', 'Product Model': 'Example', 'Price Tier': 'Budget'})
        flagged = flagged_models(sample([0, 1, 2, 3, 4, 5]))
        self.assertEqual(flagged.iloc[0]['Consecutive declining quarters'], 2)
        self.assertTrue(flagged_models(sample([0, 1, 2, 3, 4, 6])).empty)

    def test_welch_and_games_howell_match_independent_library_results(self):
        samples = [np.array([2, 4, 7, 9, 12.]), np.array([3, 4, 5, 8, 11, 15.]),
                   np.array([10, 14, 21, 30.])]
        frame = pd.concat([pd.DataFrame({'Price Tier': name, 'Units Sold': values,
                                        'Revenue ($)': values * 4, 'ASP': values / 2})
                           for name, values in zip(['A', 'B', 'C'], samples)])
        anova, pairs = tier_significance(frame)
        reference = anova_oneway(samples, use_var='unequal')
        self.assertAlmostEqual(anova.iloc[0]['F'], reference.statistic, places=10)
        self.assertAlmostEqual(anova.iloc[0]['p-value'], reference.pvalue, places=10)
        gh = stats.tukey_hsd(*samples, equal_var=False)
        for _, row in pairs[pairs['Metric'].eq('Units Sold')].iterrows():
            i, j = 'ABC'.index(row['Tier A']), 'ABC'.index(row['Tier B'])
            self.assertAlmostEqual(row['p-value'], gh.pvalue[i, j], places=10)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = AppTest.from_file(str(ROOT / 'Samsung_Dashboard.py'), default_timeout=90).run()

    def run_app(self):
        # AppTest does not serialize stateful tab blocks as widget states yet.
        # Preserve the selected label explicitly when simulating other controls.
        self.app.session_state['dashboard_tab'] = self.active_tab
        self.app.run()

    def switch(self, name):
        self.active_tab = name
        self.run_app()
        self.assertFalse(self.app.exception, [(exc.message, exc.stack_trace) for exc in self.app.exception])

    def test_01_tabs_filters_and_visual_limits(self):
        labels = ['Overview', 'Price Tier Performance', '5G Market Penetration', 'Trends and Forecast',
                  'Regional Conditions', 'Action Center', 'Data Explorer']
        self.assertEqual([tab.label for tab in self.app.tabs], labels)
        for index, label in enumerate(labels):
            with self.subTest(tab=label):
                self.switch(label)
                self.assertLessEqual(len(self.app.get('plotly_chart')), 4)
                capability = [radio for radio in self.app.radio if radio.label == '5G Capability']
                self.assertEqual(bool(capability), index in [1, 3, 6])
                horizon = next(widget for widget in self.app.selectbox if widget.label == 'Data Type')
                self.assertEqual(horizon.value, 'Actual')
                if index != 3:
                    self.assertEqual(horizon.options, ['Actual'])
        self.switch('Overview')
        cards = next(md.value for md in self.app.markdown if '<div class="kpi-container">' in md.value)
        self.assertEqual(cards.count('class="kpi-card"'), 5)

    def test_02_forecast_styles_and_no_leak_to_other_tabs(self):
        self.switch('Trends and Forecast')
        self.app.selectbox(key='filter_trend_data_type').set_value('Actual + Forecast')
        self.run_app()
        self.assertFalse(self.app.exception)
        spec = json.loads(self.app.get('plotly_chart')[0].proto.spec)
        self.assertTrue(any(trace.get('line', {}).get('dash') == 'dash' for trace in spec['data']))
        for breakdown in ['Region', 'Price Tier', 'Model', 'Total']:
            self.app.selectbox(key='trend_breakdown').set_value(breakdown)
            self.run_app()
            self.assertFalse(self.app.exception)
        self.app.radio(key='trend_growth').set_value('YoY')
        self.run_app()
        self.app.selectbox(key='filter_trend_data_type').set_value('Forecast')
        self.run_app()
        self.assertFalse(self.app.exception)
        for label in ['Overview', 'Action Center', 'Data Explorer']:
            self.switch(label)
            self.assertIn('Actual', self.app.selectbox(key='filter_actual_data_type').value)
        cleaned = self.app.dataframe[0].value
        self.assertEqual(set(cleaned['Data Type']), {'Actual'})

    def test_03_search_filters_download_and_raw_view(self):
        self.switch('Data Explorer')
        self.app.text_input(key='explorer_search').set_value('Galaxy A56')
        self.run_app()
        self.assertFalse(self.app.exception)
        data = self.app.dataframe[0].value
        self.assertGreater(len(data), 0)
        self.assertTrue(data['Product Model'].str.contains('Galaxy A56').all())
        self.app.multiselect(key='explorer_filter_columns').set_value(['Region', 'Units Sold'])
        self.run_app()
        self.app.multiselect(key='explorer_values_Region').set_value(['Europe'])
        self.run_app()
        self.assertFalse(self.app.exception)
        data = self.app.dataframe[0].value
        self.assertEqual(set(data['Region']), {'Europe'})
        self.assertTrue(self.app.get('download_button'))
        self.app.radio(key='explorer_snapshot').set_value('Cleaned')
        self.run_app()
        self.assertFalse(self.app.exception)
        self.app.text_input(key='explorer_search').set_value('no-such-model-xyz')
        self.run_app()
        self.assertTrue(self.app.dataframe[0].value.empty)
        self.assertFalse(self.app.exception)
        self.app.button[0].click()
        self.run_app()
        self.assertEqual(self.app.text_input(key='explorer_search').value, '')

    def test_04_pdf_revision_views(self):
        self.switch('Price Tier Performance')
        figures = [json.loads(element.proto.spec) for element in self.app.get('plotly_chart')]
        self.assertEqual(len(figures), 2)
        self.assertEqual([trace['name'] for trace in figures[0]['data']],
                         ['Units Sold', 'Gross Revenue', 'Blended ASP'])
        self.assertEqual(len(figures[0]['data'][0]['x']), 6)
        self.assertEqual(figures[1]['data'][0]['type'], 'heatmap')
        self.switch('5G Market Penetration')
        self.assertFalse(any(md.value == '#### Model Portfolio' for md in self.app.markdown))
        self.assertTrue(any('Regional Sales & Lifecycle Breakdown' in item.label for item in self.app.expander))
        portfolio = next(item for item in self.app.expander if item.label == 'Comprehensive Product Portfolio Matrix')
        self.assertEqual(list(portfolio.dataframe[0].value.columns),
                         ['Model', 'Price Tier', '5G', 'Share of 5G units (%)', 'Units Sold', 'Revenue ($)', 'Derived ASP ($)'])
        for metric in ['Units Sold', 'Total Revenue', 'Derived ASP', 'Avg Market Share', 'Adoption Rate']:
            self.app.selectbox(key='portfolio_rank_metric').set_value(metric)
            self.run_app()
            self.assertFalse(self.app.exception)
            ranking = json.loads(self.app.get('plotly_chart')[-1].proto.spec)
            self.assertEqual(sum(len(trace['y']) for trace in ranking['data']), 12)
        self.switch('Regional Conditions')
        self.assertEqual(list(self.app.dataframe[0].value.columns), ['Scope', 'Coverage', 'Subscribers', 'Speed', 'Preference'])
        self.assertEqual(len(self.app.dataframe[0].value), 6)
        self.assertTrue(any(item.label == 'Regional Commercial Performance Matrix' for item in self.app.expander))
        self.assertFalse(any('Heatmap' in item.label for item in self.app.expander))
        self.switch('Action Center')
        self.assertTrue(any(item.label == 'Strategic Playbook & Implementation Roadmap' for item in self.app.expander))
        self.assertFalse(any('Health Classification' in item.label for item in self.app.expander))

    def test_05_version4_requested_changes(self):
        self.switch('Overview')
        cards = next(md.value for md in self.app.markdown if '<div class="kpi-container">' in md.value)
        self.assertEqual(cards.count('QoQ'), 5)
        self.assertEqual(cards.count('YoY'), 5)
        self.assertTrue('▲' in cards or '▼' in cards)
        self.switch('Price Tier Performance')
        self.assertFalse(any('Games' in item.label or 'ANOVA' in item.label for item in self.app.expander))
        self.switch('5G Market Penetration')
        area = json.loads(self.app.get('plotly_chart')[0].proto.spec)
        self.assertEqual(area['layout']['xaxis']['tickvals'], area['layout']['xaxis']['categoryarray'])
        self.assertEqual(len(area['layout']['xaxis']['tickvals']), 30)
        self.assertFalse(any("Welch's t-test p-value" in item.value.columns for item in self.app.dataframe))
        self.switch('Trends and Forecast')
        self.assertFalse(any('Econometric Forecasting Engine' in item.label for item in self.app.expander))
        self.assertTrue(any('#### Econometric Forecasting Engine' in md.value for md in self.app.markdown))
        self.assertTrue(self.app.selectbox(key='hw_fc_metric'))
        fig = json.loads(self.app.get('plotly_chart')[-1].proto.spec)
        boundary = fig['layout']['shapes'][0]
        self.assertEqual(boundary['x0'], '2026-Q3')
        self.assertEqual(boundary['line']['color'], '#B91C1C')
        self.assertEqual(boundary['line']['dash'], 'dash')
        self.assertTrue(any(trace.get('line', {}).get('dash') == 'dash' for trace in fig['data']))
        self.switch('Regional Conditions')
        results = self.app.dataframe[0].value.iloc[:, 1:]
        self.assertFalse(results.astype(str).apply(lambda col: col.str.contains('p=|p <|n=', regex=True)).any().any())
        self.assertTrue(results.astype(str).apply(lambda col: col.str.contains('link')).any().any())

    def test_06_empty_and_narrow_filters(self):
        for label in ['Overview', 'Price Tier Performance', '5G Market Penetration', 'Trends and Forecast',
                      'Regional Conditions', 'Action Center', 'Data Explorer']:
            self.switch(label)
            self.app.multiselect(key='filter_regions').set_value([])
            self.run_app()
            self.assertFalse(self.app.exception)
            self.assertTrue(any('No records' in info.value for info in self.app.info))
            self.app.multiselect(key='filter_regions').set_value(['Europe'])
            self.run_app()
            self.app.multiselect(key='filter_tiers').set_value(['Budget Legacy 4G'])
            self.run_app()
            self.app.slider(key='filter_year_range').set_value((2026, 2026))
            self.run_app()
            self.assertFalse(self.app.exception)
            self.app.button[0].click()
            self.run_app()


if __name__ == '__main__':
    unittest.main()
