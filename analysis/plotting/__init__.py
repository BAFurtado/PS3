import os

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import conf
from . import geo
from ..output import OUTPUT_DATA_SPEC

# map mun code -> name
mun_codes = pd.read_csv('input/names_and_codes_municipalities.csv', sep=';')
mun_codes = dict(zip(mun_codes['cod_mun'], mun_codes['cod_name']))

plt.set_cmap('viridis')
plt.style.use('ggplot')


class MissingDataError(Exception):
    pass


class Plotter:
    """Manages all plotting of simulation outputs"""
    def __init__(self, run_paths, output_path, params, avg=None):
        # Keep track of params that generated these plots for annotating/titling
        self.params = params

        # List of (label, input data path)
        self.labels, self.run_paths = zip(*run_paths)
        self.avg = avg

        # Create output path (if necessary) for plots
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def save_fig(self, fig, name, clear=True):
        fig.savefig(
            os.path.join(self.output_path, '{}.{}'.format(name, conf.RUN['PLOT_FORMAT'])),
            format=conf.RUN['PLOT_FORMAT'],
            dpi=conf.RUN['PLOT_DPI'],
            bbox_inches='tight')

        if clear:
            # reset figure
            plt.close('all')

    def make_plot(self, datas, title, labels, y_label=None, q1=None, q3=None, colors=None):
        """Create plot based on input data"""
        p_pop = self.params.get('PERCENTAGE_ACTUAL_POP', conf.PARAMS['PERCENTAGE_ACTUAL_POP'])
        title = '{}\nAgents : {}% of Population'.format(title, p_pop * 100)

        fig, ax = plt.subplots()

        for i, data in enumerate(datas):
            idx = pd.to_datetime(data.index)
            color = colors[i] if colors else None
            label = labels[i]

            # Plot main line
            ax.plot(idx, data.values, label=label, color=color)

            # Fill between for bounds
            if q1 and q3 and len(q1) > i and len(q3) > i:
                q1_vals = q1[i].reindex(idx).values
                q3_vals = q3[i].reindex(idx).values
                ax.fill_between(idx, q1_vals, q3_vals, alpha=0.2, color=color)

        ax.legend(loc='best', ncol=3, fancybox=True, shadow=False, framealpha=.25)
        ax.set_title(title)
        ax.set_xlabel('Time')
        if y_label:
            ax.set_ylabel(y_label)
        fig.set_size_inches(15, 10)
        return fig

    def _prepare_data(self, path, columns):
        # Just read the data
        try:
            dat = pd.read_csv(path,  sep=';', decimal='.', header=None)
        except FileNotFoundError:
            raise MissingDataError
        dat.columns = columns

        # # Time to be eliminated (adjustment of the model)
        if conf.RUN['TIME_TO_BE_ELIMINATED'] > 0:
            dat = dat.loc[len(dat['month']) * conf.RUN['TIME_TO_BE_ELIMINATED']:, :]
        return dat

    def _prepare_datas(self, fname, columns):
        paths = [(label, os.path.join(path, fname)) for label, path in zip(self.labels, self.run_paths)]
        paths = [(label, self._prepare_data(path, columns))
                 for label, path in paths
                 if os.path.exists(path)]
        if not paths:
            raise MissingDataError(f"No files found for {fname}")
        labels, dats = zip(*paths)
        return labels, dats

    def _load_multiple_runs(self, key, fname):
        spec = OUTPUT_DATA_SPEC[key]
        labels, dats = self._prepare_datas(fname, spec['columns'])
        if self.avg:
            avg_type, avg_path = self.avg
            cols = spec['avg']['columns']
            if cols == 'ALL':
                cols = spec['columns']
            else:
                cols = spec['avg']['groupings'] + cols
            avg_dat = self._prepare_data(os.path.join(avg_path, fname), cols)
            labels = [avg_type] + list(labels)
            dats = [avg_dat] + list(dats)
        return labels, dats

    def _load_single_run(self, key, fname):
        """Some plots can only plot one run at a time.
        Use average if available, otherwise default to first run data."""
        spec = OUTPUT_DATA_SPEC[key]
        if self.avg:
            avg_type, avg_path = self.avg
            cols = spec['avg']['columns']
            if cols == 'ALL':
                cols = spec['columns']
            else:
                cols = spec['avg']['groupings'] + cols
            # return avg_type, self._prepare_data(os.path.join(avg_path, fname), cols)
            return self._prepare_data(os.path.join(avg_path, fname), cols)
        else:
            # return 'run', self._prepare_data(os.path.join(self.run_paths[0], fname), spec['columns'])
            return self._prepare_data(os.path.join(self.run_paths[0], fname), spec['columns'])

    def plot_general(self):
        dats_q1, dats_q3 = None, None
        labels, dats = self._load_multiple_runs('stats', 'stats.csv')
        if len(self.run_paths) > 1:
            try:
                # This is the case that we have comparisons, such as sensitivity
                _, dats_q1 = self._load_multiple_runs('stats', 'q1_stats.csv')
                _, dats_q3 = self._load_multiple_runs('stats', 'q3_stats.csv')
            except ValueError:
                if self.avg:
                    # This is for single run
                    temp_path = self.avg[1]
                else:
                    # This is for plotting within each comparison value
                    temp_path = self.output_path.replace('plots', 'avg')
                dats_q1 = [self._prepare_data(os.path.join(temp_path, 'q1_stats.csv'),
                                              dats[0].columns).set_index('month')]
                dats_q3 = [self._prepare_data(os.path.join(temp_path, 'q3_stats.csv'),
                                              dats[0].columns).set_index('month')]

        cols = ['pop', 'price_index', 'gdp_index', 'gdp_growth', 'unemployment', 'median_workers',
                'families_median_wealth', 'families_wages_received',
                'families_commuting', 'families_savings', 'families_helped', 'amount_subsidised',
                'firms_profit', 'firms_median_stock', 'firms_avg_eco_eff', 'firms_median_wage_paid',
                'firms_median_innovation_investment', 'emissions',
                'gini_index',
                'average_utility', 'pct_zero_consumption', 'rent_default', 'inflation', 'average_qli', 'house_vacancy',
                'house_price', 'house_rent', 'affordable', 'p_delinquent', 'equally', 'locally', 'fpm', 'bank',
                'emissions_fund',
                'ext_amount_sold']
        titles = ['Total population', 'Average goods and services prices\' level', 'GDP absolute value',
                  'GDP growth in monthly perc.',
                  'Unemployment',
                  'Average workers per firm', 'Families\' median permanent income',
                  'Families\' median wages received', 'Families\' total commuting',
                  'Families\' absolute savings', 'Number of families receiving policy help',
                  'Amount of $ implemented by policy',
                  'Firms\' profit', 'Firms\' median stock',
                  'Firms\' average eco efficiency',
                  'Firms\' median wages paid',
                  'Firms\' median innovation investments',
                  'Total amount of emissions by firms', 'GINI index',
                  'Average families\' utility', 'Percentual families zero consumption',
                  'Percentual default among renting families', 'Monthly goods and services inflation',
                  'Average QLI index value', 'House vacancies', 'House prices', 'House rent prices',
                  'Affordable rent (less than 30% permanent income)',
                  'Percentual of delinquent loans', 'Taxes invested equally', 'Taxes invested locally',
                  'Taxes invested via FPM', 'Taxes paid by the banks on top of interests',
                  'Fund from taxes on emissions and subsidies on innovation efficiency',
                  'Total amount of intermediate goods sold by other metropolis',
                  ]

        # General plotting
        dats = [d.set_index('month') for d in dats]

        for col, title in zip(cols, titles):
            if dats_q1:
                if isinstance(dats_q1[0], pd.DataFrame):
                    fig = self.make_plot([d[col] for d in dats], title, labels,
                                         q1=[d[col] for d in dats_q1],
                                         q3=[d[col] for d in dats_q3])
            else:
                fig = self.make_plot([d[col] for d in dats], title, labels)
            self.save_fig(fig, '{}'.format(title))

    def plot_banks(self):
        labels, dats = self._load_multiple_runs('banks', 'banks.csv')

        cols = ['balance', 'active_loans', 'mortgage_rate', 'p_delinquent_loans',
                'mean_loan_age', 'mean_loan']
        titles = ['Bank balance', 'Bank active loans', 'Mortage rate',
                  'Bank perc. delinquent loans', 'Bank mean loan age', 'Bank mean loan amount']
        dats = [d.set_index('month') for d in dats]
        for col, title in zip(cols, titles):
            fig = self.make_plot([d[col] for d in dats], title, labels)
            self.save_fig(fig, 'banks_{}'.format(title))

    def plot_houses(self):
        dat = self._load_single_run('houses', 'houses.csv')

        to_plot = {
            'price': {
                'title': 'Mean house prices by month',
                'name': 'prices'
            },
            'on_market': {
                'title': 'Mean house time on market by month',
                'name': 'time_on_market'
            }
        }
        df = dat.groupby(['month', 'mun_id'], as_index=False).mean()
        for k, d in to_plot.items():
            title = d['title']
            name = d['name']
            dat_to_plot = df.pivot(index='month', columns='mun_id', values=k).astype(float)
            names_mun = [mun_codes[v] for v in list(dat_to_plot.columns.values)]
            fig = self.make_plot([dat_to_plot], title, labels=names_mun, y_label='Mean {}'.format(name))
            self.save_fig(fig, 'houses_{}'.format(name))

    def plot_families(self):
        dat = self._load_single_run('families', 'families.csv')
        dat['renting'] = pd.notna(dat['house_rent'])

        to_plot = {
            'house_rent': {
                'title': 'Mean rent value by month',
                'name': 'house_rents'
            },
            'total_wage': {
                'title': 'Mean total wage by month',
                'name': 'total_wages'
            },
            'savings': {
                'title': 'Mean savings by month',
                'name': 'savings'
            }
        }

        df = dat.groupby(['month', 'mun_id'], as_index=False).mean()
        for k, d in to_plot.items():
            title = d['title']
            name = d['name']
            dat_to_plot = df.pivot(index='month', columns='mun_id', values=k).astype(float)
            dats_to_plot = [dat_to_plot[c] for c in dat_to_plot.columns.values]
            names_mun = [mun_codes[v] for v in list(dat_to_plot.columns.values)]
            fig = self.make_plot(dats_to_plot, title, labels=names_mun, y_label='Median {}'.format(name))
            self.save_fig(fig, 'families_{}'.format(name))

    def plot_regional_stats(self):
        try:
            labels, dats = self._load_multiple_runs('regional', 'regional.csv')
        except MissingDataError:
            return  # Skip plotting if regional data is missing

        dats_q1, dats_q3 = None, None
        if self.avg:
            try:
                _, dats_q1 = self._load_multiple_runs('regional', 'q1_regional.csv')
                _, dats_q3 = self._load_multiple_runs('regional', 'q3_regional.csv')
            except MissingDataError:
                temp_path = self.avg[1] if self.avg else self.output_path.replace('plots', 'avg')
                dats_q1 = []
                dats_q3 = []
                for label, path in zip(self.labels, self.run_paths):
                    avg_path = path if label == self.avg[0] else path
                    q1_file = os.path.join(avg_path, 'q1_regional.csv')
                    q3_file = os.path.join(avg_path, 'q3_regional.csv')
                    if os.path.exists(q1_file) and os.path.exists(q3_file):
                        df_q1 = self._prepare_data(q1_file, dats[0].columns).set_index('month')
                        df_q3 = self._prepare_data(q3_file, dats[0].columns).set_index('month')
                        dats_q1.append(df_q1)
                        dats_q3.append(df_q3)

        cols = ['commuting', 'gdp_region', 'regional_gini', 'regional_house_values',
                'gdp_percapita', 'regional_unemployment', 'qli_index', 'pop',
                'treasure', 'licenses']
        titles = ['Commute', 'GDP', 'GINI', 'House values', 'per capita GDP',
                  'Unemployment', 'QLI index', 'Population', 'Total Taxes', 'Land licenses']

        base_cmap = cm.get_cmap('tab20', 40)

        for col, title in zip(cols, titles):
            title = 'Evolution of {} by region, monthly'.format(title)
            dats_to_plot, q1_to_plot, q3_to_plot = [], [], []
            region_labels = []
            colors = []
            color_index = 0

            for i, d in enumerate(dats):
                df = d.pivot(index='month', columns='mun_id', values=col).astype(float)
                q1_df, q3_df = None, None
                if dats_q1 and dats_q3:
                    q1_df = dats_q1[i].pivot(index='month', columns='mun_id', values=col)
                    q3_df = dats_q3[i].pivot(index='month', columns='mun_id', values=col)

                for c in df.columns:
                    dats_to_plot.append(df[c])
                    label = f"{labels[i]} - {mun_codes.get(c, c)}"
                    region_labels.append(label)
                    if q1_df is not None and q3_df is not None:
                        q1_to_plot.append(q1_df[c])
                        q3_to_plot.append(q3_df[c])
                    colors.append(base_cmap(color_index % base_cmap.N))
                    color_index += 1
            fig = self.make_plot(dats_to_plot, title, labels=region_labels,
                                 y_label='Regional {}'.format(col),
                                 q1=q1_to_plot if q1_to_plot else None,
                                 q3=q3_to_plot if q3_to_plot else None,
                                 colors=colors)

            self.save_fig(fig, 'regional_{}'.format(col))

    def plot_firms(self):
        dat = self._load_single_run('firms', 'firms.csv')

        cols = ['amount_produced', 'price']
        titles = ['Cumulative sum of amount produced by firm, by month', 'Price values by firm, by month']
        for col, title in zip(cols, titles):
            dat_to_plot = dat.pivot(index='month', columns='firm_id', values=col).astype(float)
            dats_to_plot = [dat_to_plot[c] for c in dat_to_plot.columns.values]
            labels = ['Firm {}'.format(i) for i, _ in enumerate(dat_to_plot.columns.values)]
            fig = self.make_plot(dats_to_plot, title, labels=labels, y_label='Values in units')
            self.save_fig(fig, '{}'.format(col))

        title = 'Median of number of employees by firm, by month'
        firms_stats = dat.groupby(['month', 'firm_id'], as_index=False).median()
        dat_to_plot = firms_stats.pivot(index='month', columns='firm_id', values='number_employees').astype(float)
        dats_to_plot = [dat_to_plot[c] for c in dat_to_plot.columns.values]
        labels = ['Firm {}'.format(i) for i, _ in enumerate(dat_to_plot.columns.values)]
        fig = self.make_plot(dats_to_plot, title, labels=labels, y_label='Median of employees')
        self.save_fig(fig, 'median_number_of_employees_by_firm_index')

    def plot_sectors(self):
        pass

    def plot_construction(self):
        dat = self._load_single_run('construction', 'construction.csv')
        cols = ['amount_produced', 'price']
        titles = ['Cumulative sum of amount produced by construction firms, by month',
                  'Price values of houses by firm, by month']
        for col, title in zip(cols, titles):
            dat_to_plot = dat.pivot(index='month', columns='firm_id', values=col).astype(float)
            dats_to_plot = [dat_to_plot[c] for c in dat_to_plot.columns.values]
            labels = ['Firm {}'.format(i) for i, _ in enumerate(dat_to_plot.columns.values)]
            fig = self.make_plot(dats_to_plot, title, labels=labels, y_label='Values in units')
            self.save_fig(fig, 'construction_{}'.format(col))

        title = 'Median of number of employees by construction firms, by month'
        firms_stats = dat.groupby(['month', 'firm_id'], as_index=False).median()
        dat_to_plot = firms_stats.pivot(index='month', columns='firm_id', values='number_employees').astype(float)
        dats_to_plot = [dat_to_plot[c] for c in dat_to_plot.columns.values]
        labels = ['Firm {}'.format(i) for i, _ in enumerate(dat_to_plot.columns.values)]
        fig = self.make_plot(dats_to_plot, title, labels=labels, y_label='Median of employees')
        self.save_fig(fig, 'construction_median_number_of_employees_by_firm_index')

    def plot_geo(self, sim, text):
        """Generate spatial plots"""
        figs = geo.plot(sim, text)
        for name, fig in figs:
            fig.savefig(os.path.join(self.output_path,
                                     'spatial_plot_{}_{}.{}'.format(name, text, conf.RUN['PLOT_FORMAT'])),
                        format=conf.RUN['PLOT_FORMAT'], dpi=600)
            # Reset figure
            plt.close(fig)
        return plt

    def plot_sectorial(self):
        pass
