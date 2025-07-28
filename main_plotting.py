import glob
import json
import os
from collections import defaultdict

import pandas as pd

import conf
from analysis import report
from analysis.output import OUTPUT_DATA_SPEC
from analysis.plotting import Plotter, MissingDataError


def conf_to_str(conf, delimiter='\n'):
    """Represent a configuration dict as a string"""
    parts = []
    for k, v in sorted(conf.items()):
        v = ','.join(v) if isinstance(v, list) else str(v)
        part = '{}={}'.format(k, v)
        parts.append(part)
    return delimiter.join(parts)


def average_run_data(path, avg='mean', n_runs=1):
    """Average the run data for a specified output path"""
    output_path = os.path.join(path, 'avg')
    os.makedirs(output_path)

    # group by filename
    file_groups = defaultdict(list)
    keep_files = {'{}.csv'.format(k): k for k in conf.RUN['AVERAGE_DATA']}
    for file in glob.glob(os.path.join(path, '**/*.csv')):
        fname = os.path.basename(file)
        if fname in keep_files:
            file_groups[fname].append(file)

    # merge
    for fname, files in file_groups.items():
        spec = OUTPUT_DATA_SPEC[keep_files[fname]]
        dfs = []
        for f in files:
            df = pd.read_csv(f,  sep=';', decimal='.', header=None)
            dfs.append(df)
        df = pd.concat(dfs)
        df.columns = spec['columns']

        # Saving date before averaging
        avg_cols = spec['avg']['columns']
        if avg_cols == 'ALL':
            avg_cols = [c for c in spec['columns'] if c not in spec['avg']['groupings']]

        # Ensure these columns are numeric
        df[avg_cols] = df[avg_cols].apply(pd.to_numeric)

        dfg = df.groupby(spec['avg']['groupings'])
        dfg = dfg[avg_cols]
        df = getattr(dfg, avg)()
        if n_runs > 1 and conf.RUN['SAVE_PLOTS_FIGURES']:
            std = getattr(dfg, 'std')()
            q1 = df - (2 * std)
            q3 = df + (2 * std)
        # "ungroup" by
        df = df.reset_index()
        df.to_csv(os.path.join(output_path, fname), header=False, index=False, sep=';')
        if n_runs > 1 and conf.RUN['SAVE_PLOTS_FIGURES']:
            q1 = q1.reset_index()
            q3 = q3.reset_index()
            q1.to_csv(os.path.join(output_path, 'q1_{}'.format(fname)), header=False, index=False, sep=';')
            q3.to_csv(os.path.join(output_path, 'q3_{}'.format(fname)), header=False, index=False, sep=';')
    return output_path


def plot(input_paths, output_path, params, logger, avg=None, sim=None, only=None):
    """Generate plots based on data in specified output path"""
    logger.info('Plotting to {}'.format(output_path))
    plotter = Plotter(input_paths, output_path, params, avg=avg)

    if conf.RUN['DESCRIPTIVE_STATS_CHOICE']:
        report.stats('')

    keys = ['general', 'firms',
            'regional_stats',
            'construction', 'houses',
            'families', 'banks']
    if only is not None:
        keys = [k for k in keys if k in only]

    if conf.RUN['SAVE_PLOTS_FIGURES']:
        for k in keys:
            try:
                logger.info('Plotting {}...'.format(k))
                getattr(plotter, 'plot_{}'.format(k))()
            except MissingDataError:
                logger.warn('Missing data for "{}", skipping.'.format(k))
                if avg is not None:
                    logger.warn('You may need to add "{}" to AVERAGE_DATA.'.format(k))

        if conf.RUN['PLOT_REGIONAL']:
            logger.info('Plotting regional...')
            plotter.plot_regional_stats()

    # Checking whether to plot or not
    if conf.RUN['SAVE_SPATIAL_PLOTS'] and sim is not None:
        logger.info('Plotting spatial...')
        plotter.plot_geo(sim, 'final')


def plot_runs_with_avg(run_data, logger, only=None):
    """Plot results of simulations sharing a configuration,
    with their average results"""
    # individual runs
    labels_paths = list(enumerate(run_data['runs']))

    # output to the run directory + /plots
    output_path = os.path.join(run_data['path'], 'plots')

    # plot
    only = ['general'] + only if only is not None else ['general']
    plot(input_paths=labels_paths,
         output_path=output_path,
         params={},
         logger=logger,
         avg=(run_data['avg_type'], run_data['avg']),
         only=only)


def plot_results(output_dir, logger):
    """Plot results of multiple simulations"""
    logger.info('Plotting results...')
    results = json.load(open(os.path.join(output_dir, 'meta.json'), 'r'))
    avgs = []
    for r in results:
        if not conf.RUN.get('SKIP_PARAM_GROUP_PLOTS'):
            plot_runs_with_avg(r, logger=logger, only=conf.RUN.get('AVERAGE_DATA'))

        # group averages, with labels, to plot together
        label = conf_to_str(r['overrides'], delimiter='\n')
        avgs.append((label, r['avg']))

    # plot averages
    if len(avgs) > 1:
        output_path = os.path.join(output_dir, 'plots')
        plot(input_paths=avgs,
             output_path=output_path,
             params={},
             logger=logger,
             only=conf.RUN.get('AVERAGE_DATA', ['general']))
