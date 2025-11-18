"""
This is the module to make entrance and that organizes the full simulation.
It handles all the choices of the model, set at the 'params' module.

Disclaimer:
This code was generated for research purposes only.
It is licensed under GNU v3 license
"""
import copy
import datetime
import itertools
import json
import logging
import os
import secrets

from glob import glob
from itertools import product

import click
import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import conf
import main_plotting
from simulation import Simulation
# from web import app

matplotlib.pyplot.close('all')
matplotlib.use('agg')

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO)


def single_run(params, path):
    """Run a simulation once for given parameters"""
    if conf.RUN['PRINT_STATISTICS_AND_RESULTS_DURING_PROCESS']:
        logging.basicConfig(level=logging.INFO)
    sim = Simulation(params, path)
    sim.initialize()
    sim.run()

    if conf.RUN['PLOT_EACH_RUN']:
        logger.info('Plotting run...')
        main_plotting.plot(input_paths=[('run', path)], output_path=os.path.join(path, 'plots'),
                           params=params, logger=logger, sim=sim)


def multiple_runs(overrides, runs, cpus, output_dir, fix_seeds=False):
    """Run multiple configurations, each `runs` times"""
    # overrides is a list of dictionaries with parameter name and value
    logger.info('Running simulation {} times'.format(len(overrides) * runs))

    if fix_seeds:
        seeds = [secrets.randbelow(2 ** 32) for _ in range(runs)]
    else:
        seeds = []

    # calculate output paths and params with overrides
    paths = [os.path.join(output_dir, main_plotting.conf_to_str(o, delimiter=';'))
             for o in overrides]
    params = []
    for o in overrides:
        p = copy.deepcopy(conf.PARAMS)
        p.update(o)
        params.append(p)

    # run simulations in parallel
    if cpus == 1:
        # run serially if cpus==1, easier debugging
        for p, path in zip(params, paths):
            for i in range(runs):
                if seeds:
                    p['SEED'] = seeds[i]
                single_run(p, os.path.join(path, str(i)))
    else:
        jobs = []
        for p, path in zip(params, paths):
            for i in range(runs):
                if seeds:
                    p['SEED'] = seeds[i]
                jobs.append((delayed(single_run)(p, os.path.join(path, str(i)))))
        Parallel(n_jobs=cpus, prefer='processes', backend='multiprocessing', batch_size=1)(jobs)

    logger.info('Averaging run data...')
    results = []
    for path, params, o in zip(paths, params, overrides):
        # save configurations
        with open(os.path.join(path, 'conf.json'), 'w') as f:
            json.dump({
                'RUN': conf.RUN,
                'PARAMS': params
            }, f,
                indent=4,
                default=str)

        # average run data and then plot
        runs = [p for p in glob('{}/*'.format(path)) if os.path.isdir(p)]
        avg_path = main_plotting.average_run_data(path, avg=conf.RUN['AVERAGE_TYPE'], n_runs=len(runs))

        # return result data, e.g. paths for plotting
        results.append({
            'path': path,
            'runs': runs,
            'params': params,
            'overrides': o,
            'avg': avg_path,
            'avg_type': conf.RUN['AVERAGE_TYPE']
        })
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(results, f,
                  indent=4,
                  default=str)

    main_plotting.plot_results(output_dir, logger)

    # link latest sim to convenient path
    latest_path = os.path.join(conf.RUN['OUTPUT_PATH'], 'latest')
    if os.path.isdir(latest_path):
        os.remove(latest_path)

    try:
        os.symlink(os.path.join('..', output_dir), latest_path)
    except OSError: # Windows requires special permissions to symlink
        pass

    logger.info('Finished.')
    return results


def gen_output_dir(command):
    timestamp = datetime.datetime.now().isoformat().replace(':', '_')
    run_id = '{}__{}'.format(command, timestamp)
    return os.path.join(conf.RUN['OUTPUT_PATH'], run_id)


@click.group()
@click.pass_context
@click.option('-n', '--runs', help='Number of simulation runs', default=1)
@click.option('-c', '--cpus', help='Number of CPU cores to use', default=1)
@click.option('-p', '--params', help='JSON of params override')
@click.option('-r', '--config', help='JSON of run config override')
def main(ctx, runs, cpus, params, config):
    if conf.RUN['SAVE_AGENTS_DATA'] is None:
        logger.warn('Warning!!! Are you sure you do NOT want to save AGENTS\' data?')

    # apply any top-level overrides, if specified
    if params:
        with open(params, 'r') as infile:
            params = json.load(infile)
    else:
        params = {}
    # params = json.loads(params) if params is not None else {}
    config = json.loads(config) if config is not None else {}
    conf.PARAMS.update(params) # applied per-run
    conf.RUN.update(config)    # applied globally

    ctx.obj = {
        'output_dir': gen_output_dir(ctx.invoked_subcommand),
        'runs': runs,
        'cpus': cpus
    }


@main.command()
@click.pass_context
def run(ctx):
    """
    Basic run(s) with different seeds
    """
    multiple_runs([{}], ctx.obj['runs'], ctx.obj['cpus'], ctx.obj['output_dir'])


@main.command()
@click.argument('params', nargs=-1)
@click.pass_context
def sensitivity(ctx, params):
    """
    Continuous param syntax: NAME:MIN:MAX:STEP
    Boolean param syntax: NAME
    """
    my_dict, permutations_dicts = dict(), list()
    p_name, p_vals = None, None
    for param in params:
        flag = None
        ctx.obj['output_dir'] = gen_output_dir(ctx.command.name)
        # if ':' present, assume continuous param
        if ':' in param:
            p_name, p_min, p_max, p_step = param.split(':')
            p_min, p_max = float(p_min), float(p_max)
            p_vals = np.linspace(p_min, p_max, int(p_step))
            # round to 8 decimal places
            p_vals = [round(v, 8) for v in p_vals]
        # TODO: Fix plots for starting-day sensitivity analysis.
        #  Yearly information refers to 2010-2020. Should go the whole period.
        elif "EMISSIONS" in param: 
            # The EMISSIONS sensitivity configuration follows the structure: 
            # python main.py -n x -c x sensitivity EMISSIONS... 
            # TODO: Add ACP groups as (Capital or all) 
            flag = True 
                # Define MCMV scenarios and define available interest values 
            my_dict = {'TAX_EMISSION': [0, .005], 
                        'TARGETED_TAX_SUBSIDIES': [False, True], 
                        'CARBON_TAX_RECYCLING': [False, True], 
                        'ECO_INVESTMENT_SUBSIDIES': [0,.15]} 
            ps = list(my_dict.keys()) 
            keys, values = zip(*my_dict.items()) 
            all_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)] 
            # Filter so that if POLICY_MCMV is False, the use only INTEREST='media' 
            for p in all_permutations: 
                if p['ECO_INVESTMENT_SUBSIDIES'] == 0 and (p['TARGETED_TAX_SUBSIDIES'] or p['CARBON_TAX_RECYCLING']): 
                    continue 
                #It's a valid combination 
                permutations_dicts.append(p) 
        elif param == 'STARTING_DAY':
            p_name = param
            p_vals = [datetime.date(2000, 1, 1), datetime.date(2010, 1, 1)]
        elif param == 'POLICIES':
            p_name = param
            p_vals = ['buy', 'rent', 'wage', 'no_policy']
        elif param == 'INTEREST':
            p_name = param
            p_vals = ['real', 'nominal', 'fixed']
        elif '-' in param:
            p_name = 'PROCESSING_ACPS'
            p_vals = [[i] for i in param.split('-')[1:]]
        elif '*' in param:
            flag = True
            # One should include first the params, separated by '+', then '*' and then the list of values also '+'
            # Such as 'param1+param2*1+2*10+20'.
            # Thus producing the dict: {'param1': ['10', '20'], 'param2': ['10', '20']}
            ps = param.split('*')[0]
            my_dict = {ps.split('+')[i]: [float(f) for f in param.split('*')[i + 1].split('+')]
                       for i in range(len(ps.split('+')))}
            keys, values = zip(*my_dict.items())
            permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        # Else, assume boolean
        else:
            p_name = param
            p_vals = [True, False]
        if not flag:
            ctx.obj['output_dir'] = ctx.obj['output_dir'].replace('sensitivity', p_name)
            confs = [{p_name: v} for v in p_vals]
        else:
            p_name = ps
            p_vals = my_dict.values()
            ctx.obj['output_dir'] = ctx.obj['output_dir'].replace('sensitivity', '_'.join(k for k in keys))
            confs = permutations_dicts.copy()
        # Fix the same seed for each run
        conf.RUN['KEEP_RANDOM_SEED'] = True
        # conf.RUN['FORCE_NEW_POPULATION'] = False # Ideally this is True, but it slows things down a lot
        conf.RUN['SKIP_PARAM_GROUP_PLOTS'] = True

        logger.info('Sensitivity run over {} for values: {}, {} run(s) each'.format(p_name, p_vals, ctx.obj['runs']))
        multiple_runs(confs, ctx.obj['runs'], ctx.obj['cpus'], ctx.obj['output_dir'])


@main.command()
@click.pass_context
def distributions(ctx):
    """
    Run across ALTERNATIVE0/FPM_DISTRIBUTION combinations
    """
    confs = [{
        'ALTERNATIVE0': ALTERNATIVE0,
        'FPM_DISTRIBUTION': FPM_DISTRIBUTION
    } for ALTERNATIVE0, FPM_DISTRIBUTION in product([True, False], [True, False])]

    logger.info('Varying distributions, {} run(s) each'.format(ctx.obj['runs']))
    multiple_runs(confs, ctx.obj['runs'], ctx.obj['cpus'], ctx.obj['output_dir'])


@main.command()
@click.pass_context
def distributions_acps(ctx):
    """
    Run across taxes combinations for all ACPs
    """
    confs = []
    dis = [{
        'ALTERNATIVE0': ALTERNATIVE0,
        'FPM_DISTRIBUTION': FPM_DISTRIBUTION
    } for ALTERNATIVE0, FPM_DISTRIBUTION in product([True, False], [True, False])]

    # ACPs with just one municipality
    exclude_list = ['CAMPO GRANDE', 'CAMPO DOS GOYTACAZES', 'FEIRA DE SANTANA', 'MANAUS',
                    'PETROLINA - JUAZEIRO', 'TERESINA', 'UBERLANDIA', 'SAO PAULO']
    all_acps = pd.read_csv('input/ACPs_BR.csv', sep=';', header=0)
    acps = set(all_acps.loc[:, 'ACPs'].values.tolist())
    acps = list(acps)
    for acp in acps:
        if acp not in exclude_list:
            dic0 = {'PROCESSING_ACPS': [acp]}
            for each in dis:
                confs.append(dict(dic0, **each))

    logger.info('Varying distributions, {} run(s) each'.format(ctx.obj['runs']))
    multiple_runs(confs, ctx.obj['runs'], ctx.obj['cpus'], ctx.obj['output_dir'])


@main.command()
@click.pass_context
def acps(ctx):
    """
    Run across ACPs
    """
    confs = []
    # ACPs with just one municipality
    exclude_list = ['SAO PAULO', 'RIO DE JANEIRO', 'BELO HORIZONTE']
    all_acps = pd.read_csv('input/ACPs_BR.csv', sep=';', header=0)
    acps = set(all_acps.loc[:, 'ACPs'].values.tolist())
    acps = list(acps)
    for acp in acps:
        if acp not in exclude_list:
            confs.append({
                'PROCESSING_ACPS': [acp]
            })
        else:
            confs.append({
                'PROCESSING_ACPS': [acp],
                'PERCENTAGE_ACTUAL_POP': .005
            })
    logger.info('Running over ACPs, {} run(s) each'.format(ctx.obj['runs']))
    multiple_runs(confs, ctx.obj['runs'], ctx.obj['cpus'], ctx.obj['output_dir'])


@main.command()
@click.argument('params', nargs=-1)
def make_plots(params):
    """
    (Re)generate plots for an output directory
    """
    output_dir = params[0]
    main_plotting.plot_results(output_dir, logger)
    if len(params) > 1:
        results = json.load(open(os.path.join(output_dir, 'meta.json'), 'r'))
        keys = ['general', 'firms', 'construction', 'houses', 'families', 'banks', 'regional_stats']
        for res in results:
            for i in range(len(res['runs'])):
                main_plotting.plot(input_paths=[('run', res['runs'][i])],
                                   output_path=os.path.join(res['runs'][i], 'plots'),
                                   params=res['params'],
                                   logger=logger,
                                   only=keys)
    else:
        print('To plot internal maps: enter True after output directory')


# @main.command()
# def web():
#     app.run(debug=False)


if __name__ == '__main__':
    main()
