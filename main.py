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
from pathlib import Path
from joblib import Parallel, delayed

import conf
import main_plotting
from checkpoint import save_jobs, pending_jobs
from simulation import Simulation

# from web import app

matplotlib.pyplot.close('all')
matplotlib.use('agg')

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO)


def ensure_population_exists(params, path):
    """
    Ensure the population file exists BEFORE parallel execution.
    This runs serially.
    """
    os.makedirs(path, exist_ok=True)

    sim = Simulation(params, path)
    save_file = f"{sim.output.save_name}.agents"

    if not os.path.isfile(save_file) or conf.RUN["FORCE_NEW_POPULATION"]:
        logger.info("Pre-generating shared population...")
        sim.generate()   # <-- runs ONCE
    else:
        logger.info("Population already exists. Skipping generation.")


def single_run(params, path):
    """Run a simulation once for given parameters"""
    if conf.RUN['PRINT_STATISTICS_AND_RESULTS_DURING_PROCESS']:
        logging.basicConfig(level=logging.INFO)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'conf.json'), 'w') as f:
        json.dump({
            'RUN': conf.RUN,
            'PARAMS': params
        }, f, indent=4, default=str)
    sim = Simulation(params, path)
    sim.initialize()
    sim.run()
    open(os.path.join(path, 'DONE'), 'w').close()

    if conf.RUN['PLOT_EACH_RUN']:
        logger.info('Plotting run...')
        main_plotting.plot(input_paths=[('run', path)], output_path=os.path.join(path, 'plots'),
                           params=params, logger=logger, sim=sim)


def multiple_runs(overrides, runs, cpus, output_dir, fix_seeds=None):
    """Run multiple configurations, each `runs` times"""
    # overrides is a list of dictionaries with parameter name and value
    logger.info('Running simulation {} times'.format(len(overrides) * runs))

    # calculate output paths and params with overrides
    paths = [os.path.join(output_dir, main_plotting.conf_to_str(o))
             for o in overrides]
    params = []
    for o in overrides:
        p = copy.deepcopy(conf.PARAMS)
        p.update(o)
        params.append(p)

    for p, path in zip(params, paths):
        # use base path, not per-run path
        ensure_population_exists(p, path)

    job_specs = [
        {"path": os.path.join(path, str(i)), "params": p}
        for p, path in zip(params, paths)
        for i in range(runs)
    ]
    save_jobs(output_dir, job_specs, cpus)

    # run simulations in parallel
    if cpus == 1:
        # run serially if cpus==1, easier debugging
        for p, path in zip(params, paths):
            for i in range(runs):
                p = copy.deepcopy(p)
                if fix_seeds:
                    p['SEED'] = fix_seeds[i]
                single_run(p, os.path.join(path, str(i)))
    else:
        jobs = []
        for p, path in zip(params, paths):
            for i in range(runs):
                p = copy.deepcopy(p)
                if fix_seeds:
                    p['SEED'] = fix_seeds[i]
                jobs.append((delayed(single_run)(p, os.path.join(path, str(i)))))
        Parallel(n_jobs=cpus, prefer='processes', backend='loky', batch_size=1)(jobs)

    logger.info('Averaging run data...')
    results = []
    for path, base_params, o in zip(paths, params, overrides):
        per_run_confs = []
        run_dirs = [d for d in glob(f"{path}/*") if os.path.isdir(d)]
        run_dirs.sort(key=lambda d: int(os.path.basename(d)))  # ensure order is 0,1,...

        for run_path in run_dirs:
            conf_path = os.path.join(run_path, 'conf.json')
            if os.path.exists(conf_path):
                with open(conf_path, 'r') as f:
                    run_conf = json.load(f)
                    per_run_confs.append(run_conf['PARAMS'])
            else:
                per_run_confs.append(base_params)

        avg_path = main_plotting.average_run_data(path, avg=conf.RUN['AVERAGE_TYPE'], n_runs=len(run_dirs))

        results.append({
            'path': path,
            'runs': run_dirs,
            'params': per_run_confs,
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
    except OSError:  # Windows requires special permissions to symlink
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
    if conf.RUN['SAVE_DATA_PERIDIOCITY'] is None:
        logger.warn('Warning!!! Are you sure you do NOT want to save AGENTS\' data?')

    # apply any top-level overrides, if specified
    if params:
        with open(params, 'r') as infile:
            params = json.load(infile)
    else:
        params = {}
    # params = json.loads(params) if params is not None else {}
    config = json.loads(config) if config is not None else {}
    conf.PARAMS.update(params)  # applied per-run
    conf.RUN.update(config)  # applied globally

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

    # Create unique experiment directory (NO command prefix)
    run_id = datetime.datetime.now().isoformat().replace(':', '_')
    base_output = Path(conf.RUN['OUTPUT_PATH']) / run_id
    base_output.mkdir(parents=True, exist_ok=True)

    for param in params:

        flag = False
        my_dict = {}
        permutations_dicts = []
        p_name, p_vals = None, None

        # ----------------------------
        # PARAM PARSING
        # ----------------------------

        if ':' in param:
            p_name, p_min, p_max, p_step = param.split(':')
            p_min, p_max = float(p_min), float(p_max)
            p_vals = np.linspace(p_min, p_max, int(p_step))
            p_vals = [round(v, 8) for v in p_vals]

        elif '*' in param:
            flag = True
            parts = param.split('*')
            keys = parts[0].split('+')
            value_blocks = parts[1:]

            for key, block in zip(keys, value_blocks):
                raw_vals = block.split('+')

                if key in ['PROCESSING_ACPS', 'INTEREST', 'POLICIES']:
                    if key == 'PROCESSING_ACPS':
                        my_dict[key] = [[v] for v in raw_vals]
                    else:
                        my_dict[key] = raw_vals
                else:
                    my_dict[key] = [float(v) for v in raw_vals]

            keys, values = zip(*my_dict.items())
            permutations_dicts = [
                dict(zip(keys, v))
                for v in itertools.product(*values)
            ]

            p_name = "_".join(keys)
            p_vals = list(my_dict.values())

        elif "PLANHAB" in param:
            flag = True
            cities = param.split('-')[1:]
            capitais = [
                'ARACAJU', 'BELEM', 'BELO HORIZONTE', 'BRASILIA', 'CAMPO GRANDE',
                'CUIABA', 'CURITIBA', 'FLORIANOPOLIS', 'FORTALEZA', 'GOIANIA',
                'JOAO PESSOA', 'MACAPA', 'MACEIO', 'MANAUS', 'NATAL',
                'PORTO ALEGRE', 'RECIFE', 'SALVADOR', 'SAO LUIS', 'TERESINA', 'VITORIA'
            ]
            if cities[0].lower() == "capitais":
                cities = capitais
            my_dict = {
                "PROCESSING_ACPS": [[c] for c in cities],
                "POLICY_MELHORIAS": [True, False],
                "FUNDS_AVAILABILITY": [
                    "pessimista",
                    "tendencial",
                    "otimista"
                ]
            }

            keys, values = zip(*my_dict.items())
            permutations_dicts = [
                dict(zip(keys, v))
                for v in itertools.product(*values)
            ]
            p_name = "_".join(keys)
            p_vals = list(my_dict.values())

        elif '-' in param:
            p_name = 'PROCESSING_ACPS'
            p_vals = [[i] for i in param.split('-')[1:]]

        else:
            p_name = param
            p_vals = [True, False]

        # ----------------------------
        # BUILD CONFIGURATIONS
        # ----------------------------

        if not flag:
            confs = [{p_name: v} for v in p_vals]
        else:
            confs = permutations_dicts.copy()

        ctx.obj['output_dir'] = str(base_output)

        logger.info(
            f"Sensitivity run over {p_name} for values: {p_vals}, "
            f"{ctx.obj['runs']} run(s) each"
        )

        if conf.RUN.get('KEEP_RANDOM_SEED', False):
            fixed_seeds = [
                secrets.randbelow(2 ** 32)
                for _ in range(ctx.obj['runs'])
            ]
        else:
            fixed_seeds = []

        multiple_runs(
            confs,
            ctx.obj['runs'],
            ctx.obj['cpus'],
            ctx.obj['output_dir'],
            fix_seeds=fixed_seeds
        )


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
    all_acps = pd.read_csv('input/CONCURBs_BR.csv', header=0)
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
        keys = ['general', 'firms', 'construction', 'houses', 'families', 'banks', 'regional']
        for res in results:
            for i in range(len(res['runs'])):
                main_plotting.plot(input_paths=[('run', res['runs'][i])],
                                   output_path=os.path.join(res['runs'][i], 'plots'),
                                   params=res['params'],
                                   logger=logger,
                                   only=keys)
    else:
        print('To plot internal maps: enter True after output directory')


@main.command()
@click.argument('root_dir')
@click.option('-c', '--cpus', default=1, help='Number of CPU cores to use')
def resume(root_dir, cpus):
    """Resume an interrupted run from root_dir/jobs.json."""
    pending, cleaned = pending_jobs(root_dir)

    if not pending:
        logger.info('All jobs already completed — nothing to resume.')
        return

    logger.info(f'Cleaned {cleaned} partial run(s). Resuming {len(pending)} job(s)...')

    if cpus == 1:
        for job in pending:
            single_run(job['params'], job['path'])
    else:
        jobs = [delayed(single_run)(job['params'], job['path']) for job in pending]
        Parallel(n_jobs=cpus, prefer='processes', backend='loky', batch_size=1)(jobs)

    logger.info('Finished.')


# @main.command()
# def web():
#     app.run(debug=False)


if __name__ == '__main__':
    main()
