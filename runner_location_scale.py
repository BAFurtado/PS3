"""
Calibration of the location choice and the commute acceptance: Teresina and Goiania, 2010-2020, seeds SEED + i
(i = 0-2, the seeds of runner_mcmv2026). Every run: CAR_OWNERSHIP census, TRANSPORT_WEIGHTS trips, base network (no
schedule), LOCATION_CHOICE logit, VACANCY_SCOPE region, TRANSPORT_MONEY fare. Settings:
    s0.03, s0.1, s0.3, s1.0   LOCATION_CHOICE_SCALE at that value, COMMUTE_ACCEPTANCE off
    acc2010, acc2022          LOCATION_CHOICE_SCALE 1, COMMUTE_ACCEPTANCE census / census2022
Targets: Census 2022 motorised commuters by mode (input/commute_mode_2022.csv): private-vehicle share, share up to
30 min of private and of transit commuters, mean minutes by mode (input/car_congestion_2022.csv). Saved: stats,
regional and the agents parquet every January.

Modes (cwd = repository root):
    python runner_location_scale.py lote CPUS [SETTING,...]   the batch, all settings by default; rerun the same
                                                              command to resume (finished runs skipped)

Output: output/location_scale/<CITY>/<setting>/<i>/; queue manifest and log in output/location_scale/queue/.
Local conf/params.py and conf/run.py are ignored.
"""
import copy
import datetime
import logging
import os
import shutil
import sys

import runner_funding2026 as R

OUTPUT = 'output/location_scale'
END = datetime.date(2021, 1, 1)
CITIES = ['GOIANIA', 'TERESINA']
SETTINGS = {'s{}'.format(x): {'LOCATION_CHOICE_SCALE': x, 'COMMUTE_ACCEPTANCE': 'off'} for x in (0.03, 0.1, 0.3, 1.0)}
SETTINGS['acc2010'] = {'LOCATION_CHOICE_SCALE': 1.0, 'COMMUTE_ACCEPTANCE': 'census'}
SETTINGS['acc2022'] = {'LOCATION_CHOICE_SCALE': 1.0, 'COMMUTE_ACCEPTANCE': 'census2022'}
RUNS = 3
COMMON = {
    'CAR_OWNERSHIP': 'census',
    'TRANSPORT_WEIGHTS': 'trips',
    'TRANSPORT_TIME': False,
    'TRANSPORT_SCHEDULE': None,
    'LOCATION_CHOICE': 'logit',
    'VACANCY_SCOPE': 'region',
    'TRANSPORT_MONEY': 'fare',
    'TOTAL_DAYS': (END - R.conf.PARAMS['STARTING_DAY']).days,
}
R.conf.RUN['SAVE_DATA'] = ['agents']
logger = logging.getLogger('main')


def run_path(city, setting, i):
    return os.path.join(OUTPUT, city, setting, str(i))


def build_jobs(settings):
    """Seed by seed, largest city first, so a batch stopped halfway holds complete grids."""
    jobs = []
    for i in range(RUNS):
        for city in CITIES:
            for setting in settings:
                path = run_path(city, setting, i)
                if os.path.exists(os.path.join(path, 'DONE')):
                    continue
                if os.path.exists(path):
                    shutil.rmtree(path)
                p = copy.deepcopy(R.conf.PARAMS)
                p.update(COMMON)
                p.update(SETTINGS[setting])
                p['PROCESSING_ACPS'] = [city]
                p['SEED'] = R.SEED + i
                jobs.append({'params': p, 'path': path})
    return jobs


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4) or sys.argv[1] != 'lote':
        sys.exit(__doc__)
    settings = sys.argv[3].split(',') if len(sys.argv) == 4 else list(SETTINGS)
    unknown = set(settings) - set(SETTINGS)
    if unknown:
        sys.exit('Unknown settings {}. Settings: {}'.format(sorted(unknown), ', '.join(SETTINGS)))
    from main import save_jobs, _run_jobs_parallel
    cpus = int(sys.argv[2])
    manifest = os.path.join(OUTPUT, 'queue')
    os.makedirs(manifest, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    jobs = build_jobs(settings)
    logger.info('%d of %d runs to do, %d at a time', len(jobs), RUNS * len(CITIES) * len(settings), cpus)
    save_jobs(manifest, jobs, cpus)
    _run_jobs_parallel(jobs, cpus)
