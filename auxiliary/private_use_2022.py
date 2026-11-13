"""Private-vehicle use by ACP and work-income decile at the Census 2022 level.

Shape: the Census 2010 probabilities of a private vehicle by ACP and group (input/car_ownership_2010.csv, p_private,
group 0 without work income, 1-10 the deciles of work income).
Level: the private-vehicle share of the Census 2022 motorised commuters of the ACP (input/commute_mode_2022.csv,
Sidra table 10330: car, motorcycle, taxi and motorcycle taxi over these plus bus, BRT, van and rail).
Each ACP's odds are multiplied by one factor, the same for the eleven groups, so that the mean probability of groups
1-10 equals the 2022 share.

    python auxiliary/private_use_2022.py
writes input/private_use_2022.csv: acp;group;p_private;p_2010;share_2022;odds_factor
"""
import pandas as pd
from scipy.optimize import brentq

OWNERSHIP = 'input/car_ownership_2010.csv'
MODE_2022 = 'input/commute_mode_2022.csv'
OUT = 'input/private_use_2022.csv'
PRIVATE = ['car', 'motorcycle', 'taxi', 'motorcycle_taxi']
TRANSIT = ['bus', 'brt', 'van', 'rail']


def scaled(p, factor):
    odds = factor * p / (1.0 - p)
    return odds / (1.0 + odds)


def main():
    mode = pd.read_csv(MODE_2022, sep=';')
    persons = mode.groupby(['acp', 'mode']).persons.sum().unstack()
    share = persons[PRIVATE].sum(axis=1) / persons[PRIVATE + TRANSIT].sum(axis=1)
    cars = pd.read_csv(OWNERSHIP, sep=';')
    out = []
    for acp, target in share.items():
        t = cars[cars.acp == acp].sort_values('group')
        p = t.p_private.clip(1e-6, 1 - 1e-6)
        deciles = p[t.group > 0]
        factor = brentq(lambda f: scaled(deciles, f).mean() - target, 1e-4, 1e4)
        out.append(pd.DataFrame({'acp': acp, 'group': t.group, 'p_private': scaled(p, factor).round(4),
                                 'p_2010': t.p_private, 'share_2022': round(target, 4),
                                 'odds_factor': round(factor, 4)}))
        print('%-17s deciles 2010 %.3f, share 2022 %.3f, odds factor %.2f' % (acp, deciles.mean(), target, factor))
    pd.concat(out).to_csv(OUT, sep=';', index=False)


if __name__ == '__main__':
    main()
