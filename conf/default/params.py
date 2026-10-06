import datetime

# MODEL PARAMETERS

# CLOSURE ######################################################
# Closure of the ACP's money and goods circuit. 'legacy': every parameter as set. 'open': the ACP as a small open
# economy, CLOSURE_OPEN overriding the parameters it names.
CLOSURE = 'legacy'
CLOSURE_OPEN = {
    'EXTERNAL_DEMAND_SPREAD': 'stock',
    'EXPORTS_REAL': True,
    'EXTERNAL_RECYCLING_SHARE': 0.0,
    'SHORTAGE_IMPORTS': True,
    'PRICE_DEMAND_RESPONSE': 0.1,
    'PRICE_INDEX': 'staffed',
    'IMPORT_PARITY_PRICING': True,
    'GOV_EXTERNAL_WAGE': 'national',
    'FUNDS_REAL': True,
    'INTEREST_HOUSING': 'real',
    'BANK_NATIONAL': True,
    'WEALTH_NORM': 'symmetric',
    'INITIAL_MONEY': 'target',
    'FIRM_PAYOUT': 'staff',
    'PI_START': 'census',
}

# FIRMS #########################################################
# Production function, labor with decaying exponent, Alpha for K. [0, 1]
PRODUCTIVITY_EXPONENT = 0.65
# How a firm's wage bill is split among its staff (also profit shares, own-account pool pay and the public pay unit).
# 'q_alpha': qualification ** PRODUCTIVITY_EXPONENT. 'census': that, times exp(b1 age + b2 age^2) and a persistent
# individual factor exp(e), e ~ N(0, sd^2); b1, b2 and sd per ACP from the Census 2010 work income of the employed
# (input/wage_dispersion_2010.csv, auxiliary/wage_dispersion.py), and each area's initial Census income is shared
# among its families in proportion to their members' weights instead of per person. Production keeps q ** alpha.
WAGE_SPLIT = 'q_alpha'
# How partners are matched. 'random': the start deals adults to families in random order and marriages pair random
# agents. 'census': each family's second adult at the start, and each marriage partner, has an education level drawn
# from the Census 2010 spouses of the partner's level in the ACP (input/spouse_education_2010.csv,
# auxiliary/spouse_education.py), the nearest level with agents if none is left.
FAMILY_MATCHING = 'random'
# Order of magnitude correction of production. Production divided by parameter
PRODUCTIVITY_MAGNITUDE_DIVISOR = 1
# Where the level of private output comes from. 'divisor': PRODUCTIVITY_MAGNITUDE_DIVISOR as set. 'municipal': after
# start-up hiring, the divisor that makes the value added of the private staff's capacity (national input coefficients)
# equal the IBGE 2010 value added per resident of the run's municipalities, net of imputed rent
# (input/municipal_va_2010.csv, auxiliary/municipal_va.py), in model money; replaces PRODUCTIVITY_MAGNITUDE_DIVISOR.
PRODUCTIVITY_LEVEL = 'divisor'
# GENERAL CALIBRATION PARAMETERS
# INTEREST: market/SELIC scenario. Choose: 'real', 'media', 'fixed'
INTEREST = "real"
# INTEREST_HOUSING: SBPE/FGTS regulated rate scenario for PlanHab. Choose: 'alta', 'media', 'baixa', or 'real': the
# 'media' SBPE and FGTS rates and the market mortgage rate deflated by expected inflation (auxiliary/real_housing_rates.py),
# the mortgage rate replacing the INTEREST file's
INTEREST_HOUSING = "media"
# By how much percentage to increase prices
MARKUP = 0.1
# Frequency firms change prices. Probability < than parameter
STICKY_PRICES = .7
# Price ruggedness a positive value (below 1) that multiplies the magnitude of price reduction
# Reflects a reluctance of businesses to lower prices. Amount estimated for reduction multiplied by parameter
PRICE_RUGGEDNESS = 0.1
# Maximum premium above market average before the inventory-driven price rise is capped.
# Decouples the inventory signal from the relative-price gate: firms respond to low inventory
# freely up to avg_prices * (1 + PRICE_MARKUP_CAP), then stop. Set to 0 for old joint-condition.
PRICE_MARKUP_CAP = 0.0875
# Safety-stock buffer: fraction of monthly sales firms want to hold above productive capacity.
# Higher values keep more firms in "low inventory" mode → more hiring signals, less deflation risk.
INVENTORY_TARGET_RATIO = 0.2
# Demand signal for production and hiring (step 2b). True: sales plus the quantity refused for lack of stock, so demand a
# firm could not serve (household, input or external) asks for more output. False (old model): sales only.
DEMAND_SIGNAL_UNMET = False
# Firms in the average goods price (avg_prices: markup ceiling, fall threshold, price of unstocked sectors; and the
# price_level / inflation in stats.csv). 'stocked': firms with staff and stock. 'staffed': firms with staff, stocked
# out or not.
PRICE_INDEX = 'stocked'
# Price response to refused demand, θ (step 2b/3). A firm that refused buyers for lack of stock this month raises its
# price, when it revises it (STICKY_PRICES), by θ × refused / (sold + refused), beyond PRICE_MARKUP_CAP, and does not
# lower it that month. Goods firms only (not Construction). 0 (old model): refusals do not move prices.
PRICE_DEMAND_RESPONSE = 0.0
# Number of firms consulted before consumption
SIZE_MARKET = 5
# A household refused (or served only in part) for lack of stock by the firm it picked tries the other stocked firms of
# the same sample, in the order of its strategy (price or distance), until the money is spent. False (old model): the
# rest goes back to savings. Refused quantity stays recorded at each firm that refused it (its demand signal).
HOUSEHOLD_RETRY = False
# Number of firms to buy from in the INTERMEDIATE market
INTERMEDIATE_SIZE_MARKET = 10
# Frequency firms enter the market
LABOR_MARKET = 0.8

# Monthly probability an employed worker separates (quits, contract end, etc.).
NATURAL_SEPARATION_RATE = 0.010
# Firms that pay no wages for this many consecutive months fire one worker per adjustment (0 = off).
# Tolerates short revenue droughts from erratic, small-sample demand; sheds staff only when they persist.
FIRE_UNPAID_MONTHS = 3
# Firm capital, in months of a firm's monthly cost (its staff's output at current price; Construction at least one
# median project: land plus the building cost it advances as wages before the first sale). Initial firms are sized
# after start-up hiring; entrants are funded from their sector's incumbents' capital above this buffer (reinvested
# earnings) and do not enter when that surplus is short, so entry creates no money; builders buy land only from cash
# above the buffer and not owed as wages; firms with no revenue advance 1/FIRM_CAPITAL_MONTHS of capital as wages.
# 0 = original: beta(1.5, 10) x 1e6 x IDHM for initial firms and entrants alike (~5,000 months of revenue, created
# at entry), 0.1% advanced.
FIRM_CAPITAL_MONTHS = 3
# Firm cash above the capital buffer. 'none': kept by the firm. 'staff': each month a private firm pays FIRM_PAYOUT_RATE
# of it to its employees, split as wages; the share counts in their families' permanent income. 'national': each month
# all of it leaves the firm: the corporate investment rate (FBCF / gross operating surplus, input/investment_rate_2015.csv)
# is spent the next month as investment demand with the national FBCF composition (final_demand.csv), the rest goes to
# owners outside the ACP (money_profits_out); entrants' capital comes from those owners (money_firm_entry).
FIRM_PAYOUT = 'none'
FIRM_PAYOUT_RATE = 1 / 6
# With FIRM_CAPITAL_MONTHS > 0: a construction firm's capital in months of its cost, and at least one median project
# (land plus the wages it advances before the first sale). Builders have no production credit and a 2-3 year project
# cycle, so they hold more than other firms. Their land purchases are recovered from revenue before wages over
# CONSTRUCTION_ACC_CASH_FLOW months, which rebuilds the buffer; the land gate itself keeps FIRM_CAPITAL_MONTHS.
CONSTRUCTION_CAPITAL_MONTHS = 12
# A firm exits after this many consecutive months insolvent (balance <= 0) or idle (no staff and no sales); its staff
# become unemployed, its remaining capital goes to its sector's firms, and it moves to sim.firm_grave. Government never
# exits; Construction only with no house for sale or under construction. 0 = original, no exit.
FIRM_EXIT_MONTHS = 6
# Firm output. 'capacity' (old model): every firm produces what its staff can each month, whatever its stock, and sheds
# staff only when it is overstocked and losing money. 'sales': private firms other than builders top their stock up to
# last month's sold plus refused quantity times (1 + INVENTORY_TARGET_RATIO), and to at least INVENTORY_TARGET_RATIO x
# capacity, within capacity, buying inputs for that output only; when it adjusts its labour force, a firm whose capacity
# alone exceeds that need sheds the excess workers, at most half its staff.
PRODUCTION_PLAN = 'capacity'
# Builders under PRODUCTION_PLAN 'sales'. 'pipeline': they produce at capacity and hire and shed on their house pipeline
# (pending houses short of stock, a profitable plot found, too many houses for sale). 'sales': they plan as the other
# private firms, their demand being goods sold plus refused plus the stock their completed houses used, and their stock
# target adding the cost of their cheapest pending house.
CONSTRUCTION_PLAN = 'pipeline'
# Firm count by sector. 'rais': RAIS 2010 shares grouped as Trade = CNAE G+I, Business = J+M+N, OtherServices = R+S+T,
# Government = O+P+Q+U. 'ibge12': the same shares in the IBGE nível 12 classification of the input-output matrix
# (Trade = G, Business = J, Government = O and public P/Q, OtherServices = I, M, N, R, S, T, U and private P/Q).
# 'census': Census 2010 employees aged 17-69 (domestic workers included) by sector in the nível 12 classification,
# public P/Q in Government as in 'ibge12'.
SECTOR_SHARES = 'rais'
# Education of the agents. 'pooled': each weighting area's distribution of years of study for people of all ages, those
# under 10 counted as without instruction (input/qualification_APs_2010.csv), drawn once per age and sex in each area;
# children gain a year at each birthday from 8 to 17 unless they drop out (17 %), newborns draw gamma(3, 3) years.
# 'census': the level is drawn per agent for its age group, the area's 10+ distribution (input/education_AP_2010.csv,
# Sidra 1554) reweighted by its municipality's age profile (input/education_age_2010.csv, Sidra 3572;
# auxiliary/education_age.py); under 25 the 25-29 level is drawn as the final one, held from the school completion ages
# (15, 18, 22); newborns draw from their mother's area (world/education.py).
EDUCATION = 'pooled'
# Output per unit of labour by sector: national output per job relative to the mean, IBGE national accounts 2015
# (input/sector_productivity.csv); builders keep 1. Needs SECTOR_SHARES 'ibge12' or 'census'. False = the same in every
# sector.
SECTOR_PRODUCTIVITY = False
# Firms refill workers lost to natural separation or death (one post each) unless shrinking. False = off.
REPLACE_SEPARATIONS = True
# Growing firms post the vacancies their production plan needs (gap between sales plus stock target and current
# output, over output per worker), capped at doubling headcount in a month. False = original one post a month.
PLANNED_GROWTH_POSTS = True
# Revised public sector. (1) Government headcount is set only by gov_hire_fire (RAIS target), not by the profit and
# insolvency firing, which emptied Government within a few years. (2) Balanced budget, per municipality
# (Funds.settle_government_budget): public revenue pays the public payroll, then government purchases (input-output
# ratio to payroll), then policy money, and the rest is spent as public investment (input-output FBCF shares), also
# recorded as the regions' applied public money for the QLI fiscal leg. False = original, which multiplied the
# equally-divided share by the number of municipalities, never paid the FPM and local shares, destroyed the regions'
# share, and let Government firms spend their start-up capital as demand.
GOV_REVISED = True
# Public wage rule (GOV_REVISED).
# 'premium': Government pays what private firms pay per unit of qualification (qualification ** PRODUCTIVITY_EXPONENT,
#   the same split as make_payment) times (1 + P), for the qualification it actually employs; it offers job seekers
#   (1 + P) x the municipality's mean private wage. P is the municipality's mix of public jobs by level of government
#   (input/gov_levels.csv, Ipea Atlas do Estado Brasileiro 2021; auxiliary/gov_levels.py) weighted by the premia below,
#   which are conditional on schooling, age, gender and race (World Bank, Um Ajuste Justo, 2017: federal 67 %, state
#   over 30 %, municipal none). Composition comes from the model's own sorting, not from the observed raw ratio.
# 'cempre_ratio': GOV_WAGE_RATIO x the observed raw public/private wage ratio (input/gov_wage_ratio.csv, IBGE CEMPRE;
#   auxiliary/gov_wage_ratio.py) x the mean private wage. It counts composition twice, as higher-paying firms also hire
#   the most qualified candidates first.
# 'uniform': GOV_WAGE_RATIO x the mean private wage.
GOV_WAGE_RULE = 'premium'
GOV_PREMIUM_FEDERAL = 0.67
GOV_PREMIUM_STATE = 0.30
GOV_PREMIUM_MUNICIPAL = 0.0
GOV_WAGE_RATIO = 1.0
# Federal and state staff are paid from national and state revenue, not from the taxes raised in the ACP. True: when a
# municipality's budget cannot pay its public payroll (and the purchases and inputs that go with it), the shortfall is
# funded from outside the ACP, up to the non-municipal share of that cost; the inflow is counted in
# Funds.external_public_funding. False: the budget caps the public wage.
GOV_EXTERNAL_FUNDING = True
# What federal and state staff are paid. 'local': the municipality's private pay per unit of qualification (per worker
# for 'cempre_ratio'/'uniform'), as municipal staff. 'real': the ACP's private pay divided by the average goods price
# (avg_prices, see PRICE_INDEX), in units of the import price (P_imp = 1); municipal staff keep local pay, and
# GOV_EXTERNAL_FUNDING pays at most the federal and state staff's cost. 'national': the observed multiple of private
# pay for each level (input/gov_pay.csv: Ipea Atlas do Estado Brasileiro state pay per level over the ACP's CEMPRE
# private pay, 2010; auxiliary/gov_pay.py) times the ACP's private pay per worker, the mean over GOV_PAY_BASE_MONTHS
# months after GOV_PAY_BURN_IN, then fixed in real terms; municipal staff and the cap as for 'real'.
GOV_EXTERNAL_WAGE = 'local'
GOV_PAY_BURN_IN = 12
GOV_PAY_BASE_MONTHS = 12
# The taxes pooled 'equally' (labour and firm taxes net of the FPM share, the state share of consumption tax, tax on
# bank interest, the import tax that returns) are federal and state revenue. True: they leave the ACP through the
# external account (money_public_taxes_out), so the net public transfer is GOV_EXTERNAL_FUNDING minus these. False: they
# stay and fund the municipalities' budgets, while GOV_EXTERNAL_FUNDING still pays in (money grows ~25 %/yr in PALMAS).
# Off until the model has the federal spending that comes back (pensions, Bolsa Família, SUS): with only the payroll
# transfer, BH and Goiânia pay out ~10x what returns and unemployment reaches 35 % (step test 2026-09-29).
PUBLIC_TAXES_OUT = False
# Federal benefits paid to residents from outside the ACP. 'off': none. 'data': RGPS benefits, BPC and Bolsa Família at
# the 2010 beneficiaries per resident and mean benefit of each municipality (input/social_transfers_2010.csv,
# auxiliary/social_transfers.py), fixed in 2010 R$: RGPS to the oldest, BPC to those without RGPS aged 65 or more and
# then in the poorest families, Bolsa Família to the poorest families (world/social_transfers.py); part of permanent
# income, counted in money_social_transfers.
SOCIAL_TRANSFERS = 'off'
# Public jobs per municipality and year (the Government sector's headcount target). 'rais': RAIS public jobs by
# employer's municipality (input/qtde_vinc_gov_rais_stable_from_2020_onwards.csv). 'census': Census 2010 public servants
# by municipality of residence times the national RAIS / Census ratio, growing along the ACP's own RAIS trend 2010-2020
# (input/gov_headcount_census.csv, auxiliary/gov_headcount.py).
GOV_HEADCOUNT = 'rais'
# INTERREGIONAL_TRADE 'iioas': when the trade base (local shares, exports) is set. 'census': month 1, household spending
# from the Census permanent income. 'rebase': recomputed once at month 4 with that spending scaled by the household
# income the model paid in months 2-3 (wages, profit shares, social transfers) over the month-1 permanent income,
# output at month-1 staff capacity.
TRADE_BASE = 'census'
# Private firms' wage bill as a share of value added (revenue - inputs). 'unemployment': exp(-unemployment x
# RELEVANCE_UNEMPLOYMENT_SALARIES). 'tru': the sector's (remunerations + gross mixed income) / value added in the 2015
# national accounts (input/firm_income_2015.csv, auxiliary/firm_income.py); RELEVANCE_UNEMPLOYMENT_SALARIES unused.
# Government keeps its own rule.
WAGE_SHARE = 'unemployment'
# The wages a family counts as income (permanent income, savings buffer). 'last': each member's last wage, kept after
# the member leaves the job or the firm stops paying. 'month': the wages paid in the latest monthly payroll.
FAMILY_WAGE = 'last'
# How newborns' sex is labelled and who counts as a man in mortality and fertility. 'mixed': newborns 'Male' / 'Female',
# only 'Male' takes male mortality, so agents generated at the start ('male') take female mortality and fertility.
# 'lower': newborns 'male' / 'female' as generated; men take male mortality and no fertility.
GENDER_LABELS = 'mixed'
# Where immigration's per-municipality population targets act. 'acp': immigrants filling a municipality's shortfall
# rent any vacant house in the ACP and a municipality's excess is removed from residents anywhere in the ACP.
# 'municipal': both within the municipality.
IMMIGRATION = 'acp'
# Population each municipality's immigration and emigration steer to. 'projection': the yearly estimate in
# input/Demografia/4_Pop_Estimatives_Munic at the run's scale. 'census': its population at the start grown at its
# yearly rate between the 2010 and 2022 Censuses (input/census_population_2010_2022.csv).
POP_TARGET = 'projection'
# Agents per area at the start. 'nearest': each area x sex x age cell of the Census rounded to the nearest agent.
# 'remainder': the area's total rounded once and its cells filled by largest remainder.
POP_ROUNDING = 'nearest'
# Own-account work (world/own_account.py), the Census 2010 share of the employed of each education level at the start.
# 'off': a resident without a firm job is unemployed. 'firms': own-account workers run one-person firms in their
# sector, opened on the Harris-Todaro comparison of earnings with (1 - u) times the private wage and funded from family
# savings; buyers draw sellers in proportion to their staff. 'pool': the own-account workers of a sector share its
# own-account part of every purchase of it (Census 2010 own-account share of its work income x its labour share);
# agents join and leave one at a time on the same comparison.
OWN_ACCOUNT = 'off'
# Vale-transporte (Lei 7.418/1985). True: employers pay each employee on public transport the fare above 6 % of the
# gross wage. False: workers pay all their fare.
VALE_TRANSPORTE = False
# Public investment, the municipal budget's residual after payroll and purchases. 'local': whatever the budget leaves.
# 'real': after GOV_PAY_BURN_IN + GOV_PAY_BASE_MONTHS months, held at its base months' real level, the difference paid
# from (or to) outside the ACP: federal and state spending there is set nationally, not by local revenue.
GOV_SPENDING = 'local'
# Education of a vacancy. 'off': none. 'census': each vacancy draws its level from the Census 2010 employees of its
# sector in the run's municipalities and is filled from applicants of that level; it stays open if none applies.
POSTING_EDUCATION = 'off'
# Percentage of employees' firms hired by distance
PCT_DISTANCE_HIRING = 0.2
# Ignore unemployment in wage base calculation if parameter is zero, else discount unemployment times parameter
RELEVANCE_UNEMPLOYMENT_SALARIES = 1.5
# Candidate sample size for the labor market
HIRING_SAMPLE_SIZE = 20

# Reduction size in case of eco innovation success: multiplies firm parameters
ENVIRONMENTAL_EFFICIENCY_STEP = .99
# Innovation process probability: 1 - exp(lambda * investment / wage_base)
ECO_INVESTMENT_LAMBDA = 10
# Adjustment factor for emissions within firms
EMISSIONS_PARAM = 1000

# GOVERNMENT ####################################################################
# ALTERNATIVE OF DISTRIBUTION OF TAXES COLLECTED. REPLICATING THE NOTION OF A COMMON POOL OF RESOURCES ################
# Alternative0 is True, municipalities are just normal as INPUT
# Alternative0 is False, municipalities are all together
ALTERNATIVE0 = True
# Apply FPM distribution as current legislation assign TRUE
# Distribute locally, assign FALSE
FPM_DISTRIBUTION = True
# alternative0  TRUE,           TRUE,       FALSE,  FALSE
# fpm           TRUE,           FALSE,      TRUE,   FALSE
# Results     fpm + eq. + loc,  locally,  fpm + eq,   eq

# POLICIES #######################################################################
# POVERTY POLICIES. If POLICY_COEFFICIENT=0, do nothing.
# Size of the budget designated to the policy
POLICY_COEFFICIENT = 0
# Policies alternatives may include: 'buy', 'rent' or 'wage' or 'no_policy'. For no policy set to empty strings ''
# POLICY_COEFFICIENT needs to be > 0.
POLICIES = "no_policy"

# POLICY_MCMV indicates whether MCMV is active
POLICY_MCMV = True
OGU_INVESTMENT = {'otimista': .25,
                  'tendencial': .09,
                  'pessimista': .04}
# FUNDS AVAILABILITY can be 'otimista', 'tendencial' or 'pessimista' [positive, tendencial or negative perspectives].
# NOTICE: It interferes on both OGU investment and FGTS AND SBPE investments
FUNDS_AVAILABILITY = 'tendencial'
# Programme funds from outside the ACP. False: the OGU and the FGTS and SBPE lines are their shares of last month's
# municipal GDP, and FGTS and SBPE instalments are paid to the local bank. True: the same shares of last month's GDP
# for FUNDS_BURN_IN + FUNDS_BASE_MONTHS months; then of the municipality's real GDP over the base months (GDP / the
# national real GDP index, input/national_real_gdp.csv) times this year's index; FGTS and SBPE instalments go back to
# the national funds, out of the ACP.
FUNDS_REAL = False
FUNDS_BURN_IN = 12
FUNDS_BASE_MONTHS = 12
INCOME_MODALIDADES = {'faixa1': .38,
                      # 'rural': .38,
                      'melhorias': .38,
                      'fgts': .65,
                      'sbpe': .85
                      }
# Scalar aliases — income quantile ceilings for subsidised credit channels.
# Families with permanent_income BELOW this quantile of the current income distribution
# qualify for that channel. Mirrors INCOME_MODALIDADES['fgts'/'sbpe'] but exposed as
# individual scalars so OAT sensitivity can sweep them without touching the dict.
FGTS_INCOME_QUANTILE = 0.70
SBPE_INCOME_QUANTILE = 0.85
# Mirrors INCOME_MODALIDADES['melhorias'] but exposed as a scalar so OAT sensitivity
# can sweep it without touching the dict.
MELHORIAS_INCOME_QUANTILE = 0.38
TOTAL_TARGETING_POLICY = False
POLICY_MELHORIAS = True
# Share of the works' market price the state pays for a melhorias upgrade. The works
# are the .5 -> 1 quality delta, worth `size * .5 * region.index` under
# House.update_price, which is the house's own pre-upgrade price. At 1 the builder
# earns the same revenue per unit of construction capacity as it would by putting that
# capacity into a new house and selling it, so the contract is capacity-neutral.
UPGRADE_COST = 1
POLICY_DAYS = 360
# Days until environmental policies start
ECO_POLICY_DAYS = 360 * 5
# Size of the poorest families to be helped
POLICY_QUANTILE = 0.2
# Change of policy for collecting consumption tax at:
# firms' municipalities origin (True) or destiny (consumers' municipality)
TAX_ON_ORIGIN = True
# BNDES test with (True) and without (False) TRANSPORT investments.
# Variation in time_travel implemented in labor market decisions -- BNDES test
TRANSPORT_TIME = False
# Opening schedule of the network (world/transport.py): list of [date, spec], date 'YYYY-MM-DD' or a year,
# spec a matrix state ('base', 'nec', ...) or phi in [0, 1] blending base into nec. Base before the first date.
# None: static network for the whole run, chosen by TRANSPORT_TIME.
# E.g. 14-year phase-in from 2026: transport.linear_phase_in(2026, 14)
TRANSPORT_SCHEDULE = None
# LOANS ##############################################################################
# Maximum age of borrower at the end of the contract
MAX_LOAN_AGE = 70
# Used to calculate monthly payment for the families, thus limiting maximum loan by number of months and age
# Because permanent income includes wealth, it should be just a small percentage,
# otherwise compromises monthly consumption.
LOAN_PAYMENT_TO_PERMANENT_INCOME = 0.35
# Refers to the maximum loan monthly payment to total wealth
# MAX_LOAN_PAYMENT_TO_WEALTH=.4
# Refers to the maximum rate of the loan on the value of the estate
MAX_LOAN_TO_VALUE = 0.8
# Subsidised credit channels allow higher LTV — FGTS/MCMV effectively 0-5% down payment
MAX_LOAN_TO_VALUE_FGTS = 0.95
MAX_LOAN_TO_VALUE_SBPE = 0.90
# This parameter refers to the total amount of resources available at the bank.
MAX_LOAN_BANK_PERCENT = 0.6
BANK_DEPOSIT_RESERVE = .2
# The bank's relation with the rest of Brazil. False: the bank's cash earns the policy rate every month, and a
# deposit earns, when withdrawn, that month's rate compounded over its age. True: deposits earn each month's rate every
# month; the cash earns nothing; every month the bank's equity (cash plus the remaining principal of market loans,
# minus deposits) returns to its initial value, the difference leaving the ACP or covered from outside.
BANK_NATIONAL = False

# HOUSING AND REAL ESTATE MARKET #############################################################
CAPPED_TOP_VALUE = 1.3
CAPPED_LOW_VALUE = 0.7
# Vacancy-price sensitivity: formula is 1 + (VACANCY_PRICE_REFERENCE - vacancy) * OFFER_SIZE_ON_PRICE
# Symmetric around the reference: tight markets (vacancy < reference) generate a premium;
# slack markets (vacancy > reference) generate a discount.
OFFER_SIZE_ON_PRICE = 5
# Vacancy rate at which prices sit at base level — the market equilibrium point.
VACANCY_PRICE_REFERENCE = 0.08
# TOO LONG ON THE MARKET:
# value=(1 - MAX_OFFER_DISCOUNT) * e ** (ON_MARKET_DECAY_FACTOR * MONTHS ON MARKET) + MAX_OFFER_DISCOUNT
# AS SUCH (-.02) DECAY OF 1% FIRST MONTH, 10% FIRST YEAR. SET TO 0 TO ELIMINATE EFFECT
ON_MARKET_DECAY_FACTOR = -0.02
# LOWER BOUND, THAT IS, AT LEAST 60% PERCENT OF VALUE WILL REMAIN AT END OF PERIOD, IF PARAMETER IS .6
MAX_OFFER_DISCOUNT = 0.65
# UPPER BOUND: maximum price premium in tight markets (vacancy near zero)
MAX_OFFER_PREMIUM = 1.3
# How strong construction firms respond to vacancy.
# Used in exponential suppression: P(skip) = 1 - exp(-vacancy * sensitivity).
# At equilibrium vacancy (8%) → ~65% skip; at 15% → ~86% skip; at 25% → ~96% skip.
BUILD_VACANCY_SENSITIVITY = 13
# Percentage of households pursuing new location (on average families move about once every 20 years)
# Brazilian households move on average every 15-20 years → 0.4-0.5% per month.
# At 2.5% the buyer pool (~470/month in BH) far exceeds monthly housing supply (~14),
# so the wealthiest buyers absorb all supply regardless of need-based scoring.
# At 0.5% the pool (~94/month) is closer to supply, allowing some months where
# available houses exceed active buyers and vacancy begins to accumulate.
PERCENTAGE_ENTERING_ESTATE_MARKET = 0.005
NEIGHBORHOOD_EFFECT = 0.2

# RENTAL #######################
# Reais (2010) per model money unit (κ). Converts Census income into each family's initial permanent income, which
# drives the first rental market at generation (#30). Step 1 of the units plan measured κ ≈ 880-1,100 from household
# income in Goiânia (≈ 1,100 from rents, 1,300-1,500 from wages)
REAIS_PER_MONEY_UNIT = 1000
INITIAL_RENTAL_SHARE = 0.40
# Monthly rent as a fraction of house price.
# At 0.003 this is 3.6% annual gross yield — in line with Brazilian urban rental markets.
# Also calibrates the financial attractiveness comparison in decision_enter_house_market:
# when the bank rate exceeds this yield, depositing savings is more profitable than buying.
INITIAL_RENTAL_PRICE = 0.002
# House values. 'legacy': price = size x quality x region index, monthly rent = price x INITIAL_RENTAL_PRICE, builders'
# cost in construction output = size x quality x productivity x index / HOUSE_PRODUCTION_ADEQUACY. 'data': rents keep
# that level, prices are rent x 12 / RENTAL_YIELD, and builders' cost is size x the Sinapi 2010 cost per m² of the state
# (input/sinapi_2010.csv) x the CUB/m² 2010 ratio of the quality's finish standard to the normal one (input/cub_2010.csv)
# x productivity over its mean, in money, plus land at LOT_COST of the house value (world/house_values.py)
HOUSE_VALUES = 'legacy'
# Gross rental yield, annual rent / price, FipeZAP 2010 national (HOUSE_VALUES 'data')
RENTAL_YIELD = 0.0664
# Maximum fraction of permanent income a household will commit to rent when choosing to move.
# 0.3 matches the Brazilian "comprometimento de renda" standard used in PlanHab/MCMV eligibility.
# Applies only to already-housed families in maybe_move; homeless families are unaffected.
MAX_RENT_TO_INCOME_RATIO = 0.3

# HOUSING PURCHASE DECISION #######################
# Minimum fraction of target house price that must be held in savings + bank deposits
# before a family enters the housing market. Enforces equity accumulation before buying
# (consistent with MAX_LOAN_TO_VALUE = 0.80, which already requires 20% equity at negotiation).
MIN_DOWN_PAYMENT_FRACTION = 0.20
# Months of current wages kept as a liquid emergency buffer before depositing surplus
# in the bank. Anchored to wages (not permanent income) so the buffer tracks actual
# cash needs rather than compounding with house appreciation or interest returns.
# 3 months matches standard financial-planning guidance for employed households.
SAVINGS_BUFFER_MONTHS = 2
# Scales the opportunity-cost term in decision_enter_house_market.
# opportunity_cost = max(0, bank_rate - INITIAL_RENTAL_PRICE) × HOUSING_FINANCIAL_WEIGHT
# This is now an absolute-difference formula (not normalized), so the weight is larger than
# the old normalized version. At SELIC ≈ 10% annual (bank_rate ≈ 0.008/month):
#   opportunity_cost ≈ (0.008 - 0.002) × 25 = 0.15
# A renter (housing_need=1.0) scores 0.85 > 0 → enters.
# A comfortable owner (housing_need=0) scores −0.15 → excluded.
# A crowded owner (crowding_bonus=0.7) scores 0.55 → enters to upgrade.
# At low SELIC (≈ 2%, bank_rate ≈ 0.0017): opportunity_cost ≈ 0 → some owners enter.
HOUSING_FINANCIAL_WEIGHT = 60
# Minimum months of permanent income that must remain liquid after the down payment.
# Discourages families from locking all savings into a house and being cash-poor.
# At 6 months: a family spending 100% of available savings on a down payment scores
# a full liquidity_penalty of 1.0, reducing their entry score significantly.
LIQUIDITY_BUFFER_MONTHS = 6

# CONSTRUCTION #################################################################################
# LICENSES ARE URBANIZED LOTS AVAILABLE FOR CONSTRUCTION PER NEIGHBORHOOD PER MONTH.
# Expected number of NEW licenses created monthly by region (neighborhood). Set to 0 for no licenses.
# Reduced from 3 → 1: at 3/region the pool accumulated so fast that licenses never
# constrained which regions could be built in; at 1/region repeatedly chosen profitable
# regions can run short, providing a secondary throttle alongside BUILD_VACANCY_SENSITIVITY.
EXPECTED_LICENSES_PER_REGION = 1.5
# Minimum total licenses issued city-wide per month, regardless of number of regions.
# Small cities (few neighborhoods) have proportionally more free urban land and should not
# be starved by low per-region rates. Effective rate = max(EXPECTED_LICENSES_PER_REGION, floor/n_regions).
# At 6: an 8-region city gets max(0.65, 0.75)=0.75/region; a 76-region city is unchanged at 0.65.
# Set to 0 to disable (pure per-region Poisson with no floor).
LICENSE_MIN_CITY_MONTHLY = 10
# PERCENT_CONSTRUCTION_FIRMS = 0.07 This has been deprecated with the introduction of sectors
# Months that construction firm will divide its income into monthly revenue installments.
# Although prices are accounted for at once.
CONSTRUCTION_ACC_CASH_FLOW = 12
# Cost of lot in PERCENTAGE of construction
LOT_COST = 0.15
# Initial percentage of vacant houses
HOUSE_VACANCY = 0.1
# MAX_NUMBER OF HOUSES IN STOCK
MAX_HOUSE_STOCK = 36
# Categories of submarkets for the housing markets
PERC_HOUSE_CATEGORIES = [0.4, 0.3, 0.2, 0.1]
# HOW LARGER IS CONSTRUCTION FIRMS PROFIT RELATIVE TO USUAL MARKUP (firms' productivity, given current prices)
CONSTRUCTION_FIRM_MARKUP_MULTIPLIER = 5
# Bridges the scale gap between construction firm labor output (sum of qual^alpha per month, ~3-8 units)
# and building_size in square metres (~60-200 m²). Without this divisor a median house requires ~190
# production-units, meaning 25-60 months of dedicated firm output — far too slow.
# At 15: median cost ≈ 13 units → ~4 months throughput per house for a 10-employee firm,
# equivalent to maintaining 5 concurrent projects each individually taking ~20 months.
HOUSE_PRODUCTION_ADEQUACY = 12

# POPULATION AND DEMOGRAPHY
# Families run parameters (on average) for year 2000, or no information. 2010 uses APs average data
EXOGENOUS_HEAD_RATE = False
MEMBERS_PER_FAMILY = 2.5
MARRIAGE_CHECK_PROBABILITY = 0.03

# CONSUMPTION #############################################################
# Fraction of permanent income actually spent on goods; remainder flows to savings.
CONSUMPTION_PROPENSITY = 1
# Household wealth norm, on liquid wealth (cash and bank deposits) against WEALTH_TARGET_MONTHS of permanent income.
# 'off': families consume their permanent income. 'symmetric': spending moves by WEALTH_ADJUSTMENT of the gap each
# month, up above the target and down below it, drawing on deposits for any spending cash cannot cover. 'dissave': only
# the excess above the target is spent.
WEALTH_NORM = 'off'
WEALTH_TARGET_MONTHS = 4.5
WEALTH_ADJUSTMENT = 1 / 12
# Months from the start before the norm applies
WEALTH_NORM_BURN_IN = 24
# Start of the permanent-income average. 'reset': the first monthly update, before any wage is paid, replaces the
# Census permanent income. 'census': the 24-month average starts full of the Census permanent income.
PI_START = 'reset'
# Employment at the start: initial hiring stops when those aged 17-69 without a job fall to this share of them, the
# definition of the unemployment statistic. 'legacy': 0.086, the mean unemployment rate of six metropolitan regions in
# January 2000. 'census': the Census 2010 share in the run's municipalities (input/nonemployment_2010.csv, Sidra 1572;
# auxiliary/census_nonemployment.py).
INITIAL_EMPLOYMENT = 'legacy'
# Labour force. 'off': everyone aged 17-69 without a job seeks one and counts as unemployed. 'census': an agent aged
# 17-69 is active with the Census 2010 share for its sex, age group and municipality (input/participation_2010.csv,
# Sidra 3573; auxiliary/census_participation.py), from a draw it keeps; the inactive do not seek jobs, leave the job they
# hold and are left out of unemployment, which becomes the share of the active without a job. Under 'census',
# INITIAL_EMPLOYMENT 'census' starts at the Census share of the active without a job.
PARTICIPATION = 'off'
# Initial money. 'lognormal': each agent holds a lognormal(3, 0.5) draw of model money. 'target': agents aged 10+ hold
# WEALTH_TARGET_MONTHS of their area's Census income per person, times their draw over its mean; younger ones none.
# Immigrants hold the same, at the ACP's initial income per person.
INITIAL_MONEY = 'lognormal'
# Fraction of accumulated balance government firms spend each month; remainder carried forward.
GOVERNMENT_EXECUTION_RATE = 1

# TAXES ##################################################################
TAX_CONSUMPTION = 0.15
TAX_LABOR = 0.15
TAX_ESTATE_TRANSACTION = 0.004
TAX_FIRM = 0.15
TAX_PROPERTY = 0.012
TAX_TRANSPORT = 0

# EMISSIONS POLICIES ######################################################
# Taxes on emission are given by tax * total_emissions. Roughly R$ * tonCO2. .1 is about R$10
# Brazil taxes no emissions: 0 in the baseline; policy arms set it (runner_emissions.py)
TAX_EMISSION = 0
# Subsidies in (0,1) is the amount of investment paid by the gov(subsidies * total_invested)
# 0 is none, 1 is full. With TAX_EMISSION = 0 a subsidy is paid unfunded (Firm.decision_on_eco_efficiency),
# so the baseline keeps both off
ECO_INVESTMENT_SUBSIDIES = 0
TARGETED_SUBSIDIES = False
TARGETED_SECTORS = ['Agriculture', 'Transport', 'Utilities']
CARBON_TAX_RECYCLING = False
CARBON_RECYCLING_QUANTILE = 0.25

# Consumption_equal: ratio of consumption tax distributed at state level (equal)
# Fpm: ratio of 'labor' and 'firm' taxes distributed per the fpm ruling
TAXES_STRUCTURE = {"consumption_equal": 0.1875, "fpm": 0.235}

# TRANSPORT ######################################################################################
# Cobb-Douglas parameters for matching utility:
# log(U) = α log_qualification + β log_commuting + γ log_wages
# GAMMA is 1 - alpha - beta
# Emphasizes qualification and wages (0.4) equally, with lesser weight (0.2) on commuting time.
CB_QUALIFICATION = .35
CB_COMMUTING = .2

# Wage deciles that place job candidates in WAGE_TO_CAR_OWNERSHIP_QUANTILES, from a half sample of agents.
# 'all': every sampled agent's last wage, zero included. 'employed': sampled agents in a job with a positive wage.
CAR_DECILES = 'all'
WAGE_TO_CAR_OWNERSHIP_QUANTILES = [
    0.1174,
    0.1429,
    0.2303,
    0.2883,
    0.3395,
    0.4667,
    0.5554,
    0.6508,
    0.7779,
    0.9135,
]
# PUBLIC_TRANSIT_COST and PRIVATE_TRANSIT_COST reflect perceived commuting penalties,
# with higher values indicating greater sensitivity to distance when evaluating job offers.
PRIVATE_TRANSIT_COST = .25
PUBLIC_TRANSIT_COST = .05
REGIONAL_FREIGHT_COST = .3
# Price of goods bought from the rest of Brazil. 'flat': every import costs P_imp = 1 plus REGIONAL_FREIGHT_COST, and
# so does the import-parity ceiling of tradables (IMPORT_PARITY_PRICING). 'margins': imports cost P_imp, since the
# national input coefficients and final-demand shares buy transport margins from Transport; the import-parity ceiling
# of a tradable product is 1 plus its national transport margin (input/transport_margins.csv, IBGE TRU 2015;
# auxiliary/transport_margins.py).
FREIGHT = 'flat'
# Trade with the rest of Brazil (defect #27). True: firms buy the imported part of their inputs, from the external->local
# block of the regional input-output matrix, so local + imported inputs sum to the national coefficients. False (old
# model): they read the local->external block, which is ~0, and buy only the local share of their inputs.
IO_IMPORTS = False
# Share of the ACP's monthly import bill (net of the import tax that returns) that comes back as demand for its products
# from the rest of Brazil, split across sectors like its exports. 1: balanced trade. 0: imports leave the ACP for good
# (old model). Between: a trade deficit, recorded in stats.csv ext_net_position.
EXTERNAL_RECYCLING_SHARE = 0.0
# How the rest of Brazil's demand (exports and recycled imports) reaches local firms (step 2b). 'stock': every firm of
# the sector with stock, in proportion to the value of its stock. 'cheapest' (old model): split equally over the 10
# cheapest stocked firms of a sample of 3 x SIZE_MARKET, which run out while the rest of the sector keeps its stock.
EXTERNAL_DEMAND_SPREAD = 'cheapest'
# Real Estate in household goods demand. True (old model): households spend the input-output share of Real Estate (actual
# and imputed rent) at Real Estate firms. False: that share is 0 and the other sectors' shares are rescaled to sum to 1;
# households pay rent only in the rental market.
HOUSEHOLD_REAL_ESTATE = True
# Households buy a share of their consumption of each tradable sector from the rest of Brazil (step 2b): the ACP's import
# share of that product in the regionalised technical matrix, imported / (local + imported) over all buying sectors (the
# location quotients IO_IMPORTS uses). Services stay local. Household totals: BH 0.15, GYN 0.17, BSB 0.22, PMW 0.24,
# SP 0.05. The final-demand files cannot give it: their rest-of-Brazil -> local household block is 0 in every ACP.
# False (old model): households buy only from local firms.
HOUSEHOLD_IMPORTS = False
HOUSEHOLD_IMPORT_SECTORS = ['Agriculture', 'Mining', 'Manufacturing']
# Price of imported inputs (step 2b). 'exogenous': 1, the initial goods price held in real terms, plus freight, so local
# price rises do not feed back into import prices. 'local' (old model): the local seller's price plus freight.
IMPORT_PRICE = 'local'
# Exports. False: each sector's external demand is its export multiplier (final-demand files) times this month's
# internal demand. True: that rule for EXPORTS_BURN_IN + EXPORTS_BASE_MONTHS months; then the base months' mean
# quantity, times the national real GDP index (input/national_real_gdp.csv) relative to the base months, times
# (sector price / P_imp) ** -EXPORTS_PRICE_ELASTICITY, with P_imp = 1.
EXPORTS_REAL = False
# Sectors whose goods are traded with the rest of Brazil at P_imp = 1 plus freight (FREIGHT).
TRADABLE_SECTORS = ['Agriculture', 'Mining', 'Manufacturing']
# True: household and government spending on TRADABLE_SECTORS that no local firm served (no stock, or refused after
# the retry) is bought outside at P_imp plus freight. False: households keep it, government funds carry it over.
SHORTAGE_IMPORTS = False
# True: firms in TRADABLE_SECTORS price against the tradable average, their ceiling is the lower of that average times
# (1 + PRICE_MARKUP_CAP) and import parity (FREIGHT), and refused demand does not raise
# their price (PRICE_DEMAND_RESPONSE); the other firms price against the non-tradable average. False: every firm
# prices against the average of all firms.
IMPORT_PARITY_PRICING = False
EXPORTS_BURN_IN = 12
EXPORTS_BASE_MONTHS = 12
EXPORTS_PRICE_ELASTICITY = 1.0
# Trade with the rest of Brazil. 'files' (old model): IO_IMPORTS, HOUSEHOLD_IMPORTS and the export rule above.
# 'iioas': the rule of the Brazilian interstate input-output system (Haddad et al.) for every buyer. The local share of
# product i is s_i = TRADE_POTENTIAL[i] x min(local output_i / local demand_i, 1); firms buy s_i of their national input
# coefficients locally and the rest outside, households and government import 1 - s_i of their spending, at P_imp = 1
# plus freight (FREIGHT). Exports of i = local output_i - s_i x local demand_i. Output (staff capacity) and demand
# (inputs, household and government spending, fares) are measured in month 1, which uses s_i = TRADE_POTENTIAL[i]; the
# shares are then fixed, and exports held in quantity times the national real GDP index relative to month 1, times
# (price / P_imp) ** -EXPORTS_PRICE_ELASTICITY. Construction and Government: s_i = TRADE_POTENTIAL[i], no exports.
# Written to trade_base.csv.
INTERREGIONAL_TRADE = 'files'
# Haddad et al. (2019, p. 614), F: 0.5 for products 1-87 (agriculture, mining, manufacturing), 0.9 for products 88-128
# (utilities, construction, trade and services)
TRADE_POTENTIAL = {'Agriculture': 0.5, 'Mining': 0.5, 'Manufacturing': 0.5, 'Utilities': 0.9, 'Construction': 0.9,
                   'Trade': 0.9, 'Transport': 0.9, 'Business': 0.9, 'Financial': 0.9, 'RealEstate': 0.9,
                   'OtherServices': 0.9, 'Government': 0.9}

# RUN DETAILS ###############################################################################
# Percentage of actual population to run the simulation
# Minimum value to run depends on the size of municipality 0,001 is recommended minimum
PERCENTAGE_ACTUAL_POP = 0.01

# QLI / IDHM DEVELOPMENT ######################################################################
# Monthly growth rate scaling factor. Calibrated so a municipality at the reference
# development level (QLI_GDP_NORM) grows ≈ +0.0007/month = +0.008/year, consistent
# with Brazilian IDHM improvement of ~0.006–0.010/year in 2010–2020.
# The logistic ceiling naturally slows growth as QLI approaches QLI_MAX.
QLI_GROWTH_RATE = 0.002
# Theoretical ceiling for the QLI/IDHM index (HDI max = 1.0).
QLI_MAX = 1.0
# Reference GDP per capita (model units) representing a typical Brazilian state capital
# at mid-calibration (≈ 2015). Sets the scale so that economic_driver ≈ 1.0 for an
# average city. Richer cities (higher GDP/pop) develop faster; poorer ones slower.
# Estimated from wave calibration data: median GDP/pop across capitals ≈ 3.5.
QLI_GDP_NORM = 3.5
# Weight of the FISCAL leg of the QLI driver: the share of the monthly growth impulse
# that comes from public spending per capita rather than from GDP per capita.
#   driver = (1 - w) * sqrt(gdp_pc / QLI_GDP_NORM) + w * sqrt(spend_pc / QLI_SPEND_NORM)
# At w = 0 the model behaves exactly as before this parameter existed, so the fiscal
# channel is a designed arm, not a silent change of the baseline. It is what lets a
# government move neighbourhood quality — and through region.index, house prices — by
# collecting or transferring differently; FPM is redistributive, so spend per capita
# varies across municipalities independently of GDP per capita.
QLI_TAX_WEIGHT = 0.0
# Reference public spending per capita (model units, monthly), the fiscal counterpart of
# QLI_GDP_NORM. Set so that fiscal_driver ≈ economic_driver for an average city, which is
# what keeps the w = 1 baseline comparable with the w = 0 one:
#   QLI_SPEND_NORM = mean(spend_pc / gdp_pc) * QLI_GDP_NORM
# Measured at w = 0 over 2010-2040 on GOIANIA / MACAPA / CUIABA / ARACAJU: mean of the four
# city means of spend_pc/gdp_pc is 0.3342, so 0.3342 x 3.5 = 1.170. Cities are weighted
# equally rather than by municipality count, so Goiania's 15 municipalities do not set it
# alone. See model_defects.md §QLI.
QLI_SPEND_NORM = 1.170

# Write exactly like the list below
PROCESSING_ACPS = ["GOIANIA"]

# Selecting the starting year to build the Agents can be: 1991, 2000 or 2010
STARTING_DAY = datetime.date(2010, 1, 1)

# The Maximum running time (restrained by official data) is 30 years,
TOTAL_DAYS = (datetime.date(2040, 1, 1) - STARTING_DAY).days

# Select the possible ACPs (Population Concentration Areas) from the list below.
# Actually, they are URBAN CONCENTRATION AREAS FROM IBGE, 2022

"""
ABAETETUBA
ACAILANDIA
ALAGOINHAS
AMERICANA - SANTA BARBARA D'OESTE
ANAPOLIS
ANGRA DOS REIS
APUCARANA
ARACAJU
ARACATUBA
ARAGUAINA
ARAGUARI
ARAPIRACA
ARAPONGAS
ARARAQUARA
ARARAS
ARARUAMA
ATIBAIA
BACABAL
BAGE
BAIXADA SANTISTA
BARBACENA
BARREIRAS
BARRETOS
BAURU
BELEM
BELO HORIZONTE
BENTO GONCALVES
BIRIGUI
BLUMENAU
BOA VISTA
BOTUCATU
BRAGANCA
BRAGANCA PAULISTA
BRASILIA
BRUSQUE
CABO FRIO
CACHOEIRO DE ITAPEMIRIM
CAMETA
CAMPINA GRANDE
CAMPINAS
CAMPO GRANDE
CAMPOS DOS GOYTACAZES
CARAGUATATUBA - UBATUBA - SAO SEBASTIAO
CARUARU
CASCAVEL
CASTANHAL
CATALAO
CATANDUVA
CAXIAS
CAXIAS DO SUL
CHAPECO
CODO
COLATINA
CONSELHEIRO LAFAIETE
CRICIUMA
CUIABA
CURITIBA
DIVINOPOLIS
DOURADOS
EUNAPOLIS
FEIRA DE SANTANA
FLORIANOPOLIS
FORMOSA
FORTALEZA
FRANCA
GARANHUNS
GOIANIA
GOVERNADOR VALADARES
GUARAPARI
GUARAPUAVA
GUARATINGUETA
ILHEUS
IMPERATRIZ
INDAIATUBA
INTERNACIONAL DE CORUMBA
INTERNACIONAL DE FOZ DO IGUACU
INTERNACIONAL DE PEDRO JUAN CABALLERO
INTERNACIONAL DE SANT'ANA DO LIVRAMENTO
INTERNACIONAL DE URUGUAIANA
IPATINGA
ITABIRA
ITABUNA
ITAJAI - BALNEARIO CAMBORIU
ITAJUBA
ITAPETININGA
ITAPIPOCA
ITATIBA
ITU - SALTO
JARAGUA DO SUL
JAU
JEQUIE
JI-PARANA
JOAO PESSOA
JOINVILLE
JUAZEIRO DO NORTE
JUIZ DE FORA
JUNDIAI
LAGES
LAJEADO
LAVRAS
LIMEIRA
LINHARES
LONDRINA
MACAE - RIO DAS OSTRAS
MACAPA
MACEIO
MANAUS
MARABA
MARILIA
MARINGA
MOGI GUACU - MOGI MIRIM
MONTES CLAROS
MOSSORO
MURIAE
NATAL
NOVA FRIBURGO
OURINHOS
PALMAS
PARANAGUA
PARAUAPEBAS
PARINTINS
PARNAIBA
PASSO FUNDO
PASSOS
PATOS
PATOS DE MINAS
PAULO AFONSO
PELOTAS
PETROLINA
PETROPOLIS
PIRACICABA
POCOS DE CALDAS
PONTA GROSSA
PORTO ALEGRE
PORTO SEGURO
PORTO VELHO
POUSO ALEGRE
PRESIDENTE PRUDENTE
RECIFE
RESENDE
RIBEIRAO PRETO
RIO BRANCO
RIO CLARO
RIO DE JANEIRO
RIO GRANDE
RIO VERDE
RONDONOPOLIS
SALVADOR
SANTA CRUZ DO SUL
SANTA MARIA
SANTAREM
SAO BENTO DO SUL - RIO NEGRINHO
SAO CARLOS
SAO JOAO DEL REI
SAO JOSE DO RIO PRETO
SAO JOSE DOS CAMPOS
SAO LUIS
SAO MATEUS
SAO PAULO
SAO ROQUE - MAIRINQUE
SERTAOZINHO
SETE LAGOAS
SINOP
SOBRAL
SOROCABA
TAQUARA - PAROBE - IGREJINHA
TATUI
TEIXEIRA DE FREITAS
TEOFILO OTONI
TERESINA
TERESOPOLIS
TOLEDO
TRAMANDAI - OSORIO
TRES LAGOAS
TRES RIOS - PARAIBA DO SUL
TUBARAO - LAGUNA
UBA
UBERABA
UBERLANDIA
UMUARAMA
VARGINHA
VITORIA
VITORIA DA CONQUISTA
VITORIA DE SANTO ANTAO
VOLTA REDONDA - BARRA MANSA

"""

