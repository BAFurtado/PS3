import datetime

# MODEL PARAMETERS

# FIRMS #########################################################
# Production function, labor with decaying exponent, Alpha for K. [0, 1]
PRODUCTIVITY_EXPONENT = 0.65
# Order of magnitude correction of production. Production divided by parameter
PRODUCTIVITY_MAGNITUDE_DIVISOR = .5
# GENERAL CALIBRATION PARAMETERS
# INTEREST. Choose either: 'nominal', 'real' or 'fixed'. Default 'real'
# FOR CENARIOS PLANHAB, choose either interests: 'alta', 'media' ou 'baixa'
# Assumption. Mortgage assumed lower than SELIC (general rate).
INTEREST = "media"
# By how much percentage to increase prices
MARKUP = 0.1
# Frequency firms change prices. Probability < than parameter
STICKY_PRICES = .7
# Price ruggedness a positive value (below 1) that multiplies the magnitude of price reduction
# Reflects a reluctance of businesses to lower prices. Amount estimated for reduction multiplied by parameter
PRICE_RUGGEDNESS = 0.1
# Safety-stock buffer: fraction of monthly sales firms want to hold above productive capacity.
# Higher values keep more firms in "low inventory" mode → more hiring signals, less deflation risk.
INVENTORY_TARGET_RATIO = 0.2
# Number of firms consulted before consumption
SIZE_MARKET = 5
# Number of firms to buy from in the INTERMEDIATE market
INTERMEDIATE_SIZE_MARKET = 10
# Frequency firms enter the market
LABOR_MARKET = 0.8

# Monthly probability an employed worker separates (quits, contract end, etc.).
NATURAL_SEPARATION_RATE = 0.025
# Percentage of employees' firms hired by distance
PCT_DISTANCE_HIRING = 0.2
# Ignore unemployment in wage base calculation if parameter is zero, else discount unemployment times parameter
RELEVANCE_UNEMPLOYMENT_SALARIES = .5
# Candidate sample size for the labor market
HIRING_SAMPLE_SIZE = 20

# Reduction size in case of eco innovation success: multiplies firm parameters
ENVIRONMENTAL_EFFICIENCY_STEP = .99
# Innovation process probability: 1 - exp(lambda * investment / wage_base)
ECO_INVESTMENT_LAMBDA = 50
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
TOTAL_TARGETING_POLICY = False
POLICY_MELHORIAS = True
UPGRADE_COST = .2
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

# HOUSING AND REAL ESTATE MARKET #############################################################
CAPPED_TOP_VALUE = 1.3
CAPPED_LOW_VALUE = 0.7
# Vacancy-price sensitivity: formula is 1 + (VACANCY_PRICE_REFERENCE - vacancy) * OFFER_SIZE_ON_PRICE
# Symmetric around the reference: tight markets (vacancy < reference) generate a premium;
# slack markets (vacancy > reference) generate a discount.
OFFER_SIZE_ON_PRICE = 3
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
NEIGHBORHOOD_EFFECT = 0.3

# RENTAL #######################
INITIAL_RENTAL_SHARE = 0.40
# Monthly rent as a fraction of house price.
# At 0.003 this is 3.6% annual gross yield — in line with Brazilian urban rental markets.
# Also calibrates the financial attractiveness comparison in decision_enter_house_market:
# when the bank rate exceeds this yield, depositing savings is more profitable than buying.
INITIAL_RENTAL_PRICE = 0.002
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
SAVINGS_BUFFER_MONTHS = 3
# Scales the opportunity-cost term in decision_enter_house_market.
# opportunity_cost = max(0, bank_rate - INITIAL_RENTAL_PRICE) × HOUSING_FINANCIAL_WEIGHT
# This is now an absolute-difference formula (not normalized), so the weight is larger than
# the old normalized version. At SELIC ≈ 10% annual (bank_rate ≈ 0.008/month):
#   opportunity_cost ≈ (0.008 - 0.002) × 100 = 0.6
# A renter (housing_need=1.0) scores 0.4 > 0 → enters.
# A comfortable owner (housing_need=0) scores −0.6 → excluded.
# A crowded owner (crowding_bonus=0.7) scores 0.1 → enters to upgrade.
# At low SELIC (≈ 2%, bank_rate ≈ 0.0017): opportunity_cost ≈ 0 → some owners enter.
HOUSING_FINANCIAL_WEIGHT = 100
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
HOUSE_PRODUCTION_ADEQUACY = 6

# POPULATION AND DEMOGRAPHY
# Families run parameters (on average) for year 2000, or no information. 2010 uses APs average data
EXOGENOUS_HEAD_RATE = False
MEMBERS_PER_FAMILY = 2.5
MARRIAGE_CHECK_PROBABILITY = 0.03

# CONSUMPTION #############################################################
# Fraction of permanent income actually spent on goods; remainder flows to savings.
CONSUMPTION_PROPENSITY = 1
# Fraction of accumulated balance government firms spend each month; remainder carried forward.
GOVERNMENT_EXECUTION_RATE = 1

# TAXES ##################################################################
TAX_CONSUMPTION = 0.15
TAX_LABOR = 0.15
TAX_ESTATE_TRANSACTION = 0.004
TAX_FIRM = 0.15
TAX_PROPERTY = 0.008
TAX_TRANSPORT = 0

# EMISSIONS POLICIES ######################################################
# Taxes on emission are given by tax * total_emissions. Roughly R$ * tonCO2. .1 is about R$10
TAX_EMISSION = 1
# Subsidies in (0,1) is the amount of investment paid by the gov(subsidies * total_invested)
# 0 is none, 1 is full
ECO_INVESTMENT_SUBSIDIES = 0.2
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

# RUN DETAILS ###############################################################################
# Percentage of actual population to run the simulation
# Minimum value to run depends on the size of municipality 0,001 is recommended minimum
PERCENTAGE_ACTUAL_POP = 0.005

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

# Write exactly like the list below
PROCESSING_ACPS = ["ARACAJU"]

# Selecting the starting year to build the Agents can be: 1991, 2000 or 2010
STARTING_DAY = datetime.date(2010, 1, 1)

# The Maximum running time (restrained by official data) is 30 years,
TOTAL_DAYS = (datetime.date(2030, 1, 1) - STARTING_DAY).days

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

