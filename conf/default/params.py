import datetime

# MODEL PARAMETERS

# FIRMS #########################################################
# Production function, labour with decaying exponent, Alpha for K. [0, 1]
PRODUCTIVITY_EXPONENT = .8
# Order of magnitude correction of production. Production divided by parameter
PRODUCTIVITY_MAGNITUDE_DIVISOR = .2
# GENERAL CALIBRATION PARAMETERS
# Order of magnitude parameter of input into municipality investment
MUNICIPAL_EFFICIENCY_MANAGEMENT = 0.00001
# INTEREST. Choose either: 'nominal', 'real' or 'fixed'. Default 'real'
# FOR CENARIOS PLANHAB, choose either: 'alta', 'media' ou 'baixa'
### Assumption. Mortgage assumed at 80% of regular economy interest rates.
INTEREST = "media"
# By how much percentage to increase prices
MARKUP = 0.07
# Frequency firms change prices. Probability < than parameter
STICKY_PRICES = .9
# Price ruggedness a positive value (below 1) that multiplies the magnitude of price reduction
# Reflects a reluctance of businesses to lower prices. Amount estimated for reduction multiplied by parameter
PRICE_RUGGEDNESS = 0.1
# Number of firms consulted before consumption
SIZE_MARKET = 15
# Number of firms to buy from in the INTERMEDIATE market
INTERMEDIATE_SIZE_MARKET = 15
# Frequency firms enter the market
LABOR_MARKET = 0.3
# Percentage of employees' firms hired by distance
PCT_DISTANCE_HIRING = 0.2
# Ignore unemployment in wage base calculation if parameter is zero, else discount unemployment times parameter
RELEVANCE_UNEMPLOYMENT_SALARIES = 4
# Candidate sample size for the labor market
HIRING_SAMPLE_SIZE = 20

# Reduction size in case of eco innovation success: multiplies firm parameters
ENVIRONMENTAL_EFFICIENCY_STEP = 1
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
POLICY_MCMV = True
POLICY_MCMV_PERCENTAGE = 0.01
TOTAL_TARGETING_POLICY = False

# HOUSING_POLICY can be ('baseline', )
HOUSING_POLICY = 'baseline'
HOUSING_INTEREST = 'baseline'
# Quantiles for families to be included in the different programs
INCOME_MODALIDADES = {'faixa1': .1,
                      'rural': .1,
                      'melhorias': .2,
                      'fgts': .6,
                      'sbpe': .8
                      }
POLICY_MELHORIAS = True
UPGRADE_COST = .2
POLICY_DAYS = 360
# Size of the poorest families to be helped
POLICY_QUANTILE = 0.2
# Change of policy for collecting consumption tax at:
# firms' municipalities origin (True) or destiny (consumers' municipality)
TAX_ON_ORIGIN = True
# BNDES test with (True) and without (False) TRANSPORT investments.
# Variation in time_travel implemented in labor market decisions
READ_TRANSPORT_MATRIX = True
TRANSPORT_POLICY = True
# LOANS ##############################################################################
# Maximum age of borrower at the end of the contract
MAX_LOAN_AGE = 70
# Used to calculate monthly payment for the families, thus limiting maximum loan by number of months and age
# Because permanent income includes wealth, it should be just a small percentage,
# otherwise compromises monthly consumption.
LOAN_PAYMENT_TO_PERMANENT_INCOME = 0.07
# Refers to the maximum loan monthly payment to total wealth
# MAX_LOAN_PAYMENT_TO_WEALTH=.4
# Refers to the maximum rate of the loan on the value of the estate
MAX_LOAN_TO_VALUE = 0.35
# This parameter refers to the total amount of resources available at the bank.
MAX_LOAN_BANK_PERCENT = 0.6

# HOUSING AND REAL ESTATE MARKET #############################################################
CAPPED_TOP_VALUE = 1.3
CAPPED_LOW_VALUE = 0.7
# Influence of vacancy size on house prices
# It can be True or 1 or if construction companies consider vacancy strongly it might be 2 [1 - (vacancy * VALUE)]
OFFER_SIZE_ON_PRICE = 2
# TOO LONG ON THE MARKET:
# value=(1 - MAX_OFFER_DISCOUNT) * e ** (ON_MARKET_DECAY_FACTOR * MONTHS ON MARKET) + MAX_OFFER_DISCOUNT
# AS SUCH (-.02) DECAY OF 1% FIRST MONTH, 10% FIRST YEAR. SET TO 0 TO ELIMINATE EFFECT
ON_MARKET_DECAY_FACTOR = -0.02
# LOWER BOUND, THAT IS, AT LEAST 50% PERCENT OF VALUE WILL REMAIN AT END OF PERIOD, IF PARAMETER IS .5
MAX_OFFER_DISCOUNT = 0.6
# Percentage of households pursuing new location (on average families move about once every 20 years)
PERCENTAGE_ENTERING_ESTATE_MARKET = 0.05
NEIGHBORHOOD_EFFECT = 2

# RENTAL #######################
INITIAL_RENTAL_SHARE = 0.25
INITIAL_RENTAL_PRICE = 0.002

# CONSTRUCTION #################################################################################
# LICENSES ARE URBANIZED LOTS AVAILABLE FOR CONSTRUCTION PER NEIGHBORHOOD PER MONTH.
# Expected number of NEW licenses created monthly by region (neighborhood). Set to 0 for no licenses.
EXPECTED_LICENSES_PER_REGION = 2
# PERCENT_CONSTRUCTION_FIRMS = 0.07 This has been deprecated with the introduction of sectors
# Months that construction firm will divide its income into monthly revenue installments.
# Although prices are accounted for at once.
CONSTRUCTION_ACC_CASH_FLOW = 24
# Cost of lot in PERCENTAGE of construction
LOT_COST = 0.15
# Initial percentage of vacant houses
HOUSE_VACANCY = 0.1
# MAX_NUMBER OF HOUSES IN STOCK
MAX_HOUSE_STOCK = 12
# HOW LARGER IS CONSTRUCTION FIRMS PROFIT RELATIVE TO USUAL MARKUP (firms' productivity, given current prices)
CONSTRUCTION_FIRM_MARKUP_MULTIPLIER = 10

# POPULATION AND DEMOGRAPHY
# Families run parameters (on average) for year 2000, or no information. 2010 uses APs average data
EXOGENOUS_HEAD_RATE = False
MEMBERS_PER_FAMILY = 2.5
MARRIAGE_CHECK_PROBABILITY = 0.03

# TAXES ##################################################################
TAX_CONSUMPTION = 0.15
TAX_LABOR = 0.15
TAX_ESTATE_TRANSACTION = 0.004
TAX_FIRM = 0.15
TAX_PROPERTY = 0.004

# EMISSIONS POLICIES ######################################################
# Taxes on emission are given by tax * total_emissions. Roughly R$ * tonCO2. .1 is about R$10
TAX_EMISSION = 0
# Subsidies in (0,1) is the amount of investment paid by the gov(subsidies * total_invested)
# 0 is none, 1 is full
ECO_INVESTMENT_SUBSIDIES = 0

# Consumption_equal: ratio of consumption tax distributed at state level (equal)
# Fpm: ratio of 'labor' and 'firm' taxes distributed per the fpm ruling
TAXES_STRUCTURE = {"consumption_equal": 0.1875, "fpm": 0.235}

# TRANSPORT ######################################################################################
# Cobb-Douglas parameters for matching utility:
# log(U) = α log_qualification + β log_commuting + γ log_wages
# GAMMA is 1 - alpha - beta
# Emphasizes qualification and wages (0.4) equally, with lesser weight (0.2) on commuting time.
CB_QUALIFICATION = .4
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

# Write exactly like the list below
PROCESSING_ACPS = ["BRASILIA"]

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

