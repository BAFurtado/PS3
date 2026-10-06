import math
from zlib import crc32

import numpy as np

from world.population import marriage_data
from .bank import Central
from .family import Family
from .firm import (
    AgricultureFirm,
    MiningFirm,
    ManufacturingFirm,
    UtilitiesFirm,
    ConstructionFirm,
    TradeFirm,
    TransportFirm,
    BusinessFirm,
    FinancialFirm,
    RealEstateFirm,
    OtherServicesFirm,
    GovernmentFirm,
)
from .house import House
from .region import Region


class Agent:
    """
    This class represent the general citizen of the model. Individual workers.
    Agents have the following variables:
    (a) fixed: id, gender, month of birth, qualification, family_id
    (b) variable: age, money (amount owned at any given moment), saving,
    firm_id, utility, address, distance, region_id.
    """

    # FIRM_PAYOUT: last month's profit share from the employer. Class-level default so agents unpickled from an older
    # cache have it.
    last_profit_share = 0.0
    # Wage received in the latest payroll it was paid in; FAMILY_WAGE 'month' zeroes it before each month's payroll
    wage_paid = 0.0
    # SOCIAL_TRANSFERS: this month's federal benefits (world/social_transfers.py)
    last_transfer = 0.0
    # EDUCATION 'census': years of study the agent finishes with (world/education.py); None under 'pooled'
    target = None
    # WAGE_SPLIT 'census': persistent individual earnings factor exp(e), drawn the first time a weight is needed
    earnings = None
    # WAGE_SPLIT 'census': (age coefficient, age-squared coefficient, residual sd, run seed); None under 'q_alpha'
    wage_profile = None

    # Class for Agents. Citizens of the model
    # Agents live in families, work in firms, consume
    def __init__(
        self,
        id,
        gender,
        age,
        qualification,
        money,
        month,
        firm_id=None,
        family=None,
        distance=0,
        has_car=False
    ):
        self.id = id
        self.gender = gender
        self.age = age
        self.month = month  # Birthday month
        self.qualification = qualification
        self.money = money
        self.firm_id = firm_id
        self.distance = distance
        self.family = family
        self.last_wage = 0
        self.p_marriage = marriage_data.p_marriage(self)
        self.head = False
        self.has_car = has_car

    def wage_factor(self):
        """WAGE_SPLIT 'census': the Census age profile times the persistent earnings factor exp(e), e ~ N(0, residual
        sd ** 2); 1 under 'q_alpha'"""
        profile = Agent.wage_profile
        if profile is None:
            return 1.0
        b1, b2, sd, seed = profile
        if self.earnings is None:
            # Own stream per agent and run, apart from Participation's draw on the same id
            self.earnings = math.exp(np.random.RandomState([seed, crc32(str(self.id).encode()), 2]).normal(0.0, sd))
        return math.exp(b1 * self.age + b2 * self.age ** 2) * self.earnings

    def wage_weight(self, alpha):
        """The agent's weight in its employer's wage split: qualification ** alpha times wage_factor"""
        if Agent.wage_profile is None:
            return self.qualification ** alpha
        return self.qualification ** alpha * self.wage_factor()

    @property
    def address(self):
        return self.family.address

    @property
    def region_id(self):
        return self.family.region_id

    @property
    def is_minor(self):
        return self.age < 16

    @property
    def is_retired(self):
        return self.age > 70

    # Commute in the units the transit costs are calibrated in (see set_commute). None, the default for
    # agents unpickled from older caches, charges on distance as before.
    commute_cost_units = None

    def pay_transport(self, money, params, regions):
        # TODO. Check how much money is being collected as well as proportion of distance/wages
        units = self.distance if self.commute_cost_units is None else self.commute_cost_units
        if self.has_car:
            cost_transport = units * params['PRIVATE_TRANSIT_COST']
        else:
            cost_transport = units * params['PUBLIC_TRANSIT_COST']
        # Collect taxes to the municipality
        regions[self.family.house.region_id].collect_taxes(cost_transport, 'transport')
        return money - cost_transport

    def grab_money(self, params, regions):
        money = self.money
        self.money = 0
        if self.is_employed:
            money = self.pay_transport(money, params, regions)
        return money

    @property
    def belongs_to_family(self):
        return self.family is not None

    def set_head_family(self):
        self.head = True

    @property
    def is_employed(self):
        return self.firm_id is not None

    @property
    def is_employable(self):
        return not self.is_retired and not self.is_minor and not self.is_employed

    def set_commute(self, firm, transport=None):
        """Set (cache) commute according to their employer firm.
        With a travel-time matrix, distance holds minutes and the cost is charged on minutes converted to
        distance units (TransportNetwork.cost_factor), so the transit cost parameters keep their calibration."""
        if firm is None:
            self.distance = 0
            self.commute_cost_units = 0
        elif transport is not None and transport.matrix is not None:
            self.distance = transport.minutes_between(self.family.house.region_id, firm.region_id)
            self.commute_cost_units = self.distance * transport.cost_factor
        else:
            self.distance = self.distance_to_firm(firm)
            self.commute_cost_units = self.distance

    def __repr__(self):
        return (
            "Ag. ID: %s, %s, Qual. %s, Age: %s, Money $ %.2f, Firm: %s"
            % (
                self.id,
                self.gender,
                self.qualification,
                self.age,
                self.money,
                self.firm_id
            )
        )

    def distance_to_firm(self, firm):
        return self.family.house.distance_to_firm(firm)
