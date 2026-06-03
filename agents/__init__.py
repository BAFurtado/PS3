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

    def pay_transport(self, money, params, regions):
        # TODO. Check how much money is being collected as well as proportion of distance/wages
        if self.has_car:
            cost_transport = self.distance * params['PRIVATE_TRANSIT_COST']
        else:
            cost_transport = self.distance * params['PUBLIC_TRANSIT_COST']
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

    def set_commute(self, firm):
        """Set (cache) commute according to their employer firm"""
        if firm is not None:
            self.distance = self.distance_to_firm(firm)
        else:
            self.distance = 0

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
