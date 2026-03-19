""" Introducing a Central Bank that sells titles and provide interest set by the Government
    Eventually, it will loan to other banks

    Banks will serve to offer mortgage and capitalize on deposits
    """
import datetime
from collections import defaultdict

import numpy as np
import numpy_financial as npf
import pandas as pd
import conf


class Loan:
    def __init__(self, principal, mortgage_rate, months, house, loan_type='market'):
        self.age = 0
        self.months = months
        self.principal = principal
        self.my_mortgage_rate = mortgage_rate
        self.payment = list()
        self.payment_schedule()
        # House value is updated
        self.collateral = house
        self.paid_off = False
        self.delinquent = False
        self.loan_type = loan_type

    def balance(self):
        return sum(self.payment)

    def payment_schedule(self):
        # Implementation of SAC Brazilian system. Amortization is constant with decreasing interest.
        amortiza = round(self.principal / self.months, 6)
        balance = self.principal
        for i in range(self.months):
            interest = balance * self.my_mortgage_rate
            self.payment.append(amortiza + interest)
            balance -= amortiza

    def current_collateral(self):
        bal = sum(self.payment)
        if bal <= 0:
            return 1 + self.my_mortgage_rate
        return min(self.collateral.price / bal, 1 + self.my_mortgage_rate)

    def pay(self, amount):
        for i in range(len(self.payment)):
            if amount > self.payment[i]:
                amount -= self.payment[i]
                self.payment[i] = 0
            else:
                self.payment[i] -= amount
                break
        # Introducing 180 days (6 months) rule, before considering delinquent
        month = self.age - 6 if self.age > 5 else 0
        if sum(self.payment[:month]) > 0:
            self.delinquent = True
        else:
            self.delinquent = False

        # Fully paid off
        self.paid_off = self.balance() <= 0
        return self.paid_off


class Central:
    """ The Central Bank
        Given a set rate of real interest rates, it provides capital remuneration
        (exogenously provided for the moment)
        """

    def __init__(self, id_, balance=0):
        self.id = id_
        self.balance = balance
        self.interest = 0
        self.wallet = defaultdict(list)
        self.taxes = 0
        self.mortgage_rate = 0
        self.i_sbpe = 0
        self.i_fgts = 0
        self.loan_stats = defaultdict(int)
        self._outstanding_loans = 0
        # IBGE codes got only 6 digits
        funding_data = pd.read_csv(f'input/planhab_funds/fgts_sbpe_pct_{conf.PARAMS['FUNDS_AVAILABILITY']}.csv')
        self.funding = (funding_data.set_index(['ano', 'cod_ibge'])[['recursos_sbpe', 'recursos_fgts']]
                        .to_dict(orient='index'))
        self.monthly_funding_available = defaultdict(float)
        self.monthly_funding_used = defaultdict(float)
        self.tax_firm = conf.PARAMS['TAX_FIRM']

        self.loan_to_income = conf.PARAMS['LOAN_PAYMENT_TO_PERMANENT_INCOME']

        # Track remaining loan balances
        self.loans = defaultdict(list)

    def funding_usage_month(self, year, month, regions):
        fgts_used = 0
        fgts_avail = 0
        sbpe_used = 0
        sbpe_avail = 0
        for r in regions:
            fgts_used += self.monthly_funding_used.get((year, month, int(r.id[:6]), 'recursos_fgts'), 0)
            fgts_avail += self.monthly_funding_available.get((year, month, int(r.id[:6]), 'recursos_fgts'), 0)

            sbpe_used += self.monthly_funding_used.get((year, month, int(r.id[:6]), 'recursos_sbpe'), 0)
            sbpe_avail += self.monthly_funding_available.get((year, month, int(r.id[:6]), 'recursos_sbpe'), 0)

        perc_fgts = fgts_used / fgts_avail if fgts_avail > 0 else 0
        perc_sbpe = sbpe_used / sbpe_avail if sbpe_avail > 0 else 0

        return perc_fgts, perc_sbpe

    def set_interest(self, interest, mortgage, sbpe, fgts):
        self.interest, self.mortgage_rate, self.i_sbpe, self.i_fgts = interest, mortgage, sbpe, fgts

    def pay_interest(self, client, y, m):
        """ Updates interest to the client
        """
        # Compute future values
        interest = 0
        for amount, date in self.wallet[client]:
            interest += npf.fv(self.interest,
                               (datetime.date(y, m, 1) - date).days // 30,
                               0,
                               amount * -1)
            interest -= amount

        # Compute taxes
        tax = interest * self.tax_firm
        self.taxes += tax
        self.balance -= interest - tax
        return interest - tax

    def collect_taxes(self):
        """ This function withdraws monthly collected taxes from investments, at tax firm rates and
            resets the counter back to 0.
            """
        amount, self.taxes = self.taxes, 0
        return amount

    def deposit(self, client, amount, date):
        """ Receives the money of the client
        """
        self.wallet[client].append((amount, date))
        self.balance += amount

    def withdraw(self, client, y, m):
        """ Gives the money back to the client
        """
        interest = self.pay_interest(client, y, m)
        amount = self.sum_deposits(client)
        del self.wallet[client]
        self.balance -= amount
        return amount + interest

    def sum_deposits(self, client):
        return np.sum(amount for amount, _ in self.wallet[client])

    def loan_balance(self, family_id):
        """Get total loan balance for a family"""
        return np.sum(l.balance() for l in self.loans.get(family_id, []))

    def n_loans(self):
        return np.sum(len(ls) for ls in self.loans.values())

    def outstanding_loan_balance(self):
        return np.sum(l.balance() for l in self.all_loans())

    def all_loans(self):
        for ls in self.loans.values():
            yield from ls

    def active_loans(self):
        return [l for l in self.all_loans() if not l.paid_off]

    def delinquent_loans(self):
        return [l for l in self.active_loans() if l.delinquent]

    def outstanding_active_loan(self):
        return sum([l.balance() for l in self.active_loans() if l])

    def mean_collateral_rate(self):
        mean_collateral = sum([l.current_collateral() * l.balance() for l in self.active_loans() if l]) / \
                          self.outstanding_active_loan() if self.outstanding_active_loan() else 0
        return min(1 + self.mortgage_rate, mean_collateral)

    def prob_default(self):
        # Sum of loans of clients who are currently missing any payment divided by total outstanding loans.
        return np.sum([l.balance() for l in self.delinquent_loans()]) / self.outstanding_active_loan() \
            if self.outstanding_active_loan() else 0

    def calculate_monthly_mortgage_rate(self):
        if not self.loans:
            return
        default = self.prob_default()
        # First three months, few loans
        # self.interest is economy rate, fixed by monetary policy. Rate of reference
        if default == 1:
            return
        self.mortgage_rate = (1 + self.mortgage_rate - default * self.mean_collateral_rate()) / (1 - default) - 1

    def loan_stats_summary(self):
        loans = self.active_loans()
        amounts = [l.principal for l in loans]
        if amounts:
            mean = np.sum(amounts) / len(amounts)
            return min(amounts), max(amounts), mean
        return 0, 0, 0

    def request_loan(self, family, house, amount, ano, month):
        # register loan request
        self.loan_stats["requested"] += 1

        # If they have outstanding loans, don't lend
        if self.loans[family.id]:
            self.loan_stats["denied_existing_loan"] += 1
            return False, None

        # Family-side cap
        max_amount, max_months = self.max_loan(family, flag=family.loan_rate)
        if max_months <= 0:
            self.loan_stats["denied_invalid_term"] += 1
            return False, None

        amount = min(amount, max_amount)

        if amount <= 0:
            self.loan_stats["denied_zero_capped_amount"] += 1
            return False, None

        # Criteria related to consumer
        monthly_payment = self._max_monthly_payment(family)
        if amount / max_months > monthly_payment:
            self.loan_stats["denied_affordability"] += 1
            return False, None

        # Source-of-funds checks
        if family.loan_rate == 'market':
            required_reserve = conf.PARAMS['BANK_DEPOSIT_RESERVE'] * self.total_deposits()
            available_cash = self.balance - required_reserve

            if amount > available_cash:
                self.loan_stats["denied_liquidity_reserve"] += 1
                return False, None

            if self._outstanding_loans + amount > self.balance * conf.PARAMS['MAX_LOAN_BANK_PERCENT']:
                self.loan_stats["denied_bank_limit"] += 1
                return False, None

        else:
            loan_type = "recursos_" + family.loan_rate
            region = int(house.region_id[:6])
            try:
                if amount > self.funding[(ano, region)][loan_type]:
                    self.loan_stats[f"denied_{loan_type}"] += 1
                    return False, None
            except KeyError:
                self.loan_stats["denied_funding_keyerror"] += 1
                return False, None

        # Approve
        self.loan_stats["approved"] += 1

        rate = {
            'market': self.mortgage_rate,
            'sbpe': self.i_sbpe,
            'fgts': self.i_fgts
        }.get(family.loan_rate, self.mortgage_rate)

        self.loans[family.id].append(Loan(amount, rate, max_months, house, loan_type=family.loan_rate))

        if family.loan_rate == 'market':
            self.balance -= amount
            self._outstanding_loans += amount
        else:
            region = int(house.region_id[:6])
            loan_type = 'recursos_' + family.loan_rate
            self.funding[(ano, region)][loan_type] -= amount
            self.monthly_funding_used[(ano, month, region, loan_type)] += amount

        return True, amount

    def _max_monthly_payment(self, family):
        # Max % of income on loan repayments
        return family.get_permanent_income() * self.loan_to_income

    def max_loan(self, family, flag='market'):
        """Estimate maximum loan for family"""
        income = self._max_monthly_payment(family)
        max_years = conf.PARAMS['MAX_LOAN_AGE'] - max([m.age for m in family.members.values()])
        # Longest possible mortgage period is limited to 30 years (360 months).
        max_months = min(max_years * 12, 360)
        if max_months <= 0:
            return 0, 0
        max_total = income * max_months
        rate = {
            'market': self.mortgage_rate,
            'sbpe': self.i_sbpe,
            'fgts': self.i_fgts
        }.get(flag, self.mortgage_rate)

        max_total = income * max_months
        max_principal = max_total / (1 + rate)

        return max_principal, max_months

    def collect_loan_payments(self, sim):
        for family_id, loans in self.loans.items():
            if not loans:
                continue
            family = sim.families[family_id]
            remaining_loans = []
            for loan in loans:
                if loan.paid_off:
                    continue
                loan.age += 1
                if family.savings < sum(loan.payment[:loan.age]):
                    family.savings += family.grab_savings(self, sim.clock.year, sim.clock.months)
                payment = min(family.savings, sum(loan.payment[:loan.age]))
                done = loan.pay(payment)
                if done:
                    family.have_loan = None
                family.savings -= payment

                # Add to bank balance
                self.balance += payment

                # Remove loans that are paid off
                if not done:
                    remaining_loans.append(loan)
            self.loans[family_id] = remaining_loans
        self.recompute_outstanding_market_loans()

    def recompute_outstanding_market_loans(self):
        total = 0
        for loans in self.loans.values():
            for loan in loans:
                if not loan.paid_off and loan.loan_type == 'market':
                    total += loan.balance()
        self._outstanding_loans = total

    def total_deposits(self):
        return sum(sum(amount for amount, _ in deposits) for deposits in self.wallet.values())


class Bank(Central):
    """ Market banks
        Yet to be designed

        May benefit from methods available at the Central Bank
        """
    pass
