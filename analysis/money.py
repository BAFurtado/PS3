"""Money stock of the ACP and the ledger of money that crosses its boundary.

Every unit of money inside the model is held by one of the holders summed in `money_stock`. Money enters or leaves
only through the channels in `LEDGER_CHANNELS`, each recorded in `sim.ledger` (cumulative, signed: inflows positive)
where it happens. So at any month end

    money_stock_total(sim) - sim.money_initial == sum(sim.ledger.values())

and `money_unexplained` in stats.csv, the difference, must stay at rounding level. A non-zero value means some code
creates or destroys money without a channel.
"""

# Cumulative, signed; stats.csv writes each as money_<channel>
LEDGER_CHANNELS = (
    'public_transfers',  # GOV_EXTERNAL_FUNDING: federal and state payroll paid in from outside the ACP
    'public_taxes_out',  # PUBLIC_TAXES_OUT: the federal and state share of the taxes collected in the ACP
    'bank_interest',     # Central.remunerate_liquid_balance: the bank's liquid balance remunerated at the policy rate
    'ogu',               # MCMV and melhorias budget lines (federal)
    'fgts_sbpe',         # FGTS and SBPE loans, funded outside the local bank
    'immigrants',        # money immigrants bring
    'exports',           # sales to the rest of Brazil, recycled demand included
    'imports',           # inputs bought from the rest of Brazil, freight included
    'import_tax',        # consumption tax on imports returned to the municipalities
    'firm_entry',        # start-up capital of entrants when FIRM_CAPITAL_MONTHS = 0
    'firm_writeoff',     # negative balances written off at firm exit
    'eco_investment',    # eco-efficiency investment, bought from no one
)


def money_stock(sim):
    """Money by holder group. Households: members' wallets and family savings (bank deposits are in `bank`, which
    holds the depositors' money as cash). Public: Government firms' budget funds, region treasuries, revenue waiting
    for the budget, policy pots and interest tax not yet collected."""
    funds = sim.funds
    public = sum(r.total_treasure for r in sim.regions.values())
    public += sum(sum(d.values()) for d in funds.pending_public_money.values())
    for pot in ('policy_money', 'policy_money_mcmv', 'policy_money_melhorias'):
        public += sum(getattr(funds, pot, {}).values())
    public += sim.central.taxes
    firms = 0.0
    for f in sim.firms.values():
        firms += f.total_balance
        if f.sector == 'Government':
            public += f.purchase_fund + f.investment_fund + f.input_fund
    return {
        'households': sum(a.money for a in sim.agents.values()) + sum(f.savings for f in sim.families.values()),
        'firms': firms,
        'public': public,
        'bank': sim.central.balance,
    }


def money_stock_total(sim):
    return sum(money_stock(sim).values())
