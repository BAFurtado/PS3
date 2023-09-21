## LIST OF CHANGES

### Sep 2023

#### Developer: Rafa

1. creation of a 'RegionalMarket' class focused on firms trading with each other and its consequences;
2. proposed creation of a 'search_goods_market' method to return best options for buying needed inputs;
3. creation of the 'buy_inputs' method (self-explanatory, based on technical matrix from the I-O matrix);
   1. jury still out on its placement, probably should move to Firm class;
   2. needs the previous method to work for this one to work.
4. creation of the 'create_externalities' method to return externalities for a given money output of an activity;
5. creation of the 'production_function' method to call upon the previous two listed functions;
   1. needs to take labour into account;
   2. probably will be absorbed by the existing Firm class production function.
6. creation of the 'market_balancing' method to keep demands by agent type (sectors, households, government, etc.)
   1. needs simulated market distributions to work;
   2. maybe a 12-month moving average needs to keep below a certain threshold TBD.
7. proposed classes 'OtherRegions' and 'ForeignSector'.

  
