## LIST OF CHANGES

### Sep 20 2023

#### Developer: Rafa

1. creation of a RegionalMarket class focused on firms trading with each other and its consequences;
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
7. proposed classes OtherRegions and ForeignSector.

### Sep 29 2023

#### Developer: Rafa

1. started moving methods from RegionalMarket class module to Firm class;
   1. need to test all these new methods with debug mode or unit tests;
   2. might still move other methods to 'firm' too.
2. changed names of sectors or consumer types to remove accented characters.

### Oct 06 2023

#### Developer: Rafa

1. moved methods from RegionalMarket class to Firm class;
   1. everything still untested;
   2. might load technical, demand and externalities matrix still in some other way instead of attribute from Simulation, too clunky.
2. renamed headers from the technical and demand matrix to English names;
3. created the other eleven firm types by sector, still empty just copying the Firm class;
4. with the new firm classes, rewrote the 'create_new_firms' from the Generator class;
   1. needs testing too;
   2. if design is approved, its structure might be reused to create consumer relationships.