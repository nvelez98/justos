#path with the data about insurance claims in Brasil
F_PATH = 'C:/Users/nvele/PycharmProjects/Justos/Justos_DS_takehome_dataset.gz.parquet'

ALL_COLS = ['age' , 'policy_holder_zipcode',
        'policy_exposure_days', 'policy_claims_num_reported',
        'policy_claims_num_paid', 'policy_holder_gender', 'policy_holder_bonus_clas',
        'vehicle_brand', 'vehicle_model', 'vehicle_make_year', 'vehicle_tarif_class',
        'policy_holder_residence_region', 'policy_holder_residence_city', 'vehicle_value_brl']

CATEGORICAL_VARIABLES = ['age' , 'policy_holder_zipcode', 'policy_holder_gender',
        'vehicle_brand', 'vehicle_model', 'vehicle_make_year', 'vehicle_tarif_class',
        'policy_holder_residence_region', 'policy_holder_residence_city']


ANALYSIS_COLS = ['age' , 'age_group', 'vehicle_price_group','vehicle_brand', 'vehicle_tarif_class', 'policy_holder_residence_region', 'policy_holder_gender', 'policy_holder_bonus_clas', 'vehicle_make_year', 'policy_holder_residence_city']


RANGE_COLUMNS = ['policy_start_date', 'policy_holder_birth_date', 'vehicle_value_brl']

AGE_RANGES = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
AGE_LABELS = ['<20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80', '80+']


PRICE_RANGES = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000,  130000,  140000,  150000, 200000, 300000, 500000, float('inf')]
PRICE_LABELS = ['0-5K', '5-10K', '10-15K', '15-20K', '20-25K', '25-30K', '30-35K', '35-40K', '40-45K', '45-50K', '50-55K', '55-60K', '60-65K', '65-70K', '70-75K', '75-80K', '80-85K', '85-90K', '90-95K', '95-100K', '100-105K', '105-110K',
                 '110-115K', '115-120K', '120-130K', '130-140K', '140-150K', '150-200K', '200-300K', '300-500K', '>500K']

AGE_RANGES_REGRESSION = [0, 35, 80, float('inf')]
AGE_LABELS_REGRESSION = ['<35', '35-80', '>=80']

MAKE_YEAR_RANGES = [0,2000, float('inf')]
MAKE_YEAR_RANGES = ['Old', 'New']

VEHICLE_TARIFF_CLASS_RANGES = [0,200, 400, float('inf')]
VEHICLE_TARIFF_CLASS_RANGES = ['low_tariff_class', 'medium_tariff_class', 'high_tariff_class']

REGRESSION_INPUT_CATEGORIES = ['age' , 'policy_holder_gender', 'policy_holder_bonus_clas',
        'vehicle_brand',  'vehicle_make_year', 'vehicle_tarif_class',
        'policy_holder_residence_region']

REPLACE_LIST = ['vehicle_brand',  'vehicle_tarif_class',
        'policy_holder_residence_region']

ANOVA_OW_CATEGORIES = ['policy_holder_gender', 'vehicle_brand', 'vehicle_tarif_class', 'policy_holder_residence_region']

CLUSTER_CATEGORIES = ['vehicle_tarif_class', 'policy_holder_residence_region',  'vehicle_make_year', 'vehicle_brand']
NUM_CLUSTS = {'vehicle_tarif_class':3, 'policy_holder_residence_region':3,  'vehicle_make_year':2, 'vehicle_brand':3}
