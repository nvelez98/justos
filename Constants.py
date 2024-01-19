#path with the data about insurance claims in Brasil
F_PATH = 'C:/Users/nvele/PycharmProjects/Justos/Justos_DS_takehome_dataset.gz.parquet'

# ANALYSIS_COLS = ['age' , 'age_group', 'policy_holder_zipcode',
#        'policy_exposure_days', 'policy_claims_num_reported',
#        'policy_claims_num_paid', 'policy_holder_gender', 'policy_holder_bonus_clas',
#        'vehicle_brand', 'vehicle_model', 'vehicle_make_year', 'vehicle_tarif_class',
#        'policy_holder_residence_region', 'policy_holder_residence_city']
ANALYSIS_COLS = ['age' , 'age_group', 'vehicle_price_group','vehicle_brand', 'vehicle_tarif_class', 'policy_holder_residence_region']

RANGE_COLUMNS = ['policy_start_date', 'policy_holder_birth_date', 'vehicle_value_brl']

AGE_RANGES = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
AGE_LABELS = ['<20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80', '80+']

PRICE_RANGES2 = [0, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000, 44000, 48000, 52000, 56000, 60000, 64000, 68000, 72000, 76000, 80000, 84000, 88000, 92000, 96000, 100000, 104000, 108000, 112000, 116000, 120000, 150000, 300000, float('inf')]
PRICE_LABELS2 = ['0K-4K', '4K-8K', '8K-12K', '12K-16K', '16K-20K', '20K-24K', '24K-28K', '28K-32K', '32K-36K', '36K-40K', '40K-44K', '44K-48K', '48K-52K', '52K-56K', '56K-60K', '60K-64K', '64K-68K', '68K-72K', '72K-76K', '76K-80K', '80K-84K', '84K-88K', '88K-92K',
                '92K-96K', '96K-100K', '100K-104K', '104K-108K', '108K-112K', '112K-116K', '116K-120K', '120K-150K', '150K-300K', '>300K']

PRICE_RANGES = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000,  130000,  140000,  150000, 200000, 300000, 500000, float('inf')]
PRICE_LABELS = ['0-5K', '5-10K', '10-15K', '15-20K', '20-25K', '25-30K', '30-35K', '35-40K', '40-45K', '45-50K', '50-55K', '55-60K', '60-65K', '65-70K', '70-75K', '75-80K', '80-85K', '85-90K', '90-95K', '95-100K', '100-105K', '105-110K',
                 '110-115K', '115-120K', '120-130K', '130-140K', '140-150K', '150-200K', '200-300K', '300-500K', '>500K']
# l = []
# for i in range(len(PRICE_RANGES2)-1):
#     x = str(int(PRICE_RANGES2[i]/1000)) + '-' + str(int(PRICE_RANGES2[i+1]/1000)) + 'K'
#     l.append(x)
