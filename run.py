
from Constants import *
from DataFrameTransformer import DataFrameTransformer

'''Instance of the class that transforms the data'''
transformer = DataFrameTransformer()

'''read data from parquet file'''
transformer.read_data_parquet(F_PATH)

'''calculate the age of the policy holders, and create age groups in 5Y intervals'''
transformer.calculate_age('policy_holder_birth_date')

'''Plot distribution of vehicle prices to get an idea of the behaviour of the variable'''
transformer.plot_histogram('vehicle_value_brl', 'Distribution of vehicle values in insurance policies', limit = True)

'''Plot distribution of age of policy holders'''
transformer.plot_histogram('age', 'Distribution of the age of policy holders')

'''Define vehicle price groups'''
transformer.vehicle_price_groups(PRICE_RANGES, PRICE_LABELS)

'''calculate loss ratio by each of the analysis categories'''
for col in ANALYSIS_COLS:
    transformer.calculate_loss_ratio_by_category('policy_claims_total_amount_paid_brl', 'policy_premium_received_brl', col)

'''calculate loss ratio for combination of gender and age group'''
transformer.loss_ratio_composed('policy_claims_total_amount_paid_brl', 'policy_premium_received_brl', 'age_group', 'policy_holder_gender')

'''plot bar chart for age group loss ratio'''
transformer.plot_barchart(transformer.dfs[1], 'age_group', 'Loss ratio by age group', AGE_LABELS)

'''plot composed bar chart for gender and age group loss ratio'''
transformer.plot_barchart_composed(transformer.dfs[-1], 'policy_holder_gender', 'age_group', AGE_LABELS, 'Loss ratio by gender and age group')

'''Bar chart for vehicle price groups'''
transformer.plot_barchart(transformer.dfs[2], 'vehicle_price_group', 'Loss ratio by price groups', PRICE_LABELS, rot = 'vertical', colr = 'dodgerblue')

'''plot vehicle brand'''
transformer.dfs[3]  = transformer.srt_and_filter(transformer.dfs[3], qnt = 300)
transformer.plot_barchart(transformer.dfs[3], 'vehicle_brand', 'Loss ratio by brand', rot = 'vertical', colr = 'salmon')
transformer.plot_size_scatter(transformer.dfs[3], 'Loss ratio by vehicle brand class and number of insurances', 'vehicle_brand', rot = 'vertical', factor = 0.005, fsize = 8 ,adjust = 0.2)

'''graphs for tarif class'''
transformer.dfs[4]  = transformer.srt_and_filter(transformer.dfs[4])
transformer.plot_size_scatter(transformer.dfs[4], 'Loss ratio by residence tariff class and number of insurances', 'Tariff class', rot = 'vertical', factor = 0.005, fsize = 8 ,adjust = 0.35, clr = 'tomato')

'''Graphs for residence region'''
transformer.dfs[5]  = transformer.srt_and_filter(transformer.dfs[5], qnt =300)
transformer.plot_size_scatter(transformer.dfs[5], 'Loss ratio by residence region and number of insurances', 'Residence region',rot = 'vertical', factor = 0.005, fsize = 8, adjust = 0.35)


print('Finishes data analysis')


'''Notas por categor'ia:
    Policy exposure days. Maybe pedir m'as info de a que hace referencia esta variable
    policy claims num reported, quizas da m'as info el count, habria que comparar contra otras variables, pero no creo que haga falta
    Lo mismo que lo anterior, interesante ver el reported vs paid, aunque igual serria extra y no creo que haga falta.
    binus class, interesante ver con linea de tendencia
    vehicle make year, ver con linea de tendencia - grafico de puntos unidos
    Vehicle tariff class - interesante analizar'''

