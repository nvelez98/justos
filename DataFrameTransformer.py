import pandas as pd
import datetime
from Constants import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway
from kmodes.kmodes import KModes
import statsmodels.api as sm
from sklearn.metrics import r2_score

class DataFrameTransformer:
    def __init__(self):
        self.df = pd.DataFrame()
        self.NUM_INSURANCES = None
        self.dfs = []
        self.avg_dfs = []
        self.avg_loss_ratio = None
        self.male_lr = None
        self.female_lr = None
        self.paid_df = None
        self.regression_df = None
        self.anova_results = []
        self.regression_output_df = pd.DataFrame()

    def read_data_parquet(self, path):
        self.df = pd.read_parquet(path)
        self.NUM_INSURANCES = len(self.df)

    def calc_avg_loss_ratio(self, paid_col, rcv_col):
        self.avg_loss_ratio = self.df[paid_col].sum() / self.df[rcv_col].sum()

    def get_paid_df(self):
        self.paid_df = self.df[self.df['policy_claims_total_amount_paid_brl'] > 0]
        self.paid_df['pay size'] = self.paid_df['policy_claims_total_amount_paid_brl']/self.paid_df['policy_premium_received_brl']

    def calculate_age(self, date_of_birth_col, current_date=None):
        if current_date is None:
            current_date = datetime.datetime.now()

        self.df['date_of_birth'] = pd.to_datetime(self.df[date_of_birth_col], format='%Y%m%d')
        self.df['age'] = (current_date - self.df['date_of_birth']).dt.days // 365
        # Grupos de edad por intervalose de 5 anios
        self.df['age_group'] = pd.cut(self.df['age'], bins=AGE_RANGES, labels=AGE_LABELS, right=False)

    def vehicle_price_groups(self, bins, labels):
        self.df['vehicle_price_group'] = pd.cut(self.df['vehicle_value_brl'], bins=bins, labels=labels, right=False)


    def calculate_loss_ratio_by_category(self, paid_col, received_col, col):

        loss_ratio = self.df.groupby(col).apply(lambda x: x[paid_col].sum() / x[
            received_col].sum()).reset_index(name='loss_ratio')
        count = self.df.groupby(col).size().reset_index(name='count')
        count['percent'] = count['count'] / self.NUM_INSURANCES
        loss_ratio = pd.merge(loss_ratio, count, on=col)
        self.dfs.append(loss_ratio)

    def loss_ratio_composed(self, paid_col, received_col, col1, col2):
        loss_ratio = self.df.groupby([col1,col2]).apply(lambda x: x[paid_col].sum() / x[
            received_col].sum()).reset_index(name='loss_ratio')
        count = self.df.groupby([col1, col2]).size().reset_index(name='count')
        count['percent'] = count['count'] / self.NUM_INSURANCES
        loss_ratio = pd.merge(loss_ratio, count, on=[col1,col2])
        self.dfs.append(loss_ratio)

    def srt_and_filter(self, df, qnt = None):
        df = df.sort_values(by='loss_ratio')
        if qnt is not None:
            df = df[df['count'] > qnt]
        return df

    def set_loss_ratio_by_gener(self, df):
        self.male_lr = df[df['policy_holder_gender'] == 'M']['loss_ratio'].iloc[0]
        self.female_lr = df[df['policy_holder_gender'] == 'F']['loss_ratio'].iloc[0]

    def copy_into_regression(self):
        self.regression_df = self.df.copy()

    def calculate_avg_loss_by_category(self, paid_col, col):
        count = self.regression_df.groupby(col).size().reset_index(name='count')
        if col in REPLACE_LIST:
            categories_to_replace = count[count['count'] < 1000][col].to_list()
            self.regression_df[col] = self.regression_df[col].replace(categories_to_replace, 'others')
            copy_df = self.regression_df.copy()
            copy_df[col] = copy_df[col].replace(categories_to_replace, 'others')
            avg_loss = copy_df.groupby(col)[paid_col].mean().reset_index(name='avg_loss')
            avg_loss = avg_loss.sort_values(by='avg_loss')
        else:
            avg_loss = self.regression_df.groupby(col)[paid_col].mean().reset_index(name='avg_loss')

        self.avg_dfs.append(avg_loss)

    def calculate_vehicle_age(self):
        self.regression_df['vehicle_age'] = CURRENT_YEAR - self.regression_df['vehicle_make_year']

    def exclude_age_anomalies(self):
        self.regression_df = self.regression_df[self.regression_df['age'] < 100]


##############################################################################################
    '''Pricing model functions'''


    def perform_one_anova(self, metric, col):
        dfs = []
        unique_values = self.df[col].unique()
        for unq in unique_values:
            dfs.append(self.df[self.df[col]== unq][metric])
        anova_result = f_oneway(*dfs)
        self.anova_results.append(anova_result)
        print(anova_result)

    def create_clusters(self, col , metric, clusters = 3):

        df_cluster = self.regression_df[[col,metric]]
        n_clusters = clusters
        km = KModes(n_clusters=n_clusters, init='random', n_init=5, verbose=1)
        clusters = km.fit_predict(df_cluster)
        col_name = '{col}_clusters'.format(col = col)
        self.regression_df[col_name] = clusters

    def map_to_clusters(self):
        '''age clusters'''
        self.regression_df['age_clusters'] = pd.cut(self.regression_df['age'], bins=AGE_RANGES_REGRESSION, labels=AGE_LABELS_REGRESSION, right=False)

        '''tariff_class_clusters'''
        self.avg_dfs[5]['tariff_clusters'] = pd.cut(self.avg_dfs[5]['avg_loss'], bins=VEHICLE_TARIFF_CLASS_RANGES,
                                                    labels=VEHICLE_TARIFF_CLASS_LABELS, right=False)
        tarif_dict = dict(zip(self.avg_dfs[5]['vehicle_tarif_class'], self.avg_dfs[5]['tariff_clusters']))
        self.regression_df['tariff_clusters'] = self.regression_df['vehicle_tarif_class'].map(tarif_dict)

        '''brand_ranges'''
        self.avg_dfs[3]['brand_clusters'] = pd.cut(self.avg_dfs[3]['avg_loss'], bins=VEHICLE_BRAND_RANGES,
                                                    labels=VEHICLE_BRAND_LABELS, right=False)
        brand_dict = dict(zip(self.avg_dfs[3]['vehicle_brand'], self.avg_dfs[3]['brand_clusters']))
        self.regression_df['brand_clusters'] = self.regression_df['vehicle_brand'].map(brand_dict)

        '''region_ranges'''
        self.avg_dfs[6]['region_clusters'] = pd.cut(self.avg_dfs[6]['avg_loss'], bins=REGION_RANGES,
                                                   labels=REGION_LABELS, right=False)
        region_dict = dict(zip(self.avg_dfs[6]['policy_holder_residence_region'], self.avg_dfs[6]['region_clusters']))
        self.regression_df['region_clusters'] = self.regression_df['policy_holder_residence_region'].map(region_dict)



    def regression_model(self):

        df = self.regression_df[REGRESSION_COLUMNS]
        df = df.dropna()
        df1 = df[REGRESSION_COLUMNS]
        df = pd.get_dummies(df1, columns=CATEGORICAL_REGRESSION_COLUMNS, drop_first=True)
        dict = {True:1, False:0}
        for col in BOOL_COLS:
            df[col] = df[col].map(dict)
        y = df['policy_claims_total_amount_paid_brl']
        X = df.drop(['policy_claims_total_amount_paid_brl'], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        print(model.summary())
        while model.pvalues.max() > 0.05:
            variable_to_remove = model.pvalues.idxmax()
            X = X.drop([variable_to_remove], axis=1)
            model = sm.OLS(y, X).fit()
        print(model.summary())
        self.regression_output_df = df1
        self.regression_output_df['prediction'] = model.predict(X)
        self.regression_output_df['predicted_price'] = self.regression_output_df['prediction']/self.avg_loss_ratio






###############################################################################################

    #GRAPH FUNCTIONS

    def plot_histogram(self, col, title, limit=False, dframe = None):

        if dframe is None:
            df = self.df
        else:
            df = dframe

        if limit == True:
            df2 = df[df[col] < np.percentile(df[col], 99.9)]
        else:
            df2 = df
        iqr = np.percentile(df2[col], 75) - np.percentile(df2[col], 25)
        bin_width = 2 * iqr / (len(df2[col]) ** (1 / 3))
        num_bins = int((df2[col].max() - df2[col].min()) / bin_width)

        plt.hist(df2[col], bins=num_bins,
                 edgecolor='black')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(title)
        plt.ticklabel_format(style='plain', axis='x')
        plt.show()

    def plot_barchart(self, df, col,  title, labels = None, rot = 'horizontal', colr = 'skyblue', fsize = None, adjust = None, vals = 'loss_ratio', tiks = True):
        if labels is None:
            labels = df[df.columns[0]]
        plt.figure(figsize=(10, 6))
        bar_positions = np.arange(len(labels))
        bars = plt.bar(bar_positions, df[vals], width=0.7, color=colr)
        if adjust is not None:
            plt.subplots_adjust(bottom=adjust)

        plt.bar(bar_positions, df[vals], width=0.7, color=colr)
        plt.xlabel(col)
        plt.ylabel(vals)
        plt.title(title)
        plt.xticks(bar_positions, labels, rotation=rot)
        if fsize is not None:
            plt.xticks(fontsize=fsize)

        if tiks == True:
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
        plt.show()

    def plot_barchart_composed(self, df, col1, col2, labels, title):


        df = df[[col1,col2, 'loss_ratio']]
        df = df.pivot(index=col2, columns=col1, values='loss_ratio').reset_index()
        df['M'] = df['M'].fillna(0)
        df['F'] = df['F'].fillna(0)
        # Fill missing values with zeros

        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        bar_positions_1 = np.arange(len(labels))
        bar_positions_2 = bar_positions_1 + bar_width

        bars_male = plt.bar(bar_positions_1, df['M'], width=0.35, color='lightseagreen')
        bars_female = plt.bar(bar_positions_2, df['F'], width=0.35, color='coral')

        plt.bar(bar_positions_1, df['M'], width=bar_width,
                label='M', color='lightseagreen')
        plt.bar(bar_positions_2, df['F'], width=bar_width,
                label='F', color='coral')

        plt.xlabel(col1)
        plt.ylabel('loss_ratio')
        plt.title(title)
        plt.xticks(bar_positions_1 + bar_width / 2, labels)
        plt.legend()

        for bar, position in zip(bars_male, bar_positions_1):
            yval = bar.get_height()
            plt.text(position + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

        for bar, position in zip(bars_female, bar_positions_2):
            yval = bar.get_height()
            plt.text(position + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

        plt.show()

    def plot_pie(self, df , title,lbls = None):
        if lbls is None:
            lbls = df[df.columns[0]]
        plt.pie(df['count'], labels=lbls, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.show()


    def plot_size_scatter(self,  df, title, col, rot = 'horizontal', factor =1, fsize = None, adjust = None, clr = 'blue'):

        fig, ax = plt.subplots(figsize=(10, 6))
        if adjust is not None:
            fig.subplots_adjust(bottom=adjust)

        categories = df[df.columns[0]]
        metric = df['loss_ratio']
        counts = df['count']

        scatter = ax.scatter(categories, metric, s=np.array(counts) * factor, alpha=0.7, c=clr,
                             edgecolors='black')

        categories = categories.reset_index()[df.columns[0]]
        metric = metric.reset_index().round(2)['loss_ratio']

        for i, txt in enumerate(metric):
            ax.annotate(txt, (categories[i], metric[i]), textcoords="offset points", xytext=(0, 5), ha='center')

        plt.xlabel(col)
        plt.ylabel('Loss ratio')
        plt.title(title)
        plt.xticks(rotation=rot)
        if fsize is not None:
            plt.xticks(fontsize=fsize)
        plt.show()

    def plot_joined_dot(self, df, col, title, clr = 'blue', tik = False):

        x = df[col]
        y= df['loss_ratio']
        plt.plot(x, y, marker='o', linestyle='-', color = clr)
        if tik == True:
            y = y.round(2)
            for xv, yv in zip(x, y):
                plt.annotate(f'{yv}', (xv, yv), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8,
                             color='black')
        plt.xlabel(col)
        plt.ylabel('loss_ratio')
        plt.title(title)
        plt.show()

    def plot_boxplot(self, df, col, title):
        df.boxplot(column=[col])

        plt.xlabel(col)
        plt.ylabel('Values')
        plt.title(title)
        plt.show()






