import pandas as pd
import datetime
from Constants import *
import matplotlib.pyplot as plt
import numpy as np


class DataFrameTransformer:
    def __init__(self):
        self.df = pd.DataFrame()
        self.NUM_INSURANCES = None
        self.dfs = []
        self.avg_loss_ratio = None
        self.male_lr = None
        self.female_lr = None
        self.paid_df = None

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






