import pandas as pd
import pickle
import os
import numpy as np


class DataProcessor:

    def __init__(self, name, date_format):
        self.date_format = date_format
        self.name = name
        days = [i for i in range(7)]
        rest = [0 for i in range(17)]
        days += rest
        hours = [i for i in range(24)]
        ext = np.array([days, hours])
        ext = np.moveaxis(ext, 0, 1)
        ext = pd.DataFrame(ext, columns=['day', 'hour'])
        ext['date'] = 0
        ext['price'] = 0
        ext['diff'] = 0
        ext['diff_per'] = 0
        self.ext = ext

    def form_data(self, data):
        date_format = self.date_format
        data = data[['Gmt time', 'Open', 'Close']]
        data.columns = ['date', 'open', 'close']
        data['date'] = pd.to_datetime(data['date'], format=date_format)
        data['price'] = data['close']
        data = data[['date', 'price']]
        return data

    def form_data_local(self, data):
        date_format = self.date_format
        pair = self.name[:3] + '/' + self.name[3:]
        data = data.loc[data.pair == pair]
        data = data[['date', 'sell']]
        data['date'] = pd.to_datetime(data['date'], format=date_format)
        data.columns = ['date', 'price']
        return data

    def to_datetime(self, data):
        date_format = self.date_format
        data['date'] = pd.to_datetime(data['date'], format=date_format)
        return data

    def transform(self, data, period, form=False):
        if form:
            data = self.form_data(data)

        if period is not None:
            data = data.resample(period, on='date').last()
            data = data.dropna()
            data = data[['price']]
            data = data.reset_index()
        return data

    def get_features(self, data, local=False):
        data = data.copy()
        tmp1 = data[:-1].reset_index(drop=True)
        tmp2 = data[1:].reset_index(drop=True)
        data['diff'] = 0
        data = data.iloc[1:, ].reset_index(drop=True)
        data['diff'] = (tmp2['price']-tmp1['price'])
        data['diff_per'] = (tmp2['price']-tmp1['price'])/tmp1['price']
        if not local:
            data['day'] = data['date'].dt.dayofweek
            data['hour'] = data['date'].dt.hour
        data['target'] = 0
        tmp = data.loc[1:, 'price'].reset_index(drop=True)
        data = data[:-1].reset_index(drop=True)
        data['target'] = tmp
        data = self._transform(data, local)
        return data

    def _transform(self, data, local=False):
        change_per = 0
        # normalization
        # data['price'] = self.norm(data['price'], change_per)
        # data['diff'] = self.norm(data['diff'], change_per)
        data['diff_per'] = self.norm(data['diff_per'], change_per, fn='workspace/data/model/min_max_diff_per.p')
        # data['target'] = self.norm(data['target'], change_per)

        # one hot key encoding
        if not local:
            l_ = len(data)
            data = pd.concat([data, self.ext], sort=False)
            data = pd.get_dummies(data, columns=['day', 'hour'])
            data = data.head(l_)

        # reorder columns
        columns = list(data.columns)
        columns.remove('target')
        columns.append('target')
        data = data.reindex(columns=columns)
        return data

    def norm(self, series, change_per=0, fn=None):
        # input: pandas Series
        try:
            with open(fn, 'rb') as f:
                min_, max_ = pickle.load(f)
        except FileNotFoundError:
            max_ = series.max()
            min_ = series.min()
            with open(fn, 'wb') as f:
                pickle.dump((min_, max_), f)
        max_ = max_ + abs(max_-min_)*change_per
        min_ = min_ - abs(max_-min_)*change_per
        series = (series-min_)/(max_-min_)
        return series


def gen_data():
    sec_fns = ['EURUSD_Candlestick_1_s_BID_19.08.2019-19.08.2019.csv',
            'EURUSD_Candlestick_1_s_BID_20.08.2019-20.08.2019.csv', 
            'EURUSD_Candlestick_1_s_BID_21.08.2019-21.08.2019.csv']
    min_fns = ['EURUSD_Candlestick_1_M_BID_01.08.2017-16.08.2019.csv',
           'EURUSD_Candlestick_1_M_BID_01.08.2007-01.08.2010.csv',
           'EURUSD_Candlestick_1_M_BID_01.08.2011-01.08.2014.csv',
           'EURUSD_Candlestick_1_M_BID_01.08.2004-01.08.2007.csv',
           'EURUSD_Candlestick_1_M_BID_01.08.2001-01.08.2004.csv']
    date_format = '%d.%m.%Y %H:%M:%S.%f'
    dp = DataProcessor('EURUSD', date_format)
    dir_name = 'workspace/data/EURUSD/'
    cnt = 0
    # sec data
    for fn in sec_fns:
        data = pd.read_csv(dir_name+fn)
        data = dp.form_data(data)
        data1s = dp.transform(data, period=None)
        data1s = dp.get_features(data1s, local=True)
        data5s = dp.transform(data, period='5S')
        data5s = dp.get_features(data5s, local=True)

        data10s = dp.transform(data, period='10S')
        data10s = dp.get_features(data10s, local=True)
        data30s = dp.transform(data, period='30S')
        data30s = dp.get_features(data30s, local=True)
        data1s.to_csv(dir_name+'1s/1s_'+str(cnt)+'.csv', index=False)
        data5s.to_csv(dir_name+'5s/5s_'+str(cnt)+'.csv', index=False)
        data10s.to_csv(dir_name+'10s/10s_'+str(cnt)+'.csv', index=False)
        data30s.to_csv(dir_name+'30s/30s_'+str(cnt)+'.csv', index=False)
        cnt += 1

    cnt = 0
    # min data
    for fn in min_fns:
        data = pd.read_csv(dir_name+fn)
        data = dp.form_data(data)
        data1m = dp.transform(data, period=None)
        data1m = dp.get_features(data1m)
        data5m = dp.transform(data, period='5min')
        data5m = dp.get_features(data5m)

        data10m = dp.transform(data, period='10min')
        data10m = dp.get_features(data10m)
        data30m = dp.transform(data, period='30min')
        data30m = dp.get_features(data30m)
        data1m.to_csv(dir_name+'1m/1m_'+str(cnt)+'.csv', index=False)
        data5m.to_csv(dir_name+'5m/5m_'+str(cnt)+'.csv', index=False)
        data10m.to_csv(dir_name+'10m/10m_'+str(cnt)+'.csv', index=False)
        data30m.to_csv(dir_name+'30m/30m_'+str(cnt)+'.csv', index=False)
        cnt += 1

    data = pd.read_csv(dir_name+'EURUSD_Candlestick_1_Hour_BID_01.01.2003-18.08.2019.csv')
    data = dp.form_data(data)
    data1h = dp.transform(data, period=None)
    data1h = dp.get_features(data1h)
    data1h.to_csv(dir_name+'1h/1h.csv', index=False)


def gen_data_sec():
    dir_name = 'workspace/data/all/old/'
    dir_store = 'workspace/data/'
    fns = os.listdir(dir_name)
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    dp = DataProcessor('EURUSD', date_format)
    cnt = 0
    local = True
    for fn in fns:
        data = pd.read_csv(dir_name+fn)
        data = dp.form_data_local(data)
        data1s = dp.transform(data, period=None)
        data1s = dp.get_features(data1s, local=local)
        data5s = dp.transform(data, period='5S')
        data5s = dp.get_features(data5s, local=local)

        data10s = dp.transform(data, period='10S')
        data10s = dp.get_features(data10s, local=local)
        data30s = dp.transform(data, period='30S')
        data30s = dp.get_features(data30s, local=local)
        data1s.to_csv(dir_store+dp.name+'/1s/1s_'+str(cnt)+'.csv', index=False)
        data5s.to_csv(dir_store+dp.name+'/5s/5s_'+str(cnt)+'.csv', index=False)
        data10s.to_csv(dir_store+dp.name+'/10s/10s_'+str(cnt)+'.csv', index=False)
        data30s.to_csv(dir_store+dp.name+'/30s/30s_'+str(cnt)+'.csv', index=False)
        cnt += 1


def gen_online_transformed(name='EURUSD', min_records=500):
    dir_name = 'workspace/data/' + name + '/base/'
    dir_store = 'workspace/data/' + name + '/transformed_online'
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    local = True
    dp = DataProcessor(name, date_format)
    cnt = 0
    for fn in os.listdir(dir_name):
        data = pd.read_csv(dir_name+fn)
        if len(data) < min_records:
            continue
        data = dp.to_datetime(data)
        data1s = dp.transform(data, period=None)
        data1s = dp.get_features(data1s, local=local)
        data5s = dp.transform(data, period='5S')
        data5s = dp.get_features(data5s, local=local)

        data10s = dp.transform(data, period='10S')
        data10s = dp.get_features(data10s, local=local)
        data30s = dp.transform(data, period='30S')
        data30s = dp.get_features(data30s, local=local)

        data1m = dp.transform(data, period='60S')
        data1m = dp.get_features(data1m)
        data5m = dp.transform(data, period='300S')
        data5m = dp.get_features(data5m)
        data10m = dp.transform(data, period='600S')
        data10m = dp.get_features(data10m)
        data30m = dp.transform(data, period='1800S')
        data30m = dp.get_features(data30m)
        data1h = dp.transform(data, period='3600S')
        data1h = dp.get_features(data1h)

        data1s.to_csv(dir_store+'/1s/1s_'+str(cnt)+'.csv', index=False)
        data5s.to_csv(dir_store+'/5s/5s_'+str(cnt)+'.csv', index=False)
        data10s.to_csv(dir_store+'/10s/10s_'+str(cnt)+'.csv', index=False)
        data30s.to_csv(dir_store+'/30s/30s_'+str(cnt)+'.csv', index=False)
        data1m.to_csv(dir_store+'/1m/1m_'+str(cnt)+'.csv', index=False)
        data5m.to_csv(dir_store+'/5m/5m_'+str(cnt)+'.csv', index=False)
        data10m.to_csv(dir_store+'/10m/10m_'+str(cnt)+'.csv', index=False)
        data30m.to_csv(dir_store+'/30m/30m_'+str(cnt)+'.csv', index=False)
        data1h.to_csv(dir_store+'/1h/1h_'+str(cnt)+'.csv', index=False)

        cnt += 1


def run():
    gen_data()
    gen_data_sec()
    # gen_online_transformed()


if __name__ == '__main__':
    run()
