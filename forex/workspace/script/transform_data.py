import pandas as pd
import os
from os.path import isfile, join
import time
from util import Log


class transform_data:

    def __init__(self, dir_name='workspace/data/', length=10800):
        self.dir_name = dir_name
        self.length = length
        self.df = None
        self.columns = ['date', 'pair', 'change', 'buy', 'sell']

    def init(self):
        print('start init...')
        self._load_all_old2()
        print('init finish')

    def get_1s(self):
        try:
            latest_df = pd.read_csv(self.dir_name+'latest.csv')
        except:
            latest_df = pd.DataFrame()

        dir_name = self.dir_name+'/all/'
        self.files = [f for f in os.listdir(dir_name) if isfile(join(dir_name, f))]
        df = pd.DataFrame()
        if 'all.csv' in self.files:
            self.files.remove('all.csv')

        # collect all data 
        for fn in self.files:
            curr_df = pd.read_csv(dir_name+fn)
            df = pd.concat([df, curr_df], sort=False)

        df = pd.concat([df, latest_df], sort=False)
        df = self._set_types(df)
        try:
            all_df = pd.read_csv(dir_name+'all.csv')
            all_df['date'] = pd.to_datetime(all_df['date'])
            df = pd.concat([all_df, df], sort=False)
            del all_df
        except:
            pass

        df = df.drop_duplicates()
        df = df.sort_values(by='date')
        self.df = df.reset_index(drop=True)

        self.latest_time = self.df.tail(1)['date'].values[0]

        self.df = self.df.loc[(self.latest_time-self.df['date']).dt.total_seconds()<self.length]
        self.df[self.columns].to_csv(dir_name+'all.csv', index=False)
        for fn in self.files:
            os.rename(dir_name+fn, dir_name+'old/'+fn)

        del df
        del latest_df
        return self.df

    def _load_all_old(self):
        dir_name = self.dir_name+'all/old/'
        files = [f for f in os.listdir(dir_name) if isfile(join(dir_name, f))]
        df = pd.DataFrame()
        for fn in files:
            curr = pd.read_csv(dir_name+fn)
            df = pd.concat([df, curr], sort=False)

        df = self._set_types(df)
        df = df.sort_values(by='date')
        self.latest_time = df.tail(1)['date'].values[0]
        self.n_pairs = len(self.df['pair'].unique())
        return df

    def _load_all_old2(self):
        dir_name = self.dir_name+'all/old/'
        files = [f for f in os.listdir(dir_name) if isfile(join(dir_name, f))]
        df = pd.DataFrame()
        df1m = pd.DataFrame()
        df5m = pd.DataFrame()
        df30m = pd.DataFrame()
        df1h = pd.DataFrame()
        pre = pd.DataFrame()

        cnt = 0
        files.sort()
        for fn in files:
            print(fn)
            curr = pd.read_csv(dir_name+fn)
            df = pd.concat([df, curr], sort=False)
            if cnt % 10 == 9:
                big_block = pd.concat([pre, df], sort=False)
                big_block = self._set_types(big_block)
                big_block = big_block.sort_values(by='date')
                big_block = big_block.set_index('date')
                n_pairs = len(big_block['pair'].unique())

                # transform big block in different level
                tmp1m = big_block.groupby(['pair']).resample('1min').mean()
                tmp5m = big_block.groupby(['pair']).resample('5min').mean()
                tmp30m = big_block.groupby(['pair']).resample('30min').mean()
                tmp1h = big_block.groupby(['pair']).resample('1H').mean()
                tmp1m = tmp1m.sort_values(by='date')
                tmp5m = tmp5m.sort_values(by='date')
                tmp30m = tmp30m.sort_values(by='date')
                tmp1h = tmp1h.sort_values(by='date')

                # delete first and end
                tmp1m = tmp1m[n_pairs:-n_pairs].reset_index()
                tmp5m = tmp5m[n_pairs:-n_pairs].reset_index()
                tmp30m = tmp30m[n_pairs:-n_pairs].reset_index()
                tmp1h = tmp1h[n_pairs:-n_pairs].reset_index()

                tmp1m = tmp1m.dropna()
                tmp5m = tmp5m.dropna()
                tmp30m = tmp30m.dropna()
                tmp1h = tmp1h.dropna()

                df1m = pd.concat([df1m, tmp1m], sort=False)
                df5m = pd.concat([df5m, tmp5m], sort=False)
                df30m = pd.concat([df30m, tmp30m], sort=False)
                df1h = pd.concat([df1h, tmp1h], sort=False)

                # delete duplicates
                df1m = df1m.drop_duplicates()
                df5m = df5m.drop_duplicates()
                df30m = df30m.drop_duplicates()
                df1h = df1h.drop_duplicates()

                # control the length of diffferent level
                df1m = df1m[-10000*n_pairs:]
                df5m = df5m[-10000*n_pairs:]
                df30m = df30m[-10000*n_pairs:]
                df1h = df1h[-10000*n_pairs:]

                pre = df
                df = pd.DataFrame()

            cnt += 1

        # write files
        df1m[self.columns].to_csv(self.dir_name+'all/1m/all.csv', index=False)
        df5m[self.columns].to_csv(self.dir_name+'all/5m/all.csv', index=False)
        df30m[self.columns].to_csv(self.dir_name+'all/30m/all.csv', index=False)
        df1h[self.columns].to_csv(self.dir_name+'all/1h/all.csv', index=False)
        self.n_pairs = n_pairs

        del df
        del df1m
        del df5m
        del df30m
        del df1h
        del pre
        del tmp1m
        del tmp5m
        del tmp30m
        del tmp1h
        del curr

    def load_online_data(self, name, file_dir='workspace/data/', min_records=500):
        source_dir = file_dir+name+'/base/'
        self.restore_dir = file_dir + name +'/transformed_online/'
        fns = os.listdir(source_dir)
        data = []
        for fn in fns:
            curr = pd.read_csv(fn)
            if len(curr) < min_records:
                del curr
            else:
                data.append(curr)
        return data

    def _set_types(self, df):
        if 'date' not in df.columns:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df['change'] = df['change'].str.rstrip('%').astype('float')/100.0
        return df

    def _transform(self, period, times, old):
        ret = self.df.set_index(['date'])
        ret = ret.groupby(['pair']).resample(period).mean()
        ret = ret.reset_index()
        if 'date' not in ret.columns:
            print(len(ret))
            print('ret columns:', ret.columns)
        ret = ret.sort_values(by='date')
        ret = ret[self.n_pairs: -self.n_pairs]
        ret = pd.concat([old, ret], sort=False)
        ret = ret.loc[(self.latest_time-ret['date']).dt.total_seconds()<self.length*times]
        ret = ret.drop_duplicates()
        return ret

    def get_1m(self):
        old = pd.read_csv(self.dir_name+'all/1m/all.csv')
        old['date'] = pd.to_datetime(old['date'])
        df = self._transform('1min', 60, old)
        df[self.columns].to_csv(self.dir_name+'all/1m/all.csv', index=False)
        return df

    def get_5m(self):
        old = pd.read_csv(self.dir_name+'all/5m/all.csv')
        old['date'] = pd.to_datetime(old['date'])
        df = self._transform('5min', 300, old)
        df[self.columns].to_csv(self.dir_name+'all/5m/all.csv', index=False)
        return df

    def get_30m(self):
        old = pd.read_csv(self.dir_name+'all/30m/all.csv')
        old['date'] = pd.to_datetime(old['date'])
        df = self._transform('30min', 1800, old)
        df[self.columns].to_csv(self.dir_name+'all/30m/all.csv')
        return df

    def get_1h(self):
        old = pd.read_csv(self.dir_name+'all/1h/all.csv')
        old['date'] = pd.to_datetime(old['date'])
        df = self._transform('1H', 3600, old)
        df[self.columns].to_csv(self.dir_name+'all/1h/all.csv')
        return df


def order_transform(length=3600):
    start_time = time.time()
    dump = transform_data()
    logger = Log()
    while True:
        try:
            dump.init()
            for i in range(10000000):
                time_dist = int(time.time()-start_time)
                if time_dist % 55 > 52:
                    dump.get_1s()
                    dump.get_1m()

                if time_dist % 295 > 292:
                    dump.get_5m()

                if time_dist % 1795 > 1792:
                    dump.get_30m()

                if time_dist % 3595 > 2592:
                    dump.get_1h()

                time.sleep(1)
        except Exception as e:
            logger.error(str(e))
            print(str(e))


if __name__ == '__main__':
    order_transform()
