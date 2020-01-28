import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from collections import deque
from io import StringIO
from prepare_data import DataProcessor
from subprocess import check_output
import time
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class load_data:

    def __init__(self, n=0, name='EURUSD', base_len=14400, series_len=128):
        self.name = name
        self.dir_name = 'workspace/data/'+name+'/'
        self.dir_online = 'workspace/data/'
        self.n = n
        self.pair = name[:3]+'/'+name[3:]
        self.dfbase = None
        self.base_len = 14400
        self.series_len = series_len

        self.df1s = None
        self.df5s = None
        self.df10s = None
        self.df30s = None
        self.df1m = None
        self.df5m = None
        self.df10m = None
        self.df30m = None
        self.df1h = None
        date_format = '%Y-%m-%d %H:%M:%S.%f'
        self.data_proc = DataProcessor(name, date_format)

        self.online_data = None
        self.fn_online = 'workspace/data/' + name + '_new_record'

    def load_online_data(self, max_r=3):
        while True:
            data = pd.read_csv(self.fn_online, sep=' ', \
                    header=None, \
                    names=['date', 'time', 'pair', 'change', 
                        'sell', 'buy'])
            if len(data)>0:
                break
            time.sleep(0.1)
        data['date'] = data['date']+' ' + data['time']
        data = data[['date', 'pair', 'change', 'sell', 'buy']]
        data = self.data_proc.form_data_local(data)
        self.online_data = pd.concat([self.online_data, data])
        self.online_data = self._hold_last_data(self.online_data, 3)

    def _hold_last_data(self, data, secs):
        if len(data) == 0:
            return data
        latest_time = data.tail(1)['date'].values[0]
        data = data.loc[(latest_time-data['date']). \
                dt.total_seconds() < secs]
        return data

    def load_online_data_by_level(self, level):

        if level == '1s':
            return self.get1s_online()
        elif level == '5s':
            return self.get5s_online()
        elif level == '10s':
            return self.get10s_online()
        elif level == '30s':
            return self.get30s_online()
        elif level == '1m':
            return self.get1m_online()
        elif level == '5m':
            return self.get5m_online()
        elif level == '10m':
            return self.get10m_online()
        elif level == '30m':
            return self.get30m_online()
        elif level == '1h':
            return self.get1h_online()

    def load_by_level(self, level, online=False):
        if level == '1s':
            return self.get1s(online)
        elif level == '5s':
            return self.get5s(online)
        elif level == '10s':
            return self.get10s(online)
        elif level == '30s':
            return self.get30s(online)
        elif level == '1m':
            return self.get1m(online)
        elif level == '5m':
            return self.get5m(online)
        elif level == '10m':
            return self.get10m(online)
        elif level == '30m':
            return self.get30m(online)
        elif level == '1h':
            return self.get1h(online)

    def load_base(self, fn='workspace/data/latest.csv'):
        self.dfbase = pd.read_csv(fn)
        self.df1s = self.dfbase
        self.df1s['date'] = pd.to_datetime(self.df1s['date'])
        print('df1s:', self.df1s.columns)

    def get1s(self, online):
        if online:
           pass 
        else:
            fns = os.listdir(self.dir_name+'1s/')
            for fn in fns:
                yield pd.read_csv(self.dir_name+'1s/'+fn)

    def get5s(self, online):
        if online:
            pass
        else:
            fns = os.listdir(self.dir_name+'5s/')
            for fn in fns:
                yield pd.read_csv(self.dir_name+'5s/'+fn)

    def get10s(self, online):
        if online:
            pass
        else:
            fns = os.listdir(self.dir_name+'10s/')
            for fn in fns:
                yield pd.read_csv(self.dir_name+'10s/'+fn)

    def get30s(self, online):
        if online:
            pass
        else:
            fns = os.listdir(self.dir_name+'30s/')
            for fn in fns:
                yield pd.read_csv(self.dir_name+'30s/'+fn)

    def get1m(self, online=False):
        if online:
            pass
        else:
            fns = os.listdir(self.dir_name+'1m/')
            for fn in fns:
                yield pd.read_csv(self.dir_name+'1m/'+fn)

    def get5m(self, online=False):
        if online:
            pass
        else:
            fns = os.listdir(self.dir_name+'5m/')
            for fn in fns:
                yield pd.read_csv(self.dir_name+'5m/'+fn)

    def get10m(self, online=False):
        if online:
            pass
        else:
            fns = os.listdir(self.dir_name+'10m/')
            for fn in fns:
                yield pd.read_csv(self.dir_name+'10m/'+fn)

    def get30m(self, online=False):
        if online:
            pass
        else:
            fns = os.listdir(self.dir_name+'30m/')
            for fn in fns:
                yield pd.read_csv(self.dir_name+'30m/'+fn)

    def get1h(self, online=False):
        if online:
            pass
        else:
            yield pd.read_csv(self.dir_name+'1h.csv')

    def load_tail(self, fn, n):
        with open(fn, 'r') as f:
            q = deque(f, n)
            return pd.read_csv(StringIO(''.join(q)), header=None) 

    def get1s_online(self):
        data = self.data_proc.transform(self.online_data, period='1S')
        if len(data) < 2 and self.df1s is None:
            return None
            print('online_data is shorter than 2')
        else:

            self.df1s = pd.concat([self.df1s, data.iloc[-2:-1, :]], sort=False)
            self.df1s = self._hold_last_data(self.df1s, self.base_len)
        self.dfbase = self.df1s

        """
        self.dfbase = pd.concat([self.dfbase, self.df1s.iloc[-1:]],
                sort=False)

        self.dfbase = self._hold_last_data(self.dfbase, self.base_len)
        """
        data = self.data_proc.get_features(self.df1s, local=True)
        return data.iloc[-self.series_len:, :]

    def get5s_online(self):
        if self.df1s is None:
            return None
        data = self.df1s.iloc[-15:]
        data = self.data_proc.transform(data, period='5S')
        if len(data) < 2:
            return None

        self.df5s = pd.concat([self.df5s, data.iloc[-2:-1, :]])
        self.df5s = self.df5s.iloc[-self.series_len:]
        # self.df5s = self._hold_last_data(self.df5s, 30)

        data = self.data_proc.get_features(self.df5s, local=True)
        # return data.iloc[-1:, :]
        return data

    def get10s_online(self):
        if self.df5s is None:
            return None
        data = self.df5s.iloc[-6:]
        data = self.data_proc.transform(self.df5s, period='10S')
        if len(data) < 2:
            return None

        self.df10s = pd.concat([self.df10s, data.iloc[-2:-1, :]])
        self.df10s = self.df10s.iloc[-self.series_len:]
        # self.df10s = self._hold_last_data(self.df10s, 90)

        data = self.data_proc.get_features(self.df10s, local=True)
        # return data.iloc[-1:, :]
        return data

    def get30s_online(self):
        if self.df10s is None:
            return None
        data = self.df10s.iloc[-9:]
        data = self.data_proc.transform(self.df10s, period='30S')
        if len(data) < 2:
            return None

        self.df30s = pd.concat([self.df30s, data.iloc[-2:-1, :]])
        self.df30s = self.df30s.iloc[-self.series_len:]
        # self.df30s = self._hold_last_data(self.df30s, 180)

        data = self.data_proc.get_features(self.df30s, local=True)
        # return data.iloc[-1:, :]
        return data

    def get1m_online(self):
        if self.df30s is None:
            return None
        data = self.df30s.iloc[-6:]
        data = self.data_proc.transform(self.df30s, period='60S')
        if len(data) < 2:
            return None

        self.df1m = pd.concat([self.df1m, data.iloc[-2:-1, :]])
        self.df1m = self.df1m.iloc[-self.series_len:]
        # self.df1m = self._hold_last_data(self.df1m, 900)

        data = self.data_proc.get_features(self.df1m, local=False)
        # return data.iloc[-1:, :]
        return data

    def get5m_online(self):
        if self.df1m is None:
            return None
        data = self.df1m.iloc[-15:]
        data = self.data_proc.transform(self.df1m, period='300S')
        if len(data) < 2:
            return None

        self.df5m = pd.concat([self.df5m, data.iloc[-2:-1, :]])
        self.df5m = self.df5m.iloc[-self.series_len:]
        # self.df5m = self._hold_last_data(self.df5m, 1800)

        data = self.data_proc.get_features(self.df5m, local=False)
        # return data.iloc[-1:, :]
        return data

    def get10m_online(self):
        if self.df5m is None:
            return None
        data = self.df5m.iloc[-6:]
        data = self.data_proc.transform(self.df5m, period='600s')
        if len(data) < 2:
            return None

        self.df10m = pd.concat([self.df10m, data.iloc[-2:-1, :]])
        self.df10m = self.df10m.iloc[-self.series_len:]
        # self.df10m = self._hold_last_data(self.df10m, 5400)

        data = self.data_proc.get_features(self.df10m, local=False)
        # return data.iloc[-1:, :]
        return data

    def get30m_online(self):
        if self.df10m is None:
            return None
        data = self.df10m.iloc[-9:]
        data = self.data_proc.transform(self.df10m, period='1800S')
        if len(data) < 2:
            return None

        self.df30m = pd.concat([self.df30m, data.iloc[-2:-1, :]])
        # self.df30m = self._hold_last_data(self.df30m, 10800)
        self.df30m = self.df30m.iloc[-self.series_len:]

        data = self.data_proc.get_features(self.df30m, local=False)
        # return data.iloc[-1:, :]
        return data

    def get1h_online(self):
        if self.df30m is None:
            return None
        data = self.df30m.iloc[-6:]
        data = self.data_proc.transform(self.df30m, period='3600S')
        if len(data) < 2:
            return None

        self.df1h = pd.concat([self.df1h, data.iloc[-2:-1, :]])
        # self.df1h = self._hold_last_data(self.df1h, 10800)
        self.df1h = self.df1h.iloc[-self.series_len:]

        data = self.data_proc.get_features(self.df1h, local=False)
        # return data.iloc[-1:, :]
        return data

    def max_min_online(self, data, level, new_data, length=128, root=False):
        limit = level*length
        data = self._hold_last_data(data, limit)
        data = pd.concat([data, new_data])
        data = data.sort_values(['price']).reset_index(drop=True)
        if len(data) > 2 and not root:
            data = pd.concat([data.iloc[:2], data.iloc[-2:]])
        max_ = data.iloc[0]['price']
        min_ = data.iloc[-1]['price']
        return data, max_, min_

    def save_base(self, pre_time, force=False):
        limit = self.base_len
        curr = time.time()
        if pre_time is None:
            return curr
        time_dist = curr - pre_time
        if time_dist >= limit or force:
            self.dfbase.to_csv('workspace/data/'+self.name+'/base/'+str(datetime.now())+'.csv', index=False)
            return curr
        return pre_time

    def save_latest(self, fn='workspace/data/latest.csv'):
        self.dfbase.to_csv(fn, index=False)

    def test(self):
        print('test')
        data = self.load_tail(self.dir_online+'latest.csv', 200)
        print(data.head())


class Predictor:

    def __init__(self, name, level, input_dim=2, hidden_dim=5, output_dim=1):
        self.name = name
        self.model = Model(input_dim, hidden_dim, output_dim)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.level = level

    def offline(self, data, is_test, epoch=20, local=False, verbose=True):
        len_batch = 128
        n_mini = 128

        if not is_test:
            if verbose:
                print('training start...')
            self.train(data, len_batch, n_mini, epoch, local, verbose)
        else:
            print('test start...')
            self.test(data, len_batch, n_mini, local)

    def predict(self, inputs):
        output = self.model(inputs)
        return output[-1]

    def predict_with_all_out(self, inputs):
        output = self.model(inputs)
        return output

    def train(self, data, len_batch, n_mini, epoch, local=False, verbose=True):
        len_batch = len_batch
        n_mini = n_mini
        data = data.drop(['date'], axis=1)
        data = torch.Tensor(data.values)
        if local:
            data = batch_gen_local(data, len_batch=len_batch)
        else:
            data = batch_gen(data, len_batch=len_batch, n_mini=n_mini)

        cnt = 0
        data_copy = data
        for i in range(epoch):
            data, data_copy = itertools.tee(data_copy)
            if verbose:
                print('epoch:', i)
            for tmp in data:
                # tmp = self.norm(tmp, [0, 1, 2, -1])
                tmp = self.norm(tmp, [0, 1, -1])
                inputs = tmp[:, :, :-1]
                targets = tmp[:, :, -1:]
                self.model.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()
                if cnt % 100 == 1 and verbose:
                    print(loss)
                    print('std of targets:', torch.std(targets))
                    print('mean diff:', torch.abs(outputs-targets).mean())
                    # print('outputs:', outputs)
                    # print('targets:', targets)
                    print('-----------------')
                cnt += 1

    def gen_inter_data(self, data, len_batch, n_mini, local=False):
        len_batch = len_batch
        n_mini = n_mini
        copy = data.copy()
        data = data.drop(['date'], axis=1)
        data = torch.Tensor(data.values)
        # print(data.shape)
        if local:
            data = batch_gen_local(data, len_batch=len_batch)
        else:
            data = batch_gen(data, len_batch=len_batch, n_mini=n_mini)

        df_tmp = torch.Tensor()
        for tmp in data:
            max_, _ = torch.max(tmp, dim=0)
            min_, _ = torch.min(tmp, dim=0)
            max_ = max_[0, 0]
            min_ = min_[0, 0]
            interval = max_-min_
            tmp = self.norm(tmp, [0, 1, -1])
            inputs = tmp[:, :, :-1]
            preds = self.predict_with_all_out(inputs)
            preds = preds*interval+min_
            df_tmp = torch.cat((df_tmp, preds), 0)
        df_tmp = df_tmp.reshape(-1)
        df_tmp = pd.Series(df_tmp.detach().numpy())
        copy['pred'] = df_tmp
        copy['diff'] = (copy.pred-copy.price)/copy.price
        copy['acc'] = (copy.pred-copy.target)/copy.price
        tmp1 = copy.iloc[:-1]['acc']
        copy = copy.iloc[1:].reset_index(drop=True)
        copy['acc'] = tmp1
        # print(copy.tail())
        copy = copy[['date', 'price', 'diff', 'acc']]
        copy['date'] = pd.to_datetime(copy['date'])
        # print(copy.tail())
        # print(copy['diff'].mean(), copy['acc'].mean())

        return copy

    def test(self, data, len_batch, n_mini, local=False):
        len_batch = len_batch
        n_mini = n_mini
        data = data.drop(['date'], axis=1)
        data = torch.Tensor(data.values)
        if local:
            data = batch_gen_local(data, len_batch=len_batch)
        else:
            data = batch_gen(data, len_batch=len_batch, n_mini=n_mini)
        loss = []

        for tmp in data:
            tmp = self.norm(tmp, [0, 1, -1])
            inputs = tmp[:, :, :-1]
            targets = tmp[-1, :, -1:]
            preds = self.predict(inputs)
            curr_loss = torch.mean(torch.abs(preds-targets))
            loss.append(curr_loss)

        print('mean loss:', torch.mean(torch.Tensor(loss)))

    def online_predict(self, data):
        data = data.drop(['date'], axis=1)
        data = torch.Tensor(data.values)
        data = data.unsqueeze(1)
        data = self.norm(data, [0, 1, -1])
        inputs = data[:, :, :-1]
        preds = self.predict(inputs)
        return preds

    def online_train(self, data):
        pass

    def norm(self, data, columns_index=[0, 1, -1]):
        tmp = data[:, :, columns_index]
        shape = tmp.shape

        max_, _ = torch.max(tmp, dim=0)
        min_, _ = torch.min(tmp, dim=0)

        # unify the scale of target and price 
        cols = [0, -1]
        max_tmp, _ = torch.max(max_[:, cols], dim=1)
        max_[:, cols] = max_tmp.unsqueeze(1).repeat(1, 2)

        min_tmp, _ = torch.min(min_[:, cols], dim=1)
        min_[:, cols] = min_tmp.unsqueeze(1).repeat(1, 2)
        #
        interval = max_ - min_
        interval = interval.unsqueeze(0).repeat(shape[0], 1, 1)
        min_ = min_.unsqueeze(0).repeat(shape[0], 1, 1)
        tmp = (tmp-min_)/interval
        tmp[tmp != tmp] = 0
        data[:, :, columns_index] = tmp
        return data

    def form_online_data(data, max_, min_):
        data = data.drop(['date', 'target'], axis=1)
        data['price'] = (data['price']-min_)/(max_-min_)
        data = data.fillna(0)
        data = torch.Tensor(data.values).unsqueeze(1)
        return data

    def save(self, fn='workspace/data/model/'):
        torch.save(self.model.state_dict(), fn+self.name+'_' + self.level)

    def load(self, fn='workspace/data/model/'):
        try:
            self.model.load_state_dict(torch.load(fn+self.name+'_' + self.level))
        except Exception as e:
            print(e)


class Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.affine_dim = 3
        self.affine_dim2 = 3
        self.hidden_dim = hidden_dim
        self.affine2affine = nn.Linear(input_dim, self.affine_dim)
        self.affine2rep = nn.Linear(self.affine_dim, self.affine_dim2)
        self.lstm = nn.LSTM(self.affine_dim2, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        affine = torch.tanh(self.affine2affine(inputs))
        rep = torch.sigmoid(self.affine2rep(affine))
        lstm_out, _ = self.lstm(rep)
        out_space = self.hidden2out(lstm_out)
        out_scores = torch.sigmoid(out_space)
        return out_scores

    def reset_grad(self):
        self.lstm.zero_grad()
        self.affine2affine.zero_grad()
        self.affine2rep.zero_grad()
        self.hidden2out.zero_grad()


def batch_gen(data, n_batch=None, n_mini=None, len_batch=None):
    length = data.shape[0]
    if n_batch is not None:
        len_batch = int(length/n_batch)
    elif len_batch is not None and n_mini is not None:
        n_batch = int(length/(len_batch*n_mini))

    else:
        if length < 1000000:
            yield data
            return
        len_batch = 1000000
        n_batch = int(length/len_batch)

    rest = length % len_batch

    for i in range(n_batch):
        yield data[i*len_batch*n_mini:(i+1)*len_batch*n_mini].reshape(len_batch, n_mini, -1).permute(1, 0, 2)

    if rest != 0:
        rest_len = len(data[-rest:])
        n_mini = int(rest_len / len_batch)
        r_rest = rest_len-n_mini*len_batch
        if n_mini == 0:
            return
        else:
            yield data[-rest:-r_rest].reshape(len_batch, n_mini, -1).permute(1, 0, 2)


def batch_gen_local(data, n_batch=None, len_batch=None):
    # a batch generator for local data (short data)
    length = data.shape[0]
    if n_batch is not None:
        len_batch = int(length/n_batch)
    elif len_batch is not None:
        n_batch = int(length/len_batch)

    else:
        if length < 1000000:
            yield data
            return
        len_batch = 1000000
        n_batch = int(length/len_batch)

    rest = length % len_batch

    for i in range(n_batch):
        yield data[i*len_batch: (i+1)*len_batch].unsqueeze(1)

    if rest != 0:
        yield data[-rest:].unsqueeze(1)


def run():
    n_data = 5
    reader = load_data(n_data)
    levels = [#'1m', 
            '5m', '10m', '30m', #'1h'
            ]
    # levels = []
    local_levels = ['1s', 
            '5s', '10s', '30s'
            ]
    local_levels = []

    for level in levels:
        print('data level:', level)
        data = reader.load_by_level(level)
        predictor = Predictor('EURUSD', level, input_dim=34)
        predictor.load()
        cnt2 = 0
        for data in data:
            cnt2 += 1
            if cnt2 == n_data:
                is_test = True
            else:
                is_test = False
            print('#####################')
            predictor.offline(data, is_test, epoch=100)
        predictor.save()

    n_data = 180
    reader = load_data(n_data)
    local = True

    for level in local_levels:
        print('data level:', level)
        data = reader.load_by_level(level)
        predictor = Predictor('EURUSD', level, input_dim=3)
        predictor.load()
        cnt2 = 0
        for data in data:
            cnt2 += 1
            if cnt2 == n_data:
                is_test = True
            else:
                is_test = False
            print('#####################')
            predictor.offline(data, is_test, epoch=50, local=local, verbose=True)
        predictor.save()


if __name__ == '__main__':
    run()
