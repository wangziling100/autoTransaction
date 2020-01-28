from analyse import Predictor
from prepare_data import gen_online_transformed
import os
import pandas as pd
import numpy as np
from util import interest_margin, fold_data
import pdb
import torch
from robot import Qvalue


def prepare_robot_data(name='EURUSD', data_length=3600, save=True):

    clear_robot_data()
    # generate intermediate data
    gen_online_transformed()
    save_dir = 'workspace/data/'+name+'/robot_data/'

    levels = ['1s', '5s', '10s', '30s', '1m', '5m', '10m',
            '30m', '1h']
    dir_name = 'workspace/data/' + name + '/transformed_online/'
    # a list of filename from different levels
    # like [fns(1s), fns(5s), ...]
    data_names = []
    cnt = 0
    pred1s = Predictor(name, '1s', input_dim=3)
    pred5s = Predictor(name, '5s', input_dim=3)
    pred10s = Predictor(name, '10s', input_dim=3)
    pred30s = Predictor(name, '30s', input_dim=3)

    pred1m = Predictor(name, '1m', input_dim=34)
    pred5m = Predictor(name, '5m', input_dim=34)
    pred10m = Predictor(name, '10m', input_dim=34)
    pred30m = Predictor(name, '30m', input_dim=34)
    pred1h = Predictor(name, '1h', input_dim=34)

    pred1s.load()
    pred5s.load()
    pred10s.load()
    pred30s.load()
    pred1m.load()
    pred5m.load()
    pred10m.load()
    pred30m.load()
    pred1h.load()

    for level in levels:
        tmp_dir = dir_name+level+'/'
        tmp = os.listdir(tmp_dir)
        tmp = [tmp_dir+i for i in tmp]
        data_names.append(tmp)

    local = True
    train_test_cnt = 0

    for fn1s, fn5s, fn10s, fn30s, fn1m, fn5m, fn10m, fn30m, fn1h in zip(*data_names):
        df1s = pd.read_csv(fn1s)
        df5s = pd.read_csv(fn5s)
        df10s = pd.read_csv(fn10s)
        df30s = pd.read_csv(fn30s)

        df1m = pd.read_csv(fn1m)
        df5m = pd.read_csv(fn5m)
        df10m = pd.read_csv(fn10m)
        df30m = pd.read_csv(fn30m)
        df1h = pd.read_csv(fn1h)
        
        f1s = pred1s.gen_inter_data(df1s, 128, 128, local)
        f5s = pred5s.gen_inter_data(df5s, 128, 128, local)
        f10s = pred10s.gen_inter_data(df10s, 128, 128, local)
        f30s = pred30s.gen_inter_data(df30s, 128, 128, local)

        first_time = f1s.loc[0, 'date']
        last_time = f1s.iloc[-1]['date']

        f5s = add_first_last_rec(f5s, first_time, last_time)
        f10s = add_first_last_rec(f10s, first_time, last_time)
        f30s = add_first_last_rec(f30s, first_time, last_time)

        f1s = f1s.resample('1S', on='date').last().ffill()
        f5s = f5s.resample('1S', on='date').last().ffill()
        f10s = f10s.resample('1S', on='date').last().ffill()
        f30s = f30s.resample('1S', on='date').last().ffill()

        shape = f1s.shape
        cols = ['date', 'price', 'diff', 'acc']
        pattern_df = pd.DataFrame(np.ones(shape), columns=cols)
        pattern_df *= 100

        if len(df1m) < 2:
            f1m = pattern_df
        else:
            f1m = pred1m.gen_inter_data(df1m, 128, 128, local)
            f1m = add_first_last_rec(f1m, first_time, last_time)
            f1m = f1m.resample('1S', on='date').last().ffill()
        if len(df5m) < 2:
            f5m = pattern_df
        else:
            f5m = pred5m.gen_inter_data(df5m, 128, 128, local)
            f5m = add_first_last_rec(f5m, first_time, last_time)
            f5m = f5m.resample('1S', on='date').last().ffill()
        if len(df10m) < 2:
            f10m = pattern_df
        else:
            f10m = pred10m.gen_inter_data(df10m, 128, 128, local)
            f10m = add_first_last_rec(f10m, first_time, last_time)
            f10m = f10m.resample('1S', on='date').last().ffill()
        if len(df30m) < 2:
            f30m = pattern_df
        else:
            f30m = pred30m.gen_inter_data(df30m, 128, 128, local)
            f30m = add_first_last_rec(f30m, first_time, last_time)
            f30m = f30m.resample('1S', on='date').last().ffill()
        if len(df1h) < 2:
            f1h = pattern_df
        else:
            f1h = pred1h.gen_inter_data(df1h, 128, 128, local)
            f1h = add_first_last_rec(f1h, first_time, last_time)
            # f1h['date'] = pd.to_datetime(f1h['date'])
            f1h = f1h.resample('1S', on='date').last().ffill()

        f1s = f1s.reset_index(drop=True)
        f5s = f5s.reset_index(drop=True)
        f10s = f10s.reset_index(drop=True)
        f30s = f30s.reset_index(drop=True)
        f1m = f1m.reset_index(drop=True)
        f5m = f5m.reset_index(drop=True)
        f10m = f10m.reset_index(drop=True)
        f30m = f30m.reset_index(drop=True)
        f1h = f1h.reset_index(drop=True)

        features = ['diff', 'acc']

        res = pd.concat([f1s, f5s[features], f10s[features],
            f30s[features], f1m[features], f5m[features],
            f10m[features],
            f30m[features], f1h[features]], axis=1)

        # res = res.iloc[:, :feature_bit+2]
        if res.isnull().values.any():
            pdb.set_trace()

        base = interest_margin(name)
        curr_set = add_q_value(res, base, data_length)

        # save data, as train data or test data
        if train_test_cnt % 5 == 1:
            save_name = save_dir + 'test/' 
        else:
            save_name = save_dir + 'train/'

        train_test_cnt += 1

        if len(curr_set) == 0:
            continue

        for data in curr_set:
            if data is None:
                stop
            torch.save(data, save_name + str(cnt) + '.data')
            cnt += 1


def add_first_last_rec(data, first, last):
    head = data.head(1).copy()
    head[['date']] = first
    tail = data.tail(1).copy()
    tail[['date']] = last
    head['date'] = pd.to_datetime(head['date'])
    tail['date'] = pd.to_datetime(tail['date'])
    ret = pd.concat([head, data, tail]).reset_index(drop=True)
    return ret


def add_q_value(data, base=0.00006, length=3600, del_tail=True):
    ret = []
    data = data.drop(['date'], axis=1)
    data = torch.Tensor(data.values)
    feature_dim = data.shape[-1]
    if len(data) > length:
        rest = len(data) % length
        if rest > 120:
            rest_data = data[-rest:]
            rest_data = torch.unsqueeze(rest_data, 1)
            q_value_rest = Qvalue.q_value_max_slope(rest_data)
            if del_tail:
                q_value_rest = q_value_rest[:-100]
            ret.append(q_value_rest)
        data = data[:-rest]
        data = fold_data(data, (length, -1, feature_dim))
    else:
        data = torch.unsqueeze(data, 1)

    q_value = Qvalue.q_value_max_slope(data)
    if del_tail:
        q_value = q_value[:-100]

    if len(q_value) > 120:
        ret.append(q_value)
    return ret


def clear_robot_data(fn='workspace/data/EURUSD/robot_data/'):
    fn1 = fn + 'train/'
    fn2 = fn + 'test/'
    fns = os.listdir(fn1)

    for fn in fns:
        os.remove(fn1+fn)

    fns = os.listdir(fn2)

    for fn in fns:
        os.remove(fn2+fn)


if __name__ == '__main__':
    prepare_robot_data()
