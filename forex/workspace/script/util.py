from datetime import datetime
import traceback
import pickle
import subprocess
import string
import numpy as np
import time
import itertools
import pdb
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
import json

try:
    from analyse import load_data, Predictor
    import torch
except Exception as e:
    print(e)


class Log:
    def __init__(self, dir_name='workspace/log/'):
        self.dir_name = dir_name

    def error(self, info):
        with open(self.dir_name+'error.log', 'a') as f:
            f.write('----------------------\n')
            now = datetime.now()
            f.write(str(now)+'\n')
            f.write(info)
            f.write('\n')
            f.write('-----details-----\n')
            f.wrote(traceback.format_exc())
            f.write('\n\n')
            f.close()

    def avg_diff(self, name, info):
        with open(self.dir_name+'avg_diff.log', 'a') as f:
            f.write('------------ '+name+' -------------\n')
            levels = ['1s', '5s', '10s', '30s', '1m',
                    '5m', '10m', '30m', '1h']
            for level, ele in zip(levels, info):
                s = str(level)+':'+str(np.std(ele))+'\n'
                f.write(s)
            f.close()

    def write_available(self, name, info):
        with open(self.dir_name+'available.log', 'a') as f:
            f.write(str(datetime.now())+' '+name+': '+info)
            f.write('\n')


class Wrapper:
    def __init__(self, name):
        self.name = name

    def online_predict(self, reader, level, predictor, ret, bit):
        """
        bit:        change which bit in ret
        """
        data = reader.load_online_data_by_level(level)

        if data is None or len(data) < 2:
            pred = None
            pred_price = None
            curr_price = None
        else:
            curr_price = data.iloc[-1]['price']
            max_ = data['price'].max()
            min_ = data['price'].min()

            pred = predictor.online_predict(data)
            pred = pred.item()

            pred_price = pred*(max_-min_)+min_
        ret[bit] = pred_price
        return ret, curr_price, pred_price


def interest_margin(name):
    if name == 'EURUSD':
        return 0.00006


def fold_data(data, shape):
    shape = [shape[1], shape[0], shape[2]]
    data = data.reshape(shape)
    data = data.permute(1, 0, 2)
    return data


class Rolling:
    def rolling_max(data):
        l_ = len(data)
        tmp = torch.zeros(data.shape)
        ret = torch.cat((data, tmp), 0)
        ret = ret.unfold(0, l_, 1)
        ret = ret[:-1]
        max_, index = torch.max(ret, dim=2)
        return max_, index

    def rolling_max_slope(data, base=0.00006, action='buy'):
        if action == 'buy':
            is_buy = True
            is_sell = False
        elif action == 'sell':
            is_buy = False
            is_sell = True

        l_ = len(data)
        tmp = data[:, :, 0]
        if is_buy:
            tmp1 = torch.zeros(tmp.shape)
        elif is_sell:
            tmp1 = torch.ones(tmp.shape)*100
        ret = torch.cat((tmp, tmp1), 0)
        ret = ret.unfold(0, l_, 1)
        ret = ret[:-1]
        print(ret.shape)

        if is_buy:
            diff = ret[1:] - ret[0] - base
        if is_sell:
            diff = ret[0] - ret[1:] - base

        time = torch.range(1, len(diff))
        time = time.reshape(-1, 1, 1)

        slope = diff/time.float()
        max_slope, index = torch.max(slope, dim=0)
        time = index.float()+1
        max_slope = max_slope/time
        max_slope = max_slope.permute(1, 0)
        max_slope = max_slope.unsqueeze(2)
        data = torch.cat((data, max_slope), 2)
        return data


class Norm:

    def abs_max_no_bias(datum, f=None, load=True, save_dir='workspace/data/model/', fn='abs_max_no_bias.model'):
        # f:    a function for data filter to reduce outliers
        #
        # find out abs max data
        # max_abs = max(abs(data))
        # ret = data/max_abs

        fn = save_dir + fn
        if load:
            with open(fn, 'rb') as f:
                param = pickle.load(f)

            if type(datum) is torch.Tensor:
                try:
                    return datum/param
                except Exception as e:
                    print(e)
                    print('the feature dim should match the norm model')

            else:
                print('please input tensor data')

        else:
            if type(datum) is torch.Tensor:
                datum = [datum]

            max_ = None
            min_ = None

            for data in datum:
                if f is not None:
                    tmp = f(data)
                else:
                    tmp = data
                shape = tmp.shape
                tmp = tmp.reshape(shape[0]*shape[1], -1)

                if max_ is not None:
                    tmp = torch.cat((tmp, max_, min_), dim=0)

                max_, _ = torch.max(tmp, dim=0)
                max_ = max_.unsqueeze(0)
                min_, _ = torch.min(tmp, dim=0)
                min_ = min_.unsqueeze(0)

            min_ = torch.abs(min_)
            param = torch.cat((max_, min_), dim=0)
            param, _ = torch.max(param, dim=0)

            with open(fn, 'wb') as f:
                pickle.dump(param, f)
            return

    def minus_min_abs_no_bias(datum, f=None, load=True, save_dir='workspace/data/model/', fn='minus_min_abs_no_bias.model'):
        # f:    a function for data filter to reduce outliers
        #
        # min = min(data_-)
        # ret = data/abs(min)

        fn = save_dir + fn
        if load:
            with open(fn, 'rb') as f:
                param = pickle.load(f)

            if type(datum) is torch.Tensor:
                try:
                    return datum/param
                except Exception as e:
                    print(e)
                    print('the feature dim should match the norm model')

            else:
                print('please input tensor data')

        else:
            if type(datum) is torch.Tensor:
                datum = [datum]

            min_ = None

            for data in datum:
                if f is not None:
                    tmp = f(data)
                else:
                    tmp = data

                shape = tmp.shape
                tmp = tmp.reshape(shape[0]*shape[1], -1)

                if min_ is not None:
                    tmp = torch.cat((tmp, min_), dim=0)

                min_, _ = torch.min(tmp, dim=0)
                min_ = min_.unsqueeze(0)

            if torch.sum(min_ > 0).item() > 0:
                print('no negative vaule')
                return
            param = torch.abs(min_)

            with open(fn, 'wb') as f:
                pickle.dump(param, f)
            return


class IO:
    def add_data(data, fn, sep=' '):
        if type(data) is not list:
            data = [data]
        data = [datetime.now()] + data
        data = sep.join(str(v) for v in data)
        cmd = 'echo "' + data + '" > ' + fn
        subprocess.call(cmd, shell=True)

    def clear_data(fn):
        cmd = 'echo "" > ' + fn
        print(cmd)
        subprocess.call(cmd, shell=True)

    def read_file(fn):
        with open(fn, 'r') as f:
            return f.readline()
            f.close()

    def read_online_price():
        fn = 'workspace/data/EURUSD_new_record'
        with open(fn, 'r') as f:
            line = f.readline()
            line = line.strip('/n')
            line = line.split(' ')
            if len(line) < 2:
                return None
            t = line[1]
            price = line[-2]
            f.close()
            return t, price

    def fetch_online_price(name):
        url = 'https://financialmodelingprep.com/api/v3/forex/'+name
        response = urlopen(url)
        data = response.read().decode('utf-8')
        data = json.loads(data)
        price = data['bid']
        t = data['date']
        return t, price

    def load_prof_loss(fn='workspace/log/available.log'):
        with open(fn, 'r', errors='ignore') as f:
            lines = f.readlines()
            if len(lines) < 3:
                return 0
            line1 = lines[-1]
            line2 = lines[-2]
            line3 = lines[-3]

            line1 = line1.split(' ')[-1]
            line2 = line2.split(' ')[-1]
            line3 = line3.split(' ')[-1]

            try:
                line1 = float(line1)
                line2 = float(line2)
                line3 = float(line3)
                line1 = abs(line1)
                line2 = abs(line2)
                line3 = abs(line3)
            except ValueError:
                return 1

            return np.mean([line1, line2, line3])

    def write_response(response, response_fn='workspace/data/control/response'):
        """
        response format
        response: True/False
        open price: float
        """
        data = {}
        data['response'] = True
        for r in response:
            data[r] = response[r]

        while True:
            try:
                with open(response_fn, 'w') as f:
                    json.dump(data, f)
                    f.close()
                    return 
            except Exception:
                time.sleep(1)

    def close_response(response_fn='workspace/data/control/response'):
        data = {}
        data['response'] = False
        while True:
            try:
                with open(response_fn, 'w') as f:
                    json.dump(data, f)
                    f.close()
                    return
            except Exception:
                time.sleep(1)

    def write_request(requests, request_fn='workspace/data/control/request'):
        data = {}
        data['request'] = True
        data['order'] = requests
        """
        for r in requests:
            data[r] = True
        """

        while True:
            with open(request_fn, 'w') as f:
                try:
                    json.dump(data, f)
                    f.close()
                    return 
                except Exception:
                    time.sleep(1)

    def close_request(request_fn='workspace/data/control/request'):
        data = {}
        data['request'] = False
        while True:
            try:
                with open(request_fn, 'w') as f:
                    json.dump(data, f)
                    f.close()
                    return
            except Exception:
                time.sleep(1)

    def get_response(response_fn='workspace/data/control/response'):
        while True:
            try:
                with open(response_fn, 'r') as f:
                    response = json.load(f)
                    f.close()
                    if response['response'] is True:
                        return response
            except Exception:
                time.sleep(1)
                continue
            time.sleep(1)

    def _del_strange_char(data):
        data = ''.join(str(c) for c in data if c in string.printable)

        data = data.strip('/n')
        return data


class Tool:
    def get_all_combined(params):
        # params:   list of list, every element in the outside list
        #           is a list of all possible params of this param
        # ret:      list of list, every element of the outside list
        #           is a combination of all params
        return list(itertools.product(*params))
        """ 
        if n >= len(params):
            return [[]]
        ret = []
        param = params[n]
        n += 1
        for p_0 in param:
            for p_1 in get_all_combined(params, n):
                curr = [p_0] + p_1
                ret.append(curr)

        return ret
        """

    def list2dict(data, names):
        # data:     list of list, each element is a list which
        #           converted to dict
        # name:     dict format
        # ret:      list of dicts
        ret = []
        for d in data:
            curr = {}
            for e, n in zip(d, names):
                curr[n] = e

            ret.append(curr)
        return ret


if __name__ == '__main__':
    print(IO.load_prof_loss())
