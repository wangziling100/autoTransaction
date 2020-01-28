import torch
import os
from util import Norm, interest_margin
from robot import Robot
import numpy as np
import sys
from datetime import datetime


def train_robot(name='EURUSD', epoch=20, feature_bit=None, save_name=None, verbose=False):

    if feature_bit is None:
        n_features = 18
        feature_bit = [i for i in range(n_features)]
    else:
        n_features = len(feature_bit)

    bob = Robot(name, input_dim=n_features, hidden_dim=5, l_dim1=5, l_dim2=5)
    if save_name is None:
        save_name = str(datetime.now())
    else:
        try:
            bob.load(fn=save_name)
            bob.load_checkpoint(fn=save_name)
        except Exception:
            pass
    loss_con = []
    prof_con = []
    base_price = interest_margin(name)
    highest_prof = 0

    # output_std_con = []

    for i in range(epoch):

        datum = gen_train_data(name)
        if verbose:
            print('epoch:', i)
        for data in datum:
            # normalization
            price = data[:, :, 0]
            inputs = data[:, :, 1:-2]
            targets = data[:, :, -2:]
            inputs = Norm.abs_max_no_bias(inputs)
            # print(inputs.shape)
            inputs = inputs[:, :, feature_bit]
            # targets = Norm.minus_min_abs_no_bias(targets)
            targets = transform2bi(targets)
            # print(targets)
            # print(targets.mean())
            # print(targets.max(), targets.min())

            # loss
            outputs, loss = bob.mini_train(inputs, targets)
            loss_con.append(loss.item())
            
            # evluation
            prof = eval(price, outputs, base_price)
            prof_con.append(prof.item())

            # output std
            # output_std = torch.std(outputs, dim=0)
            # output_std_con.append(output_std)

        print('mean loss:', np.mean(loss_con))
        print('mean profit:', np.mean(prof_con))
        loss_con = []
        prof_con = []
        print('-----------------')

        test_datum = gen_test_data(name)
        loss_test_con = []
        prof_test_con = []

        for data in test_datum:
            # normalization
            price = data[:, :, 0]
            inputs = data[:, :, 1:-2]
            targets = data[:, :, -2:]
            inputs = Norm.abs_max_no_bias(inputs)
            inputs = inputs[:, :, feature_bit]
            targets = transform2bi(targets)

            # loss
            outputs, loss = bob.mini_test_return_all(inputs, targets)
            loss_test_con.append(loss.item())

            # evluation
            prof = eval(price, outputs, base_price)
            prof_test_con.append(prof.item())

        mean_prof = np.mean(prof_test_con)
        print('------------test--------------')
        print('mean loss:', np.mean(loss_test_con))
        print('mean profit:', mean_prof)
        if mean_prof > highest_prof:
            bob.save(fn=save_name)
            bob.save_checkpoint(fn=save_name)
            print('model is saved')
            highest_prof = mean_prof
        print('---------------------------')
        
        loss_test_con = []
        prof_test_con = []


def gen_train_data(name='EURUSD'):
    file_dir = 'workspace/data/'+name+'/robot_data/train/'
    fns = os.listdir(file_dir)
    for fn in fns:
        yield torch.load(file_dir+fn)


def gen_test_data(name='EURUSD'):
    file_dir = 'workspace/data/'+name+'/robot_data/test/'
    fns = os.listdir(file_dir)
    for fn in fns:
        yield torch.load(file_dir+fn)


def save_norm_model(name, fn=''):
    datum = gen_train_data(name)

    def f(data):
        data[data > 1] = 0
        data = data[:, :, 1:-2]
        return data

    save_dir = 'workspace/data/model/'
    fn = 'abs_max_no_bias_' + fn + 'bit.model'
    Norm.abs_max_no_bias(datum, f, False, save_dir, fn)

    def f(data):
        return data[:, :, -2:]
    datum = gen_train_data(name)
    fn = 'minus_min_abs_no_bias_' + fn + 'bit.model'
    Norm.minus_min_abs_no_bias(datum, f, False, save_dir, fn)


def eval(price, outputs, base=0.00006):
    # out = outputs.squeeze(1)
    # price = price .squeeze(1)
    price = price.unsqueeze(2)
    buy = outputs[:, :, 0] - outputs[:, :, 1]
    buy = buy.unsqueeze(2)

    buy[buy <= 0] = 0
    buy[buy > 0] = -1
    sell = buy + 1
    # print(buy.std().item())
    # print(sell.std().item())
    ext = torch.zeros(1, buy.shape[1], 1)

    try:
        buy = torch.cat((ext, buy), dim=0)
    except Exception as e:
        print(e)
        print(buy.shape)
        print(price.shape)
        sys.exit(1)
    buy[-1] = 0
    ext = torch.zeros(1, sell.shape[1], 1)
    sell = torch.cat((ext, sell), dim=0)
    sell[-1] = 0
    tmp1 = buy[1:]
    tmp2 = buy[:-1]
    tmp = tmp1-tmp2

    eval1 = (price*tmp).sum()
    tmp1 = sell[1:]
    tmp2 = sell[:-1]
    tmp = tmp1-tmp2
    eval2 = (price*tmp).sum()
    # calc times of handle
    tmp[tmp < 0] = 0
    tmp = tmp.sum()

    res = (eval1 + eval2 - tmp*base)/outputs.shape[1]
    # print('std of buy:', buy.std().item())
    # print('std of sell:', sell.std().item())
    # print('eval:', res.item())
    return res


def transform2bi(targets):
    buy_ = targets[:, :, [0]]
    sell_ = targets[:, :, [1]]
    buy = buy_
    sell = sell_
    buy[buy_ > sell_] = 1
    buy[buy_ <= sell_] = 0
    sell = -buy + 1
    return torch.cat((buy, sell), dim=2)


if __name__ == '__main__':
    name = 'EURUSD'
    # feature_bit = [i for i in range(14)]
    feature_bit = [#0, 1, 2, 3, 4, 5, 
            6, 7, 8, 9, 10, 11, 12, 13, 
            #14, 15, 16, 17
            ]
    train_robot(name, epoch=100, verbose=True, feature_bit=feature_bit, save_name='6_13_600l')
