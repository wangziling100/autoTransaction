from multiprocessing import Queue, Process
from online_prediction import online_prediction
from robot import Robot
import torch
from util import IO, Norm, interest_margin
from datetime import datetime
import time
from momentum_robot import MRobot
from base_robot import TableRobot, ProbRobot
from decision_maker import DecisionMaker, DecisionWriter


def data_transform(data):
    price = data[0]
    diff = data[1]
    acc = data[2]
    feature = []
    for p, d, a in zip(price, diff, acc):
        if p is None:
            exp = 100
            conf = 100
        elif d is None:
            exp = 100
            conf = a/p
        elif a is None:
            exp = d/p
            conf = 100
        else:
            exp = d/p
            conf = a/p

        feature.append(exp)
        feature.append(conf)

    return feature


def main():

    ################### define robot ################
    # name = 'EURUSD'
    """
    feature_bit = [6, 7, 8, 9, 10, 11, 12, 13]
    n_features = len(feature_bit)
    bob = Robot(name, input_dim=n_features)
    bob.load(fn='6_13_600l')
    """
    bob = ProbRobot('EURUSD')

    ################### init local param ############

    # feature_con = torch.Tensor()
    q = Queue()
    p = Process(target=online_prediction, args=(q,))
    p.start()
    IO.clear_data('workspace/log/available.log')
    # dmaker = DecisionMaker()
    # prof_limit = 2*10**-4
    # loss_limit = 10**-3
    history = []
    IO.write_request(['search index', 'open price'])
    open_price = IO.get_response()['open price']
    IO.close_response()
    bob.set_opening_price(open_price)
    DecisionWriter.clear_decision()
    pre_decision = 'close'
    cnt = 0



    ################ loops ##################

    while True:
        if q.empty():
            time.sleep(0.1)
            continue
        origin, price = q.get()
        if price is None:
            time.sleep(0.1)
            continue
        # price_diff = origin[1]
        history.append(price)
        if len(history) > 1000:
            history = history[-1000:]

        cnt += 1
        # print(cnt)

        """
        feature = data_transform(origin)
        feature = torch.Tensor(feature).reshape(1, 1, -1)
        feature = Norm.abs_max_no_bias(feature)
        feature = feature[:, :, feature_bit]
        feature_con = torch.cat((feature_con, feature), dim=0)
        feature_con = feature_con[-600:]
        if len(feature_con) < 240:
            continue
        """

        if IO.load_prof_loss() > 0.000000000001:
            continue

        # diff_in_5_min = price_diff[5]
        """
        if diff_in_5_min is None:
            continue
        """

        # decision = dmaker.from_rnn_feature(diff_in_5_min)
        # print(decision)

        """
        decision = bob.predict(feature_con)
        decision = decision.reshape(-1).tolist()
        decision = dmaker.max_out(decision)
        print(decision)
        """
        # dmaker.write_for_rnn_model(price, prof_limit, loss_limit)
        bob.calc_avg_price(history, period=100)
        res = bob.make_decision(price)
        if res is not None:
            decision, prof_lim, loss_lim = res
            if decision == pre_decision:
                DecisionWriter.clear_decision()
                pre_decision = 'close'
            else:
                DecisionWriter.write_decision_with_limit(price, decision, prof_lim, loss_lim)
                pre_decision = decision
                time.sleep(3)



if __name__ == '__main__':
    # load_prof()
    main()
