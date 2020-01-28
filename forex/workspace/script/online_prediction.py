from util import Wrapper, Log
from analyse import load_data, Predictor
import time
import sys


def online_prediction(q):
    """
    Input:
    q:          Queue object, used to translate data

    Output:
    ret_price:  a set of predicted price
    ret_acc:    the accuracy of predicted price 
    ret_diff:   the predicted change than last price
    curr:       current price

    """
    name = 'EURUSD'
    logger = Log()
    online_pred = Wrapper('online_predict')
    reader = load_data()
    start_time = time.time()
    pre_time = -1
    pred1s = Predictor('EURUSD', '1s', input_dim=3)
    pred5s = Predictor('EURUSD', '5s', input_dim=3)
    pred10s = Predictor('EURUSD', '10s', input_dim=3)
    pred30s = Predictor('EURUSD', '30s', input_dim=3)
    pred1s.load()
    pred5s.load()
    pred10s.load()
    pred30s.load()

    pred1m = Predictor('EURUSD', '1m', input_dim=34)
    pred5m = Predictor('EURUSD', '5m', input_dim=34)
    pred10m = Predictor('EURUSD', '10m', input_dim=34)
    pred30m = Predictor('EURUSD', '30m', input_dim=34)
    pred1h = Predictor('EURUSD', '1h', input_dim=34)
    pred1m.load()
    pred5m.load()
    pred10m.load()
    pred30m.load()
    pred1h.load()

    p1s = None
    p5s = None
    p10s = None
    p30s = None
    p1m = None
    p5m = None
    p10m = None
    p30m = None
    p1h = None

    diff1s = []
    diff5s = []
    diff10s = []
    diff30s = []
    diff1m = []
    diff5m = []
    diff10m = []
    diff30m = []
    diff1h = []

    ret_price = [None for _ in range(9)]
    # pred - curr
    ret_diff = [None for _ in range(9)]
    # last pred for curr - curr
    ret_acc = [None for _ in range(9)]

    sig_num = 0
    signals_control(sig_num)
    time.sleep(10)
    save_pre_time = None
    # reader.load_base()

    try:
        for _ in range(1000000000):
            reader.load_online_data()
            time_dist = int(time.time()-start_time)

            # 1s
            if pre_time != time_dist:
                pre_time = time_dist
                ret_price, curr, pred = online_pred.online_predict(reader, '1s', pred1s, ret_price, 0) 
                if p1s is not None:
                    diff = abs(p1s-curr)
                    diff1s.append(diff)
                    ret_acc[0] = p1s-curr

                if curr is not None:
                    ret_diff[0] = pred-curr

                p1s = pred
                ret = [ret_price, ret_diff, ret_acc]
                q.put([ret, curr])

                # 5s
                if time_dist % 5 == 1:
                    sig_num += 1
                    signals_control(sig_num)

                    ret_price, curr, pred = online_pred.online_predict(reader, '5s', pred5s, ret_price, 1)

                    if p5s is not None:
                        diff = abs(p5s-curr)
                        diff5s.append(diff)
                        ret_acc[1] = p1s-curr

                    if curr is not None:
                        ret_diff[1] = pred-curr
                    p5s = pred

                # 10s
                if time_dist % 10 == 1:
                    ret_price, curr, pred = online_pred.online_predict(reader, '10s', pred10s, ret_price, 2)

                    if p10s is not None:
                        diff = abs(p10s-curr)
                        diff10s.append(diff)
                        ret_acc[2] = p1s-curr

                    if curr is not None:
                        ret_diff[2] = pred-curr

                    p10s = pred

                # 30s
                if time_dist % 30 == 1:
                    ret_price, curr, pred = online_pred.online_predict(reader, '30s', pred30s, ret_price, 3)

                    if p30s is not None:
                        diff = abs(p30s-curr)
                        diff30s.append(diff)
                        ret_acc[3] = p1s-curr

                    if curr is not None:
                        ret_diff[3] = pred-curr

                    p30s = pred

                # 1m
                if time_dist % 60 == 1:
                    ret_price, curr, pred = online_pred.online_predict(reader, '1m', pred1m, ret_price, 4)

                    if p1m is not None:
                        diff = abs(p1m-curr)
                        diff1m.append(diff)
                        ret_acc[4] = p1s-curr

                    if curr is not None:
                        ret_diff[4] = pred-curr
                    p1m = pred

                    # save base data
                    save_pre_time = reader.save_base(save_pre_time)


                # 5m
                if time_dist % 300 == 1:
                    ret_price, curr, pred = online_pred.online_predict(reader, '5m', pred5m, ret_price, 5)

                    if p5m is not None:
                        diff = abs(p5m-curr)
                        diff5m.append(diff)
                        ret_acc[5] = p1s-curr

                    if curr is not None:
                        ret_diff[5] = pred-curr
                    p5m = pred

                    # log average diff
                    info = [diff1s, diff5s, diff10s, diff30s, diff1m,
                            diff5m, diff10m, diff30m, diff1h]
                    logger.avg_diff(name, info)

                    diff1s = []
                    diff5s = []
                    diff10s = []
                    diff30s = []
                    diff1m = []
                    diff5m = []
                    diff10m = []
                    diff30m = []
                    diff1h = []

                # 10m
                if time_dist % 600 == 1:
                    ret_price, curr, pred = online_pred.online_predict(reader, '10m', pred10m, ret_price, 6)

                    if p10m is not None:
                        diff = abs(p10m-curr)
                        diff10m.append(diff)
                        ret_acc[6] = p1s-curr

                    if curr is not None:
                        ret_diff[6] = pred-curr
                    p10m = pred

                # 30m
                if time_dist % 1800 == 1:
                    ret_price, curr, pred = online_pred.online_predict(reader, '30m', pred30m, ret_price, 7)

                    if p30m is not None:
                        diff = abs(p30m-curr)
                        diff30m.append(diff)
                        ret_acc[7] = p1s-curr

                    if curr is not None:
                        ret_diff[7] = pred-curr
                    p30m = pred

                # 1h
                if time_dist % 3600 == 1:
                    ret_price, curr, pled = online_pred.online_predict(reader, '1h', pred1h, ret_price, 8)

                    if p1h is not None:
                        diff = abs(p1h-curr)
                        diff1h.append(diff)
                        ret_acc[8] = p1s-curr

                    if curr is not None:
                        ret_diff[8] = pred-curr
                    p1h = pred

            time.sleep(0.05)

    except IndexError:
        print('Abnormal exit')
        reader.save_base(0, True)
        sys.exit(0)


def signals_control(n, fn='workspace/data/control/signals'):
    with open(fn, 'w') as f:
        f.write('save:False\n')
        f.write('count:'+str(n))
        f.close()


if __name__ == '__main__':
    online_prediction()

