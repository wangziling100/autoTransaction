from datetime import datetime
import random
from pandas.io.json import json_normalize
import pandas as pd


class BaseOp:

    def __init__(self, capital):
        self.capital = capital
        self.opening = 0
        self.closing = 0
        self.start_time = 0
        self.end_time = 0
        self.today = datetime.now().date()
        self.ana = Analyse()
        self.operating = False
        self.trend = None

    def cap2uni(self, hb_ratio, leverage):
        self.units = self.capital*hb_ratio*leverage

    def open(self, price, hb_ratio, leverage):
        self.cap2uni(hb_ratio, leverage)
        self.opening = price
        self.start_time = datetime.now()
        self.operating = True

    def detect(self, bid, ask, hb_ratio, qh_ratio, leverage):
        end_time = datetime.now()

        trend = self.ana.get_trend()
        
        if not self.operating:
            self.trend = trend
            if self.trend == 'long':
                self.open(ask, hb_ratio, leverage)
            elif self.trend == 'short':
                self.open(bid, hb_ratio, leverage)

        else:
            if trend != self.trend:
                if self.trend == 'long':
                    self.close(bid, qh_ratio)
                elif self.trend == 'short':
                    self.close(ask, qh_ratio)
            else:

                # performance
                duration = end_time-self.start_time
                if trend == 'long':
                    perf = self.ana.avg_perf(duration, self.opening, bid, self.trend, qh_ratio, self.units)
                elif trend == 'short':
                    perf = self.ana.avg_perf(duration, self.opening, ask, self.trend, qh_ratio, self.units)
                print(perf)

    def close(self, price, qh_ratio):
        self.closing = price
        self.end_time = datetime.now()
        # log the performance
        self.duration = self.end_time-self.start_time
        profit = self.ana.profit(self.opening, self.closing, self.trend, qh_ratio, self.units)
        perf = self.ana.avg_perf(self.duration, self.opening, self.closing, self.trend, qh_ratio, self.units)
        log_name = 'workspace/log/'+str(self.today)+'.log'
        with open(log_name, 'a') as f:
            f.write('----------------------\n')
            f.write('capital: '+str(self.capital)+'\n')
            f.write('opening: '+str(self.opening)+'\n')
            f.write('closing: '+str(self.closing)+'\n')
            f.write('duration: ' +str(self.duration)+'\n')
            f.write('trend: '+self.trend+'\n')
            f.write('performance: '+str(perf)+'\n')
            f.write(self.ana.info+'\n')
            f.write('profit: '+str(profit)+'\n')

        # clean the data
        self.capital += profit
        self.opening = 0
        self.closing = 0
        self.start_time = 0
        self.end_time = 0
        self.duration = 0
        self.operating = False
        self.trend = None


class Analyse:
    def __init__(self, n_index=200000):
        self.info = ''
        self.data = None
        self.n_index = n_index

    def get_trend(self):
        ret = random.choice(['long', 'short'])
        return ret

    def avg_perf(self, dur, opening, closing, trend, qh_ratio, units):
        self.info = '(average performance, point/s)'
        return self.profit(opening, closing, trend, qh_ratio, units)/(dur.seconds+0.001)

    def profit(self, opening, closing, trend, qh_ratio, units):
        if trend == 'long':
            return (closing-opening)*qh_ratio*units
        elif trend == 'short':
            return (opening-closing)*qh_ratio*units

    def collect(self, data):
        data = json_normalize(data)
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index(['date'])
        self.data = pd.concat([self.data, data])
        if len(self.data)>self.n_index:
            self.data = self.data[-self.n_index:]

    def report(self):
        self.data.to_csv('workspace/data/tmp.csv')


class Draw:
    def __init__(self, file_dir='workspace/data/figure/'):
        self.file_dir = file_dir

    def line(self, df, name):
        fn = self.file_dir+name+'.png'


if __name__ == '__main__':
    draw = Draw()
    df = pd.read_csv('workspace/data/2019-08-12\ 10:06:22.890532.csv')
