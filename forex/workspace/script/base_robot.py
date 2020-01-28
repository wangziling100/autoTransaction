import pandas as pd
import torch
import numpy as np
from util import Tool
import pdb
import re


class BaseRobot:
    def __init__(self, name):
        self.name = name

    def make_decision(self, data):
        pass


class TableRobot(BaseRobot):
    def __init__(self, name, base, n_after):
        super().__init__(name)
        self.base = base
        self.n = n_after
        # hour = time.gmtime()[3]
        # self.load(hour)
        # self._get_decision_table()

    def load(self, hour):
        base = self.base
        n = self.n
        fn = 'workspace/data/model/table_tr_'+str(base)+'_'+str(n)+'_'+str(hour)+'.csv'
        self.table = pd.read_csv(fn)

    def get_strict_decision_table(self):
        tmp = self.table.copy()
        base = self.base
        val_col = 'c' + str(base)
        tmp = tmp[tmp[val_col] != 0]

        tmp_pos = tmp.copy()
        tmp_pos.loc[tmp_pos[val_col]==1, ] = -1
        tmp_pos[val_col] = tmp_pos[val_col]/tmp_pos[val_col].abs()
        tmp_pos = tmp_pos[tmp_pos[val_col] > 0]

        tmp_neg = tmp.copy()
        tmp_neg.loc[tmp_neg[val_col]==-1, ] = 1
        tmp_neg[val_col] = tmp_neg[val_col]/tmp_neg[val_col].abs()
        tmp_neg = tmp_neg[tmp_neg[val_col] < 0]
        self.s_table = pd.concat([tmp_pos, tmp_neg]).reset_index(drop=True)
        del tmp
        return self.d_table

    def _get_decision_table(self):
        tmp = self.table.copy()
        base = self.base
        col = 'c' + str(base)
        tmp = tmp[tmp[col] != 0]

        tmp[col] = tmp[col]/tmp[col].abs()
        by = list(tmp.columns)[:base+1]
        tmp = tmp.groupby(by).sum()
        self.d_table = tmp.reset_index()
        del tmp
        return self.d_table

    def _trend_table(self, data, pre_decision):
        data = data[:3]
        if data[0] == 2:
            return 'wait'
        elif data[0] == 0:
            return 'buy'
        elif data[0] == 4:
            return 'sell'
        elif pre_decision == 'buy' and data[0] == 3:
            return 'close'
        elif pre_decision == 'sell' and data[0] == 1:
            return 'close'
        elif data[:2] == [1, 0]:
            return 'buy'
        elif data[:2] == [3, 4]:
            return 'sell'
        else:
            return 'wait'

    def make_decision(self, index, pre_decision):

        index = index[-self.base:]
        index = np.array(index).argsort()
        # index = pd.Series(index)
        # index = index.rank()-1
        # print(index)
        prob = self.search_table(index)
        # print(prob)
        data = prob.iloc[:, self.base+1:].values
        data = torch.Tensor(data)
        # print(data)
        max_, index = torch.max(data, dim=0)
        index = index.tolist()
        return self._trend_table(index, pre_decision)

    def brake(self, price, pre_decision, high, low):
        if pre_decision == 'sell' and (price-low)>10**-4:
            return 'close'
        elif pre_decision == 'buy' and (high-price)>10**-4:
            return 'close'
        else:
            return pre_decision

    def search_table(self, index):
        ret = self.table.copy()
        cnt = 0
        for i in index.tolist():
            col = 'c'+str(cnt)
            ret = ret[ret[col]==i]
            cnt += 1

        return ret


class ProbRobot(BaseRobot):

    def __init__(self, name, prob_table=None):
        super().__init__(name)
        if prob_table is None:
            fn = 'workspace/data/model/prob_in_trade_adjustment.csv'
            self.prob_table = pd.read_csv(fn)
        else:
            self.prob_table = prob_table
        self.di = None

    def set_opening_price(self, price):
        # a daily opening price, for forex we set it as end  price
        # in the last day
        self.open_price = price

    def calc_avg_price(self, history, period=100):
        if len(history) < period:
            self.avg = None
        else:
            data = history[-period:]
            self.avg = np.mean(data)

    def set_direction(self, di=None):
        self.di = di

    def make_decision(self, price):
        # search the best choice for buy and for sell
        table = self.prob_table
        last_data_num = table['data_num'].unique()[-1]
        data = table[table['data_num'] == last_data_num]
        data = data.fillna('nan')
        data = data.drop('data_num', axis=1)
        data = data.groupby(['prof', 'loss', 'open_di',\
                'fl_interval', 'mean_di']).mean()
        data = data.reset_index()
        buy_keywords, sell_keywords = self.gen_keyword(price)
        if buy_keywords is None:
            buy_best = None
        else:
            buy_table = self._search_table_by_keywords(buy_keywords)
            # pdb.set_trace()
            buy_best = buy_table.sort_values(['long_ex_prof'], ascending=False).head(1)

        if sell_keywords is None:
            sell_best = None
        else:
            sell_table = self._search_table_by_keywords(sell_keywords)
            sell_best = sell_table.sort_values(['short_ex_prof'], ascending=False).head(1)

        # compare buy and sell
        if buy_best is None and sell_best is None:
            return None
        elif buy_best is None:
            selected = 'sell'
        elif sell_best is None:
            selected = 'buy'
        else:
            try:
                buy_expected = buy_table['long_ex_prof'].tolist()[0]
                sell_expected = sell_table['short_ex_prof'].tolist()[0]
            except Exception:
                return None
            if buy_expected > sell_expected and buy_expected > 0:
                selected = 'buy'
            elif buy_expected <= sell_expected and sell_expected > 0:
                selected = 'sell'
            else:
                return None

        if selected == 'buy':
            prof = buy_best['prof'].tolist()[0]
            loss = buy_best['loss'].tolist()[0]
            return ['buy', prof, loss]
        if selected == 'sell':
            prof = sell_best['prof'].tolist()[0]
            loss = sell_best['loss'].tolist()[0]
            return ['sell', prof, loss]

    def _search_table_by_keywords(self, keywords):
        ret = pd.DataFrame()
        for keyword in keywords:
            table = self.prob_table.copy()
            for key in keyword:
                table = table[table[key] == keyword[key]]

            ret = pd.concat([ret, table])
        return ret

    def gen_keyword(self, curr_price):
        # generate some set of keywords and check if 
        # the relative operation has a positive expected
        # profit
        open_diff = curr_price - self.open_price
        if self.avg is None:
            return None, None
        mean_diff = curr_price - self.avg

        # for buy
        if open_diff > 0:
            open_di = True
        elif open_diff < 0:
            open_di = False
        else:
            open_di = 'nan'
        if mean_diff > 0:
            mean_di = True
        elif mean_diff < 0:
            mean_di = False
        else:
            mean_di = 'nan'
        o_params = ['nan', open_di]
        m_params = ['nan', mean_di]
        o_params = list(dict.fromkeys(o_params))
        m_params = list(dict.fromkeys(m_params))
        params = [o_params, m_params]
        buy_keywords = Tool.get_all_combined(params)
        names = ['open_di', 'mean_di']
        buy_keywords = Tool.list2dict(buy_keywords, names)

        # for sell
        if open_diff > 0:
            open_di = False
        elif open_diff < 0:
            open_di = True
        else:
            open_di = 'nan'
        if mean_diff > 0:
            mean_di = False
        elif mean_diff < 0:
            mean_di = True
        else:
            mean_di = 'nan'

        o_params = ['nan', open_di]
        m_params = ['nan', mean_di]
        o_params = list(dict.fromkeys(o_params))
        m_params = list(dict.fromkeys(m_params))
        params = [o_params, m_params]
        sell_keywords = Tool.get_all_combined(params)
        names = ['open_di', 'mean_di']
        sell_keywords = Tool.list2dict(sell_keywords, names)

        if self.di == 'buy':
            sell_keywords = None
        elif self.di == 'sell':
            buy_keywords = None

        return buy_keywords, sell_keywords

