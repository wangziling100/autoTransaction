import requests
import time
from datetime import datetime
from operate import BaseOp, Analyse


class pair_struct:

    def __init__(self, name):
        self.name = name
        self.ticker = ''
        self.bid = 0
        self.ask = 0
        self.open = 0
        self.low = 0
        self.high = 0
        self.changes = 0
        self.date = ''

    def load(self, info):
        self.ticker = info['ticker']
        self.bid = float(info['bid'])
        self.ask = float(info['ask'])
        self.open = float(info['open'])
        self.low = float(info['low'])
        self.high = float(info['high'])
        self.chages = float(info['changes'])
        # self.date = info['date']
        self.date = datetime.strptime(info['date'], \
                '%Y-%m-%d %H:%M:%S')


def run():
    ####################init#####################
    account_currency = 'EUR'
    forex_pairs = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'EUR/GBP', \
            'EUR/JPY', 'USD/CHF', 'USD/CAD', 'AUD/USD', 'GBP/JPY', \
            'AUD/CHF', 'AUD/JPY', 'AUD/NZD', 'CAD/CHF', 'CAD/JPY', \
            'CHF/JPY', 'EUR/AUD', 'EUR/CAD', 'EUR/NOK', 'EUR/NZD', \
            'GBP/CAD', 'GBP/CHF', 'NZD/JPY', 'NZD/USD', 'USD/NOK', \
            'USD/SEK', 'AUD/CAD', 'EUR/CHF']
    pair_structs = {}
    ana = Analyse()

    op = BaseOp(200)
    leverage = 30
    for pair in forex_pairs:
        pair_structs[pair] = pair_struct(pair)
    ######################running##########################
    for i in range(20):
        response = requests.get("https://financialmodelingprep.com/api/v3/forex")
        all_pairs = response.json()['forexList']
        for pair in all_pairs:
            pair_name = pair['ticker']
            # print(pair_name)
            pair_structs[pair_name].load(pair)

        ninja = pair_structs['USD/JPY']
        selected_pair = ninja
        base_curr, quote_curr = selected_pair.name.split('/')
        if base_curr == account_currency:
            hb_ratio = 1
        else:
            name = account_currency+'/'+base_curr
            try:
                hb_ratio = pair_structs[name].bid
            except:
                name = base_curr+'/'+account_currency
                hb_ratio = 1/pair_structs[name].ask

        if quote_curr == account_currency:
            qh_ratio = 1

        else:
            name = quote_curr+'/'+account_currency
            try:
                qh_ratio = pair_structs[name].bid
            except:
                name = account_currency+'/'+quote_curr
                qh_ratio = 1/pair_structs[name].ask

        # print(selected_pair.bid, selected_pair.ask)
        # op.detect(ninja.bid, ninja.ask, hb_ratio, qh_ratio, leverage)
        ana.collect(all_pairs)
        time.sleep(1)
        if i%100==0:
            ana.report()


if __name__ == '__main__':
    run()
