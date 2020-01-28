from util import IO


class DecisionMaker:
    def __init__(self):
        self.decision = None
        self.pre_decision = None

    def max_out(self, data):
        # data struction : [buy, sell]
        buy = data[0]
        sell = data[1]
        if buy >= sell:
            self.decision = 'buy'
            return 'buy'
        else:
            self.decision = 'sell'
            return 'sell'

    def from_rnn_feature(self, feature):
        if feature > 0:
            self.decision = 'buy'
            return 'buy'
        else:
            self.decision = 'sell'
            return 'sell'

    def write_for_rnn_model(self, price, prof, loss):
        """
        price:      current price
        prof:       profit limit
        loss:       loss limit
        """

        if self.decision == 'buy':
            upper = price + prof
            lower = price - loss

        elif self.decision == 'sell':
            upper = price + loss
            lower = price - prof

        if self.pre_decision == self.decision:
            self.pre_decision = 'close'
        else:
            self.pre_decision = self.decision
        fn = 'workspace/data/decision'
        data = [self.pre_decision, upper, lower]
        IO.add_data(data, fn, '|')


class DecisionWriter:
    def write_decision_with_limit(price, decision, prof, loss):
        if decision == 'buy':
            upper = price + prof
            lower = price - loss
            
        elif decision == 'sell':
            upper = price + loss
            lower = price - prof

        upper = round(upper, 5)
        lower = round(lower, 5)

        fn = 'workspace/data/decision'
        data = [decision, upper, lower]
        IO.add_data(data, fn, '|')

    def clear_decision():
        fn = 'workspace/data/decision'
        decision = 'close'
        data = [decision, None, None]
        IO.add_data(data, fn, '|')
