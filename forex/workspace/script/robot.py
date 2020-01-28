from util import Rolling
import random
import subprocess
from datetime import datetime
import torch
from torch import nn
import pickle


class Qvalue:

    def __init__(self):
        pass

    def q_value_max_min(data):
        # set_P = {price_0 ... price_N}
        # set_t = {price_t ... price_N}
        # set_t is in set P
        # 
        # max = max(set_t)
        # min = min(set_t)
        # q_value_t_N (for buy)   = (max-price_t)/(N-t)^2
        # q_value_t_N (for sell)  = (price_t-min)/(N-t)^2
        pass

    def q_value_max_slope(data, base=0.00006):
        # set_P = {price_0 ... price_N}
        # set_t = {price_t ... price_N}
        # set_t is in set P
        #
        # diff_t = price_t-price_0-base (for buy)
        # diff_t = price_0-price_t-base (for sell)
        # diff_t_N = price_N-price_t-base (for buy)
        # diff_t_N = price_t-price_N-base (for sell)
        # 
        # set_P_slope = {diff_0/0.0001 ... diff_N/(N+0.0001)}
        # set_t_slope = {diff_t/0.0001 ... diff_N/(N-t+0.0001)}
        # 
        # max_t = max(set_t_N_slope)
        # q_value_t_N (for buy)   = max_t/(N-t)
        # q_value_t_N (for sell)  = max_t/(N-t)
        buy = Rolling.rolling_max_slope(data)
        sell_buy = Rolling.rolling_max_slope(buy, base=base, action='sell')
        return sell_buy


class Robot:
    def __init__(self, name='EURUSD', input_dim=18, hidden_dim=5, output_dim=2, l_dim1=5, l_dim2=5):
        self.name = name
        self.actions = ['buy', 'sell', 'wait']
        self.model = RNN(input_dim, hidden_dim, output_dim, l_dim1, l_dim2)
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.params = (input_dim, hidden_dim, output_dim, l_dim1, l_dim2)

    def mini_train(self, inputs, targets):
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return outputs, loss

    def mini_test_return_all(self, inputs, targets):
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, targets)
        return outputs, loss

    def predict(self, inputs):
        outputs = self.model(inputs)
        return outputs[-1]

    def predict_return_all(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def save(self, save_dir='workspace/data/model/', fn='robot'):
        fn_ = save_dir + self.name + '_' + fn + '.model'
        torch.save(self.model.state_dict(), fn_)
        self.save_params(save_dir, fn)

    def save_checkpoint(self, save_dir='workspace/data/model/', fn='robot'):
        fn = save_dir + self.name + '_' + fn + '.checkpoint'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, fn)

    def load_checkpoint(self, save_dir='workspace/data/model/', fn='robot'):
        fn = save_dir + self.name + '_' + fn + '.checkpoint'
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()

    def save_params(self, save_dir='workspace/data/model/', fn='robot'):
        fn = save_dir + self.name + '_' + fn + '.param'
        with open(fn, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, save_dir='workspace/data/model/', fn='robot'):
        fn = save_dir + self.name + '_' + fn + '.param'
        with open(fn, 'rb') as f:
            self.params = pickle.load(f)
            input_dim, hidden_dim, output_dim, l_dim1, l_dim2 = self.params
            del self.model
            self.model = RNN(input_dim, hidden_dim, output_dim, l_dim1, l_dim2)

    def load(self, save_dir='workspace/data/model/', fn='robot'):
        try:
            fn_ = save_dir + self.name + '_' + fn + '.model'
            self.load_params(save_dir, fn)

            self.model.load_state_dict(torch.load(fn_))
        except Exception as e:
            print(e)

    def write_decision(self):
        fn = 'workspace/data/decision'
        add_data(self.decision, fn)


class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, l_dim1=5, l_dim2=5):
        super(RNN, self).__init__()
        self.affine_dim = l_dim1
        self.affine_dim2 = l_dim2
        self.hidden_dim = hidden_dim
        self.affine2affine = nn.Linear(input_dim, self.affine_dim)
        self.affine2rep = nn.Linear(self.affine_dim, self.affine_dim2)
        self.lstm = nn.LSTM(self.affine_dim2, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)
        # self.activation = torch.nn.CELU()

    def forward(self, inputs):
        affine = torch.tanh(self.affine2affine(inputs))
        rep = torch.sigmoid(self.affine2rep(affine))
        # print(rep)
        
        lstm_out, _ = self.lstm(rep)
        out_space = self.hidden2out(lstm_out)
        out_scores = torch.sigmoid(out_space)
        # print(out_scores)
        # stop
        return out_scores


class decision_maker:
    def max_out(data):
        data = data.reshape(-1)
        buy = data[0].item()
        sell = data[1].item()
        if buy >= sell:
            return 'buy'
        else:
            return 'sell'

    def random_decide(self, data):
        self.decision = random.choice(self.actions)
        return self.decision


def add_data(data, fn):
    data = [str(datetime.now())] + [data]
    data = '/'.join(str(v) for v in data)
    cmd = 'echo "' + data + '" > ' + fn
    subprocess.call(cmd, shell=True)
