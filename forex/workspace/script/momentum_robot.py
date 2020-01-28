import numpy as np
from torch import nn
import torch
from random import random


class MRobot:
    """
    this model predicts the (positive/negative) change in next 60 sec
    """
    def __init__(self, name='EURUSD', base=0.00006, input_dim=7, layers=[7, 5, 2], zoom=10000, target_zoom=5000):
        self.name = name
        self.long_len = 300
        self.n_inputs = 60
        self.short_len = self.n_inputs
        self.short_data = []
        self.long_data = []
        self.direction = None
        self.positive_model = Model(input_dim, layers)
        self.negative_model = Model(input_dim, layers)
        self.loss_f = nn.MSELoss()
        self.pos_optimizer = torch.optim.Adam(self.positive_model.parameters())
        self.neg_optimizer = torch.optim.Adam(self.negative_model.parameters())
        self.inputs = None
        self.input_con = torch.Tensor().reshape(0, 1, input_dim)
        self.zoom = zoom
        self.base = base
        self.min_loss = 10**5
        self.loss = 10**5
        self.candi = []
        self.n_candi = 10
        self.stop = False
        self.target_zoom = target_zoom
        self.output_con = torch.Tensor()

    def receive(self, data):
        if len(self.short_data)>=10 and data == np.std(self.short_data[-10:])==0:
            self.stop = True
            self.change = None
            print('receive the same data')
            return

        self.stop = False
        self.short_data.append(data)
        self.long_data.append(data)
        self.short_data = self.short_data[-self.short_len:]
        self.long_data = self.long_data[-self.long_len:]
        if len(self.short_data) < self.n_inputs:
            self.change = None
            return
        self.change = [max(self.short_data)-self.short_data[0], min(self.short_data)-self.short_data[0]]
        self.calc_inputs()
        # self.change = np.array(self.change)

    def cal_momentum(self):
        # curr momentum
        curr_mom = self.short_data[-1]-self.short_data[-2]

        # avg momentum isn 1 min
        a = np.array(self.short_data[:-1])
        b = np.array(self.short_data[1:])

        short_moms = b-a
        avg_short_mom = np.mean(short_moms)

        a = np.array(self.long_data[:-1])
        b = np.array(self.long_data[1:])

        long_moms = b-a
        avg_long_mom = np.mean(long_moms)
        return curr_mom, avg_short_mom, avg_long_mom

    def calc_mid(self):
        short_mid = np.mean(self.short_data)
        long_mid = np.mean(self.long_data)

        # calc distance to mid point
        dist_short_mom = self.short_data[-1]-short_mid
        dist_long_mom = self.long_data[-1] - long_mid
        return dist_short_mom, dist_long_mom

    def calc_inputs(self):
        """
        cm:     current momentum
        asm:    average momentum of short data
        alm:    average momentum of long data
        dsm:    distance from current price to average in short
        dlm:    distance from current price to average in long
        csd:    distance from current momentum to average in short
        cld:    distance from current momentum to average in long 
        """
        cm, asm, alm = self.cal_momentum()
        dsm, dlm = self.calc_mid()
        csd = cm-asm
        cld = cm-alm

        self.inputs = torch.Tensor([cm, alm, 
            asm, dsm, dlm, csd, cld]).reshape(1, 1, -1)*self.zoom
        self.input_con = torch.cat((self.input_con, 
            self.inputs), dim=0)
        if len(self.input_con) > self.n_inputs:
            self.input_con = self.input_con[-self.n_inputs:, :, :]

    def online_train(self, verbose=False):

        if len(self.input_con) < self.n_inputs:
            return
        if self.change is None:
            return
        if self.change[0] == 0 and self.change[1] == 0:
            return
        if abs(self.change[0]) > abs(self.change[1]):
            is_pos = True
            is_neg = False
        else:
            is_pos = False
            is_neg = True
        inputs = self.input_con[-self.n_inputs, :, :]
        inputs = inputs.reshape(1, 1, -1)
        target = torch.Tensor(self.change).reshape(1, 1, 2)
        target = target*self.target_zoom
       
        self.positive_model.zero_grad()
        self.negative_model.zero_grad()
        prob = 0.1
        if random() < prob:
            is_pos = True
            is_neg = True
        if is_pos:
            output = self.positive_model(inputs)
            loss = self.loss_f(output, target)
            loss.backward()
            self.pos_optimizer.step()
        if is_neg:
            output = self.negative_model(inputs)
            loss = self.loss_f(output, target)
            loss.backward()
            self.neg_optimizer.step()

        if verbose:
            print('loss:', loss, )
            print('target:', target)
            print('output', output)

    def predict(self):
        if self.stop or self.inputs is None:
            return
        if self.change[0] == 0 and self.change[1] == 0:
            return
        with torch.no_grad():
            d_pos = self.positive_model(self.inputs)
            d_neg = self.negative_model(self.inputs)

            max_pos = d_pos[0, 0, 0]
            min_pos = d_pos[0, 0, 1]
            max_neg = d_neg[0, 0, 0]
            min_neg = d_neg[0, 0, 1]

            diff_pos = abs(abs(max_pos)-abs(min_pos))
            diff_neg = abs(abs(max_neg)-abs(min_neg))

            if diff_pos > diff_neg:
                self.direction = d_pos
            else:
                self.direction = d_neg
            
            self.output_con = torch.cat((self.output_con, self.direction), dim=0)
            if len(self.output_con) > self.n_inputs:
                self.output_con = self.output_con[-self.n_inputs:]
            else:
                return
            
            curr_max = self.output_con[-20:, :, 0].mean()
            curr_min = self.output_con[-20:, :, 1].mean()
            
            # formal
            r = 1.2
            print(abs(curr_max), abs(curr_min)) 

            if abs(curr_max)-abs(curr_min) > r*self.base*self.target_zoom:
                return 'buy'
            elif abs(curr_max)-abs(curr_min) < r*-self.base*self.target_zoom:
                return 'sell'
            else:
                return 'wait'
      
    def save_checkpoint(self, save_dir='workspace/data/model/', fn='mrobot'):
        fn = save_dir + self.name + '_' + fn + '.checkpoint'
        torch.save({
            'pos_model_state': self.positive_model.state_dict(),
            'pos_optimizer_state': self.pos_optimizer.state_dict(),
            'neg_model_state': self.negative_model.state_dict(),
            'neg_optimizer_state': self.neg_optimizer.state_dict(),
            }, fn)

    def load_checkpoint(self, save_dir='workspace/data/model/', fn='mrobot'):
        fn = save_dir + self.name + '_' + fn + '.checkpoint'
        checkpoint = torch.load(fn)
        self.positive_model.load_state_dict(checkpoint['pos_model_state'])
        self.negative_model.load_state_dict(checkpoint['neg_model_state'])

        self.pos_optimizer.load_state_dict(checkpoint['pos_optimizer_state'])
        self.neg_optimizer.load_state_dict(checkpoint['neg_optimizer_state'])


class Model(nn.Module):
    def __init__(self, input_dim, layers_dim=[5, 5, 2]):
        super(Model, self).__init__()
        self.layers_dim = layers_dim
        self.layers = []
        last_dim = input_dim
        self.params = nn.ParameterList([])
        for dim in layers_dim:
            layer = nn.Linear(last_dim, dim)
            last_dim = dim
            self.layers.append(layer)
            self.params.append(layer.weight)
            self.params.append(layer.bias)

    def forward(self, inputs):
        out = inputs
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out




