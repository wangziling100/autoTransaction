import torch
from util import Norm, interest_margin
from robot import Robot
from train_robot import gen_train_data, transform2bi, eval, gen_test_data
import numpy as np


def test_model(name='EURUSD', verbose=False):
    input_dim = 16
    bob = Robot(name, input_dim=input_dim)
    bob.load()

    loss_con = []
    prof_con = []
    base_price = interest_margin(name)
    
    datum = gen_train_data(name)
    save_dir = 'workspace/data/model/'
    fn = 'abs_max_no_bias_18bit.model'

    for data in datum:
        price = data[:, :, 0]
        inputs = data[:, :, 1:-2]
        targets = data[:, :, -2:]
        inputs = Norm.abs_max_no_bias(inputs, save_dir=save_dir, fn=fn)


        inputs = torch.cat((inputs[:,:,:12], inputs[:,:,14:]), dim=2)

        targets = transform2bi(targets)
        #print(targets.shape)
        #print(torch.sum(targets, dim=1))

        outputs, loss = bob.mini_test_return_all(inputs, targets)
        loss_con.append(loss.item())

        prof = eval(price, outputs, base_price)
        prof_con.append(prof.item())

    mean_prof = np.mean(prof_con)
    print('-------train----------')
    print('mean loss:', np.mean(loss_con))
    print('mean profit:', mean_prof)

    prof_con = []
    loss_con = []

    datum = gen_test_data()

    for data in datum:
        price = data[:, :, 0]
        inputs = data[:, :, 1:-2]
        targets = data[:, :, -2:]
        inputs = Norm.abs_max_no_bias(inputs, save_dir=save_dir, fn=fn)


        inputs = torch.cat((inputs[:,:,:12], inputs[:,:,14:]), dim=2)
        targets = transform2bi(targets)

        outputs, loss = bob.mini_test_return_all(inputs, targets)
        loss_con.append(loss.item())

        prof = eval(price, outputs, base_price)
        prof_con.append(prof.item())
        print(outputs)

    mean_prof = np.mean(prof_con)
    print('-------test----------')
    print('mean loss:', np.mean(loss_con))
    print('mean profit:', mean_prof)


if __name__ == '__main__':
    test_model()
