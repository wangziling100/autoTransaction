from prepare_data import gen_online_transformed
from get_robot_data import prepare_robot_data
from train_robot import train_robot, save_norm_model

name = 'EURUSD'
gen_online_transformed()
prepare_robot_data(name, data_length=600)
save_norm_model(name)
# train_robot(name, epoch=20, feature_bit=feature_bit)
