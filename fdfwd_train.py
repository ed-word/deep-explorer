import argparse
import tensorflow as tf

from feedforward_time.feedforward_time import FDFWD_TN
from feedforward_time.arg_handler import arg_parser
from dataset.dataset import get_batches


# MODEL NAME
MODEL_NAME = 'FDFWD_TN'

hparams = {}
# MODEL HYPERPARAMETERS
hparams['learning_rate'] = 0.000075
hparams['num_classes'] = 2

# INPUT SIZE HYPERPARAMETERS
hparams['local_num_steps_per_tstep'] = 32
hparams['local_num_overlap_prevstep'] = 8
hparams['local_num_out_per_tsetp'] = 8

hparams['global_num_steps_per_tstep'] = 128
hparams['global_num_overlap_prevstep'] = 16
hparams['global_num_out_per_tsetp'] = 16

hparams['time_layer_out_num'] = 8
hparams['num_fc_layers'] = 2
hparams['overlap'] = True
hparams['include_global'] = True

# TRAINING HYPERPARAMETRS
hparams['batch_size'] = 128
hparams['num_epochs'] = 5


if hparams['overlap']:
    if hparams['local_num_overlap_prevstep'] > hparams['local_num_steps_per_tstep'] / 2:
        hparams['local_num_overlap_prevstep'] = int(hparams['local_num_steps_per_tstep'] / 2)
    if hparams['global_num_overlap_prevstep'] > hparams['global_num_steps_per_tstep'] / 2:
        hparams['global_num_overlap_prevstep'] = int(hparams['global_num_steps_per_tstep'] / 2)

# SESSION CONF
cpu_session = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 0},
    allow_soft_placement=False,
    log_device_placement=False
)
gpu_session = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 1},
    allow_soft_placement=False,
    log_device_placement=False
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    hparams = arg_parser(parser, hparams)
    print(hparams)
    train_init_op, val_init_op, test_init_op, next_element = get_batches(hparams)
    fdtn_network = FDFWD_TN(next_element, hyperparam=hparams)
    fdtn_network.create_graph()
    fdtn_network.create_sess()
    fdtn_network.train(train_init_op, val_init_op, print_stuff=False)
    fdtn_network.test(val_init_op, labels=True)
    # fdtn_network.test(test_init_op, True)
