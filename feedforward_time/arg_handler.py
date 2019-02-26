import argparse

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
hparams['num_epochs'] = 625


def arg_parser():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning_rate", type=float,
        default=hparams['learning_rate'],
        help="Learning rate")

    parser.add_argument(
        "--local_num_steps_per_tstep", type=int,
        default=hparams['local_num_steps_per_tstep'],
        help="Number of points in each local network time step")
    parser.add_argument(
        "--local_num_overlap_prevstep", type=int,
        default=hparams['local_num_overlap_prevstep'],
        help="Number of overlapping points between local time steps")
    parser.add_argument(
        "--local_num_out_per_tsetp", type=int,
        default=hparams['local_num_out_per_tsetp'],
        help="Output shape of each local network time step")

    parser.add_argument(
        "--global_num_steps_per_tstep", type=int,
        default=hparams['global_num_steps_per_tstep'],
        help="Number of points in each global network time step")
    parser.add_argument(
        "--global_num_overlap_prevstep", type=int,
        default=hparams['global_num_overlap_prevstep'],
        help="Number of overlapping points between global time steps")
    parser.add_argument(
        "--global_num_out_per_tsetp", type=int,
        default=hparams['global_num_out_per_tsetp'],
        help="Output shape of each global network time step")

    parser.add_argument(
        "--time_layer_out_num", type=int,
        default=hparams['time_layer_out_num'],
        help="Final output shape of local/global time network")
    parser.add_argument(
        "--num_fc_layers", type=int,
        default=hparams['num_fc_layers'],
        help="Number of fully-connected layers after time network")

    parser.add_argument(
        "--overlap", type=str2bool,
        default=hparams['overlap'],
        help="Overlap")
    parser.add_argument(
        "--include_global", type=str2bool,
        default=hparams['include_global'],
        help="Should global series be used in training")

    parser.add_argument(
        "--batch_size", type=int,
        default=hparams['batch_size'],
        help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int,
        default=hparams['num_epochs'],
        help="Number of epochs")

    args = parser.parse_args()
    for arg in vars(args):
        hparams[arg] = getattr(args, arg)
    return hparams


if __name__ == '__main__':
    hp = arg_parser()
    print(hp)
