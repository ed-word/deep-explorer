import argparse


def arg_parser(parser, hparams):
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def float2int(v):
        return int(float(v))

    parser.add_argument(
        "--learning_rate", type=float,
        default=hparams['learning_rate'],
        help="Learning rate")

    parser.add_argument(
        "--local_num_steps_per_tstep", type=float2int,
        default=hparams['local_num_steps_per_tstep'],
        help="Number of points in each local network time step")
    parser.add_argument(
        "--local_num_overlap_prevstep", type=float2int,
        default=hparams['local_num_overlap_prevstep'],
        help="Number of overlapping points between local time steps")
    parser.add_argument(
        "--local_num_out_per_tsetp", type=float2int,
        default=hparams['local_num_out_per_tsetp'],
        help="Output shape of each local network time step")

    parser.add_argument(
        "--global_num_steps_per_tstep", type=float2int,
        default=hparams['global_num_steps_per_tstep'],
        help="Number of points in each global network time step")
    parser.add_argument(
        "--global_num_overlap_prevstep", type=float2int,
        default=hparams['global_num_overlap_prevstep'],
        help="Number of overlapping points between global time steps")
    parser.add_argument(
        "--global_num_out_per_tsetp", type=float2int,
        default=hparams['global_num_out_per_tsetp'],
        help="Output shape of each global network time step")

    parser.add_argument(
        "--time_layer_out_num", type=float2int,
        default=hparams['time_layer_out_num'],
        help="Final output shape of local/global time network")
    parser.add_argument(
        "--num_fc_layers", type=float2int,
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
        "--batch_size", type=float2int,
        default=hparams['batch_size'],
        help="Batch size")
    parser.add_argument(
        "--num_epochs", type=float2int,
        default=hparams['num_epochs'],
        help="Number of epochs")
    parser.add_argument(
        "--optimizer", choices=['adam', 'sgd'],
        default=hparams['optimizer'],
        help="Optimization algorithm")
    parser.add_argument(
        "--time_activation", choices=['relu', 'sigmoid', 'tanh'],
        default=hparams['time_activation'],
        help="time activation")
    parser.add_argument(
        "--fc_activation", choices=['relu', 'sigmoid', 'tanh'],
        default=hparams['fc_activation'],
        help="FC activation")

    args = parser.parse_args()
    for arg in vars(args):
        hparams[arg] = getattr(args, arg)
    return hparams


if __name__ == '__main__':
    exit
    parser = argparse.ArgumentParser()
    hp = arg_parser(parser, {})
    print(hp)
