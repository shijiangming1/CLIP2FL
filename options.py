import argparse
import os
from Dataset.param_aug import ParamDiffAug


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    # CReFF
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--path_imagenet', type=str, default=os.path.join(path_dir, 'data/ImageNet'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--num_epochs_local_training', type=int, default=10)  #
    parser.add_argument('--batch_size_local_training', type=int, default=32)
    parser.add_argument('--match_epoch', type=int, default=100)
    parser.add_argument('--crt_epoch', type=int, default=300)
    parser.add_argument('--batch_real', type=int, default=32)
    parser.add_argument('--num_of_feature', type=int, default=100)
    parser.add_argument('--lr_feature', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    parser.add_argument('--method', type=str, default='DSA', help='DC/DSA')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    # CLIP2FL
    parser.add_argument('--scale', default=1, type=int)
    parser.add_argument('--teacher_arch', metavar='ARCH', default='resnet8')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10/cifar100/imagenet')
    parser.add_argument('--beta', default=0.998, type=float)
    parser.add_argument('--result_save', type=str, default='results')
    parser.add_argument('--T', default=3.0, type=float, help='Input the temperature: default(3.0)')
    parser.add_argument('--gpu', default=2, type=int,
                        help='GPU id to use.')
    parser.add_argument('--alpha', default=2, type=float, help='the hypeparameter for BKD2 loss')
    parser.add_argument('--center_alpha', default=0.0001, type=float, help='the hypeparameter for center alpha')
    parser.add_argument('--match_alpha', default=0.0001, type=float, help='the hypeparameter for match_gradient_alpha')
    parser.add_argument('--clip_alpha', default=1.0, type=float, help='the hypeparameter for server_clip_loss')

    parser.add_argument('--small_match_epoch', type=int, default=3)
    parser.add_argument('--small_crt_epoch', type=int, default=3)
    parser.add_argument('--ins_temp', type=float, default=0.1, help='temperature in instance-level supervision')
    parser.add_argument('--contrast_alpha', default=0.5, type=float, help='the hypeparameter for server_clip_loss')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # FedProx
    parser.add_argument('--mu', type=float, default=0.01)
    # FedAvgM
    parser.add_argument('--init_belta', type=float, default=0.97)
    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    return args
