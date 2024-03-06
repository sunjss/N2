import os
import nni
import argparse
n2_parser = argparse.ArgumentParser()
n2_parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
n2_parser.add_argument('--cuda_num', type=str, default='7', help='Which GPU')
n2_parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
n2_parser.add_argument('--testmode', type=str, default='test/', help='Export file fold choose')
n2_parser.add_argument('--seed', type=int, default=42, help='Random seed.')
n2_parser.add_argument('--resume_last', action='store_true', default=False)
n2_parser.add_argument('--resume_best', action='store_true', default=False)

n2_parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
n2_parser.add_argument('--patience', type=int, default=300, help='Patience')
n2_parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')  # 0.01
n2_parser.add_argument('--lr_step', type=float, default=1, help='Initial learning rate.')  # 0.99999
n2_parser.add_argument('--lr_lb', type=float, default=0.001, help='Initial learning rate.')
n2_parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')# 5e-4
n2_parser.add_argument('--warmup', type=int, default=1, help='Learning rate warmup')
n2_parser.add_argument('--dropout', type=float, default=0.4, help='Dropout Probability')

n2_parser.add_argument('--nbatch', type=int, default=1, help='Number of self defined batch')
n2_parser.add_argument('--fold_idx', type=int, default=1, help='fold idx for multi-split')
n2_parser.add_argument('--dataset', type=str, default='AmazonPhoto', help='Type of dataset')

n2_parser.add_argument('--nlayers', type=int, default=4, help='Number of conv layers in network')
n2_parser.add_argument('--d_model', type=int, default=128, help='Number of hidden units.')
n2_parser.add_argument('--q_dim', type=int, default=64, help='Dimension of neuronal state')
n2_parser.add_argument('--n_q', type=int, default=8, help='Number of submodel copies')
n2_parser.add_argument('--n_pnode', type=int, default=256, help='Number of submodel copies')
n2_parser.add_argument('--wo_selfloop', action='store_true', default=False)
n2_parser.add_argument('--ablation', action='store_true', default=False)
n2_parser.add_argument('--part', type=str, default='full', help='Type of dataset')
args = n2_parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num