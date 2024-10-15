import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
from derivative_wrapper import *
#  /home/zheyuan/.conda/envs/zheyuan/bin/python

parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=4 + 1) # dimension of the problem.
parser.add_argument('--dataset', type=str, default="HJB_Rosenbrock")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--epochs', type=int, default=10000) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=128) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--save_loss', type=bool, default=True) # save the optimization trajectory?
parser.add_argument('--use_sch', type=int, default=1) # use scheduler?
parser.add_argument('--N_f', type=int, default=int(1000)) # num of residual points
parser.add_argument('--N_test', type=int, default=int(200)) # num of test points
parser.add_argument('--method', type=int, default=0)
parser.add_argument('--t_std', type=float, default=1e-2)
parser.add_argument('--x_std', type=float, default=1e-2)
parser.add_argument('--train_sample_cnt', type=int, default=1024)
parser.add_argument('--eval_sample_cnt', type=int, default=128)
parser.add_argument('--t_end', type=float, default=0.3)

parser.add_argument('--trans', type=int, default=8000, help="transition epoch from biased to unbiased")
parser.add_argument('--dtype', type=str, default="float32", help="dtype")
args = parser.parse_args()
print(args)

# d = 11 N_f = 1K
device = torch.device(args.device)
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
np.random.seed(args.SEED)
assert args.dataset == "HJB_Rosenbrock"

coeffs = (np.random.rand(2, args.dim - 2))# + 0.5
   
def load_data_HJB_Rosenbrock(d):
    args.input_dim = d
    args.output_dim = 1
    N_test = args.N_test
    
    MC = int(2e5)
    W = np.random.randn(MC, 1, args.dim - 1) # MC x NC x D
    def g(X):
        return np.sum(coeffs[None, 0:1, :] * (X[:, :, :-1] - X[:, :, 1:])**2 + \
                      coeffs[None, 1:2, :] * X[:, :, 1:]**2, -1) # np.log(1 + np.sum(X**2, -1)) - np.log(2)
    def u_exact(t, x): # NC x 1, NC x D
        T = args.t_end
        return -np.log(np.mean(np.exp(-g(x + np.sqrt(2.0*np.abs(T-t))*W)),0))
    X, Y = [], []
    for _ in tqdm(range(100)):
        t = np.random.rand(1, N_test, 1) * args.t_end
        x = np.random.randn(1, N_test, d - 1)
        x = np.concatenate([x, t], axis=-1)
        u = u_exact(x[:, :, -1:], x[:, :, :-1])
        x = x.squeeze()
        X.append(x); Y.append(u)
    X, Y = np.concatenate(X, 0), np.concatenate(Y, 0)
    return X, Y

x, u = load_data_HJB_Rosenbrock(d=args.dim)
print(x.shape, u.shape, coeffs.shape)
np.savetxt("HJB_Rosenbrock_Data/bias_proj_x" + str(args.dim-1) + ".txt", x)
np.savetxt("HJB_Rosenbrock_Data/bias_proj_u" + str(args.dim-1) + ".txt", u)
np.savetxt("HJB_Rosenbrock_Data/bias_proj_coeffs" + str(args.dim-1) + ".txt", coeffs)
