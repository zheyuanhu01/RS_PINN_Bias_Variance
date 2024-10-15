import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
from derivative_wrapper import *
from datetime import datetime
from torch.func import hessian
#  /home/zheyuan/.conda/envs/zheyuan/bin/python

parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=4) # dimension of the problem.
parser.add_argument('--dataset', type=str, default="TwoBodyv2_Sin")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--epochs', type=int, default=10000) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=128) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--save_loss', type=bool, default=True) # save the optimization trajectory?
parser.add_argument('--use_sch', type=int, default=1) # use scheduler?
parser.add_argument('--N_f', type=int, default=int(100)) # num of residual points
parser.add_argument('--N_test', type=int, default=int(20000)) # num of test points
parser.add_argument('--method', type=int, default=0)
parser.add_argument('--t_std', type=float, default=1e-3)
parser.add_argument('--x_std', type=float, default=1e-3)
parser.add_argument('--train_sample_cnt', type=int, default=128)
parser.add_argument('--eval_sample_cnt', type=int, default=128)
parser.add_argument('--x_radius', type=float, default=1)
parser.add_argument('--trans', type=int, default=8500, help="transition epoch from biased to unbiased")
args = parser.parse_args()

if args.dim == 1000: args.trans = 8500
if args.dim == 10: args.trans = 2000
if args.dim == 100: args.trans = 4000

print(args)

device = torch.device(args.device)
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
np.random.seed(args.SEED)
assert args.dataset == "TwoBodyv2_Sin"

c = np.random.randn(1, args.dim - 1)
const_2 = 1
def load_data_TwoBodyv2_Sin(d):
    args.input_dim = d
    args.output_dim = 1
    def func_u(x):
        temp =  args.x_radius**2 - np.sum(x**2, 1)
        temp2 = c * np.sin(x[:, :-1] + const_2 * np.cos(x[:, 1:]) + x[:, 1:] * np.cos(x[:, :-1]))
        temp2 = np.sum(temp2, 1)
        return temp * temp2

    N_test = args.N_test

    x = np.random.randn(N_test, d)
    r = np.random.rand(N_test, 1) * args.x_radius
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * r
    u = func_u(x)
    return x, u

x, u = load_data_TwoBodyv2_Sin(d=args.dim)
print(x.shape, u.shape)
print(u.mean(), u.std())

class MLP(nn.Module):
    def __init__(self, layers:list):
        super(MLP, self).__init__()
        models = []
        for i in range(len(layers)-1):
            models.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                models.append(nn.Tanh())
        self.nn = nn.Sequential(*models)
    def forward(self, x):
        #c = torch.tanh(4 * torch.arange(args.dim - 1) / (args.dim - 1) - 2).reshape(1, args.dim - 1).to(device)
        #return (1 - torch.sum(x**2, 1, True)) * \
        #    torch.sum(c * torch.sin(x[:, :-1] + 3 * torch.cos(x[:, 1:]) + x[:, 1:] * torch.cos(x[:, :-1])), 1, True)
        # return torch.sum(c * torch.sin(x[:, :-1] + 3 * torch.cos(x[:, 1:]) + x[:, 1:] * torch.cos(x[:, :-1])), 1, True)
        return self.nn(x)

class PINN:
    def __init__(self):
        self.epoch = args.epochs
        self.adam_lr = args.lr
        self.dim = args.dim
        self.dim_ = args.dim - 1
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=False).to(device)
        self.u = torch.tensor(u, dtype=torch.float32, requires_grad=False).to(device).reshape(-1, 1)
        # Initalize Neural Networks
        layers = [args.input_dim] + [args.PINN_h] * (args.PINN_L - 1) + [args.output_dim]
        self.u_net = MLP(layers).to(device)
        self.net_params_pinn = list(self.u_net.parameters())
        self.saved_loss = []
        self.saved_l2 = []

        self.x_std, self.t_std = args.x_std, args.t_std
        self.train_sample_cnt, self.eval_sample_cnt = args.train_sample_cnt, args.eval_sample_cnt

        self.grad_estimator = ImprovedSteinsWrapper(
            self.u_net,
            self.t_std, self.x_std, 
            self.train_sample_cnt, self.eval_sample_cnt
        )

    def Resample(self): # sample random points at the begining of each iteration
        N_f = args.N_f # Number of collocation points

        xf = np.random.randn(N_f, args.dim)
        rf = np.random.rand(N_f, 1) * args.x_radius
        xf = xf / np.linalg.norm(xf, axis=1, keepdims=True) * rf
        x = xf

        u1 = args.x_radius**2 - np.sum(x**2, 1, keepdims=True)
        du1_dx = -2 * x
        d2u1_dx2 = -2

        x1, x2 = x[:, :-1], x[:, 1:]
        coeffs = c
        u2 = coeffs * np.sin(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1))
        u2 = np.sum(u2, 1, keepdims=True)
        du2_dx_part1 = coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * \
                (1 - x2 * np.sin(x1))
        du2_dx_part2 = coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * \
                (-const_2 * np.sin(x2) + np.cos(x1))
        du2_dx = np.zeros((N_f, args.dim))
        du2_dx[:, :-1] += du2_dx_part1
        du2_dx[:, 1:] += du2_dx_part2
        d2u2_dx2_part1 = -coeffs * np.sin(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (1 - x2 * np.sin(x1))**2 + \
                coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (- x2 * np.cos(x1))
        d2u2_dx2_part2 = -coeffs * np.sin(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (-const_2 * np.sin(x2) + np.cos(x1))**2 + \
                coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (-const_2 * np.cos(x2))
        d2u2_dx2 = np.zeros((N_f, args.dim))
        d2u2_dx2[:, :-1] += d2u2_dx2_part1
        d2u2_dx2[:, 1:] += d2u2_dx2_part2
        ff = u1 * d2u2_dx2 + 2 * du1_dx * du2_dx + u2 * d2u1_dx2
        ff = np.sum(ff, 1)
        u = (u1 * u2).reshape(-1)
        ff = ff + np.sin(u)
        ff = ff.reshape(-1, 1)

        self.xf = torch.tensor(xf, dtype=torch.float32, requires_grad=True).to(device)
        self.ff = torch.tensor(ff, dtype=torch.float32, requires_grad=True).to(device)
        return

    def TwoBodyv2_Sin(self):
        x = self.xf
        u, u_x, u_xx = self.grad_estimator.dE(x)
        u_xx = -2 * u  - 4 * x * u_x + (1 - torch.sum(x**2, 1, True)) * u_xx
        u = (1 - torch.sum(x**2, 1, True)) * u
        residual_pred = torch.sum(u_xx, 1, True) + torch.sin(u) - self.ff
        return residual_pred

    def num_params(self):
        num_pinn = 0
        for p in self.net_params_pinn:
            num_pinn += len(p.reshape(-1))
        return num_pinn

    def get_loss_pinn(self):
        f = self.TwoBodyv2_Sin()
        mse_f = f.square().mean()
        return mse_f, mse_f
    
    def get_loss_pinn_unbiased(self):
        f1 = self.TwoBodyv2_Sin()
        f2 = self.TwoBodyv2_Sin()
        mse_f = (f1 * f2).mean()
        saved_loss = f1.square().mean()
        return mse_f, saved_loss

    def train_adam(self):
        optimizer = torch.optim.Adam(self.net_params_pinn, lr=self.adam_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
        lr_lambda = lambda epoch: 1-epoch/args.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        L2, L1 = self.L2_pinn()
        print('Initialization: l2: %e, l1: %e'%(L2, L1))
        self.saved_loss.append(0)
        self.saved_l2.append([L2, L1])
    
        for n in tqdm(range(self.epoch)):
            self.Resample()
            if args.method == 0:
                loss, saved_loss = self.get_loss_pinn() 
            elif args.method == 1:
                loss, saved_loss = self.get_loss_pinn_unbiased()
            elif args.method == 99:
                if n <= args.trans:
                    loss, saved_loss = self.get_loss_pinn() 
                elif n > args.trans:
                    loss, saved_loss = self.get_loss_pinn_unbiased() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.use_sch:
                scheduler.step()
            current_loss = saved_loss.item()
            if n % 100 == 0:
                L2, L1 = self.L2_pinn()
                print('epoch %d, loss: %e, l2: %e, l1: %e'%(n, current_loss, L2, L1))
            if args.save_loss:
                self.saved_loss.append(current_loss)
                if n % 100 != 0:
                    L2, L1 = self.L2_pinn()
                self.saved_l2.append([L2, L1])

    def predict_pinn(self):
        f = self.grad_estimator(self.x) * (1 - torch.sum(self.x**2, 1, True))
        return f
    
    def L2_pinn(self):
        pred_u = self.predict_pinn()
        pred_u = self.u - pred_u
        L2, L1 = torch.norm(pred_u) / torch.norm(self.u), \
            torch.norm(pred_u, p=1) / torch.norm(self.u, 1)
        L2, L1 = L2.item(), L1.item()
        return L2, L1

model = PINN()
print("Num params:", model.num_params())
model.train_adam()

if args.save_loss:
    model.saved_loss = np.asarray(model.saved_loss)
    model.saved_l2 = np.asarray(model.saved_l2)
    info_dict = {"loss": model.saved_loss, "L2": model.saved_l2[:, 0], "L1": model.saved_l2[:, 1]}
    df = pd.DataFrame(data=info_dict, index=None)
    filename = "saved_loss_l2/"+args.dataset+"_dim="+str(args.dim)+"_method="+str(args.method)+"_SEED="+str(args.SEED)+".xlsx"
    if args.method == 99:
         filename = "saved_loss_l2/" + args.dataset + "_dim=" + str(args.dim) + "_method=99" + \
             "_SEED=" + str(args.SEED) + "_trans=" + str(args.trans) + ".xlsx"
    df.to_excel(filename, index=False)

