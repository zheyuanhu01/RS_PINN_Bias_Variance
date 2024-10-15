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
parser.add_argument('--dim', type=int, default=3 + 1) # dimension of the problem.
parser.add_argument('--dataset', type=str, default="HJB")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--epochs', type=int, default=10000) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=128) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--save_loss', type=bool, default=True) # save the optimization trajectory?
parser.add_argument('--use_sch', type=int, default=1) # use scheduler?
parser.add_argument('--N_f', type=int, default=int(1000)) # num of residual points
parser.add_argument('--N_test', type=int, default=int(20000)) # num of test points
parser.add_argument('--method', type=int, default=2)
parser.add_argument('--t_std', type=float, default=1e-2)
parser.add_argument('--x_std', type=float, default=1e-2)
parser.add_argument('--train_sample_cnt', type=int, default=1024)
parser.add_argument('--eval_sample_cnt', type=int, default=128)
parser.add_argument('--t_end', type=float, default=1)

parser.add_argument('--trans', type=int, default=8000, help="transition epoch from biased to unbiased")
parser.add_argument('--dtype', type=str, default="float32", help="dtype")
args = parser.parse_args()
print(args)

# d = 11 N_f = 1K
device = torch.device(args.device)
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
np.random.seed(args.SEED)
assert args.dataset == "HJB"
   
def load_data_HJB(d):
    args.input_dim = d
    args.output_dim = 1
    def func_u(x):
        # X^2 / (4t + 1) + (d - 1) / 2 log (1 + 4t)
        #  
        return np.sum(x[:, :-1]**2, 1) / (1 + 4 * args.t_end - 4 * x[:, -1]) + (d - 1) / 2 * np.log(1 + 4 * args.t_end - 4 * x[:, -1])
    N_test = args.N_test
    t = np.random.rand(N_test, 1) * args.t_end
    x = np.random.randn(N_test, d - 1)
    x = np.concatenate([x, t], axis=-1)
    u = func_u(x)
    return x, u

x, u = load_data_HJB(d=args.dim)
print(x.shape, u.shape)

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
        #X, t = x[:, :-1], x[:, -1:]
        #return torch.mean(X**2, 1, True) * (-4 / (5 - 4 * t)) + torch.log(5 - 4 * t) / 2 / (args.t_end - t)
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
        #layers1 = [1] + [args.PINN_h] * (args.PINN_L - 1)
        #layers2 = [args.dim - 1] + [args.PINN_h] * (args.PINN_L - 1)
        self.u_net = MLP(layers).to(device)  
        #self.u_net = MLP2(layers1, layers2).to(device)  
        self.net_params_pinn = list(self.u_net.parameters())
        self.saved_loss = []
        self.saved_l2 = []

        self.x_std, self.t_std = args.x_std, args.t_std
        self.train_sample_cnt, self.eval_sample_cnt = args.train_sample_cnt, args.eval_sample_cnt

        self.grad_estimator = ImprovedSteinsWrapper(self.u_net, \
            self.t_std, self.x_std, self.train_sample_cnt, self.eval_sample_cnt)

    def Resample(self): # sample random points at the begining of each iteration
        tf = torch.rand(args.N_f, 1) * args.t_end
        xf = torch.randn(args.N_f, args.dim - 1)
        self.xf = torch.cat([xf, tf], dim=-1).to(device).requires_grad_()
        return

    def HJB(self):
        x, t = self.xf[:, :-1], self.xf[:, -1:]
        
        u, u_x, u_xx = self.grad_estimator.dE(self.xf)
        u_xx, u_x, u_t = u_xx[:, :-1], u_x[:, :-1], u_x[:, -1:]

        u_t = -self.dim_ * u + self.dim_ * u_t * (args.t_end - t)
        u_x = self.dim_ * (args.t_end-t) * u_x + 2 * x
        u_xx = self.dim_ * (args.t_end-t) * u_xx + 2

        residual_pred = u_t.reshape(-1) + torch.sum(u_xx, dim=1) - torch.sum(u_x**2, dim=1)
        return residual_pred

    def HJB_2(self):
        x, t = self.xf[:, :-1], self.xf[:, -1:]

        u, u_x, u_xx = self.grad_estimator.dE(self.xf)
        u_xx, u_x, u_t = u_xx[:, :-1], u_x[:, :-1], u_x[:, -1:]

        """
        d = args.dim - 1
        d * self.nn(x) * (1 - t) + torch.sum(X**2, 1, keepdim=True)
        """

        u_t = -self.dim_ * u + self.dim_ * u_t * (args.t_end - t)
        u_x = self.dim_ * (args.t_end-t) * u_x + 2 * x
        u_xx = self.dim_ * (args.t_end-t) * u_xx + 2

        _, u_x_2, _ = self.grad_estimator.dE(self.xf)
        u_x_2 = u_x_2[:, :-1]
        u_x_2 = self.dim_ * (args.t_end-t) * u_x_2 + 2 * x

        residual_pred = u_t.reshape(-1) + torch.sum(u_xx, dim=1) - torch.sum(u_x*u_x_2, dim=1)
        return residual_pred
    
    def num_params(self):
        num_pinn = 0
        for p in self.net_params_pinn:
            num_pinn += len(p.reshape(-1))
        return num_pinn

    def get_loss_pinn(self):
        f = self.HJB()
        mse_f = f.square().mean()
        return mse_f, mse_f
    
    def get_loss_pinn_unbiased(self):
        """f1 = self.HJB().detach()
        f2 = self.HJB()
        mse_f = 2 * (f1 * f2).mean()"""
        f1 = self.HJB()
        f2 = self.HJB()
        mse_f = (f1 * f2).mean()
        saved_loss = f1.square().mean()
        return mse_f, saved_loss
    
    def get_loss_pinn_unbiased_2(self):
        """f1 = self.HJB_2().detach()
        f2 = self.HJB_2()
        mse_f = 2 * (f1 * f2).mean()"""
        f1 = self.HJB_2()
        f2 = self.HJB_2()
        mse_f = (f1 * f2).mean()
        saved_loss = f1.square().mean()
        return mse_f, saved_loss
    
    def get_loss_pinn_unbiased_3(self):
        f = self.HJB_2()
        mse_f = f.square().mean()
        return mse_f
    
    def train_adam(self):
        optimizer = torch.optim.Adam(self.net_params_pinn, lr=self.adam_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
        # lr_lambda = lambda epoch: 1-epoch/args.epochs
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
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
            elif args.method == 2:
                loss, saved_loss = self.get_loss_pinn_unbiased_2()
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
        f = self.dim_ * self.grad_estimator(self.x, sample_cnt=args.eval_sample_cnt) * (args.t_end - self.x[:, -1:]) + torch.sum(self.x[:, :-1]**2, 1, True)
        #f = self.dim_ * self.grad_estimator(self.x) * (args.t_end - self.x[:, -1:]) + torch.sum(self.x[:, :-1]**2, 1, True)
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

