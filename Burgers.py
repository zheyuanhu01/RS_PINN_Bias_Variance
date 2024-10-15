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
parser.add_argument('--dim', type=int, default=2 + 1) # dimension of the problem.
parser.add_argument('--dataset', type=str, default="Burgers")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--epochs', type=int, default=10000) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=128) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--save_loss', type=bool, default=True) # save the optimization trajectory?
parser.add_argument('--use_sch', type=int, default=1) # use scheduler?
parser.add_argument('--N_f', type=int, default=int(100)) # num of residual points
parser.add_argument('--N_test', type=int, default=int(200)) # num of test points
parser.add_argument('--method', type=int, default=0)
parser.add_argument('--t_std', type=float, default=1e-2)
parser.add_argument('--x_std', type=float, default=1e-2)
parser.add_argument('--train_sample_cnt', type=int, default=1024)
parser.add_argument('--eval_sample_cnt', type=int, default=1024)
parser.add_argument('--t_end', type=float, default=1)

parser.add_argument('--trans', type=int, default=8000, help="transition epoch from biased to unbiased")
parser.add_argument('--dtype', type=str, default="float32", help="dtype")
args = parser.parse_args()
print(args)
if args.dtype == "float64": torch.set_default_dtype(torch.float64)
device = torch.device(args.device)
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
np.random.seed(args.SEED)
assert args.dataset == "Burgers"
   
def load_data_Burgers(d):
    args.input_dim = d
    args.output_dim = 1
    def u(x, t):
        return (np.sum(x, 1) + 1) / (1 + t)

    N_f = args.N_f # Number of collocation points
    N_test = args.N_test

    tf = np.random.rand(N_f, 1) * args.t_end
    xf = np.random.randn(N_f, d - 1) * np.sqrt(2 * args.t_end - 2 * tf) # ZHEYUAN: DO NOT CHANGE THE SAMPLES!
    xf = np.concatenate([xf, tf], axis=-1)
    ff = np.zeros(N_f)

    t = np.random.rand(N_test, 1) * args.t_end
    x = np.random.randn(N_test, d - 1) * np.sqrt(2 * args.t_end - 2 * t) # ZHEYUAN: DO NOT CHANGE THE SAMPLES!
    X = np.concatenate([x, t], axis=-1)
    u = u(X[:, :-1], X[:, -1])
    return X, u, xf, ff

x, u, xf, ff = load_data_Burgers(d=args.dim)
print(x.shape, u.shape, xf.shape, ff.shape)

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
        # t = x[:, -1:]; x = x[:, :-1]
        # return - (torch.sum(x, 1, True) + 1) / (1 + t)
        return self.nn(x)

class PINN:
    def __init__(self):
        self.epoch = args.epochs
        self.adam_lr = args.lr
        self.dim = args.dim
        self.dim_ = args.dim
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=False).to(device)
        self.u = torch.tensor(u, dtype=torch.float32, requires_grad=False).to(device)
        self.xf = torch.tensor(xf, dtype=torch.float32, requires_grad=True).to(device)
        self.ff = torch.tensor(ff, dtype=torch.float32, requires_grad=False).to(device).reshape(-1, 1)
        # Initalize Neural Networks
        layers = [args.input_dim] + [args.PINN_h] * (args.PINN_L - 1) + [args.output_dim]
        self.u_net = MLP(layers).to(device)  
        self.net_params_pinn = list(self.u_net.parameters())
        self.saved_loss = []
        self.saved_l2 = []

        self.x_std, self.t_std = args.x_std, args.t_std
        self.train_sample_cnt, self.eval_sample_cnt = args.train_sample_cnt, args.eval_sample_cnt

        self.grad_estimator = ImprovedSteinsWrapper(self.u_net, \
            self.t_std, self.x_std, self.train_sample_cnt, self.eval_sample_cnt)
        #self.grad_estimator = PinnWrapper(self.u_net, args.dim-1)

    def Resample(self): # sample random points at the begining of each iteration
        tf = torch.rand(args.N_f, 1) * args.t_end
        xf = torch.randn(args.N_f, args.dim - 1) * torch.sqrt(2 * args.t_end - 2 * tf) # ZHEYUAN: DO NOT CHANGE THE SAMPLES!
        self.xf = torch.cat([xf, tf], dim=-1).to(device).requires_grad_()
        return

    def Burgers(self): # ZHEYUAN: PLEASE IMPLEMENT THE DERIVATIVES BASED ON EQ 23
        x, t = self.xf[:, :-1], self.xf[:, -1:]
        u, u_x, u_xx = self.grad_estimator.dE(self.xf)
        u_xx, u_x, u_t = u_xx[:, :-1], u_x[:, :-1], u_x[:, -1:]

        u_t = u + u_t * t
        u_x = t * u_x + 1
        u_xx = t * u_xx
        u = u * t + torch.sum(x, 1, True) + 1
        residual_pred = u_t  + u * torch.mean(u_x, 1, True) - torch.sum(u_xx, 1, True)
        return residual_pred

    def Burgers2(self): # ZHEYUAN: PLEASE IMPLEMENT THE DERIVATIVES BASED ON EQ 23
        x, t = self.xf[:, :-1], self.xf[:, -1:]
        u, u_x, u_xx = self.grad_estimator.dE(self.xf)
        u_2 = self.grad_estimator(self.xf)
        u_xx, u_x, u_t = u_xx[:, :-1], u_x[:, :-1], u_x[:, -1:]

        u_t = u + u_t * t
        u_x = t * u_x + 1
        u_xx = t * u_xx
        u = u * t + torch.sum(x, 1, True) + 1
        u_2 = u_2 * t + torch.sum(x, 1, True) + 1
        residual_pred = u_t  + u_2 * torch.mean(u_x, 1, True) - torch.sum(u_xx, 1, True)
        return residual_pred

    def num_params(self):
        num_pinn = 0
        for p in self.net_params_pinn:
            num_pinn += len(p.reshape(-1))
        return num_pinn

    def get_loss_pinn(self):
        f = self.Burgers()
        mse_f = f.square().mean()
        return mse_f, mse_f
    
    def get_loss_pinn_unbiased(self):
        f1 = self.Burgers()
        f2 = self.Burgers()
        mse_f = (f1 * f2).mean()
        saved_loss = f1.square().mean()
        return mse_f, saved_loss
    
    def get_loss_pinn_unbiased_2(self):
        f1 = self.Burgers2()
        f2 = self.Burgers2()
        mse_f = (f1 * f2).mean()
        saved_loss = f1.square().mean()
        return mse_f, saved_loss
    
    def train_adam(self):
        optimizer = torch.optim.Adam(self.net_params_pinn, lr=self.adam_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
        L2, L1 = self.L2_pinn()
        print('Initialization: l2: %e, l1: %e'%(L2, L1))
        self.saved_loss.append(0)
        self.saved_l2.append([L2, L1])
    
        for n in tqdm(range(self.epoch)):
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
            self.Resample()
            if n % 100 == 0:
                L2, L1 = self.L2_pinn()
                print('epoch %d, loss: %e, l2: %e, l1: %e'%(n, current_loss, L2, L1))
            if args.save_loss:
                self.saved_loss.append(current_loss)
                if n % 100 != 0:
                    L2, L1 = self.L2_pinn()
                self.saved_l2.append([L2, L1])

    def predict_pinn(self): # ZHEYUAN: CHANGE HOW THE MODEL PREDICT
        # print("x.shape",torch.exp(torch.sum(self.x[:, :-1],1)).shape)
        # print("u.shape",(self.grad_estimator(self.x)*self.x[:, -1:]).shape)
        # #f = self.u
        # print("1",(self.grad_estimator(self.x)*self.x[:, -1:]).shape)
        # print("2",(1/(1 + torch.exp(torch.sum(self.x[:, :-1],1) / (2*args.mu)))).shape)
        # print("3",(1/(1 + torch.exp(torch.sum(self.x[:, :-1],1,True) / (2*args.mu)))).shape)
        
        f = self.grad_estimator(self.x) * self.x[:, -1:] + torch.sum(self.x[:, :-1],1,True) + 1
        # f = self.grad_estimator(self.x)
        #f = self.grad_estimator(self.x).reshape(-1) * self.x[:, -1] + torch.exp(torch.sum(self.x[:, :-1], 1) + args.t_end) / (1 + torch.exp(torch.sum(self.x[:, :-1], 1) + args.t_end))
        #f = torch.exp(torch.mean(self.x[:, :-1], 1) + args.t_end - self.x[:, -1]) / (1 + torch.exp(torch.mean(self.x[:, :-1], 1) + args.t_end - self.x[:, -1]))
        return f
    
    def L2_pinn(self):
        pred_u = self.predict_pinn()
        pred_u = self.u.squeeze() - pred_u.squeeze()
        # print(self.u.squeeze() -  pred_u.squeeze())
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
