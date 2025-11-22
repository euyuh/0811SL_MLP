# ============================================================================
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob 
from pandas import DataFrame
import random
import numpy as np
import os
import pandas as pd
from quan_utils_v4 import register_activation_ema_hooks
# from quantization_0805 import Float4Quantizer
# from modulation_v2 import Modulator
from quan_utils_v4 import (
    ActQuantization, DifferentialRound, update_activation_ema,
    get_activation_range, update_bn_ema, FoldLinear, quan_scheme,
    quantize_tensor, quantize_activation, quantize_linear_layer_from_tensors,
    inference_quantized, export_quantized_model, print_BN_EMA_status, FakeQuantLinear, ACT_EMA, BN_EMA
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================  
program = "SL_MLP_quantized"
print("starting to train\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

num_users = 1
epochs =2
frac = 1
lr = 0.001

#=====================================================================================================
#                           Model Definitions (使用 FakeQuantLinear)
#=====================================================================================================
class MLPClient(nn.Module):
    def __init__(self):
        super(MLPClient, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            FakeQuantLinear(64*64, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.4),
            FakeQuantLinear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.35),
            FakeQuantLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            FakeQuantLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            FakeQuantLinear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
    
    def forward(self, x, epoch=None, batch_idx=None, user_idx=None):  # 修改函数签名
        activation = self.layers(x)
        return activation

class MLPServer(nn.Module):
    def __init__(self):
        super(MLPServer, self).__init__()
        self.layers = nn.Sequential(
            FakeQuantLinear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            FakeQuantLinear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.35),
            FakeQuantLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            FakeQuantLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            FakeQuantLinear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

net_glob_client = MLPClient()
if torch.cuda.device_count() > 1:
    net_glob_client = nn.DataParallel(net_glob_client)
net_glob_client.to(device)

net_glob_server = MLPServer() 
if torch.cuda.device_count() > 1:
    net_glob_server = nn.DataParallel(net_glob_server)
net_glob_server.to(device)


# 如果使用了 DataParallel，请注册 module；否则直接注册模型


# 客户端模型注册
if isinstance(net_glob_client, torch.nn.DataParallel):
    register_activation_ema_hooks(net_glob_client.module, prefix="client_layers")
else:
    register_activation_ema_hooks(net_glob_client, prefix="client_layers")

# 服务器模型注册
if isinstance(net_glob_server, torch.nn.DataParallel):
    register_activation_ema_hooks(net_glob_server.module, prefix="server_layers")
else:
    register_activation_ema_hooks(net_glob_server, prefix="server_layers")


#===================================================================================
# Training Utilities
criterion = nn.CrossEntropyLoss()
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []
criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    return 100. * correct.float() / preds.shape[0]

acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

idx_collect = []
l_epoch_check = False
fed_check = False

# Server-side training function
def train_server(noisy_symbols_act, act_shape, 
                 y, l_epoch_count, l_epoch, idx, len_batch, batch_idx):
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user
    
    # 直接接收客户端传来的float32激活值（或伪量化后反量化的float）
    fx_client = noisy_symbols_act.to(device)
    y = y.to(device)
     
    # 需要跟踪激活值的梯度
    fx_client.requires_grad_(True)
    
    net_glob_server.train()
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr=lr)
    optimizer_server.zero_grad()
    
    # 前向传播 (服务器端) - 此处服务器端层的 FakeQuantLinear 会对其权重进行伪量化
    fx_server = net_glob_server(fx_client)
    
    # === 更新 EMA ===
    update_bn_ema(net_glob_server, prefix="server_layers")                 # 更新 BN 层统计
    # update_activation_ema(f"server_{idx}_out", fx_server.detach())  # 更新激活范围统计 注释掉这行，因为在hook中已经统计

    # 计算损失和准确率
    loss = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)
    
    # 反向传播
    loss.backward()
    
    # 获取客户端激活值的梯度（服务器端对输入的梯度）
    dfx_client = fx_client.grad.clone().detach()
    
    # 更新服务器模型
    optimizer_server.step()
    
    # 训练状态跟踪
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    count1 += 1
    
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        batch_acc_train, batch_loss_train = [], []
        count1 = 0
        
        prRed(f'Client{idx} Train => Local Epoch: {l_epoch_count} \tAcc: {acc_avg_train:.3f} \tLoss: {loss_avg_train:.4f}')
        
        if l_epoch_count == l_epoch-1:
            l_epoch_check = True
            loss_train_collect_user.append(loss_avg_train)
            acc_train_collect_user.append(acc_avg_train)
            if idx not in idx_collect:
                idx_collect.append(idx)
    
    # 联邦学习轮次检查
    if len(idx_collect) == num_users and fed_check:
        fed_check = False
        acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
        loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
        loss_train_collect.append(loss_avg_all_user_train)
        acc_train_collect.append(acc_avg_all_user_train)
        acc_train_collect_user, loss_train_collect_user = [], []
    
    # 返回梯度张量（float32）
    return dfx_client.detach()


# Server-side evaluation function
def evaluate_server(noisy_symbols_act, act_shape, 
                   y, idx, len_batch, ell, mode='val'):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, num_users, idx_collect
    global loss_test_collect_user, acc_test_collect_user
    global fed_check
    
    net_glob_server.eval()
  
    with torch.no_grad():
        fx_client = noisy_symbols_act.to(device)
        y = y.to(device) 
        
        fx_server = net_glob_server(fx_client)
        update_bn_ema(net_glob_server, prefix = "server_layers")
        # === 服务器端伪量化 (可选) ===
        # update_activation_ema(f"server_eval_out", fx_server.detach())
        last_server_layer = list(net_glob_server.layers.named_modules())[-1][0]
        act_key_server = f"server_layers/{last_server_layer}_out"
        act_max, act_min = get_activation_range(act_key_server, -6.0, 6.0)
        fx_server = ActQuantization(fx_server, FloatMax=act_max, FloatMin=act_min, num_bits=quan_scheme.act_bits)
        
        loss = criterion(fx_server, y)
        acc = calculate_accuracy(fx_server, y)
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            if mode == 'test':
                prGreen('Client{} [TEST] => \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            else:
                prGreen('Client{} [VAL] => \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            
            acc_test_collect_user.append(acc_avg_test)
            loss_test_collect_user.append(loss_avg_test)
            
            if idx not in idx_collect:
                idx_collect.append(idx)
            
            if len(idx_collect) == num_users:
                if acc_test_collect_user:
                    acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                    loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
                else:
                    acc_avg_all_user = 0.0
                    loss_avg_all_user = 0.0
                
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                idx_collect = []
                if mode == 'test':
                    print("\n====================== FINAL TEST ========================")
                    print(' Test Round: \tAvg Accuracy {:.3f} | Avg Loss {:.3f}'.format(
                        acc_avg_all_user, loss_avg_all_user))
                    print("==========================================================\n")
         
    return


#====================================================================================================
#                                  Client Side Implementation
#====================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]

class Client(object):
    def __init__(self, net_client_model, idx, lr, device, 
                 dataset_train=None, dataset_val=None, dataset_test=None,
                 idxs_train=None, idxs_val=None, idxs_test=None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs_train), 
                                   batch_size=64, shuffle=True)
        self.ldr_val = DataLoader(DatasetSplit(dataset_val, idxs_val),
                                 batch_size=64, shuffle=False)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test),
                                  batch_size=64, shuffle=False)

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                
                # 前向传播得到激活（客户端）
                fx = net(images, epoch=iter, batch_idx=batch_idx, user_idx=self.idx)
                
                # 更新客户端的BN_EMA
                update_bn_ema(net, prefix="client_layers") 

                last_client_layer = list(net_glob_client.layers.named_modules())[-1][0]
                act_key_client = f"client_layers/{last_client_layer}_out"
                act_max, act_min = get_activation_range(act_key_client, -6.0, 6.0)
                # 客户端进行伪量化（训练中）
                fx_q = ActQuantization(fx, FloatMax=act_max, FloatMin=act_min, num_bits=quan_scheme.act_bits)

                # 将激活传输/发送给服务器（此处用 fx_q.detach() 表示发送数值）
                dfx = train_server(
                    fx_q.detach(), fx_q.shape, labels,
                    iter, self.local_ep, self.idx, len_batch, batch_idx
                )

                # 客户端接收来自服务器的梯度（dfx），并做本地反向步
                fx.backward(dfx.to(self.device))
                optimizer_client.step()
        
        return net.state_dict()

    def evaluate(self, net, ell, mode='val'):
        net.eval()
        loader = self.ldr_test if mode == 'test' else self.ldr_val
        with torch.no_grad():
            len_batch = len(loader)
            for batch_idx, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                fx = net(images)
                update_bn_ema(net, prefix="client_layers")#加了个bnupdate（evaluate要update吗）
                # update_activation_ema(f"client_{self.idx}_out", fx.detach())
                last_client_layer = list(net_glob_client.layers.named_modules())[-1][0]
                act_key_client = f"client_layers/{last_client_layer}_out"
                act_max, act_min = get_activation_range(act_key_client, -6.0, 6.0)
                fx_q = ActQuantization(fx, FloatMax=act_max, FloatMin=act_min, num_bits=quan_scheme.act_bits)

                act_shape = fx_q.shape
                evaluate_server(fx_q.detach(), act_shape,
                                labels, self.idx, len_batch, ell, mode)


#====================================================================================================
#                             Data Loading Section
#====================================================================================================
class SkinData(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.df.iloc[index]['path']
        X = Image.open(img_path).resize((64,64)).convert('L')
        y = torch.tensor(int(self.df.iloc[index]['target']))
        if self.transform:
            X = self.transform(X)
        return X, y

# 数据集路径（请按实际路径修改）
train_csv = '/root/autodl-fs/0811SL_MLP/mnist_train_1.csv'
train_img_folder = '/root/autodl-fs/0811SL_MLP/train_images'
df_train_val = pd.read_csv(train_csv)
df_train_val['path'] = [os.path.join(train_img_folder, f"{i}.png") for i in range(len(df_train_val))]
df_train_val['target'] = df_train_val['label']

test_csv = '/root/autodl-fs/0811SL_MLP/mnist_test.csv'
test_img_folder = '/root/autodl-fs/0811SL_MLP/test_images' 
df_test = pd.read_csv(test_csv)
df_test['path'] = [os.path.join(test_img_folder, f"{i}.png") for i in range(len(df_test))]
df_test['target'] = df_test['label']

train_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

test_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

train_df, val_df = train_test_split(df_train_val, test_size=0.2, stratify=df_train_val['target'], random_state=SEED)
dataset_train = SkinData(train_df, train_transforms)
dataset_val = SkinData(val_df, test_transforms)
dataset_test = SkinData(df_test, test_transforms)  # 独立测试集

def dataset_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users   

dict_users_train = dataset_iid(dataset_train, num_users)
dict_users_val = dataset_iid(dataset_val, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)

# ===============================================
#               Main Training Loop
# ===============================================
best_val_loss = float('inf')
best_epoch = -1
best_client_state = None
best_server_state = None

best_model_path = '/root/autodl-fs/0811SL_MLP/best model'
os.makedirs(best_model_path, exist_ok=True)

for epoch in range(epochs):     
    idxs_users = np.random.choice(range(num_users), max(int(frac*num_users),1), replace=False)
    print(f"开始训练：第{epoch+1}轮")
    for idx in idxs_users:
        client = Client(net_glob_client, idx, lr, device,
                        dataset_train=dataset_train,
                        dataset_val=dataset_val,
                        dataset_test=dataset_test,
                        idxs_train=dict_users_train[idx],
                        idxs_val=dict_users_val[idx],
                        idxs_test=dict_users_test[idx])
        
        w_client = client.train(copy.deepcopy(net_glob_client).to(device))
        net_glob_client.load_state_dict(w_client)
        
        client.evaluate(copy.deepcopy(net_glob_client).to(device), epoch, mode='val')
    
    # print(f"\n===== Epoch {epoch+1} BN层 (client)状态 =====")调试
    # for name, m in net_glob_client.named_modules():
    #     if isinstance(m, nn.BatchNorm1d):
    #         mean_str = ", ".join([f"{x}" for x in m.running_mean[:3]])
    #         var_str  = ", ".join([f"{x}" for x in m.running_var[:3]])
    #         print(f"[{name}] mean=[{mean_str}], var=[{var_str}]")
    # print("=====================================\n")
    # print(f"\n===== Epoch {epoch+1} BN层 (server)状态 =====")
    # for name, m in net_glob_server.named_modules():
    #     if isinstance(m, nn.BatchNorm1d):
    #         mean_str = ", ".join([f"{x}" for x in m.running_mean[:3]])
    #         var_str  = ", ".join([f"{x}" for x in m.running_var[:3]])
    #         print(f"[{name}] mean=[{mean_str}], var=[{var_str}]")
    # print("=====================================\n")

    # print_BN_EMA_status(BN_EMA, title="BN_EMA 状态")
    
    # 验证损失检查
    if loss_test_collect_user:
        current_val_loss = sum(loss_test_collect_user) / len(loss_test_collect_user)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_client_state = copy.deepcopy(net_glob_client.state_dict())
            best_server_state = copy.deepcopy(net_glob_server.state_dict())
            print(f"Epoch {epoch+1}: Validation loss improved to {best_val_loss:.4f}. Updated best model in memory.")
        else:
            print(f"Epoch {epoch+1}: Validation loss did not improve (best: {best_val_loss:.4f})")
    
    loss_test_collect_user = []
    acc_test_collect_user = []

    # print("\n====== [ACT_EMA key list as Python literal] ======")
    # print(list(ACT_EMA.keys()))
    # print("==================================================\n")
    
# ================= 保存最佳模型到硬盘 =================
print("\n=============== Saving Best Model ===============")
if best_client_state is not None and best_server_state is not None:
    torch.save(best_client_state, os.path.join(best_model_path, "client_best.pth"))
    torch.save(best_server_state, os.path.join(best_model_path, "server_best.pth"))
    print(f"Saved best model (from epoch {best_epoch+1}) to disk")
else:
    print("Warning: No best model found, saving final model")
    torch.save(net_glob_client.state_dict(), os.path.join(best_model_path, "client_final.pth"))
    torch.save(net_glob_server.state_dict(), os.path.join(best_model_path, "server_final.pth"))

# ================= 加载最佳模型 =================
print("\n=============== Loading Best Model ===============")
if best_client_state is not None and best_server_state is not None:
    print(f"Loading best model from epoch {best_epoch+1}")
    net_glob_client.load_state_dict(best_client_state)
    net_glob_server.load_state_dict(best_server_state)
else:
    print("Warning: No best model found, using final model")

# ================= 使用最佳模型进行最终测试 =================
print("\n=============== Testing Best Model ===============")
idx_collect = []
fed_check = True
for idx in range(num_users):
    client = Client(net_glob_client, idx, lr, device,
                    dataset_train=dataset_train,
                    dataset_val=dataset_val,
                    dataset_test=dataset_test,
                    idxs_train=dict_users_train[idx],
                    idxs_val=dict_users_val[idx],
                    idxs_test=dict_users_test[idx])
    client.evaluate(copy.deepcopy(net_glob_client).to(device), best_epoch, mode='test')

if acc_test_collect:
    test_acc_value = acc_test_collect[-1]
else:
    print("警告：测试结果未正确收集，使用默认值0")
    test_acc_value = 0.0

results_df = pd.DataFrame({
    'round': range(1, len(acc_train_collect)+1),
    'train_acc': acc_train_collect,
    'val_acc': acc_test_collect[:len(acc_train_collect)],
    'test_acc': [test_acc_value] * len(acc_train_collect)
})

results_df.to_excel(f"{program}_results.xlsx", index=False)
print("训练完成! 结果已保存")

# # =================== 推理阶段量化验证 ===================这部分代码调用函数内部出错：named_modules/named_children
# print("开始进行量化推理模拟...")
# sample_image, sample_label = next(iter(client.ldr_test))# 取一批图像
# sample_image = sample_image.to(device)

# output = inference_quantized(net_glob_client, net_glob_server, sample_image, ACT_EMA, num_bits=8)
# print("量化推理输出（示例）:", output)

# =================== 导出量化模型参数 ===================
# print("\n====== [ACT_EMA key list as Python literal] ======")
# print(list(ACT_EMA.keys()))
# print("==================================================\n")

print("正在执行模型量化导出...")
# print("1️⃣ ACT_EMA 层数:", len(ACT_EMA))
# print("2️⃣ 示例键:", list(ACT_EMA.keys())[:5])
# print("3️⃣ 检查客户端prefix:", hasattr(net_glob_client, "layers"), type(net_glob_client.layers))
# print("4️⃣ 检查DataParallel:", isinstance(net_glob_client, torch.nn.DataParallel))
# print("5️⃣ ACT_EMA中第一层min/max:", list(ACT_EMA.values())[0] if len(ACT_EMA)>0 else "空")
# print("\n====== [ACT_EMA key list as Python literal] ======")
# print(list(ACT_EMA.keys()))
# print("==================================================\n")
export_path = export_quantized_model(
    net_glob_client,     # 客户端模型
    net_glob_server,     # 服务器模型
    export_dir="./quantized_export",  # 输出目录
    num_bits=8           # 量化位宽
)
print(f"导出完成，文件保存于: {export_path}")