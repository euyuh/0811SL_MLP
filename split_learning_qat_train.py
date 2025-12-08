# split_learning_qat_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 从你的工具包导入
try:
    from quant_utils import (
        FakeQuantLinear, CustomBatchNorm1d, register_activation_ema_hooks,
        ActQuantization, quan_scheme, get_activation_range,
        export_quantized_model, ACT_EMA
    )
except ImportError:
    raise ImportError("请确保 quant_utils.py 在同一目录下")

# ==========================================
# 1. 配置与超参数
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
LOG_DIR = "./runs/split_learning_qat"
DATA_ROOT = "./data"
VAL_SPLIT = 5000  # 从训练集划出 5000 张做验证

# 确保实验可复现
torch.manual_seed(1234)

# ==========================================
# 2. 模型定义 (Split Model)
# ==========================================

class ClientPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            FakeQuantLinear(64*64, 2048),
            CustomBatchNorm1d(2048, prefix="client_layers", name_in_module="layers.2"),
            nn.ReLU6(),
            nn.Dropout(0.4),
            FakeQuantLinear(2048, 1024),
            CustomBatchNorm1d(1024, prefix="client_layers", name_in_module="layers.6"),
            nn.ReLU6(),
            nn.Dropout(0.35),
            FakeQuantLinear(1024, 512),
            CustomBatchNorm1d(512, prefix="client_layers", name_in_module="layers.10"),
            nn.ReLU6(),
            nn.Dropout(0.3),
            FakeQuantLinear(512, 256),
            CustomBatchNorm1d(256, prefix="client_layers", name_in_module="layers.14"),
            nn.ReLU6(),
            nn.Dropout(0.25),
            FakeQuantLinear(256, 256),
            CustomBatchNorm1d(256, prefix="client_layers", name_in_module="layers.18"),
            nn.ReLU6() # 最后一层是 ReLU6，输出非负
        )

    def forward(self, x):
        return self.layers(x)

class ServerPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 输入维度匹配 Client 输出 (256)
            FakeQuantLinear(256, 512),
            CustomBatchNorm1d(512, prefix="server_layers", name_in_module="layers.1"),
            nn.ReLU6(),
            nn.Dropout(0.4),
            FakeQuantLinear(512, 512),
            CustomBatchNorm1d(512, prefix="server_layers", name_in_module="layers.5"),
            nn.ReLU6(),
            nn.Dropout(0.35),
            FakeQuantLinear(512, 256),
            CustomBatchNorm1d(256, prefix="server_layers", name_in_module="layers.9"),
            nn.ReLU6(),
            nn.Dropout(0.3),
            FakeQuantLinear(256, 128),
            CustomBatchNorm1d(128, prefix="server_layers", name_in_module="layers.13"),
            nn.ReLU6(),
            FakeQuantLinear(128, 10) 
        )

    def forward(self, x):
        return self.layers(x)

# ==========================================
# 3. 训练与评估流程
# ==========================================

def train_one_epoch(client_model, server_model, train_loader, 
                    opt_client, opt_server, criterion, epoch, writer):
    client_model.train()
    server_model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        opt_client.zero_grad()
        opt_server.zero_grad()
        
        # --- Step 1: Client Forward ---
        client_out = client_model(data)
        
        # --- Step 2: Interface Quantization ---
        cut_layer_key = "client/layers.19_out"
        # [修改] 默认最小值改为 0.0，因为上一层是 ReLU6
        act_min, act_max = get_activation_range(cut_layer_key, 0.0, 6.0)
        
        # 模拟量化传输
        client_out_q = ActQuantization(client_out, FloatMax=act_max, FloatMin=act_min)
        
        # --- Step 3: Send to Server ---
        server_input = client_out_q.detach().clone()
        server_input.requires_grad_(True)
        
        # --- Step 4: Server Forward ---
        server_out = server_model(server_input)
        loss = criterion(server_out, target)
        
        # --- Step 5: Server Backward ---
        loss.backward()
        
        # --- Step 6: Return Gradient ---
        server_grad = server_input.grad.clone()
        
        # --- Step 7: Client Backward ---
        client_out.backward(server_grad)
        
        opt_server.step()
        opt_client.step()
        
        # 统计
        running_loss += loss.item() * data.size(0)
        preds = server_out.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += data.size(0)
        
        if batch_idx % 50 == 0:
            global_step = (epoch-1)*len(train_loader) + batch_idx
            writer.add_scalar('Batch/Loss', loss.item(), global_step)

    return running_loss / total, 100. * correct / total

def evaluate(client_model, server_model, dataloader, criterion):
    client_model.eval()
    server_model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            client_out = client_model(data)
            
            # [修改] 推理时同样使用 0.0 作为默认最小值
            cut_layer_key = "client_layers/layers.19_out"
            act_min, act_max = get_activation_range(cut_layer_key, 0.0, 6.0)
            client_out = ActQuantization(client_out, FloatMax=act_max, FloatMin=act_min)
            
            server_out = server_model(client_out)
            
            loss = criterion(server_out, target)
            running_loss += loss.item() * data.size(0)
            preds = server_out.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += data.size(0)
            
    return running_loss / total, 100. * correct / total

# ==========================================
# 4. 主程序
# ==========================================

def main():
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # [修改] 数据加载与划分
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])
    
    # 加载完整训练集和测试集
    mnist_train_full = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(DATA_ROOT, train=False, transform=transform)
    
    # 划分 Train / Val
    train_len = len(mnist_train_full) - VAL_SPLIT
    train_dataset, val_dataset = random_split(mnist_train_full, [train_len, VAL_SPLIT])
    
    print(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(mnist_test)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 初始化模型
    client_model = ClientPart().to(DEVICE)
    server_model = ServerPart().to(DEVICE)
    
    register_activation_ema_hooks(client_model, prefix="client_layers")
    register_activation_ema_hooks(server_model, prefix="server_layers")
    
    opt_client = optim.Adam(client_model.parameters(), lr=LR)
    opt_server = optim.Adam(server_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_server, mode='min', factor=0.5, patience=2, verbose=True)
    
    best_val_acc = 0.0
    best_client_path = os.path.join(LOG_DIR, "best_client.pth")
    best_server_path = os.path.join(LOG_DIR, "best_server.pth")
    
    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            client_model, server_model, train_loader, 
            opt_client, opt_server, criterion, epoch, writer
        )
        
        # [修改] 使用 Val 集进行验证
        val_loss, val_acc = evaluate(client_model, server_model, val_loader, criterion)
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(client_model.state_dict(), best_client_path)
            torch.save(server_model.state_dict(), best_server_path)
            print(f"Saved Best Model (Val Acc: {val_acc:.2f}%)")
            
    print(f"Training Complete. Best Val Acc: {best_val_acc:.2f}%")
    writer.close()

    # ==========================================
    # [新增] 5. 加载最佳模型并在测试集上测试
    # ==========================================
    print("\n" + "="*40)
    print("Running Final Evaluation on Test Set...")
    
    if os.path.exists(best_client_path) and os.path.exists(best_server_path):
        client_model.load_state_dict(torch.load(best_client_path))
        server_model.load_state_dict(torch.load(best_server_path))
        print("Loaded best checkpoint.")
    
    test_loss, test_acc = evaluate(client_model, server_model, test_loader, criterion)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Acc : {test_acc:.2f}%")
    print("="*40 + "\n")

    # ==========================================
    # 6. 导出量化参数
    # ==========================================
    print("正在导出量化模型参数...")
    export_dir = "./quantized_export"
    export_quantized_model(client_model, server_model, export_dir=export_dir, num_bits=8)

if __name__ == "__main__":
    main()