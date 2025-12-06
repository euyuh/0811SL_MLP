# client_full_mnist_train.py
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# Try import quantization helpers from your uploaded file. If you don't want them,
# you can replace FakeQuantLinear/custom BN with nn.Linear/nn.BatchNorm1d easily.
try:
    from quant_utils import (FakeQuantLinear, CustomBatchNorm1d, 
                             register_activation_ema_hooks, ACT_EMA, BN_EMA, 
                             update_bn_ema, export_quantized_model)
    _HAS_QUANT_UTILS = True
except Exception as e:
    print("Warning: couldn't import FakeQuantLinear / CustomBatchNorm1d from quant_utils.py:", e)
    print("Falling back to nn.Linear / nn.BatchNorm1d for compatibility.")
    _HAS_QUANT_UTILS = False
    FakeQuantLinear = lambda in_f, out_f: nn.Linear(in_f, out_f)
    CustomBatchNorm1d = lambda num_features, prefix=None, name_in_module=None, use_ema_for_norm=False: nn.BatchNorm1d(num_features)

# ----------------------------
# Config / hyperparams
# ----------------------------
seed = 1234
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_root = "./data"          # 指向你已下载 MNIST 的目录
download = False              # 已下载 -> 不再尝试下载

batch_size = 128
val_split = 5000              # 从训练集中划出多少做 validation
num_workers = 4
epochs = 30                   # 可按需增大（建议 10-30）
# 推荐学习率：对于完整 MNIST，一般 Adam 可用 1e-3 ~ 1e-4
# 如果你在量化/特殊层上容易不稳定，先用 1e-4；若模型/训练稳定，可尝试 1e-3
lr = 1e-3
weight_decay = 1e-5
grad_clip_norm = 5.0          # 可选梯度裁剪，防止震荡

log_dir = "./runs/client_full_mnist"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# ----------------------------
# Data transforms and datasets
# ----------------------------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

# use existing downloaded data (download=False)
mnist_train_full = datasets.MNIST(root=data_root, train=True, transform=transform, download=download)
mnist_test = datasets.MNIST(root=data_root, train=False, transform=transform, download=download)

# split train -> train/val
train_len = len(mnist_train_full) - val_split
val_len = val_split
train_dataset, val_dataset = random_split(mnist_train_full, [train_len, val_len],
                                         generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(mnist_test)}")

# ----------------------------
# Client model definition
# ----------------------------
class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            FakeQuantLinear(64*64, 2048),
            CustomBatchNorm1d(2048, prefix="client_layers", name_in_module="layers.2", use_ema_for_norm=True),
            nn.ReLU6(),
            nn.Dropout(0.4),
            FakeQuantLinear(2048, 1024),
            CustomBatchNorm1d(1024, prefix="client_layers", name_in_module="layers.6", use_ema_for_norm=True),
            nn.ReLU6(),
            nn.Dropout(0.35),
            FakeQuantLinear(1024, 512),
            CustomBatchNorm1d(512, prefix="client_layers", name_in_module="layers.10", use_ema_for_norm=True),
            nn.ReLU6(),
            nn.Dropout(0.3),
            FakeQuantLinear(512, 256),
            CustomBatchNorm1d(256, prefix="client_layers", name_in_module="layers.14", use_ema_for_norm=True),
            nn.ReLU6(),
            nn.Dropout(0.25),
            FakeQuantLinear(256, 256),
            CustomBatchNorm1d(256, prefix="client_layers", name_in_module="layers.18", use_ema_for_norm=True),
            nn.ReLU6()
        )
        # Temporary classification head on client side
        self.classifier = nn.Linear(256, 10)

    def forward(self, x, return_features=False):
        features = self.layers(x)
        if return_features:
            return features
        logits = self.classifier(features)
        return logits


net = ClientModel().to(device)

# If using DataParallel, register on .module
if isinstance(net, torch.nn.DataParallel):
    register_activation_ema_hooks(net.module, prefix="client_layers")
else:
    register_activation_ema_hooks(net, prefix="client_layers")

print(net)

# ----------------------------
# Optimizer / Loss / Scheduler
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
# optional scheduler - reduce lr on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# ----------------------------
# Utilities: train / eval / accuracy
# ----------------------------
def compute_accuracy(output, target):
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct, preds

def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            c, _ = compute_accuracy(logits, labels)
            correct += c
            total += images.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

# ----------------------------
# Training loop
# ----------------------------
best_val_acc = 0.0
best_checkpoint = os.path.join(log_dir, "best_client_model.pth")
start_time = time.time()

for epoch in range(1, epochs + 1):
    net.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)             # logits
        loss = criterion(outputs, labels)
        loss.backward()

        # diagnostic: compute classifier grad norm (after backward, before step)
        cls_grad_norm = 0.0
        for name, p in net.named_parameters():
            if "classifier" in name and p.grad is not None:
                cls_grad_norm += float(p.grad.detach().norm().item())**2
        cls_grad_norm = cls_grad_norm**0.5 if cls_grad_norm > 0 else 0.0

        # gradient clipping
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        # accumulate stats
        running_loss += loss.item() * images.size(0)
        c, _ = compute_accuracy(outputs, labels)
        running_correct += c
        running_total += images.size(0)

        # optionally log batch-level scalars every N batches
        if (batch_idx + 1) % 200 == 0:
            batch_loss = running_loss / running_total
            batch_acc = running_correct / running_total
            global_step = (epoch-1) * len(train_loader) + batch_idx
            writer.add_scalar("train/loss_batch", batch_loss, global_step)
            writer.add_scalar("train/acc_batch", batch_acc, global_step)
            writer.add_scalar("train/classifier_grad_norm", cls_grad_norm, global_step)

    # epoch metrics
    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    print(f"Epoch {epoch:03d} Train Loss: {epoch_loss:.6f} Train Acc: {100.0*epoch_acc:.2f}% cls_grad_norm={cls_grad_norm:.4e}")

    # validate
    val_loss, val_acc = evaluate(net, val_loader, device)
    print(f"Epoch {epoch:03d} Val   Loss: {val_loss:.6f} Val   Acc: {100.0*val_acc:.2f}%")

    # tensorboard logging
    writer.add_scalar("train/loss", epoch_loss, epoch)
    writer.add_scalar("train/acc", epoch_acc, epoch)
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/acc", val_acc, epoch)
    writer.add_scalar("train/classifier_grad_norm_epoch", cls_grad_norm, epoch)

    # scheduler step (ReduceLROnPlateau uses metric)
    scheduler.step(val_loss)

    # save best checkpoint by val acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, best_checkpoint)
        print(f"Saved best model (val_acc={100.0*val_acc:.2f}%) -> {best_checkpoint}")

end_time = time.time()
print("Training finished in {:.1f} minutes. Best val acc: {:.2f}%".format((end_time - start_time)/60.0, 100.0*best_val_acc))

# ----------------------------
# Load best checkpoint and evaluate on test set
# ----------------------------
if os.path.exists(best_checkpoint):
    ckpt = torch.load(best_checkpoint, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    print("Loaded best checkpoint from epoch", ckpt.get('epoch', '?'), " val_acc=", ckpt.get('val_acc', 0.0))

test_loss, test_acc = evaluate(net, test_loader, device)
print(f"Test Loss: {test_loss:.6f} Test Acc: {100.0*test_acc:.2f}%")
writer.add_scalar("test/loss", test_loss, 0)
writer.add_scalar("test/acc", test_acc, 0)

# save final model
torch.save(net.state_dict(), os.path.join(log_dir, "client_final.pth"))
writer.close()
print("All done. TensorBoard logs are in", log_dir)

# run a few batches to populate activation and BN EMAs if not already present
need_calib = (len(ACT_EMA) == 0) or (len(BN_EMA) == 0)
if need_calib:
    print("Running short calibration to fill ACT_EMA / BN_EMA ...")
    net.train()  # ensure BN layers compute batch stats and update BN_EMA inside CustomBatchNorm1d
    cal_batches = 20
    with torch.no_grad():
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            _ = net(imgs)   # forward -> hooks update ACT_EMA, CustomBatchNorm1d updates BN_EMA
            # Optionally call explicit update_bn_ema if you want to ensure using module running_*
            update_bn_ema(net, prefix="client_layers")
            if i + 1 >= cal_batches:
                break
    net.eval()
    print("Calibration done. ACT_EMA keys example:", list(ACT_EMA.keys())[:6])
    print("BN_EMA keys example:", list(BN_EMA.keys())[:6])

# ---------------- Export quantized params ----------------
export_dir = "./quantized_export"
os.makedirs(export_dir, exist_ok=True)
export_path = export_quantized_model(net, server_model=nn.Sequential(), export_dir=export_dir, num_bits=8)
print("Exported quantized client model to:", export_path)