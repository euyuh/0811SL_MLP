# single_client_test.py
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# import from your quant_utils (ensure quant_utils.py 在同目录，或者在 PYTHONPATH)
from quant_utils import FakeQuantLinear, CustomBatchNorm1d, register_activation_ema_hooks, ActQuantization, get_activation_range, quan_scheme


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# Build tiny dataset: MNIST, pick 1 image per class
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

# Download (or use cached) MNIST train set
mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=False)

# select one sample per class (0..9)
selected_idxs = {}
for idx, (img, label) in enumerate(mnist_train):
    if label not in selected_idxs:
        selected_idxs[label] = idx
    if len(selected_idxs) == 10:
        break

selected_list = [selected_idxs[i] for i in range(10)]
print("Selected sample indices for classes 0..9:", selected_list)

class TinyMNIST(Dataset):
    def __init__(self, full_dataset, idxs):
        self.full = full_dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        return self.full[self.idxs[i]]

tiny_train = TinyMNIST(mnist_train, selected_list)
train_loader = DataLoader(tiny_train, batch_size=2, shuffle=True)  # batch_size small for 10 images

# ---------------------------
# Define client model (copy structure from your MLPClient, plus classifier head)
# ---------------------------
class SingleClientModel(nn.Module):
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

    def forward(self, x):
        x = self.layers(x)
        logits = self.classifier(x)
        return logits

net = SingleClientModel().to(device)

# If you want activation EMA hooks for CustomBatchNorm1d, ensure the register function includes this type.
# Here we try to register hooks on the model; if your quant_utils.register_activation_ema_hooks does not detect CustomBatchNorm1d,
# you may need to modify that function (as noted in analysis).
try:
    register_activation_ema_hooks(net, prefix="client_layers")
except Exception as e:
    print("register_activation_ema_hooks failed (non-fatal).", e)

# ---------------------------
# Training setup
# ---------------------------
epochs = 50
lr = 1e-5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# TensorBoard writer
writer = SummaryWriter(log_dir="./runs/single_client_test")

# ---------------------------
# Training loop (per epoch print train loss & write to tensorboard)
# ---------------------------
for epoch in range(1, epochs + 1):
    net.train()
    epoch_loss = 0.0
    num_batches = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)  # logits
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / max(1, num_batches)
    print(f"Epoch {epoch:03d} Train Loss: {avg_loss:.6f}")
    writer.add_scalar("train/loss", avg_loss, epoch)
    # --- Debug prints: sample logits/labels and grad norms ---
    # print one batch predictions to see if logits vary
    net.eval()
    with torch.no_grad():
        imgs_sample, labels_sample = next(iter(train_loader))
        imgs_sample = imgs_sample.to(device)
        labels_sample = labels_sample.to(device)
        sample_logits = net(imgs_sample)
        preds = sample_logits.argmax(dim=1)
    net.train()

    # compute grad norm of classifier params (check they are being updated)
    # total_grad_norm = 0.0
    # for name, p in net.named_parameters():
    #     if p.grad is not None:
    #         total_grad_norm += p.grad.detach().float().norm().item()**2
    # total_grad_norm = total_grad_norm**0.5

    # print(f"Epoch {epoch:03d} | avg_loss={avg_loss:.6f} | sample_labels={labels_sample.tolist()} | sample_preds={preds.tolist()} | grad_norm={total_grad_norm:.6e}")

    # optional: stop early if loss converged
    # if avg_loss < 1e-4:
    #     break
    classifier_grad_norm = 0.0
    for name, p in net.named_parameters():
        if "classifier" in name:
            if p.grad is not None:
                classifier_grad_norm += p.grad.detach().float().norm().item()**2
    classifier_grad_norm = classifier_grad_norm**0.5
    print("classifier_grad_norm =", classifier_grad_norm)


writer.close()

# ---------------------------
# Quick evaluation on the same tiny set
# ---------------------------
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        out = net(images)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy on tiny 10-sample set: {correct}/{total} = {100.0 * correct / total:.2f}%")
print("TensorBoard logs written to ./runs/single_client_test (run `tensorboard --logdir runs` to view)")
