import os
from glob import glob
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
import torchvision.transforms as transforms

# (Optional) for AUROC
try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ----------------------------
# Config
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If your dataset is folder-based like:
#   ECG_ROOT/
#     risky/
#       img1.png ...
#     normal/
#       img2.png ...
ECG_ROOT = r"C:\Users\lenovo\Desktop\HeartAttackProject\ECG_Images"

IMG_SIZE = 224
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10

# Binary classification: 0 = normal, 1 = risky
CLASS_TO_IDX = {"normal": 0, "risky": 1}   # change names to match your folders


# ----------------------------
# Dataset
# ----------------------------
class ECGImageFolderDataset(Dataset):
    """
    Loads ECG images from class folders:
      root/normal/*.png
      root/risky/*.png
    """
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.samples = []
        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class folder not found: {cls_dir}")

            # Add common image extensions here
            exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            paths = []
            for e in exts:
                paths.extend(glob(os.path.join(cls_dir, e)))

            for p in paths:
                self.samples.append((p, cls_idx))

        if len(self.samples) == 0:
            raise RuntimeError("No images found. Check ECG_ROOT and folder names.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")  # convert to 3-channel

        if self.transform:
            img = self.transform(img)

        # BCEWithLogitsLoss expects float labels (0.0/1.0)
        label = torch.tensor(label, dtype=torch.float32)
        return img, label


# ----------------------------
# CNN-LSTM Model
# ----------------------------
class CNNLSTM_ECG(nn.Module):
    """
    CNN -> feature map -> treat one spatial axis as a sequence -> BiLSTM -> binary logit
    """
    def __init__(self, lstm_hidden=128, lstm_layers=1, bidirectional=True):
        super().__init__()

        # Simple CNN backbone (kept small for undergrad-style project)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 112 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 56 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # After CNN: [B, C=128, H, W] (roughly 28x28 if input 224)
        # We'll average over height (H) -> sequence over width (W)
        self.bidirectional = bidirectional
        lstm_in = 128  # channel dimension becomes LSTM input size

        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)  # binary logit

    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.cnn(x)               # [B, 128, H', W']
        feat = feat.mean(dim=2)          # avg over height -> [B, 128, W']
        feat = feat.permute(0, 2, 1)     # [B, W', 128] sequence length = W'

        out, _ = self.lstm(feat)         # [B, W', hidden*(dir)]
        last = out[:, -1, :]             # [B, hidden*(dir)]
        logit = self.fc(last).squeeze(1) # [B]
        return logit


# ----------------------------
# Training / Eval
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)  # [B]
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    probs_all = []
    labels_all = []
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()

        probs_all.extend(probs.detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    acc = correct / total if total > 0 else 0.0

    auc = None
    if SKLEARN_OK:
        # AUROC needs both classes present in labels
        if len(set(labels_all)) == 2:
            auc = roc_auc_score(labels_all, probs_all)

    return acc, auc


def main():
    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # If you want, you can normalize; for ECG plots, it's optional
        # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    dataset = ECGImageFolderDataset(ECG_ROOT, CLASS_TO_IDX, transform=transform)

    # Split: 80/20
    n_total = len(dataset)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CNNLSTM_ECG(lstm_hidden=128, lstm_layers=1, bidirectional=True).to(device)

    # Binary loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Device: {device}")
    print(f"Train: {n_train} | Val: {n_val}")

    best_val = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_auc = evaluate(model, val_loader)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "best_ecg_cnn_lstm.pt")

        auc_text = f"{val_auc:.4f}" if val_auc is not None else "N/A"
        print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | val_acc {val_acc:.4f} | val_auc {auc_text}")

    print("Saved best model to: best_ecg_cnn_lstm.pt")


if __name__ == "__main__":
    main()
