import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler
import argparse
from copy import deepcopy

from DataSetAugmentation import MammogramRawDataset, TransformDataset, Config, get_train_transforms, get_val_transforms

class DistillConfig:
    TEACHER_NAME = "efficientnet"   
    STUDENT_NAME = "resnet"        
    BATCH_SIZE = 16
    NUM_EPOCHS_TEACHER = 15
    NUM_EPOCHS_STUDENT = 20
    TEACHER_LR = 1e-4              
    STUDENT_LR = 1e-4               
    WEIGHT_DECAY = 1e-5
    TEMPERATURE = 4.0
    ALPHA = 0.7
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = Path("checkpoints_distill")
    PLOT_DIR = Path("plots_distill_kfold")
    IMAGE_SIZE = 224

def create_model(model_name, pretrained=True):
    if model_name == "efficientnet":
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 2)
        )
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "resnet":
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 2)
        )
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(DistillConfig.DEVICE)

# -------------------- Distillation loss --------------------
def distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha):
    ce_loss = nn.CrossEntropyLoss()(student_outputs, labels)
    soft_teacher = nn.functional.softmax(teacher_outputs / temperature, dim=1)
    log_soft_student = nn.functional.log_softmax(student_outputs / temperature, dim=1)
    distill_loss = nn.functional.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    total_loss = alpha * ce_loss + (1 - alpha) * distill_loss
    return total_loss, ce_loss.item(), distill_loss.item()

# -------------------- Training functions --------------------
def train_teacher(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(DistillConfig.DEVICE), labels.to(DistillConfig.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def train_student_with_distillation(student, teacher, loader, optimizer, temperature, alpha):
    student.train()
    teacher.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    ce_losses, distill_losses = [], []
    for images, labels in loader:
        images, labels = images.to(DistillConfig.DEVICE), labels.to(DistillConfig.DEVICE)
        optimizer.zero_grad()
        student_outputs = student(images)
        with torch.no_grad():
            teacher_outputs = teacher(images)
        loss, ce_loss, d_loss = distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(student_outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        ce_losses.append(ce_loss)
        distill_losses.append(d_loss)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.mean(ce_losses), np.mean(distill_losses)

def validate(model, loader):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DistillConfig.DEVICE), labels.to(DistillConfig.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1

# -------------------- 5-Fold Cross-Validation with Distillation --------------------
def distill_kfold():
    # Load raw dataset (no transforms)
    raw_dataset = MammogramRawDataset(["mass_train", "calc_train"])
    labels = [label for _, label in raw_dataset.samples]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store per-fold metrics (for student validation)
    all_val_losses = []
    all_val_accs = []
    all_val_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n========== Fold {fold+1}/5 ==========")
        
        # Create datasets
        train_raw = Subset(raw_dataset, train_idx)
        val_raw   = Subset(raw_dataset, val_idx)
        train_dataset = TransformDataset(train_raw, transform=get_train_transforms())
        val_dataset   = TransformDataset(val_raw,   transform=get_val_transforms())
        
        # Weighted sampler for class imbalance (training only)
        train_labels = np.array(labels)[train_idx]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=DistillConfig.BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=DistillConfig.BATCH_SIZE, shuffle=False)
        
        # ---------- Step 1: Train teacher on training set ----------
        print("Training teacher...")
        teacher = create_model(DistillConfig.TEACHER_NAME, pretrained=True)
        optimizer_teacher = optim.AdamW(teacher.parameters(), lr=DistillConfig.TEACHER_LR, weight_decay=DistillConfig.WEIGHT_DECAY)
        scheduler_teacher = optim.lr_scheduler.ReduceLROnPlateau(optimizer_teacher, mode='min', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        best_teacher_loss = float('inf')
        for epoch in range(DistillConfig.NUM_EPOCHS_TEACHER):
            train_loss, train_acc = train_teacher(teacher, train_loader, criterion, optimizer_teacher)
            val_loss, val_acc, val_f1 = validate(teacher, val_loader)
            scheduler_teacher.step(val_loss)
            print(f"  Teacher Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
            if val_loss < best_teacher_loss:
                best_teacher_loss = val_loss
                torch.save(teacher.state_dict(), DistillConfig.CHECKPOINT_DIR / f"teacher_fold_{fold+1}.pth")
        
        # Load best teacher for this fold
        teacher.load_state_dict(torch.load(DistillConfig.CHECKPOINT_DIR / f"teacher_fold_{fold+1}.pth"))
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        print("Teacher training complete.")
        
        # ---------- Step 2: Distill to student ----------
        print("Training student with distillation...")
        student = create_model(DistillConfig.STUDENT_NAME, pretrained=True)
        optimizer_student = optim.AdamW(student.parameters(), lr=DistillConfig.STUDENT_LR, weight_decay=DistillConfig.WEIGHT_DECAY)
        scheduler_student = optim.lr_scheduler.ReduceLROnPlateau(optimizer_student, mode='min', patience=3, factor=0.5)
        
        best_student_loss = float('inf')
        fold_val_losses = []
        fold_val_accs = []
        fold_val_f1s = []
        
        for epoch in range(DistillConfig.NUM_EPOCHS_STUDENT):
            train_loss, train_acc, ce_loss, d_loss = train_student_with_distillation(
                student, teacher, train_loader, optimizer_student,
                DistillConfig.TEMPERATURE, DistillConfig.ALPHA
            )
            val_loss, val_acc, val_f1 = validate(student, val_loader)
            scheduler_student.step(val_loss)
            
            fold_val_losses.append(val_loss)
            fold_val_accs.append(val_acc)
            fold_val_f1s.append(val_f1)
            
            print(f"  Student Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} (CE {ce_loss:.3f}, Distill {d_loss:.3f}) | Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}")
            
            if val_loss < best_student_loss:
                best_student_loss = val_loss
                torch.save(student.state_dict(), DistillConfig.CHECKPOINT_DIR / f"student_fold_{fold+1}.pth")
        
        # Store fold metrics (use best epoch? Here we store all for averaging)
        all_val_losses.append(fold_val_losses)
        all_val_accs.append(fold_val_accs)
        all_val_f1s.append(fold_val_f1s)
        
        # Plot per-fold learning curves
        epochs_range = range(1, DistillConfig.NUM_EPOCHS_STUDENT+1)
        plt.figure()
        plt.plot(epochs_range, fold_val_losses, label='Val Loss')
        plt.plot(epochs_range, fold_val_accs, label='Val Acc')
        plt.plot(epochs_range, fold_val_f1s, label='Val F1')
        plt.xlabel('Epoch')
        plt.title(f'Fold {fold+1} Student Metrics')
        plt.legend()
        plt.savefig(DistillConfig.PLOT_DIR / f"fold_{fold+1}_metrics.png")
        plt.close()
    
    # -------------------- Aggregate across folds --------------------
    # Pad sequences to same length (all should be NUM_EPOCHS_STUDENT)
    max_epochs = DistillConfig.NUM_EPOCHS_STUDENT
    val_losses_array = np.array([losses[:max_epochs] for losses in all_val_losses])
    val_accs_array   = np.array([accs[:max_epochs] for accs in all_val_accs])
    val_f1s_array    = np.array([f1s[:max_epochs] for f1s in all_val_f1s])
    
    mean_val_loss = np.mean(val_losses_array, axis=0)
    mean_val_acc  = np.mean(val_accs_array, axis=0)
    mean_val_f1   = np.mean(val_f1s_array, axis=0)
    std_val_loss  = np.std(val_losses_array, axis=0)
    
    epochs = range(1, max_epochs+1)
    
    plt.figure()
    plt.plot(epochs, mean_val_loss, label='Mean Val Loss')
    plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mean Validation Loss Across Folds (Student)')
    plt.legend()
    plt.savefig(DistillConfig.PLOT_DIR / "mean_val_loss.png")
    plt.close()
    
    plt.figure()
    plt.plot(epochs, mean_val_acc, label='Mean Val Accuracy')
    plt.fill_between(epochs, mean_val_acc - np.std(val_accs_array, axis=0), mean_val_acc + np.std(val_accs_array, axis=0), alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Mean Validation Accuracy Across Folds')
    plt.legend()
    plt.savefig(DistillConfig.PLOT_DIR / "mean_val_acc.png")
    plt.close()
    
    plt.figure()
    plt.plot(epochs, mean_val_f1, label='Mean Val F1')
    plt.fill_between(epochs, mean_val_f1 - np.std(val_f1s_array, axis=0), mean_val_f1 + np.std(val_f1s_array, axis=0), alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Mean Validation F1 Across Folds')
    plt.legend()
    plt.savefig(DistillConfig.PLOT_DIR / "mean_val_f1.png")
    plt.close()
    
    # Final summary
    final_loss = mean_val_loss[-1]
    final_acc = mean_val_acc[-1]
    final_f1 = mean_val_f1[-1]
    print("\n========== Final Cross-Validation Results (Distillation) ==========")
    print(f"Student: {DistillConfig.STUDENT_NAME} (teacher: {DistillConfig.TEACHER_NAME})")
    print(f"Mean Val Loss (last epoch): {final_loss:.4f} ± {std_val_loss[-1]:.4f}")
    print(f"Mean Val Accuracy: {final_acc:.4f} ± {np.std(val_accs_array[:,-1]):.4f}")
    print(f"Mean Val F1 Score: {final_f1:.4f} ± {np.std(val_f1s_array[:,-1]):.4f}")

if __name__ == "__main__":
    distill_kfold()