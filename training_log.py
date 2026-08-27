from pathlib import Path

from openpyxl import Workbook

LOG_DIR = Path("training_logs")

EPOCH_HEADER = ["fold", "epoch", "train_loss", "val_loss", "val_acc", "val_precision", "val_recall", "val_f1"]
FOLD_SUMMARY_HEADER = ["fold", "best_val_f1", "tta_f1", "best_threshold"]


def create_workbook(model_name):
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{model_name}_training_log.xlsx"

    wb = Workbook()
    epochs_ws = wb.active
    epochs_ws.title = "Epochs"
    epochs_ws.append(EPOCH_HEADER)

    summary_ws = wb.create_sheet("Fold Summary")
    summary_ws.append(FOLD_SUMMARY_HEADER)

    wb.save(path)
    return wb


def log_epoch(wb, path, fold, epoch, train_loss, val_loss, val_acc, val_prec, val_rec, val_f1):
    wb["Epochs"].append([fold, epoch, train_loss, val_loss, val_acc, val_prec, val_rec, val_f1])
    wb.save(path)


def log_fold_summary(wb, path, fold, best_val_f1, tta_f1, best_threshold):
    wb["Fold Summary"].append([fold, best_val_f1, tta_f1, best_threshold])
    wb.save(path)
