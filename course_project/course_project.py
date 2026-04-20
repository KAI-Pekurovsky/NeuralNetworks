
# 1. КОНФІГУРАЦІЯ

student_name = "Петренко Уляна"
variant_number = 1

# Базова модель — параметри фіксовані, не змінювати!
# epochs=50 і patience=2 задані навмисно,
# щоб базова модель гарантовано зупинилась через ранню зупинку.

baseline_config = {
    "epochs": 50,
    "learning_rate": 0.5,    
    "hidden_layers": [8],    
    "dropout": 0.0,
    "batch_size": 1024,      
    "activation": "tanh",    
    "optimizer": "sgd",      
    "patience": 3
}

# -------------------------------------------------------
# УВАГА СТУДЕНТУ:
# У покращеному конфігу ви маєте самостійно підібрати
# гіперпараметри та обґрунтувати свій вибір у звіті.
#
# Що можна змінювати:
#   epochs        — кількість епох навчання (наприклад, 50–100)
#   learning_rate — крок навчання (наприклад, 0.0001–0.01)
#   hidden_layers — список розмірів прихованих шарів
#                   (наприклад, [128, 64] або [256, 128, 64])
#   dropout       — частка відключених нейронів (0.1 – 0.5)
#   batch_size    — розмір батчу (наприклад, 64, 128, 512)
#   activation    — функція активації: "relu", "tanh", "leaky_relu"
#   optimizer     — оптимізатор: "adam" або "sgd"
#   patience      — кількість епох без покращення до зупинки (наприклад 5, 10, 15)
# -------------------------------------------------------

improved_config = {
    "epochs": 50, 
    "learning_rate": 0.005,      
    "hidden_layers": [128, 64, 32], 
    "dropout": 0.3,
    "batch_size": 64,           
    "activation": "relu",
    "optimizer": "Adam",
    "patience": 10
}



# =========================
# 1. IMPORTS
# =========================

import sys
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

# =========================
# ЛОГ-БУФЕР
# Весь вивід print() дублюється у файл training_log_*.txt
# =========================

class TeeOutput:
    """Дублює вивід одночасно в термінал і в буфер для збереження у файл."""
    def __init__(self, stream):
        self.stream = stream
        self.buffer = io.StringIO()

    def write(self, data):
        self.stream.write(data)
        self.buffer.write(data)

    def flush(self):
        self.stream.flush()

    def getvalue(self):
        return self.buffer.getvalue()

_tee = TeeOutput(sys.stdout)
sys.stdout = _tee


# =========================
# 3. КЛАСИ (UNSW-NB15)
# =========================

class_names = [
    "Normal", "Fuzzers", "Analysis", "Backdoors", "DoS",
    "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms"
]
num_classes = len(class_names)

# =========================
# 4. ЗАВАНТАЖЕННЯ ДАТАСЕТУ
# =========================

df = pd.read_csv(f"variant_{variant_number}.csv")

# =========================
# 5. ПЕРЕДОБРОБКА (ОНОВЛЕНО)
# =========================

# Визначаємо цільову змінну та ознаки
X = df.drop("attack_cat", axis=1)
y = df["attack_cat"]

# 1. Кодування цільової змінної
le = LabelEncoder()
y = le.fit_transform(y)

# 2. Автоматичне визначення категоріальних ознак
# Ми шукаємо колонки з типом 'object' (текстові)
cat_features = X.select_dtypes(include=['object']).columns.tolist()
print(f"Виявлені категоріальні ознаки: {cat_features}")

# 3. One-Hot Encoding для всіх категоріальних ознак
# Це розширить кількість колонок (наприклад, proto_tcp, proto_udp і т.д.)
X = pd.get_dummies(X, columns=cat_features)

# 4. Масштабування (StandardScaler)
# ТЕПЕР ЦЕ КРИТИЧНО: у нас є ознаки від 0.00001 до 100,000,000
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Розмірність вхідних даних після обробки: {X.shape[1]} ознак")

# --- Далі йде стандартне розбиття ---
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, random_state=42, stratify=y_train_full
)

# Конвертація у тензори
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val,   dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

# =========================
# 6. МОДЕЛЬ
# =========================

ACTIVATIONS = {
    "relu":       nn.ReLU,
    "tanh":       nn.Tanh,
    "leaky_relu": nn.LeakyReLU
}

class Net(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, dropout, activation):
        super().__init__()

        act_cls = ACTIVATIONS.get(activation, nn.ReLU)

        layers = []
        prev = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =========================
# 7. EARLY STOPPING
# =========================

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience     = patience
        self.best         = float("inf")
        self.counter      = 0
        self.stop         = False
        self.best_weights = None
        self.best_epoch   = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best:
            self.best         = val_loss
            self.counter      = 0
            self.best_epoch   = epoch + 1
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

# =========================
# 8. ФУНКЦІЯ ТРЕНУВАННЯ
# =========================

# =========================
# 8. ФУНКЦІЯ ТРЕНУВАННЯ (ОНОВЛЕНА ТА СТАБІЛЬНА)
# =========================

def train(model, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Вибір оптимізатора
    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

    # --- ДОДАЄМО SCHEDULER ---
    # Він автоматично зменшить Learning Rate, якщо val_loss не покращується 3 епохи
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Ваги для боротьби з дисбалансом
    class_weights = torch.tensor([2.0, 2.5, 1.0, 1.5, 1.0, 2.0, 0.5, 2.5, 3.0, 5.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    stopper = EarlyStopping(config["patience"])

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    train_losses = []
    val_losses   = []

    for epoch in range(config["epochs"]):
        # ---- Тренування ----
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # на всякий випадок явно на device
            optimizer.zero_grad()
            out  = model(X_batch)
            loss = loss_fn(out, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)

        # ---- Валідація ----
        model.eval()
        with torch.no_grad():
            val_out  = model(X_val.to(device))
            val_loss = loss_fn(val_out, y_val.to(device)).item()

        val_losses.append(val_loss)

        # --- КРОК ПЛАНУВАЛЬНИКА ---
        scheduler.step(val_loss)

        # Отримуємо поточний LR для логу
        current_lr = optimizer.param_groups[0]['lr']

        # Early stopping
        stopper.step(val_loss, model, epoch)

        if stopper.counter == 0:
            es_status = "✓ новий мінімум"
        else:
            es_status = f"⏳ очікування {stopper.counter}/{stopper.patience}"

        # Оновлений вивід з LR
        print(
            f"  Epoch {epoch+1:3d} | "
            f"train_loss={epoch_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"LR={current_lr:.6f} | {es_status}"
        )

        if stopper.stop:
            print(
                f"\n  ⛔ РАННЯ ЗУПИНКА на епосі {epoch + 1}.\n"
                f"     Найкраща val_loss = {stopper.best:.4f} (епоха {stopper.best_epoch})."
            )
            break
    else:
        print(f"\n  ✅ Навчання завершено після {config['epochs']} епох.")

    if stopper.best_weights is not None:
        model.load_state_dict(stopper.best_weights)

    return {"train": train_losses, "val": val_losses}

# =========================
# 9. ІНІЦІАЛІЗАЦІЯ МОДЕЛЕЙ
# =========================

input_size = X_train.shape[1]

baseline = Net(
    input_size,
    baseline_config["hidden_layers"],
    num_classes,
    baseline_config["dropout"],
    baseline_config["activation"]
)

improved = Net(
    input_size,
    improved_config["hidden_layers"],
    num_classes,
    improved_config["dropout"],
    improved_config["activation"]
)

# =========================
# 10. ТРЕНУВАННЯ
# =========================

print("=" * 55)
print("=== Тренування базової моделі ===")
print("=" * 55)
baseline_history = train(baseline, baseline_config)

print()
print("=" * 55)
print("=== Тренування покращеної моделі ===")
print("=" * 55)
improved_history = train(improved, improved_config)

# =========================
# 11. ПЕРЕДБАЧЕННЯ
# =========================

def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1).numpy()
        preds  = np.argmax(probs, axis=1)
    return preds, probs

b_pred, b_prob = predict(baseline, X_test)
i_pred, i_prob = predict(improved,  X_test)

# =========================
# 12. МЕТРИКИ
# =========================

def compute_metrics(y_true, y_pred, y_prob):
    acc = np.mean(y_true.numpy() == y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    roc   = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    return acc, f1, roc

b_metrics = compute_metrics(y_test, b_pred, b_prob)
i_metrics = compute_metrics(y_test, i_pred, i_prob)

# =========================
# 13. ДЕТАЛЬНИЙ ЗВІТ ПО КЛАСАХ
# =========================

print()
print("=" * 55)
print("=== Базова модель — звіт по класах ===")
print("=" * 55)
print(classification_report(
    y_test, b_pred,
    target_names=class_names,
    zero_division=0
))

print("=" * 55)
print("=== Покращена модель — звіт по класах ===")
print("=" * 55)
print(classification_report(
    y_test, i_pred,
    target_names=class_names,
    zero_division=0
))

# =========================
# 14. ЗБЕРЕЖЕННЯ CSV
# =========================

# Предикції покращеної моделі
results = pd.DataFrame({
    "true": y_test.numpy(),
    "pred": i_pred
})
results["correct"] = results["true"] == results["pred"]
results.to_csv(
    f"predictions_{student_name}_var{variant_number}.csv",
    index=False
)

# Порівняльна таблиця метрик обох моделей
metrics_df = pd.DataFrame({
    "Модель":    ["Базова", "Покращена"],
    "Точність":  [round(b_metrics[0], 4), round(i_metrics[0], 4)],
    "F1-міра":   [round(b_metrics[1], 4), round(i_metrics[1], 4)],
    "ROC-AUC":   [round(b_metrics[2], 4), round(i_metrics[2], 4)],
})
metrics_df.to_csv(
    f"metrics_{student_name}_var{variant_number}.csv",
    index=False
)
print("=" * 55)
print("=== Порівняльна таблиця метрик ===")
print("=" * 55)
print(metrics_df.to_string(index=False))

# =========================
# 15. ЗБЕРЕЖЕННЯ ЛОГУ У ФАЙЛ
# =========================

sys.stdout = _tee.stream   # повертаємо оригінальний stdout

log_path = f"training_log_{student_name}_var{variant_number}.txt"
with open(log_path, "w", encoding="utf-8") as f:
    f.write(_tee.getvalue())

print(f"\nЛог тренування збережено у файл: {log_path}")

# =========================
# 16. ВІЗУАЛІЗАЦІЯ (2 × 3)
# =========================

fig, ax = plt.subplots(2, 3, figsize=(20, 10))

fig.suptitle(
    f"Результати курсової роботи\n"
    f"Студент: {student_name} | Варіант: {variant_number}",
    fontsize=16
)

# ------ 1. Loss (train + val для обох моделей) ------
ax[0, 0].plot(baseline_history["train"], label="Базова — train",    color="steelblue",  linestyle="--")
ax[0, 0].plot(baseline_history["val"],   label="Базова — val",      color="steelblue")
ax[0, 0].plot(improved_history["train"], label="Покращена — train", color="darkorange", linestyle="--")
ax[0, 0].plot(improved_history["val"],   label="Покращена — val",   color="darkorange")
ax[0, 0].set_title("Функція втрат (Loss)")
ax[0, 0].set_xlabel("Епоха")
ax[0, 0].set_ylabel("Втрати")
ax[0, 0].legend(fontsize=7)

# ------ 2. Порівняння метрик ------
metric_labels = ["Точність", "F1-міра", "ROC-AUC"]
x = np.arange(3)
ax[0, 1].bar(x - 0.2, b_metrics, width=0.4, label="Базова модель")
ax[0, 1].bar(x + 0.2, i_metrics, width=0.4, label="Покращена модель")
ax[0, 1].set_title("Порівняння метрик")
ax[0, 1].set_xticks(x)
ax[0, 1].set_xticklabels(metric_labels)
ax[0, 1].set_ylim(0, 1.05)
ax[0, 1].set_ylabel("Значення")
ax[0, 1].legend()

# ------ 3. ROC-криві базової моделі ------
y_bin = label_binarize(y_test, classes=np.arange(num_classes))
ax[0, 2].set_title("ROC-криві (базова модель)")
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], b_prob[:, i])
    ax[0, 2].plot(fpr, tpr, label=class_names[i])
ax[0, 2].plot([0, 1], [0, 1], "k--")
ax[0, 2].set_xlabel("Хибнопозитивна частота")
ax[0, 2].set_ylabel("Справжньопозитивна частота")
ax[0, 2].legend(fontsize=6)

# ------ 4. Матриця помилок — базова ------
cm_b = confusion_matrix(y_test, b_pred, normalize="true")
im1 = ax[1, 0].imshow(cm_b, cmap="Blues")
ax[1, 0].set_title("Матриця помилок (базова модель)")
ax[1, 0].set_xlabel("Передбачений клас")
ax[1, 0].set_ylabel("Справжній клас")
ax[1, 0].set_xticks(range(num_classes))
ax[1, 0].set_yticks(range(num_classes))
ax[1, 0].set_xticklabels(class_names, rotation=90, fontsize=6)
ax[1, 0].set_yticklabels(class_names, fontsize=6)
plt.colorbar(im1, ax=ax[1, 0], fraction=0.046, pad=0.04)

# ------ 5. Матриця помилок — покращена ------
cm_i = confusion_matrix(y_test, i_pred, normalize="true")
im2 = ax[1, 1].imshow(cm_i, cmap="Greens")
ax[1, 1].set_title("Матриця помилок (покращена модель)")
ax[1, 1].set_xlabel("Передбачений клас")
ax[1, 1].set_ylabel("Справжній клас")
ax[1, 1].set_xticks(range(num_classes))
ax[1, 1].set_yticks(range(num_classes))
ax[1, 1].set_xticklabels(class_names, rotation=90, fontsize=6)
ax[1, 1].set_yticklabels(class_names, fontsize=6)
plt.colorbar(im2, ax=ax[1, 1], fraction=0.046, pad=0.04)

# ------ 6. ROC-криві покращеної моделі ------
ax[1, 2].set_title("ROC-криві (покращена модель)")
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], i_prob[:, i])
    ax[1, 2].plot(fpr, tpr, label=class_names[i])
ax[1, 2].plot([0, 1], [0, 1], "k--")
ax[1, 2].set_xlabel("Хибнопозитивна частота")
ax[1, 2].set_ylabel("Справжньопозитивна частота")
ax[1, 2].legend(fontsize=6)

# ------ Збереження ------
plt.tight_layout()
plt.savefig(
    f"result_{student_name}_var{variant_number}.png",
    dpi=300
)
plt.show()