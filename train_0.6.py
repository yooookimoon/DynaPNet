import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from models.dataset import BrainSNP_Dataset
from models.gnn_model import BrainSNP_GNN
import random
import numpy as np
import psutil
from torch.utils.data import WeightedRandomSampler
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch.cuda.amp import autocast, GradScaler
import gc

# python train_0.6.py --mode full
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------
# 日志工具
# -------------------------
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write("\n" + "=" * 50 + f"\n训练开始时间: {datetime.now()}\n" + "=" * 50 + "\n")

    def log(self, msg):
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")


# -------------------------
# 环境检查
# -------------------------
def check_environment(train_dataset, val_dataset, test_dataset, batch_size=1, logger=None):
    msg = []
    msg.append("=" * 50)
    msg.append("训练前环境检查")
    msg.append(f"训练集样本数: {len(train_dataset)}")
    msg.append(f"验证集样本数: {len(val_dataset)}")
    msg.append(f"测试集样本数: {len(test_dataset)}")

    def label_stats(dataset):
        labels = [int(data.y.item()) for data in dataset]
        return labels.count(0), labels.count(1)

    t0, t1 = label_stats(train_dataset)
    v0, v1 = label_stats(val_dataset)
    s0, s1 = label_stats(test_dataset)

    msg.append(f"训练集标签分布: 0={t0}, 1={t1}")
    msg.append(f"验证集标签分布: 0={v0}, 1={v1}")
    msg.append(f"测试集标签分布: 0={s0}, 1={s1}")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        msg.append(f"GPU: {device_name}")
        msg.append(f"总显存: {total_mem:.2f} GB, 已分配: {allocated:.2f} GB, 已保留: {reserved:.2f} GB")
    else:
        mem = psutil.virtual_memory()
        msg.append("未检测到 GPU，使用 CPU 训练")
        msg.append(f"CPU 内存: {mem.total / 1024 ** 3:.2f} GB, 已使用: {mem.used / 1024 ** 3:.2f} GB")

    msg.append(f"batch_size={batch_size}")
    msg.append("=" * 50)
    for line in msg:
        if logger:
            logger.log(line)
        else:
            print(line)


# -------------------------
# 训练与验证函数
# -------------------------
scaler = GradScaler()

def train_one_epoch(model, loader, criterion, optimizer, device, logger, scaler=None, print_every=10, max_grad_norm=10.0):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    accum_steps = 8
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        with autocast(enabled=(scaler is not None)):
            out, snp_scores = model(
                batch.x, batch.snp_edge_index, batch.snp_edge_attr, batch.edge_alpha,
                subject_id=batch.subject_id, return_gate=True
            )
        
            loss_cls = criterion(out, batch.y)
        
            if snp_scores is not None:
        
                sparsity_loss = snp_scores.mean()
        
                entropy_loss = -(snp_scores * torch.log(snp_scores + 1e-8)).sum(dim=-1).mean()
        
            else:
                sparsity_loss = torch.tensor(0.0, device=out.device)
                entropy_loss = torch.tensor(0.0, device=out.device)
        
            loss = loss_cls + 0.001 * sparsity_loss + 0.005 * entropy_loss
        if torch.isnan(loss) or torch.isinf(loss):
            if logger:
                logger.log(f"Batch {batch_idx} loss=nan或inf, 跳过该batch")
            continue
        if scaler is not None:
            scaler.scale(loss / accum_steps).backward()
        else:
            (loss / accum_steps).backward()
        if (batch_idx + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
        if batch_idx % print_every == 0 and logger:
            logger.log(f"Batch {batch_idx} loss: {loss.item():.4f}, acc: {(pred == batch.y).float().mean():.4f}")
        del batch, out, loss
        torch.cuda.empty_cache()
    return total_loss / total, correct / total
    
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _ = model(
                batch.x,
                batch.snp_edge_index,
                batch.snp_edge_attr,
                batch.edge_alpha,
                return_gate=True
            )
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs

            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            label = batch.y.cpu().numpy()

            y_true.extend(label)
            y_pred.extend(pred)
            y_prob.extend(prob)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sen = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)

    return total_loss / len(loader.dataset), acc, auc, sen, spe


# -------------------------
# 主函数
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "no_PEEL", "no_propagation", "only_V2V"])
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "/root/autodl-tmp/checkpoint/new_train/42"
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, "train_0.6.log"))

    corr_threshold = 0.6
    full_dataset = BrainSNP_Dataset(
        snp_dir="/root/autodl-tmp/data/SNP_bnn",
        brain_feature_dir="/root/autodl-tmp/data/brain_feature",
        corr_threshold=corr_threshold,
        split="all"
    )

    # -------------------------
    # 7:1:2 划分
    # -------------------------
    labels = np.array([data.y.item() for data in full_dataset])
    train_idx, temp_idx = train_test_split(np.arange(len(labels)), test_size=0.3, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=2 / 3, stratify=labels[temp_idx], random_state=42)

    train_set = torch.utils.data.Subset(full_dataset, train_idx)
    val_set = torch.utils.data.Subset(full_dataset, val_idx)
    test_set = torch.utils.data.Subset(full_dataset, test_idx)

    check_environment(train_set, val_set, test_set, batch_size=1, logger=logger)

    # -------------------------
    # WeightedRandomSampler
    # -------------------------
    train_labels = np.array([train_set[i].y.item() for i in range(len(train_set))])
    num_pos = (train_labels == 1).sum()
    num_neg = (train_labels == 0).sum()
    weight_per_class = [len(train_set) / (2 * num_neg), len(train_set) / (2 * num_pos)]
    sample_weights = np.array([weight_per_class[label] for label in train_labels])
    #sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # -------------------------seed
    # 训练循环
    # -------------------------
    results_summary = []
    for max_hop in [3]:
        logger.log(f"\n========== 0.6开始训练 max_hop={max_hop} ==========")
        model = BrainSNP_GNN(
            node_feature_dim=3,
            edge_feature_dim=30,
            hidden_dim=64,
            num_classes=2,
            max_hops=max_hop,
            mode=args.mode
        ).to(device)

        # -------------------------
        # CrossEntropyLoss + 类别权重
        # -------------------------
        class_weight = torch.tensor(
            [num_neg / (num_pos + num_neg), num_pos / (num_pos + num_neg)],
            device=device,
            dtype=torch.float
        )
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        # -------------------------
        # LR调度器 (可选)
        # -------------------------
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )

        train_loader = DataLoader(train_set, batch_size=1, 
                                  collate_fn=Batch.from_data_list, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                collate_fn=Batch.from_data_list, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                                 collate_fn=Batch.from_data_list, num_workers=0, pin_memory=True)

        best_acc = 0.0
        best_metrics = (0, 0, 0, 0)
        best_model_path = os.path.join(save_dir, f"0.6best_hop{max_hop}.pth")

        patience = 20
        no_improve_epochs = 0
        max_epochs = 100

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, logger, scaler)
            val_loss, val_acc, val_auc, val_sen, val_spe = evaluate(model, val_loader, criterion, device)

            scheduler.step(val_acc)

            logger.log(f"[Epoch {epoch:02d}] Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} || "
                       f"Val ACC={val_acc:.4f}, AUC={val_auc:.4f}, SEN={val_sen:.4f}, SPE={val_spe:.4f}")

            if (val_acc > best_acc) or (val_acc == best_acc and val_auc > best_metrics[0]):
                best_acc = val_acc
                best_metrics = (val_auc, val_acc, val_sen, val_spe)
                torch.save(model.state_dict(), best_model_path)
                logger.log(f"保存最佳模型 Epoch {epoch}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                logger.log(f"连续 {patience} 轮无提升，提前停止训练（早停）")
                break

        model.load_state_dict(torch.load(best_model_path))
        _, test_acc, test_auc, test_sen, test_spe = evaluate(model, test_loader, criterion, device)
        logger.log(f"测试集结果 | ACC={test_acc:.4f}, AUC={test_auc:.4f}, SEN={test_sen:.4f}, SPE={test_spe:.4f}")

        results_summary.append((max_hop, *best_metrics, test_acc, test_auc, test_sen, test_spe))

        del model, optimizer, train_loader, val_loader, test_loader, criterion, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    result_file = os.path.join(save_dir, "0.6_results_summary.txt")
    with open(result_file, "w") as f:
        f.write("max_hop\tVal_AUC\tVal_ACC\tVal_SEN\tVal_SPE\tTest_ACC\tTest_AUC\tTest_SEN\tTest_SPE\n")
        for r in results_summary:
            f.write("\t".join(map(str, r)) + "\n")

    logger.log(f"\n所有训练完成，结果已保存至 {result_file}")


if __name__ == "__main__":
    main()
