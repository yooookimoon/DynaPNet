# dataset.py
import os
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import ast
import random

class BrainSNP_Dataset(Dataset):
    def __init__(self, snp_dir="/root/autodl-tmp/data/SNP_bnn",
                       brain_feature_dir="/root/autodl-tmp/data/brain_feature",
                       label_file="/root/autodl-tmp/data/label-CNAD.txt",
                       corr_threshold=0.6,
                       transform=None,
                       split="train",
                       seed=42):
        super().__init__()
        self.snp_dir = snp_dir
        self.brain_feature_dir = brain_feature_dir
        self.labels = self._load_labels(label_file)
        all_subjects = sorted(os.listdir(snp_dir))
        self.corr_threshold = corr_threshold
        self.transform = transform
        self.region_names = self._load_region_names()
        print('corr_threshold=',self.corr_threshold)

        # ------------- 新增：缓存目录（CHANGED） -------------
        # 缓存保存在 brain_feature_dir/cache 下，文件名包含阈值，便于区分不同阈值的缓存
        self.cache_dir = os.path.join(self.brain_feature_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        # --------------------------------------------------

        # 数据集划分
        train_ids, temp_ids = train_test_split(
            all_subjects,
            test_size=0.3,
            random_state=seed
        )
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=2 / 3,
            random_state=seed
        )

        if split == "train":
            selected_ids = train_ids
        elif split == "val":
            selected_ids = val_ids
        elif split == "test":
            selected_ids = test_ids
        elif split == "all":
            selected_ids = all_subjects
        else:
            raise ValueError(f"split 必须是 'train'/'val'/'test/all'，但得到了 {split}")

        if split == "train":
            pos_ids = [sid for sid in selected_ids if self.labels[sid] == 1]
            neg_ids = [sid for sid in selected_ids if self.labels[sid] == 0]

            random.seed(seed)
            if len(neg_ids) > 200:
                neg_ids = random.sample(neg_ids, 200)

            self.subject_ids = pos_ids + neg_ids
            random.shuffle(self.subject_ids)  # 打乱
        else:
            self.subject_ids = selected_ids

    def _load_labels(self, path):
        labels = {}
        with open(path,'r') as f:
            for line in f:
                parts = line.strip().split()
                labels[parts[0]] = int(parts[1])
        return labels
        
    def _load_region_names(self):
        example_subj = sorted(os.listdir(self.brain_feature_dir))[0]
        names_path = os.path.join(self.brain_feature_dir, example_subj, "names.csv")
        names = pd.read_csv(names_path, header=None).values.squeeze().tolist()
        return names

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):
        subject_id = self.subject_ids[idx]

        # ------------- 新增：缓存读取（CHANGED） -------------
        # 缓存文件名包含 corr_threshold，避免不同阈值混淆
        cache_path = os.path.join(self.cache_dir, f"{subject_id}_th{self.corr_threshold:.2f}.pt")
        if os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, map_location='cpu')
                # 如果需要在 GPU 上使用，DataLoader/训练流程会 .to(device)
                return data
            except Exception as e:
                # 若缓存损坏或不兼容，则忽略并重建
                print(f"Warning: load cache failed for {cache_path}, rebuild. Error: {e}")
        # --------------------------------------------------

        # Step 1: Load node features
        node_feat = self._load_brain_features(subject_id)  # [num_nodes,3]
        num_nodes = node_feat.shape[0]

        # Step 2: Load SNP edges
        snp_feat = self._load_snp_edges(subject_id, num_nodes)  # [num_nodes,num_nodes,30,3]


        # Step 4: Build edge_index & edge_attr (只保留相关边)
        snp_edge_index, snp_edge_attr, edge_alpha = \
            self._build_edges(node_feat, snp_feat)

        # Step 5: Label
        y = torch.tensor(self.labels[subject_id]).long()

        # Step 6: Return PyG Data object
        data = Data(x=torch.tensor(node_feat).float(),
                    snp_edge_index=snp_edge_index,
                    snp_edge_attr=snp_edge_attr,
                    edge_alpha=edge_alpha,
                    y=y)
        data.subject_id = subject_id
        data.region_names = self.region_names

        # ------------- 新增：写入缓存（CHANGED） -------------
        try:
            torch.save(data, cache_path)
        except Exception as e:
            # 缓存写失败不阻塞训练，只打印警告
            print(f"Warning: save cache failed for {cache_path}. Error: {e}")
        # --------------------------------------------------

        return data

    # -------------------------
    # Helper functions
    # -------------------------
    def _load_brain_features(self, subject_id):
        subj_dir = os.path.join(self.brain_feature_dir, subject_id)
        vcsf = pd.read_csv(os.path.join(subj_dir,"vcsf.csv"), header=None).values
        vgm  = pd.read_csv(os.path.join(subj_dir,"vgm.csv"), header=None).values
        vwm  = pd.read_csv(os.path.join(subj_dir,"vwm.csv"), header=None).values
        node_feat = np.stack([vcsf.squeeze(), vgm.squeeze(), vwm.squeeze()], axis=-1)  # [num_nodes,3]
        return node_feat.astype(np.float32)

    def _load_snp_edges(self, subject_id, num_nodes):
        snp_dir = os.path.join(self.snp_dir, subject_id)
    
        snp_list = []
    
        for file in sorted(os.listdir(snp_dir)):
            file_path = os.path.join(snp_dir, file)
    
            # 只处理文件
            if not os.path.isfile(file_path):
                continue
    
            # 只处理txt
            if not file.endswith(".txt"):
                continue
    
            with open(file_path, "r") as f:
                line = f.readline().strip()
                if line:
                    arr = ast.literal_eval(line)
                    snp_list.append(arr)
    
        snp_array = np.array(snp_list, dtype=np.float32)
    
        #print("SNP shape:", snp_array.shape)
    
        # broadcast
        snp_feat = np.zeros((num_nodes, num_nodes, 30, 3), dtype=np.float32)
    
        for i in range(num_nodes):
              for j in range(i + 1, num_nodes): 
                snp_feat[i, j] = snp_array
    
        return snp_feat        

    def _build_edges(self, node_feat, snp_edge_feat,  threshold=None):
        if threshold is None:
            threshold = self.corr_threshold

        num_nodes = node_feat.shape[0]
        snp_edge_index, snp_edge_attr = [], []
        edge_alpha = []

        # 只保留相关性大于阈值的边，按需加载 edge_attr
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if i == j:
                    continue
                corr = np.corrcoef(node_feat[i], node_feat[j])[0, 1]
                if corr > threshold:
                    snp_edge_index.append([i, j])
                    snp_edge_attr.append(snp_edge_feat[i, j])
                    edge_alpha.append([corr])

        snp_edge_index = torch.tensor(snp_edge_index).T.long() if snp_edge_index else torch.empty((2,0), dtype=torch.long)
        snp_edge_attr = torch.tensor(np.stack(snp_edge_attr), dtype=torch.float32) if snp_edge_attr else torch.empty((0,30,3), dtype=torch.float32)
        edge_alpha = torch.tensor(np.stack(edge_alpha), dtype=torch.float32) if edge_alpha else torch.empty((0,1), dtype=torch.float32)

        return snp_edge_index, snp_edge_attr, edge_alpha
