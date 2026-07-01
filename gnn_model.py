import torch
import torch.nn as nn
from models.edge_conv import EdgeConv
from models.edge_propagation import EdgePropagation
from models.gating import Gating
from models.layers import MLP, ResidualLayer
from models.node_update import NodeUpdate
from torch_geometric.nn import global_mean_pool 
from models.only_V2V import Only_V2V

class EdgeSeqEncoder(nn.Module):
    def __init__(self, token_dim=3, seq_len=30, out_dim=64, method='conv', attn_heads=4, dropout=0.0):
        super().__init__()
        self.method = method
        self.seq_len = seq_len
        self.token_dim = token_dim
        self.out_dim = out_dim

        if method == 'conv':
            self.conv = nn.Conv1d(in_channels=token_dim,
                                  out_channels=out_dim,
                                  kernel_size=3,
                                  padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.act = nn.ReLU()
            self.ff = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU()
            )
        elif method == 'attn':
            self.token_lin = nn.Linear(token_dim, out_dim)
            self.attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=attn_heads, dropout=dropout)
            self.act = nn.ReLU()
            self.ff = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown encoder method: {method}")

    def forward(self, x):
        if self.method == 'conv':
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = self.act(x)
            x = self.pool(x)
            x = x.squeeze(-1)
            x = self.ff(x)
            return x
        else:
            x = self.token_lin(x)
            x = x.permute(1, 0, 2)
            attn_out, _ = self.attn(x, x, x)
            attn_out = attn_out.mean(dim=0)
            attn_out = self.act(attn_out)
            attn_out = self.ff(attn_out)
            return attn_out


class BrainSNP_GNN(nn.Module):
    def __init__(self,
                 node_feature_dim,
                 edge_feature_dim=30,
                 hidden_dim=64,
                 num_classes=2,
                 max_hops=3,
                 edge_token_dim=3,
                 encoder_method='conv',
                 mode="full",
                 top_k=15
                 ):
        super().__init__()
        self.mode = mode
        self.max_hops = max_hops
        self.top_k = top_k

        print(f"选择了 {self.mode} 模式")

        # 节点编码
        self.node_encoder = MLP(node_feature_dim, hidden_dim, hidden_dim)

        # 边序列编码
        
        # EdgeConv 只接收原始 edge_attr [E,30,3]
        self.edge_conv = EdgeConv(pad_size=(32, 5))
        self.edge_embedding = nn.Linear(3, hidden_dim)

        # EdgePropagation
        self.edge_propagation = EdgePropagation(
            edge_feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            node_dim=hidden_dim,
            max_hops=self.max_hops
        )

        # Gating
        self.gating = Gating(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # 节点更新+分类输出
        self.node_update = NodeUpdate(hidden_dim=hidden_dim, num_classes=num_classes)

        self.only_v2v = Only_V2V(hidden_dim, num_classes)

        self.debug = True
        self.debug_edge_id = 0

    
    def debug_print(self, name, edge_feat):
        print(f"当前 subject: {self.current_subject}")
        if not hasattr(self, "debug") or not self.debug:
            return
        
        edge_id = getattr(self, "debug_edge_id", 0)
    
        print(f"\n==== {name} ====")
        print(f"shape: {edge_feat.shape}")
    
        try:
            print(f"edge[{edge_id}] value:\n", edge_feat[edge_id].detach().cpu())
        except:
            print("edge index out of range")
            
    def forward(self,
            nodes,
            snp_edge_index,
            snp_edge_attr,
            edge_alpha,
            *,
            subject_id=None,
            return_gate=False):

        self.current_subject = subject_id
        
        # Step 1: 节点编码
        node_feat = self.node_encoder(nodes)  # [N, hidden_dim]

        if self.mode == "no_PEEL":
            snp_edge_feat = self.edge_embedding(snp_edge_attr.float())  # [E,30,hidden_dim]
            snp_scores = self.gating(snp_edge_feat)
            snp_edge_feat = (snp_edge_feat * snp_scores.unsqueeze(-1)).sum(dim=1)
            #节点更新 + 分类输出
            updated_node_feat, out = self.node_update(node_feat, snp_edge_index, snp_edge_feat)

        elif self.mode == "only_V2V":
            updated_node_feat, out = self.only_v2v(
                node_feat,
                snp_edge_index,  
            )
            snp_scores = None
            
#注意：no V2V是在模块内部改的
        # -----------------------------
        # Full 模型
        # -----------------------------
        elif self.mode == "full":
            #self.debug_print("SNP 原始输入", snp_edge_attr)
            
            # Step 2: EdgeConv → EdgePropagation
            snp_edge_feat, _ = self.edge_conv(node_feat, snp_edge_index, snp_edge_attr.float())
            #self.debug_print("SNP after EdgeConv", snp_edge_feat)
            snp_edge_feat = self.edge_embedding(snp_edge_feat)  # [E,30,hidden_dim]
        
            snp_edge_feat = self.edge_propagation(node_feat, snp_edge_index, snp_edge_feat, alpha=edge_alpha)
            #self.debug_print("SNP after EdgePropagation", snp_edge_feat)
        
            # Step 3: gating
            snp_scores = self.gating(snp_edge_feat)  # [E, 30]

            # Step 4: Attention Pooling
            snp_edge_feat = (snp_edge_feat * snp_scores.unsqueeze(-1)).sum(dim=1)  # [E, hidden_dim]
            #self.debug_print("SNP after Gating", snp_edge_feat)
        
            # Step 5: 节点更新 + 分类输出
            updated_node_feat, out = self.node_update(node_feat, snp_edge_index, snp_edge_feat)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return out, snp_scores