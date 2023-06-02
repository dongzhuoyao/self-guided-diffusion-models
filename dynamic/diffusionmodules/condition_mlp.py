from einops import rearrange
import torch
import torch.nn as nn

def get_cond_mlp(cond_dim, cond_divide):
        module_list = []
        for i in range(len(cond_divide)-1):
            module_list.extend(
                [nn.Linear(cond_dim//cond_divide[i], cond_dim // cond_divide[i+1]),
                nn.SiLU(),
                ])
        module_list = module_list[:-1]#remove last nn.ReLU
        ################### 
        model = nn.Sequential(
                *module_list)
        return model, cond_dim//cond_divide[-1]



class Label_Cluster_Joint_Embed(nn.Module):

    def __init__(self,label_dim, cluster_dim, embed_dim):
        super().__init__()
        self.label_embed = nn.Sequential(nn.Linear(label_dim, embed_dim),nn.SiLU())
        self.cluster_embed = nn.Sequential(nn.Linear(cluster_dim, embed_dim),nn.SiLU())
        
    def forward(self, label, cluster):
        _label_feat = self.label_embed(label)
        _cluster_feat = self.cluster_embed(cluster)
        r = torch.cat([_label_feat, _cluster_feat], -1)
        return r 

class Feature_Tokenize(nn.Module):
    def __init__(self,feat_dim, token_num, cond_dim):
        super().__init__()
        self.to_tokens = nn.Sequential(
            nn.Linear(feat_dim, cond_dim * token_num),
            rearrange('b (r d) -> b r d', r = token_num)
        )
        
    def forward(self, feat):
        r = self.to_tokens(feat)
        return r 
