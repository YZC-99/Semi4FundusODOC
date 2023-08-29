import torch.nn.functional as F
import torch

'''
now_feat：原始的特征
p_feat:正样本邻居的特征均值
n_feats：负样本邻居的特征均值
'''
def info_nce_loss(now_feat,p_feat,n_feats):
    # BDHW BDHW BNDHW
    temperature = 0.1
    cos_sim_p = F.cosine_similarity(now_feat,p_feat) / temperature

    # 计算余弦相似度
    cos_similarities = F.cosine_similarity(now_feat.unsqueeze(1), n_feats, dim=2) / temperature
    cos_sim_n = torch.logsumexp(cos_similarities, dim=1)

    nll = -cos_sim_p + cos_sim_n

    nll = nll.mean()

    return nll

# def info_nce_loss(now_feat,p_feat,n_feats):


if __name__ == '__main__':
    now_feat = torch.randn(4,512,64,64)
    p_feat = torch.randn(4,512,64,64)
    n_feats = torch.randn(4,3,512,64,64)
    cos_sim_p = F.cosine_similarity(now_feat, p_feat)
    # 计算余弦相似度
    cos_similarities = F.cosine_similarity(now_feat.unsqueeze(1), n_feats, dim=2)
    cos_sim_n = torch.logsumexp(cos_similarities, dim=1)
    nll = -cos_sim_p + cos_sim_n
    nll = nll.mean()
    # print(nll)
    cat_t = torch.cat([now_feat.unsqueeze(1),p_feat.unsqueeze(1)],dim=1)
    print(cat_t.shape)