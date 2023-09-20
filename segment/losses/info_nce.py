import torch.nn.functional as F
import torch

'''
now_feat：原始的特征
p_feat:正样本邻居的特征均值
n_feats：负样本邻居的特征
'''
def info_nce_loss(now_feat,p_feat,n_feats,temperature = 0.1):
    # BDHW BDHW BNDHW

    cos_sim_p = F.cosine_similarity(now_feat,p_feat) / temperature

    # 计算余弦相似度
    cos_similarities = F.cosine_similarity(now_feat.unsqueeze(1), n_feats, dim=2) / temperature
    cos_sim_n = torch.logsumexp(cos_similarities, dim=1)

    nll = -cos_sim_p + cos_sim_n

    nll = nll.mean()

    return nll

def pixel_info_nce_loss(now_feat,p_feat,n_feats,temperature = 0.1):
    # (1,num,dim),(1,num,dim),(25,num,dim)
    # num:是当前参与计算的num数目
    # dim:是当前参与计算特征的维度
    # 25:代表当前anchor有至少25个负样本

    # 筛选非0
    # 计算每个样本的元素之和，如果和不为0，则该样本不全为0
    sum_per_sample = torch.sum(torch.abs(n_feats), dim=(1, 2))
    # 找到不全为0的样本的索引
    non_zero_indices = torch.nonzero(sum_per_sample != 0).squeeze()
    # 根据非零索引筛选样本
    filtered_n_feats = n_feats[non_zero_indices]

    # 标准化版本
    now_feat = F.normalize(now_feat,dim=2)
    p_feat = F.normalize(p_feat,dim=2)
    filtered_n_feats = F.normalize(filtered_n_feats,dim=2)

    cos_sim_p = F.cosine_similarity(now_feat,p_feat) / temperature

    cos_similarities = F.cosine_similarity(now_feat, filtered_n_feats, dim=0) / temperature

    cos_sim_n = torch.logsumexp(cos_similarities, dim=0)

    nll = -cos_sim_p + cos_sim_n

    nll = nll.mean()

    return nll

def cross_nagetive_pixel_info_nce_loss(now_feat,p_feat,n_feats,temperature = 0.1):
    # (1,num,dim),(1,num,dim),(25,num,dim)
    # 计算的时候再放入gpu

    cos_sim_p = F.cosine_similarity(now_feat,p_feat) / temperature

    # 计算余弦相似度
    # dim=1貌似不太合理，因为这样就会导致触发广播机制，实际上还是一个样本一个样本的配对计算，没有达到跨minibatch
    b,num,dim = n_feats.size()
    cross_minibatch_n_feats = n_feats.unsqueeze(dim=0).repeat(num,1,1,1)
    cross_minibatch_n_feats = cross_minibatch_n_feats.reshape(-1,num,dim)
    cos_similarities = F.cosine_similarity(now_feat, n_feats, dim=0) / temperature
    cos_sim_n = torch.logsumexp(cos_similarities, dim=0)

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
    print(nll)