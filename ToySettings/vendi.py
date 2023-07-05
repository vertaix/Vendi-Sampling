import torch

def weight_K(K, p=None):
    if p is None:
        return K / K.shape[0]
    else:
        return K * torch.outer(torch.sqrt(p), torch.sqrt(p))

def normalize_K(K):
    d = torch.sqrt(torch.diagonal(K))
    return K / torch.outer(d, d)

def entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * torch.log(p_)).sum()
    if q == "inf":
        return -torch.log(torch.max(p))
    return torch.log((p_ ** q).sum()) / (1 - q)

def score_K(K, q=1, p=None, normalize=False):
    if normalize:
        K = normalize_K(K)
    K_ = weight_K(K, p)
    w, _ = torch.linalg.eigh(K_)
    return torch.exp(entropy_q(w, q=q))

def score(samples, k, q=1, p=None, normalize=False):

    K = k(samples)

    return score_K(K, p=p, q=q, normalize=normalize)

def log_score(samples, k, q=1, p=None, normalize=False):
    return torch.log(score(samples, k, q, p, normalize))

if __name__ == "__main__":
    samples = torch.tensor([1,2,3])
    samples1 = torch.unsqueeze(samples,0)
    samples2 = torch.unsqueeze(samples,1)
    K = 1. - torch.abs(samples1-samples2)/(torch.abs(samples1)+torch.abs(samples2))
    print(K)