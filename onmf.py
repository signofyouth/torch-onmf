import torch
from tqdm import tqdm

def onmf(M, k, rank=5, max_iter=10000):
    m, n = M.size()
    if rank > min(m, n): rank = min(m, n)
    
    W = low_rank_nnpca(M, k, rank=rank, max_iter=max_iter)
    #for i in range(k):
    #    support   = torch.where(W[:, col] != 0)
    #    val, _, _ = torch.svd(M[support, :])
    #    val = val[:, 0]
    #    torch.where(W[:, col]!=0, val, W)
    H = W.t() @ M
    return W, H

def low_rank_nnpca(M, k, rank=5, max_iter=1000):
    U, S, V = torch.svd(M)
    
    US = U[:,:rank] @ torch.diag_embed(S[:rank])
    M_bar = US @ V[:,:rank].t()
    
    sgns = binarize(torch.Tensor([i for i in range(2**k)]).int(), k)
    optval = -float('inf')
    W      = None
    for iter in tqdm(range(max_iter)):
        C     = sphere_sample_cartesian(rank, k)
        A     = US @ C
        W_hat = local_opt_w(A, sgns)
        val   = torch.norm(M_bar.t() @ W_hat)
        if optval < val:
            optval = val
            W = W_hat
    
    assert W is not None
    return W

def local_opt_w(A, sgns):
    optw = None
    optv = -float('inf')
    for s in sgns:
        A_prime = A @ torch.diag_embed(s).float()
        Is = [[] for i in range(A.size(1))]
        for i in range(A.size(0)):
            j_star = torch.argmax(A_prime[i])
            if A[i, j_star] >=0: Is[j_star].append(i)
            #print(j_star)
        
        W = torch.zeros_like(A_prime)
        for j in range(A.size(1)):
            if len(Is[j]) == 0: continue
            mask   = torch.zeros(A.size(0)).scatter_(0, torch.Tensor(Is[j]).long(), 1)
            W[:,j] = mask * A[:,j] / torch.norm(mask*A[:,j])
            #print(torch.norm(W[:,j]))
        v = torch.trace(torch.matmul(W.t(), A))
        if optv < v:
            optw = W
    #print(optw)
    assert optw is not None
    return optw

def binarize(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    ret  =  x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().int()
    ret[ret==0] = -1
    return ret

def sphere_sample_cartesian(rank, pow):
    C = torch.randn(rank, pow)
    C = C / torch.norm(C, dim=0, keepdim=True)
    return C


if __name__=='__main__':
    data = torch.eye(5, 5)
    W, H = onmf(data, 6, rank=5, max_iter=1000)
    
    print(W @ H)
