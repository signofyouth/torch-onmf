import torch
import torch.nn.functional as F
from tqdm import tqdm

def onmf(M, k, rank=5, max_iter=10000):
    m, n = M.size()
    if rank > min(m, n): rank = min(m, n)
    
    W = low_rank_nnpca(M, k, rank=rank, max_iter=max_iter)
    H = W.t() @ M
    return W, H

def low_rank_nnpca(M, k, rank=5, max_iter=10000, max_bs=10):
    bs = min(max_bs, max_iter)
    
    U, S, V = torch.svd(M)
    
    US = U[:,:rank] @ torch.diag_embed(S[:rank])
    M_bar = US @ V[:,:rank].t()
    
    sgns = binarize(torch.Tensor([i for i in range(2**k)]).int(), k)
    optval = -float('inf')
    W      = None
    for iter in tqdm(range(max_iter // bs + 1)):
        C     = sphere_sample_cartesian(rank, k, bs).to(M.device)
        A     = US @ C
        W_hat = local_opt_w(A, sgns)
        val   = torch.norm(M_bar.t() @ W_hat)
        if optval < val:
            optval = val
            W = W_hat
    
    assert W is not None
    return W


### for single iteration
def _local_opt_w(A, sgns):
    device = A.device
    A_prime    = A @ torch.diag_embed(sgns).float().to(device)
    mask_jstar = torch.where(
                     (A_prime >= torch.max(A_prime, dim=2, keepdim=True)[0]) * (A >= F.relu(A)),
                     torch.Tensor([1]).to(device),
                     torch.Tensor([0]).to(device)
                 )
    W  = torch.zeros_like(A_prime)
    W  = (mask_jstar * A.unsqueeze(0)) / (torch.norm(mask_jstar * A.unsqueeze(0), dim=1, keepdim=True) + 1e-12)
    tr = (torch.matmul(W.permute(0, 2, 1), A) * torch.eye(A.size(-1)).to(device)).sum(-1).sum(-1)
    
    optW = W[torch.argmax(tr)]
    return optW

### For batch process
def local_opt_w(A, sgns):
    assert A.dim() == 3
    device = A.device
    A_prime    = A.unsqueeze(1) @ torch.diag_embed(sgns).float().to(device)
    mask_jstar = torch.where(
                     (A_prime >= torch.max(A_prime, dim=3, keepdim=True)[0]) * (A >= F.relu(A)).unsqueeze(1),
                     torch.Tensor([1]).to(device),
                     torch.Tensor([0]).to(device)
                 )
    W  = torch.zeros_like(A_prime)
    W  = (mask_jstar * A.unsqueeze(1)) / (torch.norm(mask_jstar * A.unsqueeze(1), dim=2, keepdim=True) + 1e-12)
    tr = (torch.matmul(W.permute(0, 1, 3, 2), A.unsqueeze(1)) * torch.eye(A.size(-1)).to(device)).sum(-1).sum(-1)
    
    idx = torch.argmax(tr)
    _, N = tr.size()
    optW = W[idx//N, idx%N]
    
    return optW

#def local_opt_w(A, sgns):
#    optw = None
#    optv = -float('inf')
#    for s in sgns:
#        A_prime = A @ torch.diag_embed(s).float()
#        Is = [[] for i in range(A.size(1))]
#        for i in range(A.size(0)):
#            j_star = torch.argmax(A_prime[i])
#            if A[i, j_star] >=0: Is[j_star].append(i)
#            #print(j_star)
#        
#        W = torch.zeros_like(A_prime)
#        for j in range(A.size(1)):
#            if len(Is[j]) == 0: continue
#            mask   = torch.zeros(A.size(0)).scatter_(0, torch.Tensor(Is[j]).long(), 1)
#            W[:,j] = mask * A[:,j] / torch.norm(mask*A[:,j])
#            #print(torch.norm(W[:,j]))
#        v = torch.trace(torch.matmul(W.t(), A))
#        if optv < v:
#            optw = W
#    #print(optw)
#    assert optw is not None
#    return optw

def binarize(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    ret  =  x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().int()
    ret[ret==0] = -1
    return ret

def sphere_sample_cartesian(rank, pow, bs=None):
    if bs is not None:
        C = torch.randn(bs, rank, pow)
        C = C / torch.norm(C, dim=1, keepdim=True)
    else:
        C = torch.randn(rank, pow)
        C = C / torch.norm(C, dim=0, keepdim=True)
    return C


if __name__=='__main__':
    bs = 2
    m  = 3
    k  = 2
    A = torch.randn(bs, m, k)
    sgns = binarize(torch.Tensor([i for i in range(2**k)]).int(), k)
    
    #for i in range(bs):
    #    _local_opt_w(A[i], sgns)
    #print('-----------')
    #local_opt_w(A, sgns)
    
    data = torch.eye(5, 5).cuda()
    W, H = onmf(data, 6, rank=5, max_iter=10000)
    print(W @ H)
