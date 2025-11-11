import torch
import pypose as pp



def SE3_Adj(X):
    Adj = torch.zeros((X.shape[:-1]+(6, 6)), device=X.device, dtype=X.dtype, requires_grad=False)
    t, q = X[..., :3], X[..., 3:]
    R3x3 = pp.LieTensor(q, ltype=pp.SO3_type).matrix()
    tx = pp.vec2skew(t)
    Adj[..., :3, :3] = R3x3
    Adj[..., :3, 3:] = torch.matmul(tx, R3x3)
    Adj[..., 3:, 3:] = R3x3
    return Adj



def LieDifference(q1: pp.LieTensor, q2: pp.LieTensor, frame: str="LOCAL") -> pp.LieTensor:

    """
    Compute the difference between two quaternions in SO(3).

    Args:
        q1: (..., 4) LieTensor of quaternions
        q2: (..., 4) LieTensor of quaternions
        frame: "LOCAL" or "GLOBAL" - frame of reference for the difference

    Returns:
        delta_q: (..., 4) LieTensor of quaternion differences
    """
    assert q1.shape == q2.shape
    assert frame in ["LOCAL", "GLOBAL"]

    if frame == "GLOBAL":
        return (q2 * q1.Inv()).Log()
    else:  # LOCAL
        return (q1.Inv() * q2).Log()
    

def SO3_2_SE3(q: pp.LieTensor) -> pp.LieTensor:
    
    if q.ltype == pp.SE3_type:
        return q

    elif q.ltype == pp.SO3_type:
        q_ = pp.identity_SE3(*q.shape[:-1], device=q.device)
        q_[..., 3:] = q
        return q_
    
def so3_2_se3(x: pp.LieTensor) -> pp.LieTensor:
    
    if x.ltype == pp.se3_type:
        return x

    elif x.ltype == pp.so3_type:
        x_ = pp.identity_se3(*x.shape[:-1], device=x.device)
        x_[..., 3:] = x
        return x_
    
def se3_adj_dual(x):
    adj_6x6 = torch.zeros((x.shape[:-1]+(6, 6)), device=x.device, dtype=x.dtype, requires_grad=False)
    Phi = pp.vec2skew(x[..., 3:])
    adj_6x6[..., :3, :3] = Phi
    adj_6x6[..., 3:, :3] = pp.vec2skew(x[..., :3])
    adj_6x6[..., 3:, 3:] = Phi
    return adj_6x6