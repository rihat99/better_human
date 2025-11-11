import torch

from ..utils.lie import *


def compute_Jacobian(model, adj):

    batch_size, seq_length, nj, _, _ = adj.shape # (B, T, J, 6, 6)


    # j = 0 → keep all 6 columns
    J0 = adj[..., 0, :, :]                     # (B, T, 6, 6)

    # j > 0 → keep columns 3:6
    Jrest = adj[..., 1:, :, 3:6]               # (B, T, J-1, 6, 3)

    # Pack columns as [J0 | J1..J_{J-1}] → (B, T, 6, 6 + 3*(J-1))
    Jrest = Jrest.permute(0, 1, 3, 2, 4).reshape(batch_size, seq_length, 6, -1)
    J_full = torch.cat((J0, Jrest), dim=-1) # (B, T, 6, nv)

    J = J_full.unsqueeze(2).repeat(1, 1, model.nj, 1, 1)  # (B, T, J, 6, nv)
    J = J * model.J_mapping.view(1, 1, model.nj, 1, model.nv)  # (B, T, J, 6, nv)

    return J  # (B, T, 6, nv)

def get_joints_Jacobian(J, joints_world, adj_inv, reference_frame:str = "WORLD"):

    assert reference_frame in ["WORLD", "LOCAL", "LOCAL_WORLD_ALIGNED"]

    batch_size, seq_length, nj, _ = joints_world.shape

    if reference_frame == "WORLD":
        return J  # (B, T, J, 6, nv)
    
    elif reference_frame == "LOCAL":
        return adj_inv @ J
    
    else:  # LOCAL_WORLD_ALIGNED
        H = torch.zeros(batch_size, seq_length, nj, 6, 6, device=J.device, dtype=J.dtype)
        H[..., :3, :3] = torch.eye(3, device=J.device)
        H[..., 3:, 3:] = torch.eye(3, device=J.device)
        H[..., :3, 3:] = pp.vec2skew(-joints_world.translation())

        return H @ J
    
def compute_Jacobian_derivative(model, adj, joint_local_velocities):

    batch_size, seq_length, nj, _, _ = adj.shape # (B, T, J, 6, 6)

    V = (adj @ joint_local_velocities.unsqueeze(-1)).squeeze(-1)  # (B, T, J, 6)

    dJ_ = pp.lietensor.operation.se3_adj(V) @ adj  # (B, T, J, 6, 6)

    # j = 0 → keep all 6 columns
    dJ0 = dJ_[..., 0, :, :]                     # (B, T, 6, 6)
    # j > 0 → keep columns 3:6
    dJrest = dJ_[..., 1:, :, 3:6]               # (B, T, J-1, 6, 3)
    # Pack columns as [J0 | J1..J_{J-1}] → (B, T, 6, 6 + 3*(J-1))
    dJrest = dJrest.permute(0, 1, 3, 2, 4).reshape(batch_size, seq_length, 6, -1)
    dJ_full = torch.cat((dJ0, dJrest), dim=-1) # (B, T, 6, nv)

    dJ = dJ_full.unsqueeze(2).repeat(1, 1, model.nj, 1, 1)  # (B, T, J, 6, nv)
    dJ = dJ * model.J_mapping.view(1, 1, model.nj, 1, model.nv)  # (B, T, J, 6, nv)

    return dJ  # (B, T, J, 6, nv)

def get_joints_Jacobian_derivative(J, dJ, joints_world, adj_inv, joint_world_velocities, reference_frame:str = "WORLD"):

    assert reference_frame in ["WORLD", "LOCAL", "LOCAL_WORLD_ALIGNED"]

    batch_size, seq_length, nj, _ = joints_world.shape

    if reference_frame == "WORLD":
        return dJ  # (B, T, J, 6, nv)
    
    elif reference_frame == "LOCAL":
        return adj_inv @ (dJ - pp.lietensor.operation.se3_adj(joint_world_velocities) @ J)

    else:  # LOCAL_WORLD_ALIGNED
        raise NotImplementedError("LOCAL_WORLD_ALIGNED frame is not implemented for Jacobian derivative.")
    

    
    