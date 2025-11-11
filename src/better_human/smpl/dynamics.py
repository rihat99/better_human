import torch

from .base import SMPLBase, SMPLOutputs

from ..utils.lie import *


def RNEA(
        model: SMPLBase, 
        neutral_joints: torch.Tensor,
        body_inertia_matrices: torch.Tensor,
        q: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        gravity: torch.Tensor,
    ):
    """
    Computes inverse dynamics using the Recursive Newton-Euler Algorithm.
    Note: All joint configurations (q), velocities (v), and accelerations (a),
    as well as external forces (f_ext) must be provided in the Data instance.
    Forward kinematics must be computed before calling this function.

    Parameters:
        model (Model): The robot model.
        data (Data): The data representation of the robot model.
    """

    base_transform = pp.identity_SE3(q.shape[0], device=q.device)  # (B, 7)
    base_velocity = pp.identity_se3(q.shape[0], device=q.device)  # (B, 6)
    base_acceleration = -gravity
    base_acceleration = pp.se3(base_acceleration.unsqueeze(0).repeat(q.shape[0], 1))  # (B, 6)

    joint_transforms = {}
    joint_velocities = {}
    joint_accelerations = {}
    joint_internal_forces = {}
    joint_offsets = {}


    # Forward Pass

    for joint_id in model.kinematic_order:

        inertia_matrix = body_inertia_matrices[0, joint_id, :, :]  # (6, 6)

        joint_q = q[:, model.joint_q_idx[joint_id][0]: model.joint_q_idx[joint_id][1]]  # (B, nq_j)
        joint_v = v[:, model.joint_v_idx[joint_id][0]: model.joint_v_idx[joint_id][1]]  # (B, nv_j)
        joint_a = a[:, model.joint_v_idx[joint_id][0]: model.joint_v_idx[joint_id][1]]  # (B, na_j)

        if joint_id == 0:
            # Base joint, use the base transform, velocity, and acceleration
            parent_transform = base_transform
            parent_velocity = base_velocity
            parent_acceleration = base_acceleration

            joint_offset = pp.identity_SE3(q.shape[0], device=q.device)  # (B, 7)
            joint_q = pp.SE3(joint_q)
            joint_v = pp.se3(joint_v)
            joint_a = pp.se3(joint_a)

        else:
            # For other joints, use the transform, velocity, and acceleration of the parent joint
            parent_id = model.parent_tree[0][joint_id]
            parent_transform = joint_transforms[parent_id]
            parent_velocity = joint_velocities[parent_id]
            parent_acceleration = joint_accelerations[parent_id]

            joint_offset_t = neutral_joints[:, joint_id] - neutral_joints[:, parent_id]  # (B, 3)
            joint_offset = pp.identity_SE3(q.shape[0], device=q.device)
            joint_offset[:, :3] = joint_offset_t

            joint_q = SO3_2_SE3(pp.SO3(joint_q))
            joint_v = so3_2_se3(pp.so3(joint_v))
            joint_a = so3_2_se3(pp.so3(joint_a))

        ###########################################################

        parent_2_joint = joint_offset @ joint_q

        joint_transform = parent_transform @ parent_2_joint
        
        joint_velocity = pp.Adj(parent_2_joint.Inv(), parent_velocity) + joint_v

        joint_acceleration = pp.Adj(parent_2_joint.Inv(), parent_acceleration) + \
            pp.se3((pp.lietensor.operation.se3_adj(joint_velocity.tensor()) @ joint_v.tensor().unsqueeze(-1)).squeeze(-1)) + \
            joint_a
            
        ############################################################

        joint_internal_force = \
            (inertia_matrix.unsqueeze(0) @ joint_acceleration.tensor().unsqueeze(-1)).squeeze(-1) + \
            (se3_adj_dual(joint_velocity.tensor()) @ (inertia_matrix.unsqueeze(0) @ joint_velocity.tensor().unsqueeze(-1))).squeeze(-1)
        
        joint_internal_forces[joint_id] = joint_internal_force
        joint_transforms[joint_id] = joint_transform
        joint_velocities[joint_id] = joint_velocity
        joint_accelerations[joint_id] = joint_acceleration
        joint_offsets[joint_id] = joint_offset

        # print(inertia_matrix.shape)

    # Backward Pass
    for joint_id in reversed(model.kinematic_order):
        if joint_id == 0:
            continue  # Skip the base joint

        parent_id = model.parent_tree[0][joint_id]

        joint_q = q[:, model.joint_q_idx[joint_id][0]: model.joint_q_idx[joint_id][1]]  # (B, nq_j)
        joint_q = SO3_2_SE3(pp.SO3(joint_q))

        A = (joint_offsets[joint_id] @ joint_q).Inv()
        A_adj = SE3_Adj(A.tensor()).transpose(-2, -1)

        child_internal_force = joint_internal_forces[joint_id]
        
        joint_internal_forces[parent_id] += \
            (A_adj @ child_internal_force.unsqueeze(-1)).squeeze(-1)

    torques = []

    for joint_id in range(model.nj):

        if joint_id == 0:
            torques.append(joint_internal_forces[joint_id])

        else:
            torques.append(joint_internal_forces[joint_id][:, 3:])

    joint_torques = torch.cat(torques, dim=-1)

    return joint_torques  # (B, nv)