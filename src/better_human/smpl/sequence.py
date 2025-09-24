import torch
import pypose as pp
import numpy as np

from .base import SMPLBase, SMPLOutputs


def LieDifference(q1: pp.LieTensor, q2: pp.LieTensor, frame: str="LOCAL") -> pp.LieTensor:

    """
    Compute the difference between two quaternions in SO(3).

    Args:
        q1: (B, T, 4) LieTensor of quaternions
        q2: (B, T, 4) LieTensor of quaternions
        frame: "LOCAL" or "GLOBAL" - frame of reference for the difference

    Returns:
        delta_q: (B, T-1, 4) LieTensor of quaternion differences
    """
    assert q1.shape == q2.shape
    assert frame in ["LOCAL", "GLOBAL"]

    if frame == "GLOBAL":
        return (q2 * q1.Inv()).Log()
    else:  # LOCAL
        return (q1.Inv() * q2).Log()


class SMPLSequence:
    """
    Class for handling sequences of SMPL models over time.

    This class manages a sequence of SMPLBase instances, allowing for
    temporal operations and batch processing of multiple frames.
    """
    def __init__(self, model: SMPLBase, dt: float=1/30):
        self.model = model
        self.dt = dt

    def load_sequence(self, betas: torch.Tensor, q: torch.Tensor):
        
        self.batch_size, _ = betas.shape
        self.seq_length = q.shape[1]

        self.betas = betas  # (B, num_betas) same shape for all frames in the sequence

        self.q = q  # (B, T, nq)


    def forward(self) -> SMPLOutputs:

        outputs = self.model.forward(
            betas=self.betas.repeat_interleave(self.seq_length, dim=0),  # (B*T, num_betas)
            q=self.q.view(-1, self.model.nq)  # (B*T, nq)
        )

        outputs.vertices = outputs.vertices.view(self.batch_size, self.seq_length, -1, 3)  # (B, T, V, 3)
        outputs.joints_world = outputs.joints_world.view(self.batch_size, self.seq_length, -1, 7)  # (B, T, J, 7)
        outputs.joints_parent = outputs.joints_parent.view(self.batch_size, self.seq_length, -1, 7)  # (B, T, J, 7)

        return outputs
    
    def compute_joint_vel_acc(self):

        # free-flier base joint
        vel_beg_ff = LieDifference(
            pp.SE3(self.q[:, 0, :7]), 
            pp.SE3(self.q[:, 1, :7]), 
            frame="LOCAL"
        ).unsqueeze(1) / self.dt  # (B, 1, 6)

        vel_mid_ff = LieDifference(
            pp.SE3(self.q[:, :-2, :7]),
            pp.SE3(self.q[:, 2:, :7]),
            frame="LOCAL"
        ) / (2 * self.dt)  # (B, T-2, 6)

        vel_end_ff = LieDifference(
            pp.SE3(self.q[:, -2, :7]),
            pp.SE3(self.q[:, -1, :7]),
            frame="LOCAL"
        ).unsqueeze(1) / self.dt  # (B, 1, 6)

        vel_ff = torch.cat([vel_beg_ff, vel_mid_ff, vel_end_ff], dim=1)  # (B, T, 6)

        # spherical joints
         
        vel_beg_sph = LieDifference(
            pp.SO3(self.q[:, 0, 7:].reshape(self.batch_size, (self.model.nj-1), 4)),
            pp.SO3(self.q[:, 1, 7:].reshape(self.batch_size, (self.model.nj-1), 4)),
            frame="LOCAL"
        ).unsqueeze(1) / self.dt  # (B, 1, (J-1)*3, 4)

        vel_mid_sph = LieDifference(
            pp.SO3(self.q[:, :-2, 7:].reshape(self.batch_size, -1, (self.model.nj-1), 4)),
            pp.SO3(self.q[:, 2:, 7:].reshape(self.batch_size, -1, (self.model.nj-1), 4)),
            frame="LOCAL"
        ) / (2 * self.dt)  # (B, T-2, (J-1)*3, 4)

        vel_end_sph = LieDifference(
            pp.SO3(self.q[:, -2, 7:].reshape(self.batch_size, (self.model.nj-1), 4)),
            pp.SO3(self.q[:, -1, 7:].reshape(self.batch_size, (self.model.nj-1), 4)),
            frame="LOCAL"
        ).unsqueeze(1) / self.dt  # (B, 1, (J-1)*3, 4)

        vel_sph = torch.cat([vel_beg_sph, vel_mid_sph, vel_end_sph], dim=1)  # (B, T, (J-1)*3, 3)

        self.v = torch.cat([vel_ff, vel_sph.reshape(self.batch_size, self.seq_length, -1)], dim=-1)  # (B, T, nv)


        acc_beg_ff = (vel_ff[:, 1] - vel_ff[:, 0]).unsqueeze(1) / self.dt  # (B, 1, 6)
        acc_mid_ff = (vel_ff[:, 2:] - vel_ff[:, :-2]) / (2 * self.dt)  # (B, T-2, 6)
        acc_end_ff = (vel_ff[:, -1] - vel_ff[:, -2]).unsqueeze(1) / self.dt  # (B, 1, 6)

        acc_ff = torch.cat([acc_beg_ff, acc_mid_ff, acc_end_ff], dim=1)  # (B, T, 6)

        acc_beg_sph = (vel_sph[:, 1] - vel_sph[:, 0]).unsqueeze(1) / self.dt  # (B, 1, (J-1), 3)
        acc_mid_sph = (vel_sph[:, 2:] - vel_sph[:, :-2]) / (2 * self.dt)  # (B, T-2, (J-1), 3)
        acc_end_sph = (vel_sph[:, -1] - vel_sph[:, -2]).unsqueeze(1) / self.dt  # (B, 1, (J-1), 3)

        acc_sph = torch.cat([acc_beg_sph, acc_mid_sph, acc_end_sph], dim=1)  # (B, T, (J-1), 3)

        self.a = torch.cat([acc_ff, acc_sph.reshape(self.batch_size, self.seq_length, -1)], dim=-1)  # (B, T, nv)