import numpy as np
import viser
import viser.transforms as tf
import torch
import pypose as pp

from .base import SMPLOutputs, SMPLBase


def xyzw_to_wxyz(xyzw: torch.Tensor) -> torch.Tensor:
    """
    Converts quaternions from (x, y, z, w) format to (w, x, y, z) format.
    Args:
        xyzw (torch.Tensor): Input tensor of shape (..., 4) in (x, y, z, w) format.
    Returns:
        torch.Tensor: Output tensor of shape (..., 4) in (w, x, y, z) format.
    """
    assert xyzw.shape[-1] == 4, "Input tensor must have last dimension of size 4."
    wxyz = torch.zeros_like(xyzw)
    wxyz[..., 0] = xyzw[..., 3]
    wxyz[..., 1:] = xyzw[..., :3]
    return wxyz

def visualize_single(
        server: viser._viser.ViserServer, 
        model: SMPLBase,
        output: SMPLOutputs, 
        prefix: str = "/smpl",
        show_joints: bool = True,
        show_mesh: bool = True,
        show_skeleton: bool = True,
        mesh_color: tuple = (10, 220, 30),
        joint_color: tuple = (255, 0, 0),
        joint_radius: float = 0.02,
        wireframe=True,
        ):
    
    """
    Visualizes the SMPL output using the provided VisER server.
    Args:
        server (viser._viser.ViserServer): The VisER server instance for visualization.
        output (dict): The output dictionary from the forward pass containing 'vertices' and 'joints'.
        prefix (str): Prefix for naming the visualized elements.
        show_joints (bool): Whether to visualize joints.
        show_mesh (bool): Whether to visualize the mesh.
        show_skeleton (bool): Whether to visualize the skeleton.
        mesh_color (tuple): Color for the mesh visualization.
        wireframe (bool): Whether to render the mesh in wireframe mode.
    """

    if show_mesh:
        server.scene.add_mesh_simple(
            name=f"{prefix}/mesh",
            vertices=output.vertices[0].detach().cpu().numpy(),
            faces=model.faces.cpu().numpy().astype(np.int32),
            color=mesh_color,
            opacity=1.0,
            wireframe=wireframe,
        )
        if wireframe:
            server.scene.add_mesh_simple(
                name=f"{prefix}/mesh_opacity",
                vertices=output.vertices[0].detach().cpu().numpy(),
                faces=model.faces.cpu().numpy().astype(np.int32),
                color=mesh_color,
                opacity=0.4,
            )

    if show_joints:
        for i in range(model.num_joints):
            server.scene.add_icosphere(
                name=f"/smpl/joints/{i}",
                radius=joint_radius,
                color=joint_color,
                position=output.joints_world.tensor()[0, i, :3].detach().cpu().numpy()
            )


class SMPLSequenceVisualizer:
    def __init__(
            self, 
            server: viser._viser.ViserServer, 
            model: SMPLBase, 
            betas: torch.Tensor,
            prefix: str = "/smpl_sequence",
            show_joints: bool = True,
            show_mesh: bool = True,
            show_skeleton: bool = True,
            mesh_color: tuple = (10, 220, 30),
            joint_color: tuple = (255, 0, 0),
            joint_radius: float = 0.02,
            wireframe=True,
            opacity=1.0,
            ):
        
        self.server = server
        self.model = model
        self.betas = betas
        self.prefix = prefix
        self.show_joints = show_joints

        neutral_vertices, neutral_joints = self.model.forward_shape(betas)

        self.mesh_handle = self.server.scene.add_mesh_skinned(
            name=f"{self.prefix}/mesh",
            vertices=neutral_vertices[0].detach().cpu().numpy(),
            faces=self.model.faces.detach().cpu().numpy().astype(np.int32),
            bone_wxyzs=tf.SO3.identity(batch_axes=(model.num_joints,)).wxyz,
            bone_positions=neutral_joints[0].detach().cpu().numpy(),
            skin_weights=self.model.lbs_weights.detach().cpu().numpy(),
            color=mesh_color,
            opacity=opacity,
            wireframe=wireframe,
        )

        if show_joints:
            self.joint_handles = []
            for i in range(model.num_joints):
                joint_handle = self.server.scene.add_icosphere(
                    name=f"{self.prefix}/joints/{i}",
                    radius=joint_radius,
                    color=joint_color,
                    position=neutral_joints[0, i].detach().cpu().numpy(),
                    cast_shadow=False,
                    receive_shadow=False,
                )
                self.joint_handles.append(joint_handle)

    def update(self, joints_world: pp.LieTensor):
        
        wxyz = xyzw_to_wxyz(joints_world.tensor()[:, 3:7])

        for i in range(self.model.num_joints):
            self.mesh_handle.bones[i].wxyz = wxyz[i].detach().cpu().numpy()
            self.mesh_handle.bones[i].position = joints_world.tensor()[i, :3].detach().cpu().numpy()

            if self.show_joints:
                self.joint_handles[i].position = joints_world.tensor()[i, :3].detach().cpu().numpy()


