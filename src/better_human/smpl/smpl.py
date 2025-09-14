import pickle
import numpy as np
import viser

import torch
import pypose as pp
# ... (other imports)
from .base import SMPLBase, SMPLOutputs
# from ..utils.lbs import linear_blend_skinning

class SMPL(SMPLBase):
    """
    A concrete implementation of the original SMPL model.
    """
    def __init__(self, model_path: str, gender: str = 'neutral', num_betas: int = 10, **kwargs):
        self.num_betas = num_betas
        # `super().__init__` will call `_load_model_data` internally
        super().__init__(model_path=model_path, gender=gender, **kwargs)

    def _load_model_data(self, model_path: str):
        """
        Loads the SMPL model data from a .pkl file.
        """
        smpl_data = np.load(model_path, allow_pickle=True)

        # Register model parameters as buffers
        self.register_buffer('vertices_template', torch.tensor(smpl_data['vertices_template'], dtype=torch.float32)) # (6890, 3)
        
        # Faces are not used in computation but are essential for visualization
        self.register_buffer('faces', torch.tensor(smpl_data['faces'].astype(np.int64), dtype=torch.long)) # (13776, 3)

        # Shape blend shapes
        shape_blending = torch.tensor(smpl_data['shape_blending'][:, :, :self.num_betas], dtype=torch.float32)
        self.register_buffer('shape_blending', shape_blending)   # (6890, 3, num_betas)

        # Pose blend shapes
        # Original shape is (6890, 3, 207). Reshape to (6890*3, 207)
        pose_blending = torch.tensor(smpl_data['pose_blending'], dtype=torch.float32).reshape(-1, 23*9).T
        # We need ( (J-1)*9, V*3 ) for matmul, so we transpose
        self.register_buffer('pose_blending', pose_blending) # (207, 6890*3)

        # Joint regressor
        self.register_buffer('joint_regressor', torch.tensor(smpl_data['joint_regressor'], dtype=torch.float32)) # (24, 6890)

        # LBS weights
        self.register_buffer('lbs_weights', torch.tensor(smpl_data['weights'], dtype=torch.float32)) # (6890, 24)

        # Kinematic tree
        self.parent_tree = smpl_data['kintree_table']

        # Define number of joints based on loaded data
        self.num_joints = 24

        self.joint_links = [
            [0, 1], [0, 2], [1, 4], [4, 7], [7, 10], [2, 5], [5, 8], [8, 11],
            [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], 
            [9, 13], [13, 16], [16, 18], [18, 20], [20, 22],
            [9, 14], [14, 17], [17, 19], [19, 21], [21, 23]
        ]

    def forward(self, betas: torch.Tensor, body_pose: pp.LieTensor, global_transform: pp.LieTensor, **kwargs) -> dict:
        """
        Runs the full forward pass for the SMPL model.

        """

        batch_size = betas.shape[0]

        # 1. Shape deformation and joint locations
        neutral_vertices, neutral_joints = self.forward_shape(betas)

        # 2. Pose deformation
        vertices_blended = self.blend_shape(body_pose, neutral_vertices) # (B, 6890, 3)

        # 3. Compute global joint transformations
        world_transforms, parent_transforms = self.forward_skeleton(global_transform, body_pose, neutral_joints)

        # 4. Linear Blend Skinning (optimized)
        vertices_delta = torch.ones((batch_size, 6890, self.num_joints, 4), device=self.device) # (B, 6890, 24, 4)
        vertices_delta[:, :, :, :3] = vertices_blended[:, :, None, :] - neutral_joints[:, None, :, :]  # (B, 6890, 24, 4) 

        vertices_posed = torch.einsum(
            'bjxy, vj, bvjy -> bvx', 
            world_transforms[:, :, :3, :],
            self.lbs_weights,
            vertices_delta
        ) # (B, 6890, 3)

        return SMPLOutputs(
            vertices=vertices_posed, # (B, 6890, 3)
            joints_world=pp.mat2SE3(world_transforms),  # (B, 24, 7)
            joints_parent=pp.mat2SE3(parent_transforms) # (B, 24, 7)
        )

    def forward_shape(self, betas: torch.Tensor) -> torch.Tensor:
        """
        Computes the shape-deformed vertices given shape parameters.

        Args:
            betas (torch.Tensor): Shape parameters (B, num_betas).
        Returns:
            torch.Tensor: Shape-deformed vertices (B, 6890, 3).
        """
        # 1. Shape deformation
        neutral_vertices = self.vertices_template + torch.einsum('vij, bj -> bvi', self.shape_blending, betas) # (B, 6890, 3)

        # 2. Joint locations
        neutral_joints = torch.einsum('jv, bvi -> bji', self.joint_regressor, neutral_vertices) # (B, 24, 3)

        return neutral_vertices, neutral_joints
    
    def blend_shape(self, body_pose: pp.LieTensor, neutral_vertices) -> torch.Tensor:
        batch_size = body_pose.shape[0]

        # Convert body_pose (B, 23, 4) to rotation matrices (B, 23, 3, 3)
        pose_mats = body_pose.matrix()
        pose_features = pose_mats - torch.eye(3, device=self.device)
        pose_features = pose_features.reshape(-1, 23*9) # (B, 207)
        # pose_offsets = torch.einsum('bi, ij -> bj', pose_features, self.pose_blending) # (B, 6890*3)
        pose_offsets = (pose_features @ self.pose_blending).view(batch_size, -1, 3)  # (B, 6890, 3)

        vertices_blended = neutral_vertices + pose_offsets # (B, 6890, 3)

        return vertices_blended

    @property
    def joint_names(self) -> list[str]:
        # Standard SMPL joint names
        return [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
            'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
            'left_hand', 'right_hand'
        ]
    

    def visualize(
            self, 
            server: viser._viser.ViserServer, 
            output: SMPLOutputs, 
            prefix: str = "/smpl",
            show_joints: bool = True,
            show_mesh: bool = True,
            show_skeleton: bool = True,
            mesh_color: tuple = (10, 220, 30),
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
                vertices=output.vertices[0].cpu().numpy(),
                faces=self.faces.cpu().numpy().astype(np.int32),
                color=mesh_color,
                opacity=1.0,
                wireframe=wireframe,
            )
            if wireframe:
                server.scene.add_mesh_simple(
                    name=f"{prefix}/mesh_opacity",
                    vertices=output.vertices[0].cpu().numpy(),
                    faces=self.faces.cpu().numpy().astype(np.int32),
                    color=mesh_color,
                    opacity=0.4,
                )

        if show_joints:
            for i in range(self.num_joints):
                server.scene.add_icosphere(
                    name=f"/smpl/joints/{i}",
                    radius=0.02,
                    color=(255, 0, 0),
                    position=output.joints_world.tensor()[0, i, :3].cpu().numpy()
                )

        # if show_skeleton:
        #     points = output["joints"][0].cpu().numpy()[self.joint_links]

        #     server.scene.add_line_segments(
        #         name=f"{prefix}/skeleton",
        #         points=points,
        #         colors=(255, 255, 255),
        #         line_width=20,
        #     )

        