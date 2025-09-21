import torch
import pypose as pp
import viser
import numpy as np
import json
from importlib import resources


from abc import abstractmethod
from ..core.humanoid import Humanoid
from dataclasses import dataclass


@dataclass(frozen=True)
class SMPLOutputs:
    vertices: torch.Tensor  # (B, V, 3)
    joints_world: pp.LieTensor  # (B, J, 7)
    joints_parent: pp.LieTensor  # (B, J, 7)


class SMPLBase(Humanoid):
    """
    Abstract base class for SMPL-family models.

    This class extends the base Humanoid with parameters and methods
    specific to the Skinned Multi-Person Linear (SMPL) model family,
    such as shape (betas) and pose (pose) parameters.
    """
    def __init__(self):
        super().__init__()

        # Define core SMPL parameters
        # These would be initialized based on the loaded model data
        # self.betas = torch.nn.Parameter(torch.zeros(1, self.num_betas, device=self.device))
        # self.global_orient = torch.nn.Parameter(torch.zeros(1, 3, device=self.device)) # Axis-angle
        # self.body_pose = torch.nn.Parameter(torch.zeros(1, self.num_joints * 3, device=self.device)) # Axis-angle


    def _load_model_base_data(self, model_data: dict = None, model_config: str = None):

        # load config as class attributes
        with resources.files('better_human.smpl.config').joinpath(model_config).open('r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(self, key, value)

        with resources.files('better_human.smpl.config').joinpath(self.vertices_segmentation_type).open('r') as f:
            self.vertices_segmentation = json.load(f)

        # Register model parameters as buffers
        self.register_buffer('vertices_template', torch.tensor(model_data['v_template'], dtype=torch.float32)) # (V, 3)
        
        # Faces are not used in computation but are essential for visualization
        self.register_buffer('faces', torch.tensor(model_data['f'].astype(np.int64), dtype=torch.long)) # (F, 3)

        # Shape blend shapes
        shape_blending = torch.tensor(model_data['shapedirs'][:, :, :self.num_betas], dtype=torch.float32)
        self.register_buffer('shape_blending', shape_blending)   # (V, 3, num_betas)

        # Pose blend shapes
        # Original shape is (V, 3, (J-1)*9). Reshape to (V*3, (J-1)*9)
        pose_blending = torch.tensor(model_data['posedirs'], dtype=torch.float32).reshape(-1, (self.num_joints-1)*9).T
        # We need ( (J-1)*9, V*3 ) for matmul, so we transpose
        self.register_buffer('pose_blending', pose_blending) # ( (J-1)*9, V*3 )

        # Joint regressor
        self.register_buffer('joint_regressor', torch.tensor(model_data['J_regressor'], dtype=torch.float32)) # (J, V)

        # LBS weights
        self.register_buffer('lbs_weights', torch.tensor(model_data['weights'], dtype=torch.float32)) # (V, J)

        # Kinematic tree
        self.parent_tree = model_data['kintree_table']


    def _load_model_specific_data(self, model_path: str):
        """
        Loads the specific SMPL model data (e.g., templates, blend shapes)
        from a file.
        """
        
        self.num_joints = 0
        self.num_vertices = 0
        self.num_faces = 0
        self.num_betas = 0
        self.joint_names = []
        self.frame_names = []
        self.frames_vertex_ids = []

    def forward(self, betas: torch.Tensor, body_pose: pp.LieTensor, global_transform: pp.LieTensor, **kwargs) -> dict:
        """
        Runs the full forward pass for the SMPL model.

        """

        # 1. Shape deformation and joint locations
        neutral_vertices, neutral_joints = self.forward_shape(betas)

        # 2. Pose deformation
        vertices_blended = self.deform_shape(body_pose, neutral_vertices, betas) # (B, 6890, 3)

        # 3. Compute global joint transformations
        world_transforms, parent_transforms = self.forward_skeleton(global_transform, body_pose, neutral_joints)

        # 4. Linear Blend Skinning (optimized)
        vertices_posed = self.linear_blend_skinning(vertices_blended, world_transforms, neutral_joints)

        return SMPLOutputs(
            vertices=vertices_posed, # (B, 6890, 3)
            joints_world=pp.mat2SE3(world_transforms),  # (B, 24, 7)
            joints_parent=pp.mat2SE3(parent_transforms) # (B, 24, 7)
        )

    def forward_skeleton(self, root_transform: pp.LieTensor, body_pose: pp.LieTensor, neutral_joints: torch.Tensor) -> torch.Tensor:
        """
        Computes the global joint transformations (FK) for the skeleton.
        This is the core kinematic chain logic, shared by all SMPL models.

        Args:
            base_pose (pp.LieTensor): SE3 Global transformation of the root joint (B, 7).
            joints_pose (pp.LieTensor): SO3 Local transformations for each joint (B, J, 4).
            J (torch.Tensor): Joint locations from the shaped mesh (B, J, 3).

        Returns:
            torch.Tensor: Global transformation matrices G for skinning (B, J, 4, 4).
        """
        batch_size = neutral_joints.shape[0]

        # Create relative transformation matrices for each joint
        parent_transforms = torch.zeros((batch_size, self.num_joints, 4, 4), device=self.vertices_template.device) +\
              torch.eye(4, device=self.vertices_template.device) # (B, J, 4, 4)

        parent_transforms[:, 0] = root_transform.matrix()

        parent_transforms[:, 1:, :3, :3] = body_pose.matrix()

        parent_transforms[:, 0, :3, 3] += neutral_joints[:, 0]  # Root joint translation
        parent_transforms[:, 1:, :3, 3] = neutral_joints[:, 1:] - neutral_joints[:, self.parent_tree[0, 1:]]  # Relative translations

        world_transforms = parent_transforms.clone()

        # Iterate through the kinematic tree to compute the global transformations
        for i in range(1, self.num_joints):
            parent_idx = self.parent_tree[0, i]
            world_transforms[:, i] = torch.matmul(world_transforms[:, parent_idx], parent_transforms[:, i])

        return world_transforms, parent_transforms

    def forward_shape(self, betas: torch.Tensor) -> torch.Tensor:
        """
        Computes the shape-deformed vertices given shape parameters.

        Args:
            betas (torch.Tensor): Shape parameters (B, num_betas).
        Returns:
            torch.Tensor: Shape-deformed vertices (B, V, 3).
        """
        # 1. Shape deformation
        neutral_vertices = self.vertices_template + torch.einsum('vij, bj -> bvi', self.shape_blending, betas) # (B, V, 3)

        # 2. Joint locations
        neutral_joints = torch.einsum('jv, bvi -> bji', self.joint_regressor, neutral_vertices) # (B, J, 3)

        return neutral_vertices, neutral_joints

    def deform_shape(self, body_pose: pp.LieTensor, neutral_vertices, betas=None) -> torch.Tensor:
        batch_size = body_pose.shape[0]

        # Convert body_pose (B, J-1, 4) to rotation matrices (B, J-1, 3, 3)
        pose_mats = body_pose.matrix()
        pose_features = pose_mats - torch.eye(3, device=self.vertices_template.device)  # (B, J-1, 3, 3)
        pose_features = pose_features.reshape(-1, (self.num_joints-1)*9) # (B, (J-1)*9)
        pose_offsets = (pose_features @ self.pose_blending).view(batch_size, -1, 3)  # (B, V, 3)

        vertices_blended = neutral_vertices + pose_offsets # (B, V, 3)

        return vertices_blended

    def linear_blend_skinning(self, vertices: torch.Tensor, world_transforms: torch.Tensor, neutral_joints: torch.Tensor) -> torch.Tensor:
        """
        Applies Linear Blend Skinning (LBS) to deform the mesh vertices
        based on the global joint transformations.

        Args:
            vertices (torch.Tensor): The shape-deformed vertices (B, V, 3).
            world_transforms (torch.Tensor): Global joint transformations (B, J, 4, 4).
            neutral_joints (torch.Tensor): Joint locations from the shaped mesh (B, J, 3).

        Returns:
            torch.Tensor: The posed vertices after LBS (B, V, 3).
        """
        batch_size = vertices.shape[0]

        vertices_delta = torch.ones((batch_size, self.num_vertices, self.num_joints, 4), device=self.vertices_template.device) # (B, V, J, 4)
        vertices_delta[:, :, :, :3] = vertices[:, :, None, :] - neutral_joints[:, None, :, :]  # (B, V, J, 4)

        vertices_posed = torch.einsum(
            'bjxy, vj, bvjy -> bvx', 
            world_transforms[:, :, :3, :],
            self.lbs_weights,
            vertices_delta
        ) # (B, V, 3)

        return vertices_posed

    def forward_kinematics(self, *args, **kwargs) -> dict:
        """
        Computes the forward kinematics of the humanoid.

        Returns:
            dict: A dictionary containing joint locations, transformations,
                  and potentially other kinematic information.
        """
        
        return self.forward(*args, **kwargs)


    # SMPL does not typically use IK in the same way as a robot arm,
    # but you could implement an optimization-based version here.
    # def inverse_kinematics(self, *args, **kwargs):
        # raise NotImplementedError("IK for SMPL models is typically an optimization problem.")
