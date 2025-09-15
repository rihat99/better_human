import pickle
import numpy as np
import viser

import torch
import pypose as pp

from .base import SMPLBase, SMPLOutputs


class MANO(SMPLBase):
    """
    A concrete implementation of the MANO hand model.
    """
    def __init__(
            self, 
            model_path: str, 
            hand_side: str = 'right', 
            use_pca: bool = True,
            num_pca_components: int = 6,
            flat_hand_mean: bool = False,
            **kwargs):
        
        self.num_betas = 10
        self.use_pca = use_pca
        self.num_pca_components = num_pca_components
        if num_pca_components == 45:
            self.use_pca = False  # Override to use full pose if 45 components are requested
        self.flat_hand_mean = flat_hand_mean

        self.hand_side = hand_side  # 'right' or 'left'
        # `super().__init__` will call `_load_model_data` internally
        super().__init__(model_path=model_path, **kwargs)

        self.num_joints = 16  # MANO has 16 joints
        self.num_vertices = 778  # MANO has 778 vertices

    def _load_model_data(self, model_path: str):
        """
        Loads the MANO model data from a .pkl file.
        """
        mano_data = np.load(model_path, allow_pickle=True)

        # Register model parameters as buffers
        self.register_buffer('vertices_template', torch.tensor(mano_data['vertices_template'], dtype=torch.float32)) # (778, 3)

        # Faces are not used in computation but are essential for visualization
        self.register_buffer('faces', torch.tensor(mano_data['faces'].astype(np.int64), dtype=torch.long)) # (1538, 3)

        # Shape blend shapes
        shape_blending = torch.tensor(mano_data['shape_blending'][:, :, :self.num_betas], dtype=torch.float32)
        self.register_buffer('shape_blending', shape_blending)   # (778, 3, 10)

        # Pose blend shapes
        # Original shape is (778, 3, 135). Reshape to (778*3, 135)
        pose_blending = torch.tensor(mano_data['pose_blending'], dtype=torch.float32).reshape(-1, 15*9).T
        # We need ( (J-1)*9, V*3 ) for matmul, so we transpose
        self.register_buffer('pose_blending', pose_blending) # (135, 778*3)

        # Joint regressor
        self.register_buffer('joint_regressor', torch.tensor(mano_data['joint_regressor'], dtype=torch.float32)) # (16, 778)

        # LBS weights
        self.register_buffer('lbs_weights', torch.tensor(mano_data['weights'], dtype=torch.float32)) # (778, 16)

        # Kinematic tree
        self.parent_tree = mano_data['kintree_table']

        # Register hand related parameters
        self.register_buffer('hands_components', torch.tensor(mano_data['hands_components'], dtype=torch.float32)) # (45, 45)
        if self.flat_hand_mean:
            self.register_buffer('hands_mean', torch.zeros((15, 3), dtype=torch.float32)) # (15, 3)
        else:
            self.register_buffer('hands_mean', torch.tensor(mano_data['hands_mean'].reshape(15, 3), dtype=torch.float32)) # (15, 3)
        self.register_buffer('hands_coeffs', torch.tensor(mano_data['hands_coeffs'], dtype=torch.float32)) # (1554, 45)

        # Define number of joints based on loaded data
        self.num_joints = 16

        # self.joint_links = [
        #     [0, 1], [0, 2], [1, 4], [4, 7], [7, 10], [2, 5], [5, 8], [8, 11],
        #     [0, 3], [3, 6], [6, 9], [9, 12], [12, 15],
        #     [9, 13], [13, 16], [16, 18], [18, 20], [20, 22],
        #     [9, 14], [14, 17], [17, 19], [19, 21], [21, 23]
        # ]

    
    def forward(self, hand_pose: torch.Tensor, global_transform: pp.LieTensor, betas: torch.Tensor = None, **kwargs) -> SMPLOutputs:

        if self.use_pca:
            hand_pose = torch.einsum('bc, cj -> bj', hand_pose, self.hands_components[:self.num_pca_components, :])  # (B, 45)
            hand_pose = pp.so3(hand_pose.reshape(-1, 15, 3)).Exp()  # (B, 15, 4)
        else:
            hand_pose = pp.SO3(hand_pose)  # (B, 15, 4)

        hand_pose = (hand_pose.Log() + self.hands_mean).Exp()

        return super().forward(betas=betas, body_pose=hand_pose, global_transform=global_transform, **kwargs)


    def forward_shape(self, betas: torch.Tensor) -> torch.Tensor:
        """
        Computes the shape-deformed vertices given shape parameters.

        Args:
            betas (torch.Tensor): Shape parameters (B, num_betas).
        Returns:
            torch.Tensor: Shape-deformed vertices (B, 6890, 3).
        """
        # 1. Shape deformation
        neutral_vertices = self.vertices_template + torch.einsum('vij, bj -> bvi', self.shape_blending, betas) # (B, 778, 3)

        # 2. Joint locations
        neutral_joints = torch.einsum('jv, bvi -> bji', self.joint_regressor, neutral_vertices) # (B, 16, 3)

        return neutral_vertices, neutral_joints
    
    def deform_shape(self, body_pose: pp.LieTensor, neutral_vertices, betas=None) -> torch.Tensor:
        batch_size = body_pose.shape[0]

        # Convert body_pose (B, 23, 4) to rotation matrices (B, 23, 3, 3)
        pose_mats = body_pose.matrix()
        pose_features = pose_mats - torch.eye(3, device=self.device)
        pose_features = pose_features.reshape(-1, 15*9) # (B, 207)
        pose_offsets = (pose_features @ self.pose_blending).view(batch_size, -1, 3)  # (B, 778, 3)

        vertices_blended = neutral_vertices + pose_offsets # (B, 778, 3)

        return vertices_blended