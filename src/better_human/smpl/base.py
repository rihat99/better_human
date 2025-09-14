import torch
import pypose as pp
from abc import abstractmethod
from ..core.humanoid import Humanoid
from dataclasses import dataclass

class SMPLBase(Humanoid):
    """
    Abstract base class for SMPL-family models.

    This class extends the base Humanoid with parameters and methods
    specific to the Skinned Multi-Person Linear (SMPL) model family,
    such as shape (betas) and pose (pose) parameters.
    """
    def __init__(self, model_path: str, gender: str = 'neutral', device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_data(model_path)

        self.gender = gender

        # Define core SMPL parameters
        # These would be initialized based on the loaded model data
        # self.betas = torch.nn.Parameter(torch.zeros(1, self.num_betas, device=self.device))
        # self.global_orient = torch.nn.Parameter(torch.zeros(1, 3, device=self.device)) # Axis-angle
        # self.body_pose = torch.nn.Parameter(torch.zeros(1, self.num_joints * 3, device=self.device)) # Axis-angle

        self.num_joints = 24  # Default for SMPL, can be overridden in child classes

    @abstractmethod
    def _load_model_data(self, model_path: str):
        """
        Loads the specific SMPL model data (e.g., templates, blend shapes)
        from a file.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> dict:
        """
        The main forward pass for the model.
        Must be implemented by child classes (e.g., SMPL, SMPLH).
        """
        pass

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
        parent_transforms = torch.zeros((batch_size, self.num_joints, 4, 4), device=self.device) + torch.eye(4, device=self.device)

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

    @abstractmethod
    def forward_shape(self, *args, **kwargs) -> torch.Tensor:
        """
        Computes the shaped mesh vertices after skeleton is posed.
        Must be implemented by child classes.
        """
        pass

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
        
        
    def visualize(self, *args, **kwargs):
        print("Visualizing SMPL model...")
        # Your visualization logic using libraries like PyVista or Open3D
        pass


@dataclass(frozen=True)
class SMPLOutputs:
    vertices: torch.Tensor  # (B, 6890, 3)
    joints_world: pp.LieTensor  # (B, 24, 7)
    joints_parent: pp.LieTensor  # (B, 24, 7)
