import torch
from abc import ABC, abstractmethod

class Humanoid(ABC, torch.nn.Module):
    """
    Abstract base class for all humanoid models.

    This class defines the core interface for humanoid robots, including
    kinematics, state representation, and visualization.
    All specific humanoid implementations (e.g., SMPL, URDF-based)
    should inherit from this class.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward_kinematics(self, *args, **kwargs) -> dict:
        """
        Computes the forward kinematics of the humanoid.

        Returns:
            dict: A dictionary containing joint locations, transformations,
                  and potentially other kinematic information.
        """
        pass

    # @abstractmethod
    # def inverse_kinematics(self, *args, **kwargs):
    #     """
    #     Computes the inverse kinematics to find joint configurations for
    #     given end-effector targets.
    #     """
    #     pass

    # @property
    # @abstractmethod
    # def joint_names(self) -> list[str]:
    #     """
    #     Returns a list of the names of all joints in the model.
    #     """
    #     pass

    # @property
    # @abstractmethod
    # def parent_tree(self) -> torch.Tensor:
    #     """
    #     Returns the parent array for the kinematic tree.
    #     parent_tree[i] is the parent of joint i.
    #     """
    #     pass
        
    # @abstractmethod
    # def visualize(self, *args, **kwargs):
    #     """
    #     Provides a method for visualizing the humanoid model's current state.
    #     """
    #     pass