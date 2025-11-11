import torch
import pypose as pp

import numpy as np    
import trimesh


from .base import SMPLBase, SMPLOutputs


def compute_Mass_Inertia(model: SMPLBase, betas: torch.Tensor, density_system: str = "Custom") -> dict:
    
    batch_size = betas.shape[0]
    device = betas.device

    neutral_vertices, neutral_joints = model.forward_shape(betas)  #  (B, J, 3)(B, V, 3)
    all_vertices = torch.cat([neutral_vertices, neutral_joints], dim=1)  # (B, 6890 + 24, 3)
    all_vertices = all_vertices.detach().cpu().numpy()  # Convert to numpy for trimesh
    # faces = model.faces.detach().cpu().numpy().astype(np.int32)  # (F, 3)

    results = {
        "mass": torch.zeros((batch_size, model.nj), device=device),
        "inertia": torch.zeros((batch_size, model.nj, 3, 3), device=device),
        "com": torch.zeros((batch_size, model.nj, 3), device=device)
    }

    for b in range(batch_size):
        for j in range(model.nj):

            body_faces = model.segmentation_3d['body_faces'][j]

            mesh = trimesh.Trimesh(vertices=all_vertices[b], faces=body_faces)
            mesh.fix_normals()
            mesh.density = model.body_densities[density_system][str(j)]  # Set density for the mesh

            mass = mesh.mass
            # joint_frame = np.eye(4)
            # joint_frame[:3, 3] = output.joints[0].detach().cpu().numpy()[i]
            # inertia = mesh.moment_inertia_frame(joint_frame)
            inertia = mesh.moment_inertia

            com = mesh.center_mass - neutral_joints[b, j].detach().cpu().numpy()  # Center of mass relative to the joint

            results["mass"][b, j] = mass
            results["inertia"][b, j] = torch.tensor(inertia, device=device, dtype=torch.float32)
            results["com"][b, j] = torch.tensor(com, device=device, dtype=torch.float32)

    return results