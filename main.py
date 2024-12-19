"""
Polynomial Surface Visualization Assignment
----------------------------------------
This program visualizes polynomial surfaces and their differential geometry properties.
You will implement methods to:
1. Generate and sample polynomial surfaces
2. Calculate differential geometry quantities using PyTorch's autograd
3. Visualize the results using Polyscope

Before starting:
1. Make sure you understand the concept of autograd in PyTorch
2. Review basic differential geometry (first/second fundamental forms)
3. Understand how to work with PyTorch tensors and their shapes
"""

import torch
import polyscope as ps
import numpy as np
from typing import Tuple
from scipy.spatial import Delaunay

def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Normalize vectors to unit length."""
    norms = torch.norm(vectors, dim=1, keepdim=True)
    return torch.where(norms > 0, vectors / norms, vectors)

class PolynomialSurface:
    def __init__(self, coefficients: list, order: int, grid_size: int = 30):
        """
        Initialize a polynomial surface.

        Args:
            coefficients: List of polynomial coefficients
                For order 2, represents: [x², y², xy]
                For order 3, represents: [x², y², xy, x³, y³, x²y, xy²]
            order: Order of the polynomial
            grid_size: Number of points per dimension for visualization
        """
        self.coefficients = torch.tensor(coefficients, requires_grad=True)
        self.order = order
        self.grid_size = grid_size

    def generate_grid(self, range_min: float = -1.0, range_max: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a regular grid of points.

        Implementation steps:
        1. Create linearly spaced points for x and y axes:
           x_linspace = torch.linspace(range_min, range_max, self.grid_size, requires_grad=True)
           y_linspace = torch.linspace(range_min, range_max, self.grid_size, requires_grad=True)

        2. Create 2D grid:
           x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing='ij')

        3. Flatten the grids:
           x = x_grid.flatten()
           y = y_grid.flatten()

        IMPORTANT:
        - Make sure requires_grad=True is set for autograd to work
        - Use indexing='ij' in meshgrid to get correct orientation
        - Verify tensor shapes after each step

        Expected tensor shapes:
        - x_linspace, y_linspace: (grid_size,)
        - x_grid, y_grid: (grid_size, grid_size)
        - Final x, y: (grid_size², 1)
        """
        # Your code here
        pass

    def evaluate_surface(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the polynomial surface at given points.

        Implementation steps:
        1. Generate power pairs for polynomial terms:
           For order 2, you need:
           pairs = [(2,0), (0,2), (1,1)]  # x², y², xy

           For order 3, you need:
           pairs = [(2,0), (0,2), (1,1), (3,0), (0,3), (2,1), (1,2)]

        2. Initialize output:
           z = torch.zeros_like(x)

        3. Evaluate each term:
           for c, (px, py) in zip(self.coefficients, pairs):
               term = c * (x ** px) * (y ** py)
               z += term

        IMPORTANT:
        - Double check that number of coefficients matches number of terms
        - Verify tensor shape is maintained after operations
        - Make sure you're using element-wise operations

        Expected tensor shapes:
        - Input x, y: (N, 1)
        - Output z: (N, 1)
        """
        # Your code here
        pass

    def compute_derivatives(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute first and second derivatives using autograd.

        Implementation steps:
        1. Compute first derivatives:
           dz_dx, dz_dy = torch.autograd.grad(
               outputs=z.sum(),  # sum() is needed to get scalar output
               inputs=[x, y],
               create_graph=True  # Needed for second derivatives
           )

        2. Compute second derivatives from dz_dx:
           d2z_dx2, d2z_dxdy = torch.autograd.grad(
               outputs=dz_dx.sum(),
               inputs=[x, y],
               create_graph=True
           )

        3. Compute second derivatives from dz_dy:
           _, d2z_dy2 = torch.autograd.grad(
               outputs=dz_dy.sum(),
               inputs=[x, y],
               create_graph=True
           )

        IMPORTANT:
        - Always use create_graph=True for all grad() calls
        - Verify that none of the derivatives are None
        - Check tensor shapes after each computation

        Expected tensor shapes:
        - All derivatives should be (N, 1)
        """
        # Your code here
        pass

    def compute_shape_operator(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor,
                             d2z_dx2: torch.Tensor, d2z_dxdy: torch.Tensor, d2z_dy2: torch.Tensor) -> torch.Tensor:
        """
        Compute the shape operator of the surface.

        Implementation steps:
        1. Compute first fundamental form coefficients:
           E = 1 + dz_dx ** 2
           F = dz_dx * dz_dy
           G = 1 + dz_dy ** 2

        2. Compute second fundamental form coefficients:
           denominator = torch.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)
           L = d2z_dx2 / denominator
           M = d2z_dxdy / denominator
           N = d2z_dy2 / denominator

        3. Compute shape operator matrix elements:
           det = E * G - F ** 2
           shape_operator_11 = (G * L - F * M) / det
           shape_operator_12 = (G * M - F * N) / det
           shape_operator_21 = (E * M - F * L) / det
           shape_operator_22 = (E * N - F * M) / det

        4. Build matrix using torch.stack:
           shape_operator = torch.stack([
               torch.stack([shape_operator_11, shape_operator_12], dim=-1),
               torch.stack([shape_operator_21, shape_operator_22], dim=-1)
           ], dim=-2)

        Expected tensor shapes:
        - All input derivatives: (N, 1)
        - Output shape_operator: (N, 2, 2)
        """
        # Your code here
        pass

    def compute_principal_curvatures(self, shape_operator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute principal curvatures and directions.

        Implementation steps:
        1. Compute eigenvalues and eigenvectors:
           eigenvalues, eigenvectors = torch.linalg.eig(shape_operator)

        2. Extract real components:
           k1 = eigenvalues.real[..., 0]  # First principal curvature
           k2 = eigenvalues.real[..., 1]  # Second principal curvature
           v1 = eigenvectors.real[..., 0]  # First principal direction
           v2 = eigenvectors.real[..., 1]  # Second principal direction

        IMPORTANT:
        - eigenvalues/vectors might be complex, use .real
        - The principal directions obtained here are in the 2D parameter domain
        - Principal directions become orthogonal only after mapping to 3D using the Jacobian

        Expected tensor shapes:
        - Input shape_operator: (N, 2, 2)
        - Output k1, k2: (N, 1)
        - Output v1, v2: (N, 2)
        """
        # Your code here
        pass

    def compute_curvatures(self, k1: torch.Tensor, k2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and Gaussian curvatures.

        Implementation steps:
        1. Mean curvature (average of principal curvatures):
           H = (k1 + k2) / 2

        2. Gaussian curvature (product of principal curvatures):
           K = k1 * k2

        Expected tensor shapes:
        - Input k1, k2: (N, 1)
        - Output H, K: (N, 1)
        """
        # Your code here
        pass

    def compute_jacobian(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian of the surface parameterization.

        Implementation steps:
        1. Create row vectors using torch.stack:

           First row: [1, 0, dz_dx]
           row1 = torch.stack([
               torch.ones_like(dz_dx),
               torch.zeros_like(dz_dx),
               dz_dx
           ], dim=-1)

           Second row: [0, 1, dz_dy]
           Third row: [dz_dx, dz_dy, 1]

        2. Stack rows to create Jacobian:
           jacobian = torch.stack([row1, row2, row3], dim=-2)

        Expected tensor shapes:
        - Input dz_dx, dz_dy: (N, 1)
        - Output jacobian: (N, 3, 3)
        """
        # Your code here
        pass

    def map_to_3d(self, jacobian: torch.Tensor, v1_2d: torch.Tensor, v2_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map 2D vector fields to 3D using the Jacobian.

        Implementation steps:
        1. Extend 2D vectors to 3D:
           v1_2d_ext = torch.cat([v1_2d, torch.zeros_like(v1_2d[:, :1])], dim=1)
           v2_2d_ext = torch.cat([v2_2d, torch.zeros_like(v2_2d[:, :1])], dim=1)

        2. Apply Jacobian transformation:
           v1_3d = torch.einsum('...ij,...j->...i', jacobian, v1_2d_ext)
           v2_3d = torch.einsum('...ij,...j->...i', jacobian, v2_2d_ext)

        IMPORTANT:
        - This transformation maps the principal directions to the surface
        - After this mapping, the principal directions become orthogonal on the surface

        Expected tensor shapes:
        - Input jacobian: (N, 3, 3)
        - Input v1_2d, v2_2d: (N, 2)
        - v*_2d_ext: (N, 3)
        - Output v1_3d, v2_3d: (N, 3)
        """
        # Your code here
        pass

    def visualize(self):
        # Generate grid points
        x, y = self.generate_grid()
        z = self.evaluate_surface(x, y)

        # Compute all differential quantities
        derivatives = self.compute_derivatives(x, y, z)
        dz_dx, dz_dy, d2z_dx2, d2z_dxdy, d2z_dy2 = derivatives
        shape_operator = self.compute_shape_operator(*derivatives)
        k1, k2, v1_2d, v2_2d = self.compute_principal_curvatures(shape_operator)
        H, K = self.compute_curvatures(k1, k2)

        # Compute Jacobian and map vectors to 3D
        jacobian = self.compute_jacobian(dz_dx, dz_dy)
        v1_3d, v2_3d = self.map_to_3d(jacobian, v1_2d, v2_2d)

        # Prepare points for visualization
        points_3d = torch.stack([x, y, z], dim=1)
        points_2d = torch.stack([x, y, torch.zeros_like(z)], dim=1)

        # Offset 2D points for visualization
        offset = torch.tensor([2.5, 0, 0])
        points_2d = points_2d + offset

        # Convert to numpy and compute faces
        points_3d_np = points_3d.detach().numpy()
        points_2d_np = points_2d.detach().numpy()
        faces = Delaunay(points_3d_np[:, :2]).simplices

        # Initialize polyscope
        ps.init()
        ps.set_up_dir("z_up")

        # Register and visualize 3D surface mesh
        mesh_3d = ps.register_surface_mesh(
            "3D Surface - Mesh",
            vertices=points_3d_np,
            faces=faces,
            smooth_shade=True,
            enabled=True
        )

        # Register and visualize 3D point cloud
        cloud_3d = ps.register_point_cloud(
            "3D Surface - Points",
            points=points_3d_np,
            radius=0.002,
            enabled=True
        )

        # Register and visualize 2D domain mesh (hidden by default)
        mesh_2d = ps.register_surface_mesh(
            "2D Domain - Mesh",
            vertices=points_2d_np,
            faces=faces,
            smooth_shade=True,
            enabled=False
        )
        mesh_2d.set_color((0.8, 0.8, 0.8))

        # Register and visualize 2D point cloud
        cloud_2d = ps.register_point_cloud(
            "2D Domain - Points",
            points=points_2d_np,
            radius=0.002,
            enabled=True
        )
        cloud_2d.set_color((0.8, 0.8, 0.8))

        # Add scalar quantities (only to 3D surface)
        for structure in [mesh_3d, cloud_3d]:
            structure.add_scalar_quantity("Mean Curvature", H.detach().numpy(), enabled=True, cmap='coolwarm')
            structure.add_scalar_quantity("Gaussian Curvature", K.detach().numpy(), enabled=False, cmap='coolwarm')
            structure.add_scalar_quantity("k1", k1.detach().numpy(), enabled=False, cmap='coolwarm')
            structure.add_scalar_quantity("k2", k2.detach().numpy(), enabled=False, cmap='coolwarm')

        # Scale factor for vector fields
        scale = 0.1

        # Add principal directions to 3D surface
        v1_3d_norm = normalize_vectors(v1_3d).detach().numpy() * scale
        v2_3d_norm = normalize_vectors(v2_3d).detach().numpy() * scale

        # Add vectors to both mesh and point cloud for 3D surface
        for structure in [mesh_3d, cloud_3d]:
            structure.add_vector_quantity(
                "Principal Direction 1",
                v1_3d_norm.astype(np.float32),
                enabled=True,
                color=(1.0, 0.0, 0.0),
                vectortype="ambient"
            )
            structure.add_vector_quantity(
                "Principal Direction 2",
                v2_3d_norm.astype(np.float32),
                enabled=True,
                color=(0.0, 0.0, 1.0),
                vectortype="ambient"
            )

        # Add vectors to 2D domain
        # Transform 2D vectors to world space (adding offset)
        v1_2d_world = torch.cat([v1_2d, torch.zeros_like(v1_2d[:, :1])], dim=1)
        v2_2d_world = torch.cat([v2_2d, torch.zeros_like(v2_2d[:, :1])], dim=1)

        # Normalize and scale
        v1_2d_norm = normalize_vectors(v1_2d_world).detach().numpy() * scale
        v2_2d_norm = normalize_vectors(v2_2d_world).detach().numpy() * scale

        # Add vectors to both mesh and point cloud for 2D domain
        for structure in [mesh_2d, cloud_2d]:
            structure.add_vector_quantity(
                "Principal Direction 1 (2D)",
                v1_2d_norm.astype(np.float32),
                enabled=True,
                color=(1.0, 0.0, 0.0),
                vectortype="ambient"
            )
            structure.add_vector_quantity(
                "Principal Direction 2 (2D)",
                v2_2d_norm.astype(np.float32),
                enabled=True,
                color=(0.0, 0.0, 1.0),
                vectortype="ambient"
            )

def main():
    """
    Example usage and testing.

    Try these test cases:
    1. Simple paraboloid: z = x² + y²
       coefficients = [1.0, 1.0, 0.0]  # [x², y², xy]

    2. Hyperbolic paraboloid (saddle): z = x² - y²
       coefficients = [1.0, -1.0, 0.0]  # [x², y², xy]

    3. More complex surface: z = x² + y² - 0.5xy
       coefficients = [1.0, 1.0, -0.5]  # [x², y², xy]
    """
    # Example polynomial: z = x² + y² - 0.5xy
    coefficients = [1.0, 1.0, -0.5]  # Coefficients for [x², y², xy]
    order = 2

    # Create and visualize surface
    surface = PolynomialSurface(coefficients, order)
    surface.visualize()
    ps.show()

if __name__ == "__main__":
    main()