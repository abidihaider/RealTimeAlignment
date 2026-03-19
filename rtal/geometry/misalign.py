import torch
import torch.nn.functional as F


class Rotator:
    """
    Modified Rodrigues' rotation formula for multi-dimensional input.
    """
    def __init__(self, n1, n2):
        """
        Args:
          - n1, n2: shape (..., 3)
        """
        # unnormalized rotation axis, shape = (..., 3)
        self.rot_axis = torch.linalg.cross(n1, n2)
        # the cosine of theta in the Rodrigues' formula, shape = (..., 1)
        self.cos_theta = (n1 * n2).sum(dim=-1, keepdim=True) 

    def __call__(self, vecs):
        """
        Modified Rodrigues' rotation formula (batched).

        Args:
          - vecs: shape (..., 3). NOTE: the shape of vecs should match
            those of n1 and n2 in the initializer.

        Returns:
          - rotated vecs: shape (..., 3)
        """
        # cross shape = (..., 3)
        cross = torch.linalg.cross(self.rot_axis, vecs)
        # dot shape = (..., 1)
        dot = (self.rot_axis * vecs).sum(dim=-1, keepdim=True)         

        # return shape = (..., 3)
        return (vecs * self.cos_theta
                + cross
                + self.rot_axis * dot / (1 + self.cos_theta))            


class Misalign:
    def __init__(self,
                 u_ref  = torch.tensor([-1.0, 0.0, 0.0]),
                 v_ref  = torch.tensor([ 0.0, 0.0, 1.0]),
                 eps    = 1e-8, 
                 device = 'cpu', 
                 dtype  = torch.float32):
        """
        Args:
          - u_ref and v_ref are the global coordinates
            of the reference frame. Shape: (3,)
          - eps is the precision for perpendicularity.
        """
        perpendicularity = torch.dot(u_ref, v_ref)
        assert torch.abs(perpendicularity) < eps, \
            'reference u and v are NOT perpendicular enough'

        self.u_ref = F.normalize(u_ref, dim=0).to(device).to(dtype)
        self.v_ref = F.normalize(v_ref, dim=0).to(device).to(dtype)
        self.n_ref = torch.linalg.cross(self.u_ref, self.v_ref).to(device).to(dtype)

    def misalign(self, u_ref_coord, v_ref_coord, rho):
        """
        Args:
          - u_ref_coord, v_ref_coord, rho: shape (...,)
        Returns:
          - u, v: shape (..., 3)
        """
        n_ref_coord = torch.sqrt(1.0 - u_ref_coord**2 - v_ref_coord**2)  # (...,)

        # (..., 1) * (3,) broadcasts to (..., 3)
        normal = (u_ref_coord.unsqueeze(-1) * self.u_ref +
                  v_ref_coord.unsqueeze(-1) * self.v_ref +
                  n_ref_coord.unsqueeze(-1) * self.n_ref)  # (..., 3)

        u_ref_rotated, v_ref_rotated = self.get_rotated_ref(normal)  # (..., 3)

        u = (  torch.cos(rho).unsqueeze(-1) * u_ref_rotated
             + torch.sin(rho).unsqueeze(-1) * v_ref_rotated)  # (..., 3)
        v = (- torch.sin(rho).unsqueeze(-1) * u_ref_rotated
             + torch.cos(rho).unsqueeze(-1) * v_ref_rotated)  # (..., 3)

        return u, v

    def get_misalignment(self, u, v):
        """
        Args:
          - u, v: shape (..., 3)
        Returns:
          - u_ref_coord, v_ref_coord, rho: shape (...,)
        """
        u = F.normalize(u, dim=-1)  # (..., 3)
        v = F.normalize(v, dim=-1)  # (..., 3)

        normal = torch.linalg.cross(u, v)  # (..., 3)

        u_ref_coord = (normal * self.u_ref).sum(dim=-1)  # (...,)
        v_ref_coord = (normal * self.v_ref).sum(dim=-1)  # (...,)

        u_ref_rotated, v_ref_rotated = self.get_rotated_ref(normal)  # (..., 3)

        cos_rho = (u * u_ref_rotated).sum(dim=-1)  # (...,)
        sin_rho = (u * v_ref_rotated).sum(dim=-1)  # (...,)

        rho = torch.arctan2(sin_rho, cos_rho)  # (...,)

        return u_ref_coord, v_ref_coord, rho

    def get_rotated_ref(self, normal):
        """
        Args:
          - normal: shape (..., 3)
        Returns:
          - u_ref_rotated, v_ref_rotated: shape (..., 3)
        """
        rotator = Rotator(self.n_ref.expand_as(normal), normal)
        return (rotator(self.u_ref.expand_as(normal)), 
                rotator(self.v_ref.expand_as(normal)))



DATA_ROOT = '/data/rtal/rom_det-3_part-200_cont-and-rounded/'
DEVICE    = 'cuda'
DTYPE     = torch.float64

def main(data_root=DATA_ROOT, batch_size=4):

    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from rtal.datasets.dataset import ROMDataset

    mal = Misalign(device=DEVICE, dtype=DTYPE)
    eps = 1e-6
    
    dataset  = ROMDataset(data_root, split='train', num_particles=50)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for event_id, event in tqdm(enumerate(dataloader), total=len(dataloader)):
        detector_curr = event['detector_curr'].to(DEVICE).to(torch.float64)
        
        u, v = detector_curr[..., 3:6], detector_curr[..., 6:]

        u_ref_coord, v_ref_coord, rho = mal.get_misalignment(u, v)
        misalignment = torch.cat([u_ref_coord.unsqueeze(-1), 
                                  v_ref_coord.unsqueeze(-1), 
                                  rho.unsqueeze(-1)], dim=-1)
        print(misalignment.shape)

        _u, _v = mal.misalign(u_ref_coord, v_ref_coord, rho)
        
        diff_u = torch.norm(_u - u, dim=-1)
        diff_v = torch.norm(_v - v, dim=-1)
        print(diff_u)
        print(diff_v)
        break
        if (diff_u > eps).any() or (diff_v > eps).any():
            print(event_id)
            print(diff_u)
            print(diff_v)
            print('\nu:', u, '\n')
            print('\n_u:', _u, '\n')
            print('\nv:', v, '\n')
            print('\n_v:', _v, '\n')
            break

if __name__ == '__main__':
    main()