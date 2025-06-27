"""
Calculate the residuals of points to a given geometric shape (line, helix, etc)
"""
import torch


def get_center_basis(detector_params):
    """
    Description:
        Get the center and the local basis of the detectors
    Input:
        detector_parameters: Tensor of shape (batch_size, num_detectors, 9)
    Output:
        center: Tensor of shape (batch_size, num_detectors, 3)
        basis: Tensor of shape (batch_size, num_detectors, 2, 3)
    """
    center = detector_params[..., :3]
    basis_vec_x = detector_params[..., 3:6]
    basis_vec_y = detector_params[..., 6:]
    basis = torch.stack([basis_vec_x, basis_vec_y], dim=-2)
    return center, basis


def reconstruct(detector_params,
                readout,
                pitch_x=.1,
                pitch_y=.1):

    """
    Input:
        readout: Tensor of shape (batch_size, num_particles, num_detectors, 2)
        center: Tensor of shape (batch_size, num_detectors, 3)
        basis: Tensor of shape (batch_size, num_detectors, 2, 3)
    Output:
        points: Tensor of shape (batch_size, num_particles, num_detectors, 3)
    """
    center, basis = get_center_basis(detector_params)
    return torch.einsum('bpdr,bpdrl->bpdl',
                        readout * torch.tensor([pitch_x, pitch_y],
                                               device=readout.device),
                        basis.unsqueeze(1)) + center.unsqueeze(1)


def calc_triangle_ineq_residual(points, averaged=True):
    """
    Description:
        Check whether 3 points are in line by triangle inequality.
    Input
        reconstructed points of shape (batch_size, num_particles, num_detectors, 3)
        with num_detectors = 3
    Output:
        residual evaluated as the sum of two shorter edges minus the long edge
        of a triangle formed by 3 points
    """

    assert points.shape[-2] == 3, \
        'only works for 3 detectors'

    point_0, point_1, point_2 = torch.permute(points, (2, 0, 1, 3))

    # edge_i: (batch_size, num_particles)
    edge_1 = torch.linalg.norm(point_0 - point_1, dim=-1) # shorter edge
    edge_2 = torch.linalg.norm(point_2 - point_1, dim=-1) # shorter edge
    edge_3 = torch.linalg.norm(point_2 - point_0, dim=-1) # long edge

    # residual: (batch_size, num_particles)
    residual = edge_1 + edge_2 - edge_3

    # residual is not supposed to be less than zero, but it does sometimes,
    # probably due to computational precision.
    residual = torch.clamp(residual, min=0)

    if averaged:
        return residual.mean()
    return residual


def calc_least_square_residual(points,
                               particle_vertex,
                               particle_direction,
                               averaged=True,
                               return_closest_points=False):
    """
    Description:
        Given a line and a set of points and then compute the residual
        of points to the line
    Input
        points: Tensor of shape (batch_size, num_particles, num_detectors, 3)
        particle_vertex: Tensor of shape (batch_size, num_particles, 3)
        particle_direction: Tensor of shape (batch_size, num_particles, 3)
    """
    device = points.device
    # displacements: (batch_size, num_particles, num_detectors, 3)

    # vertex: (batch_size, num_particles, 3)
    #      -> (batch_size, num_particles, 1, 3)
    vertex = particle_vertex.unsqueeze(-2).to(device)
    displacements = points - vertex
    # directions: (batch_size, num_particles, 3)
    directions = (particle_direction
                  / torch.linalg.norm(particle_direction, dim=-1, keepdim=True)).to(device)
    # time: (batch_size, num_particles, num_detectors)
    time = torch.einsum('btij,btj->bti', displacements, directions.to(device))

    # closest_points = vertex + time * directions
    # vertex     : (batch_size, num_particles, 1, 3) +
    # time       : (batch_size, num_particles, num_detectors)
    #           -> (batch_size, num_particles, num_detectors, 1)
    # directions : (batch_size, num_particles, 3)
    #           -> (batch_size, num_particles, 1, 3)
    # closest_points: (batch_size, num_particles, num_detectors, 3)
    closest_points = vertex + time.unsqueeze(-1) * directions.unsqueeze(-2)


    # residual = ||closest_points - points||_2^2
    # closest_points : (batch_size, num_particles, num_detectors, 3)
    # points         : (batch_size, num_particles, num_detectors, 3)
    # residual       : (batch_size, num_particles, num_detectors)
    residual = torch.pow(closest_points - points, 2).sum(-1)

    # average the residual to one number for metric tracking/optimizing
    if averaged:
        residual = residual.mean()

    # return the closest points for further analysis
    if return_closest_points:
        return residual, closest_points

    return residual


def calc_self_least_square_residual(points, averaged=True):
    """
    Description:
        Fit a line for a given set of points and then compute the residual
        of points to the line
    Input
        points of shape (batch_size, num_particles, num_detectors, 3)
    """
    # centroid: (batch_size, num_particles, 3)
    centroids = points.mean(dim=-2)
    # displacements: (batch_size, num_particles, num_detectors, 3)
    displacements = points - centroids.unsqueeze(-2)
    # cov: (batch_size, num_particles, 3, 3)
    cov = torch.einsum('btij,btjk->btik', displacements.transpose(-1, -2), displacements)
    # largest eigenvalue: (batch_size, num_particle)
    eigenvalues = torch.linalg.eigvalsh(cov)[..., -1]

    # residual: (batch_size, num_particles)
    residual = (displacements ** 2).sum(-1).sum(-1) - eigenvalues
    # residual is not supposed to be less than zero, but it does sometimes,
    # probably due to computational precision.
    residual = torch.clamp(residual, min=0)

    if averaged:
        return residual.mean()
    return residual
