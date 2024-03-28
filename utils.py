import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
import numpy as np
from PIL import Image
import math

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def viz_seg(vertices, segmentation_labels, output_path, device):
    image_size = 256
    background_color = (0, 0, 0)
    label_colors = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]

    # Construct various camera viewpoints
    distance = 3
    elevation = 0
    azimuths = [180 - 12 * i for i in range(30)]
    rotation_matrix, translation_matrix = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance, elev=elevation, azim=azimuths, device=device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=rotation_matrix, T=translation_matrix, fov=60, device=device)

    sample_vertices = vertices.unsqueeze(0).repeat(30, 1, 1).to(torch.float)
    sample_segmentation_labels = segmentation_labels.unsqueeze(0)
    sample_colors = torch.zeros((1, sample_vertices.shape[1], 3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_segmentation_labels == i] = torch.tensor(label_colors[i])

    sample_colors = sample_colors.repeat(30, 1, 1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_vertices, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    list_of_images = []
    azimuth_values = np.linspace(0, 360, 50)
    for azimuth in azimuth_values:
        rotation_matrix, translation_matrix = pytorch3d.renderer.look_at_view_transform(2, 2, azimuth)
        current_cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=rotation_matrix, T=translation_matrix, device=device)
        rendering = renderer(point_cloud, cameras=current_cameras)
        rendering = rendering.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        image = Image.fromarray((rendering * 255).astype(np.uint8))
        list_of_images.append(np.array(image))

    images_array = np.array(list_of_images)
    imageio.mimsave(output_path, images_array, duration=30, loop=0)
    print("Output done!")


def viz_cls(vertices, output_path, device):
    image_size = 256
    background_color = (0, 0, 0)

    sample_vertices = vertices.unsqueeze(0).repeat(30, 1, 1).to(torch.float).to(device)
    sample_colors = torch.ones(sample_vertices.shape).to(device)
    point_cloud = pytorch3d.structures.Pointclouds(points=sample_vertices, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)

    list_of_images = []
    azimuth_values = np.linspace(0, 360, 50)
    for azimuth in azimuth_values:
        rotation_matrix, translation_matrix = pytorch3d.renderer.look_at_view_transform(2, 2, azimuth)
        current_cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=rotation_matrix, T=translation_matrix, device=device)
        rendering = renderer(point_cloud, cameras=current_cameras)
        rendering = rendering.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        image = Image.fromarray((rendering * 255).astype(np.uint8))
        list_of_images.append(np.array(image))

    images_array = np.array(list_of_images)
    imageio.mimsave(output_path, images_array, duration=30, loop=0)
    print("Output done!")

def rotation_matrix_z(theta_degrees):
    """
    Generate a 3x3 rotation matrix for a rotation about the z-axis by an angle theta (in degrees).
    """
    theta_radians = torch.tensor(math.radians(theta_degrees))
    cos_theta = torch.cos(theta_radians)
    sin_theta = torch.sin(theta_radians)

    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    return rotation_matrix