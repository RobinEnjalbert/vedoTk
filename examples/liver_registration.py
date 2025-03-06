from os.path import join
import numpy as np
import vedo
import open3d as o3d
import json

from vedoTk.utils.points import compute_normals_with_pca, generate_triangles
from vedoTk import register


data_dir = join('resources', 'liver')

# Create the point clouds from the RGBD data with Open3D
with open(join(data_dir, 'camera_parameters.json')) as f:
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        intrinsic=o3d.camera.PinholeCameraIntrinsic(**json.load(f)),
        image=o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(np.load(join(data_dir, 'img_rgb.npy'))),
            depth=o3d.geometry.Image(np.load(join(data_dir, 'img_d.npy'))),
            convert_rgb_to_intensity=False))

# Create the surface from points
pts = vedo.Points(inputobj=np.asarray(pcd.points), r=8).scale(1e3)
pts.pointcolors = np.asarray(pcd.colors)[:, [2, 1, 0]] * 255
pts.subsample(0.02)
compute_normals_with_pca(pts, display=False, flip=True, n=40)
target = generate_triangles(obj=pts)

# Create the source mesh
source = vedo.Mesh(join(data_dir, 'liver.obj')).clean()

# Register objects
register(source=source, target=target)
