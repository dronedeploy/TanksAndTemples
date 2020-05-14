import sys
import numpy as np
import open3d as o3d


def transform_model(pcd_file, out_pcd_file, transform_file, invert_transform=False, is_mesh=True, verbose=True):
    if verbose:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    if is_mesh:
        pcd = o3d.io.read_triangle_mesh(pcd_file)
    else:
        pcd = o3d.io.read_point_cloud(pcd_file)
    transform = np.load(transform_file)
    if invert_transform:
        transform = np.linalg.inv(transform)
    pcd.transform(transform)
    if is_mesh:
        o3d.io.write_triangle_mesh(out_pcd_file, pcd)
    else:
        o3d.io.write_point_cloud(out_pcd_file, pcd)


if __name__ == "__main__":
    pcd_file = sys.argv[1]
    out_pcd_file = sys.argv[2]
    transform_file = "transform.npy"
    if len(sys.argv) > 3:
        transform_file = sys.argv[3]
    invert_transform = False
    if len(sys.argv) > 4:
        invert_transform = bool(sys.argv[4])
    is_mesh = True
    if len(sys.argv) > 5:
        is_mesh = bool(sys.argv[5])
    transform_model(pcd_file, out_pcd_file, transform_file, invert_transform, is_mesh)
