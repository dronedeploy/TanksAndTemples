import sys
import copy
import numpy as np
import open3d as o3d


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def manual_registration(pcd1_file, pcd2_file, threshold, transform_file=None, out_pcd_file=None, verbose=True):
    if verbose:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    print("Manual ICP")
    source = o3d.io.read_point_cloud(pcd1_file)
    target = o3d.io.read_point_cloud(pcd2_file)
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.registration.TransformationEstimationPointToPoint(True)
    trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(True))
    draw_registration_result(source, target, reg_p2p.transformation)

    if transform_file != None:
        # Save transform
        np.save(transform_file, reg_p2p.transformation)
    if out_pcd_file != None:
        # Export aligned point cloud
        source.transform(reg_p2p.transformation)
        o3d.io.write_point_cloud(out_pcd_file, source)
    return reg_p2p.transformation


if __name__ == "__main__":
    pcd1_file = sys.argv[1]
    pcd2_file = sys.argv[2]
    threshold = 0.04
    if len(sys.argv) > 3:
        threshold = float(sys.argv[3])
    transform_file = "transform.npy"
    if len(sys.argv) > 4:
        transform_file = sys.argv[4]
    out_pcd_file = None
    if len(sys.argv) > 5:
        out_pcd_file = sys.argv[5]
    manual_registration(pcd1_file, pcd2_file, threshold, transform_file, out_pcd_file)
