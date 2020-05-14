# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.
import sys
import numpy as np
import open3d as o3d
import os
import glob
import argparse
from read_pcd import read_pcd


from config import scenes_tau_dict
from registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
    read_trajectory,
)
from evaluation import EvaluateHisto
from evaluation import EvaluateHistoAligned
from util import make_dir
from plot import plot_graph


def run_evaluation(dataset_dir, traj_path, ply_path, out_dir, plot_color_coding=True):
    dataset_dir = os.path.normpath(dataset_dir)
    scene = os.path.basename(dataset_dir)

    if scene not in scenes_tau_dict:
        print(dataset_dir, scene)
        raise Exception("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    dTau = scenes_tau_dict[scene]
    # put the crop-file, the GT file, the COLMAP SfM log file and
    # the alignment of the according scene in a folder of
    # the same scene name in the dataset_dir
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    gt_filen = os.path.join(dataset_dir, scene + ".ply")
    cropfile = os.path.join(dataset_dir, scene + ".json")
    map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")
    final_transform_file = os.path.join(dataset_dir, scene + '_final_transform.npy')

    traj_path = os.path.join(dataset_dir, traj_path)
    ply_path = os.path.join(dataset_dir, ply_path)
    out_dir = os.path.join(dataset_dir, out_dir)
    make_dir(out_dir)

    # Load reconstruction and according GT
    print(ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)
    print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)

    # Refine alignment by using the actual GT and MVS pointclouds
    vol = o3d.visualization.read_selection_polygon_volume(cropfile)

    # big pointclouds will be downlsampled to this number to speed up alignment
    dist_threshold = dTau

    if os.path.isfile(final_transform_file):
		# Load existing transform
        final_transform = np.load(final_transform_file)
    else:
        # Compute transform
        gt_trans = np.loadtxt(alignment)
        traj_to_register = read_trajectory(traj_path)
        gt_traj_col = read_trajectory(colmap_ref_logfile)
        trajectory_transform = trajectory_alignment(
            map_file, traj_to_register, gt_traj_col, gt_trans, scene
        )
        # Registration refinment in 3 iterations
        r2 = registration_vol_ds(
            pcd, gt_pcd, trajectory_transform, vol, dTau, dTau * 80, 20
        )
        r3 = registration_vol_ds(
            pcd, gt_pcd, r2.transformation, vol, dTau / 2.0, dTau * 20, 20
        )
        r = registration_unif(pcd, gt_pcd, r3.transformation, vol, 2 * dTau, 20)
        final_transform = r.transformation
        # Save transform
        np.save(final_transform_file, final_transform)

    # Histogramms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = EvaluateHisto(
        pcd,
        gt_pcd,
        final_transform,
        vol,
        dTau / 2.0,
        dTau,
        out_dir,
        plot_stretch,
        plot_color_coding,
        scene,
    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")

    # Plotting
    plot_graph(
        scene,
        fscore,
        dist_threshold,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )


def run_evaluation_aligned(dataset_dir, gt_ply_path, ply_path, dTau, out_dir, postfix=None, transform_path=None, refine_alignment=False, plot_color_coding=True):
    dataset_dir = os.path.normpath(dataset_dir)
    scene = os.path.basename(dataset_dir)
    if postfix != None:
        out_dir = os.path.join(out_dir, scene)
        scene = str(postfix)

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    gt_ply_path = os.path.join(dataset_dir, gt_ply_path)
    ply_path = os.path.join(dataset_dir, ply_path)
    out_dir = os.path.join(dataset_dir, out_dir)
    make_dir(out_dir)

    # Load reconstruction and according GT
    print(ply_path)
    pcd = read_pcd(ply_path)
    print(gt_ply_path)
    gt_pcd = read_pcd(gt_ply_path)

    if transform_path != None:
		# Load and apply existing transform
        transform = np.load(transform_path)
        if refine_alignment:
            # Refine existing transform
            pcd = refine_alignment(pcd, gt_pcd, dTau, transform)
        else:
            pcd.transform(transform)
    elif refine_alignment:
        # Refine alignment
        r = registration_unif(pcd, gt_pcd, np.identity(4), None, dTau * 2, 20, sample_method=None)
        pcd.transform(r.transformation)

    # Histogramms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = EvaluateHistoAligned(
        pcd,
        gt_pcd,
        dTau,
        out_dir,
        plot_stretch,
        plot_color_coding,
        scene,
    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")

    # Plotting
    plot_graph(
        scene,
        fscore,
        dTau,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )

    return eva


def run_evaluation_aligned_project(dataset_dir, gt_ply_path, ply_path, dTau, out_dir, transform_path=None, refine_alignment=False, plot_color_coding=True):
    dataset_dir = os.path.normpath(dataset_dir)
    scene = os.path.basename(dataset_dir)
    gt_ply_path = os.path.join(dataset_dir, gt_ply_path)
    ply_path = os.path.join(dataset_dir, ply_path)
    out_dir = os.path.join(dataset_dir, out_dir)

    gt_scenes = [f for f in glob.glob(gt_ply_path)]
    scenes = [f for f in glob.glob(ply_path)]
    if len(gt_scenes) != len(scenes):
        print("error: invalid project")
        return

    if len(gt_scenes) == 1:
        return run_evaluation_aligned(dataset_dir, gt_scenes[i], scenes[i], dTau, out_dir, transform_path=transform_path, refine_alignment=refine_alignment, plot_color_coding=plot_color_coding)
    evas = []
    for i in range(len(gt_scenes)):
        evas.append(run_evaluation_aligned(dataset_dir, gt_scenes[i], scenes[i], dTau, out_dir, postfix=i, transform_path=transform_path, refine_alignment=refine_alignment, plot_color_coding=plot_color_coding))
    nevas = len(evas)
    evas = np.asmatrix(evas)
    eva = np.asarray(evas.sum(axis=0)/nevas)
    np.savetxt(os.path.join(out_dir, scene + ".average.txt"), eva.T)
    eva = np.squeeze(eva)
    print("")
    print("==============================")
    print("evaluation average result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")
    return eva


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="path to a dataset/scene directory containing X.json, X.ply, ...",
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        help="path to trajectory file. See `convert_to_logfile.py` to create this file.",
    )
    parser.add_argument(
        "--ply-path",
        type=str,
        required=True,
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="evaluation",
        help="output directory, default: an evaluation directory is created in the directory of the ply file",
    )

    parser.add_argument(
        "--gt-ply-path",
        type=str,
        help="path to ground-truth ply file",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default="0.02",
        help="precision threshold",
    )
    parser.add_argument(
        "--refine-alignment",
        type=bool,
        default=False,
        help="refine alignment",
    )
    parser.add_argument(
        "--transform-path",
        type=str,
        default=None,
        help="transform from plt to gt-ply to be used",
    )
    parser.add_argument(
        "--no-plot-color-coding",
        type=bool,
        default=False,
        help="plot color coding",
    )
    
    args = parser.parse_args()

    if args.gt_ply_path == None:
        # run normal T&T evaluation
        run_evaluation(
            dataset_dir=args.dataset_dir,
            traj_path=args.traj_path,
            ply_path=args.ply_path,
            out_dir=args.out_dir,
            plot_color_coding=not args.no_plot_color_coding
        )
    else:
        # run evaluation on already aligned data
        run_evaluation_aligned_project(
            dataset_dir=args.dataset_dir,
            gt_ply_path=args.gt_ply_path,
            ply_path=args.ply_path,
            dTau=args.tau,
            out_dir=args.out_dir,
            transform_path=args.transform_path,
            refine_alignment=args.refine_alignment,
            plot_color_coding=not args.no_plot_color_coding
        )
