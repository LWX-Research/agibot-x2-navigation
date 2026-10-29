#!/usr/bin/env python3
"""Plot the simulated ground truth and FAST-LIO estimates in map coordinates."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def plot(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    samples = np.load(run_dir / "trajectories.npz")
    truth, estimated = samples["ground_truth"], samples["estimated"]
    if not len(truth) or not len(estimated):
        raise ValueError("Run has no comparable ground truth and estimated poses")
    origin = np.asarray(summary["frames"]["sim_map_origin_in_x2_map"])
    expected = np.column_stack([np.interp(estimated[:, 0], truth[:, 0], truth[:, i]) for i in (1, 2, 3)])
    quaternion = summary["frames"].get("sim_map_rotation_in_x2_map", [0., 0., 0., 1.])
    initial_rotation = Rotation.from_quat(quaternion).as_matrix()
    estimated_world = estimated[:, 1:4] @ initial_rotation.T + origin
    errors = np.linalg.norm(estimated_world - expected, axis=1)
    points = np.asarray(o3d.io.read_point_cloud(str(Path(summary["map_source"]) / "cloud.pcd")).points)
    low, high = truth[:, 1:3].min(axis=0)-3, truth[:, 1:3].max(axis=0)+3
    mask = (points[:, 0] >= low[0]) & (points[:, 0] <= high[0])
    mask &= (points[:, 1] >= low[1]) & (points[:, 1] <= high[1])
    mask &= (points[:, 2] > -0.75) & (points[:, 2] < 1.8)
    points = points[mask][::3]
    figure, (map_axes, error_axes) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    map_axes.scatter(points[:, 0], points[:, 1], s=.5, c="#8c99a4", alpha=.25, rasterized=True)
    map_axes.plot(truth[:, 1], truth[:, 2], color="#e8832e", linewidth=2.5, label="MuJoCo ground truth")
    map_axes.plot(estimated_world[:, 0], estimated_world[:, 1], color="#1768ac", linewidth=1.4, label="FAST-LIO")
    map_axes.set(xlabel="x in x2_map (m)", ylabel="y in x2_map (m)", title="Trajectory against the original scanned map")
    map_axes.set_aspect("equal")
    map_axes.legend(loc="best")
    error_axes.plot(estimated[:, 0]-summary.get("time_origin", 0.), errors, color="#1768ac", linewidth=1.)
    time_label = "Elapsed ROS time (s)" if "joint_messages" in summary else "Simulated time (s)"
    error_axes.set(xlabel=time_label, ylabel="Position error (m)", title=f"Position RMSE: {np.sqrt(np.mean(errors**2)):.4f} m")
    error_axes.grid(alpha=.25)
    title = "Native X2 whole-body locomotion + virtual lidar" if "joint_messages" in summary else "Ideal lidar + IMU simulation on a planar carrier"
    figure.suptitle(title, fontsize=13)
    figure.savefig(run_dir / "validation.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    plot(parser.parse_args().run_dir)
