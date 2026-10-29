#!/usr/bin/env python3
"""Convert an observed PCD map into a raycastable MuJoCo voxel surface mesh."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import open3d as o3d


FACES = (
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
)


def prepare(map_dir, output_dir, voxel_size=0.20):
    map_dir, output_dir = Path(map_dir).resolve(), Path(output_dir).resolve()
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")
    source = map_dir / "cloud.pcd"
    trajectory_file = map_dir / "trajectory.json"
    if not source.is_file() or not trajectory_file.is_file():
        raise FileNotFoundError("map-dir must contain cloud.pcd and trajectory.json")
    if output_dir == map_dir or map_dir in output_dir.parents:
        raise ValueError("Use an output directory outside the original map directory")
    signature = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "trajectory_sha256": hashlib.sha256(trajectory_file.read_bytes()).hexdigest(),
        "voxel_size_m": voxel_size,
        "converter_version": 2,
        "mujoco_version": mujoco.__version__,
    }
    manifest = output_dir / "scene_info.json"
    if manifest.is_file():
        old = json.loads(manifest.read_text())
        if all(old.get(k) == v for k, v in signature.items()) and all(
            (output_dir / name).is_file() for name in ("scene.xml", "scene.mjb", "map.obj")
        ):
            return old

    cloud = o3d.io.read_point_cloud(str(source))
    cloud.remove_non_finite_points()
    points = np.asarray(cloud.points)
    if len(points) == 0:
        raise ValueError("cloud.pcd contains no finite points")
    trajectory = np.asarray(json.loads(trajectory_file.read_text()), dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) < 2:
        raise ValueError("trajectory.json must contain at least two XYZ positions")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory contains non-finite positions")

    cells = np.unique(np.floor(points / voxel_size).astype(np.int32), axis=0)
    occupied = {tuple(cell) for cell in cells}
    quads = []
    for direction, corners in FACES:
        neighbours = cells + np.asarray(direction, dtype=np.int32)
        exposed = np.fromiter((tuple(cell) not in occupied for cell in neighbours), bool, len(cells))
        quads.append(cells[exposed, None, :] + np.asarray(corners, dtype=np.int32)[None, :, :])
    corners = np.concatenate(quads)
    vertices, indices = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    quads = indices.reshape(-1, 4)
    triangles = np.concatenate((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "map.obj").open("w") as stream:
        stream.write("# Observed PCD voxel boundary; original coordinates in metres\n")
        np.savetxt(stream, vertices.astype(float) * voxel_size, fmt="v %.6f %.6f %.6f")
        np.savetxt(stream, triangles + 1, fmt="f %d %d %d")

    z = float(trajectory[0, 2])
    scene = f'''<mujoco model="scanned_map_slam">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
  <statistic center="-0.5 0 0" extent="12"/>
  <visual>
    <global azimuth="135" elevation="-55"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7"/>
  </visual>
  <asset>
    <mesh name="observed_map" file="map.obj" inertia="shell" maxhullvert="64"/>
  </asset>
  <worldbody>
    <light pos="0 0 12" dir="0 0 -1" directional="true"/>
    <geom name="scanned_surfaces" type="mesh" mesh="observed_map"
          contype="0" conaffinity="0" group="0" rgba="0.55 0.70 0.80 1"/>
    <body name="sensor_carrier" pos="0 0 {z:.9f}">
      <joint name="x" type="slide" axis="1 0 0" damping="1"/>
      <joint name="y" type="slide" axis="0 1 0" damping="1"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0.02"/>
      <geom type="sphere" size="0.10" mass="1" group="1" rgba="1 0.3 0.1 1"
            contype="0" conaffinity="0"/>
      <site name="lidar_imu" pos="0 0 0" size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="x" kp="250" kv="35"/>
    <position joint="y" kp="250" kv="35"/>
    <position joint="yaw" kp="5" kv="0.3"/>
  </actuator>
  <sensor>
    <accelerometer name="accelerometer" site="lidar_imu"/>
    <gyro name="gyroscope" site="lidar_imu"/>
    <framequat name="orientation" objtype="site" objname="lidar_imu"/>
  </sensor>
</mujoco>
'''
    (output_dir / "scene.xml").write_text(scene)
    model = mujoco.MjModel.from_xml_path(str(output_dir / "scene.xml"))
    mujoco.mj_saveModel(model, str(output_dir / "scene.mjb"), None)
    info = {
        **signature,
        "source_points": len(points),
        "occupied_voxels": len(cells),
        "mesh_vertices": len(vertices),
        "mesh_triangles": len(triangles),
        "bounds_min": points.min(axis=0).tolist(),
        "bounds_max": points.max(axis=0).tolist(),
        "start_position": trajectory[0].tolist(),
        "trajectory_positions": len(trajectory),
        "geometry_scope": "Observed voxel surfaces for raycasting; no full-body contact model",
    }
    manifest.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.20)
    args = parser.parse_args()
    print(json.dumps(prepare(args.map_dir, args.output_dir, args.voxel_size), indent=2))
