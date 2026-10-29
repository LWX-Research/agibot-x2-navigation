#!/usr/bin/env python3
"""Run MuJoCo lidar/IMU simulation and FAST-LIO against an observed PCD map."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as RosPath
from rosgraph_msgs.msg import Clock
from scipy.interpolate import CubicSpline
from sensor_msgs.msg import Imu, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster
import yaml

from prepare_scene import prepare


HERE = Path(__file__).resolve().parent
DEFAULT_MAP = Path("/workspace/maps/console-20261028-074430-826594")
DT = 0.005
SCAN_STEPS = 20
RINGS = 16
COLUMNS = 360
POINT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
    ("time", "<f4"), ("ring", "<u2"), ("padding", "<u2"),
])


def stamp(seconds):
    nanoseconds = int(round(seconds * 1e9))
    return Time(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


class Trajectory:
    def __init__(self, positions, speed, warmup):
        kept = [positions[0, :2]]
        for point in positions[1:, :2]:
            if np.linalg.norm(point - kept[-1]) >= 0.12:
                kept.append(point)
        if len(kept) < 2:
            raise ValueError("Trajectory must contain at least 0.12 m of movement")
        kept = np.asarray(kept)
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(kept, axis=0), axis=1))]
        self.curve = CubicSpline(distances, kept, bc_type="natural")
        self.length = float(distances[-1])
        self.speed, self.warmup = speed, warmup
        self.duration = warmup + self.length / speed + 5.

    def target(self, seconds):
        elapsed = max(0., seconds - self.warmup)
        distance = min(self.length, self.speed * (elapsed - 2. * (1. - math.exp(-elapsed / 2.))))
        xy = self.curve(distance)
        yaw = 0.25 * math.sin(distance * 0.25)
        return np.array([xy[0], xy[1], yaw])


def mapping_parameters():
    return {
        "/**": {"ros__parameters": {
            "use_sim_time": True,
            "common.map_frame": "sim_map",
            "common.lidar_frame": "sim_lidar",
            "common.imu_frame": "sim_lidar",
            "common.base_frame": "",
            "common.visualization_frame": "imu",
            "gravity_alignment.enable_gravity_alignment": False,
            "preprocess.lidar_type": 2,
            "preprocess.scan_line": RINGS,
            "preprocess.scan_rate": 10,
            "preprocess.timestamp_unit": 0,
            "preprocess.blind": 0.3,
            "preprocess.blind_for_human_pilots": 0.0,
            "mapping.extrinsic_est_en": False,
            "mapping.extrinsic_T": [0.0, 0.0, 0.0],
            "mapping.extrinsic_R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "mapping.acc_cov": 0.0001,
            "mapping.gyr_cov": 0.000001,
            "mapping.b_acc_cov": 0.00000001,
            "mapping.b_gyr_cov": 0.0000000001,
            "mapping.fov_degree": 360.0,
            "mapping.det_range": 30.0,
            "filter_size_map": 0.15,
            "point_filter_num_for_preprocessing": 1,
            "point_filter_num": 2,
            "cube_side_length": 100.0,
            "max_iteration": 4,
            "publish.path_en": True,
            "publish.scan_publish_en": True,
            "publish.dense_publish_en": True,
            "pcd_save.pcd_save_en": False,
            "verbose": False,
            "pcl_verbose": False,
        }}
    }


class Simulator(Node):
    def __init__(self, args, scene_dir, scene_info):
        super().__init__("mujoco_sensor_simulator")
        self.args = args
        self.model = mujoco.MjModel.from_binary_path(str(scene_dir / "scene.mjb"))
        self.data = mujoco.MjData(self.model)
        self.origin = np.asarray(scene_info["start_position"])
        self.data.qpos[:2] = self.origin[:2]
        self.data.ctrl[:2] = self.origin[:2]
        mujoco.mj_forward(self.model, self.data)
        positions = np.asarray(json.loads((args.map_dir / "trajectory.json").read_text()))
        self.trajectory = Trajectory(positions, args.speed, args.warmup)
        # RK4 evaluates this at each intermediate physics time. Updating the
        # position target only once per sensor tick biases sampled acceleration.
        self.control_callback = self.apply_control
        mujoco.set_mjcb_control(self.control_callback)
        self.site_id = self.model.site("lidar_imu").id
        self.body_id = self.model.body("sensor_carrier").id
        self.lidar = self.create_publisher(PointCloud2, "/sim/lidar/points", 10)
        self.imu = self.create_publisher(Imu, "/sim/imu", 50)
        self.clock = self.create_publisher(Clock, "/clock", 10)
        self.truth_publisher = self.create_publisher(Odometry, "/sim/ground_truth", 10)
        self.truth_path_publisher = self.create_publisher(RosPath, "/sim/ground_truth_path", 10)
        reference_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.reference_publisher = self.create_publisher(PointCloud2, "/sim/reference_map", reference_qos)
        self.create_subscription(Odometry, "/mujoco_slam/odometry", self.on_odometry, 100)
        self.create_subscription(PointCloud2, "/mujoco_slam/cloud_registered", self.on_cloud, 10)
        self.tf = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.frame_id = "x2_map"
        transform.child_frame_id = "sim_map"
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = self.origin.tolist()
        transform.transform.rotation.w = 1.
        self.tf.sendTransform(transform)
        self.truth_path = RosPath(header=Header(frame_id="x2_map"))
        self.fields = [PointField(name=name, offset=POINT_DTYPE.fields[name][1],
                                 datatype=PointField.UINT16 if name == "ring" else PointField.FLOAT32,
                                 count=1) for name in ("x", "y", "z", "intensity", "time", "ring")]
        elevation = np.linspace(math.radians(-25), math.radians(25), RINGS)
        azimuth = np.arange(COLUMNS) * (2 * math.pi / COLUMNS)
        az, el = np.meshgrid(azimuth, elevation, indexing="ij")
        self.directions = np.stack((np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)), axis=-1)
        self.groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.tick = 0
        self.scan_parts = []
        self.scan_sizes = []
        self.truth = []
        self.estimates = []
        self.cloud_frames = 0
        self.map = o3d.geometry.PointCloud()
        self.acceleration_norms = []
        reference = o3d.io.read_point_cloud(str(args.map_dir / "cloud.pcd")).voxel_down_sample(0.20)
        self.reference = point_cloud2.create_cloud_xyz32(Header(frame_id="x2_map"), np.asarray(reference.points))

    def apply_control(self, model, data):
        data.ctrl[:] = self.trajectory.target(data.time)

    def on_odometry(self, message):
        t = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        p, q = message.pose.pose.position, message.pose.pose.orientation
        self.estimates.append([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    def on_cloud(self, message):
        xyz = point_cloud2.read_points_numpy(message, field_names=["x", "y", "z"], skip_nans=True)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=float))
        self.map += cloud
        self.cloud_frames += 1
        if self.cloud_frames % 20 == 0:
            self.map = self.map.voxel_down_sample(0.10)

    def step(self):
        self.tick += 1
        seconds = self.tick * DT
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        now = stamp(seconds)
        self.clock.publish(Clock(clock=now))
        acceleration = self.data.sensor("accelerometer").data.copy()
        angular_velocity = self.data.sensor("gyroscope").data.copy()
        quaternion = self.data.sensor("orientation").data.copy()
        imu = Imu(header=Header(stamp=now, frame_id="sim_lidar"))
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = acceleration.tolist()
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = angular_velocity.tolist()
        imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = quaternion.tolist()
        imu.linear_acceleration_covariance = [0.0001,0.,0.,0.,0.0001,0.,0.,0.,0.0001]
        imu.angular_velocity_covariance = [0.000001,0.,0.,0.,0.000001,0.,0.,0.,0.000001]
        self.imu.publish(imu)
        if seconds <= self.args.warmup:
            self.acceleration_norms.append(float(np.linalg.norm(acceleration)))
        position = self.data.site_xpos[self.site_id].copy()
        self.truth.append([seconds, *position.tolist(), float(self.data.qpos[2])])

        step_in_scan = (self.tick - 1) % SCAN_STEPS
        columns_per_step = COLUMNS // SCAN_STEPS
        begin = step_in_scan * columns_per_step
        local_directions = self.directions[begin:begin+columns_per_step].reshape(-1, 3)
        world_directions = np.ascontiguousarray(local_directions @ self.data.site_xmat[self.site_id].reshape(3, 3).T)
        distances = np.empty(len(world_directions), dtype=np.float64)
        geometry_ids = np.empty(len(world_directions), dtype=np.int32)
        mujoco.mj_multiRay(self.model, self.data, position, world_directions.ravel(), self.groups,
                          True, self.body_id, geometry_ids, distances, None, len(distances), 30.0)
        valid = (distances >= 0.3) & (distances <= 30.0)
        points = np.zeros(int(valid.sum()), dtype=POINT_DTYPE)
        xyz = local_directions[valid] * distances[valid, None]
        for i, name in enumerate(("x", "y", "z")):
            points[name] = xyz[:, i]
        points["intensity"] = 100.
        points["time"] = step_in_scan * DT
        points["ring"] = np.tile(np.arange(RINGS, dtype=np.uint16), columns_per_step)[valid]
        self.scan_parts.append(points)
        if step_in_scan == SCAN_STEPS - 1:
            scan = np.concatenate(self.scan_parts)
            cloud = PointCloud2(header=Header(stamp=stamp(seconds - (SCAN_STEPS-1)*DT), frame_id="sim_lidar"))
            cloud.height, cloud.width = 1, len(scan)
            cloud.fields = self.fields
            cloud.point_step, cloud.row_step = POINT_DTYPE.itemsize, len(scan)*POINT_DTYPE.itemsize
            cloud.is_dense = True
            cloud.data = scan.tobytes()
            self.lidar.publish(cloud)
            self.scan_sizes.append(len(scan))
            self.scan_parts.clear()
            truth = Odometry(header=Header(stamp=now, frame_id="x2_map"), child_frame_id="sim_ground_truth")
            truth.pose.pose.position.x, truth.pose.pose.position.y, truth.pose.pose.position.z = position.tolist()
            truth.pose.pose.orientation = imu.orientation
            self.truth_publisher.publish(truth)
            pose = PoseStamped(header=truth.header, pose=truth.pose.pose)
            self.truth_path.poses.append(pose)
            self.truth_path.header.stamp = now
            self.truth_path_publisher.publish(self.truth_path)
        if self.tick % 400 == 0:
            self.reference.header.stamp = now
            self.reference_publisher.publish(self.reference)

    def save(self, output, elapsed):
        self.map = self.map.voxel_down_sample(0.10)
        if not self.map.is_empty():
            if not o3d.io.write_point_cloud(str(output / "map_merged.pcd"), self.map):
                raise RuntimeError("Could not save the simulated map")
        truth = np.asarray(self.truth)
        estimates = np.asarray(self.estimates)
        errors = np.array([])
        if len(truth) and len(estimates):
            estimates = estimates[(estimates[:, 0] >= self.args.warmup) & (estimates[:, 0] <= truth[-1, 0])]
            if len(estimates):
                expected = np.column_stack([np.interp(estimates[:, 0], truth[:, 0], truth[:, i+1]) for i in range(3)]) - self.origin
                errors = np.linalg.norm(estimates[:, 1:4] - expected, axis=1)
        summary = {
            "map_source": str(self.args.map_dir),
            "ros_domain_id": self.args.domain,
            "simulated_seconds": self.tick*DT,
            "wall_seconds": elapsed,
            "realtime_factor": self.tick*DT/elapsed if elapsed else 0.,
            "imu_messages": self.tick,
            "lidar_scans": len(self.scan_sizes),
            "mean_points_per_scan": float(np.mean(self.scan_sizes)) if self.scan_sizes else 0.,
            "stationary_acceleration_norm": float(np.mean(self.acceleration_norms)) if self.acceleration_norms else None,
            "registered_cloud_frames": self.cloud_frames,
            "saved_map_points": len(self.map.points),
            "estimated_poses_after_warmup": len(estimates),
            "position_rmse_m": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "position_error_max_m": float(errors.max()) if len(errors) else None,
            "ground_truth_travel_m": float(np.linalg.norm(np.diff(truth[:, 1:3], axis=0), axis=1).sum()) if len(truth) else 0.,
            "frames": {"saved_map": "sim_map", "original_map": "x2_map", "sim_map_origin_in_x2_map": self.origin.tolist()},
            "scope": "Ideal MuJoCo sensors on a planar carrier; not X2 walking dynamics or navigation acceptance",
        }
        summary["validation_passed"] = bool(
            len(errors) >= 100 and summary["registered_cloud_frames"] >= 20
            and summary["saved_map_points"] > 1000 and summary["ground_truth_travel_m"] > 0.5
            and summary["position_rmse_m"] < 0.5
            and abs(summary["stationary_acceleration_norm"] - 9.81) < 0.01
        )
        np.savez_compressed(output / "trajectories.npz", ground_truth=truth, estimated=estimates)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        if len(errors):
            from plot_results import plot
            plot(output)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--duration", type=float, default=0., help="Simulated seconds; 0 follows the complete trajectory and keeps GUI windows open until exit")
    parser.add_argument("--speed", type=float, default=0.30, help="Approximate carrier speed in m/s")
    parser.add_argument("--warmup", type=float, default=3., help="Stationary IMU initialization time in seconds")
    parser.add_argument("--realtime-factor", type=float, default=1.)
    parser.add_argument("--domain", type=int, default=73, help="Isolated ROS domain for this simulation")
    parser.add_argument("--headless", action="store_true", help="Disable MuJoCo and RViz windows")
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="Exit with failure if the SLAM validation criteria fail")
    args = parser.parse_args()
    for name in ("speed", "realtime_factor", "warmup", "voxel_size"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and non-negative")
    if not 0 <= args.domain <= 232:
        parser.error("--domain must be between 0 and 232")
    args.map_dir = args.map_dir.expanduser().resolve()
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    scene_dir = args.output_dir.expanduser().resolve() / args.map_dir.name
    print("Preparing MuJoCo scene from", args.map_dir, flush=True)
    info = prepare(args.map_dir, scene_dir, args.voxel_size)
    print(f"Scene ready: {info['occupied_voxels']} occupied voxels, {info['mesh_triangles']} triangles", flush=True)
    if args.prepare_only:
        print(scene_dir / "scene.xml")
        return 0
    output = scene_dir / "runs" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    print("Run output:", output, flush=True)
    parameters = output / "mapping.yaml"
    parameters.write_text(yaml.safe_dump(mapping_parameters(), sort_keys=False))
    rclpy.init(args=[])
    simulator = Simulator(args, scene_dir, info)
    mapping = rviz = viewer = None
    log = (output / "fast_lio.log").open("w")
    rviz_log = (output / "rviz.log").open("w")
    started = time.monotonic()
    summary = None
    shutdown_requested = False
    terminal_closed = False

    def request_shutdown(signum, frame):
        nonlocal shutdown_requested, terminal_closed
        shutdown_requested = True
        terminal_closed = terminal_closed or signum == signal.SIGHUP

    previous_handlers = {
        signum: signal.signal(signum, request_shutdown)
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    }
    try:
        mapping = subprocess.Popen([
            "ros2", "run", "spark_fast_lio", "spark_lio_mapping", "--ros-args",
            "--params-file", str(parameters), "-r", "__ns:=/mujoco_slam",
            "-r", "lidar:=/sim/lidar/points", "-r", "imu:=/sim/imu",
        ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 15
        while not (simulator.lidar.get_subscription_count() and simulator.imu.get_subscription_count()):
            if shutdown_requested:
                raise KeyboardInterrupt
            if mapping.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f"FAST-LIO did not subscribe; see {output / 'fast_lio.log'}")
            simulator.clock.publish(Clock(clock=stamp(0)))
            rclpy.spin_once(simulator, timeout_sec=0.05)
        if not args.headless and os.environ.get("DISPLAY"):
            from mujoco import viewer as mujoco_viewer
            viewer = mujoco_viewer.launch_passive(simulator.model, simulator.data)
            viewer.cam.lookat[:] = [-0.5, 0., 0.]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.5, 120., -8.
            if not args.no_rviz:
                rviz = subprocess.Popen(["rviz2", "-d", str(HERE / "slam.rviz"), "--ros-args", "-p", "use_sim_time:=true"],
                                        stdout=rviz_log, stderr=subprocess.STDOUT, start_new_session=True)
        duration = args.duration or simulator.trajectory.duration
        steps = int(math.ceil(duration / (DT*SCAN_STEPS))) * SCAN_STEPS
        started = next_step = last_status = time.monotonic()
        for _ in range(steps):
            if shutdown_requested or not rclpy.ok() or (viewer is not None and not viewer.is_running()):
                break
            if mapping.poll() is not None:
                raise RuntimeError(f"FAST-LIO exited; see {output / 'fast_lio.log'}")
            simulator.step()
            for _ in range(3):
                rclpy.spin_once(simulator, timeout_sec=0.)
            if viewer is not None and simulator.tick % 10 == 0:
                viewer.sync()
            now = time.monotonic()
            if now - last_status >= 5:
                print(f"sim={simulator.tick*DT:.1f}s scans={len(simulator.scan_sizes)} registered={simulator.cloud_frames} poses={len(simulator.estimates)}", flush=True)
                last_status = now
            next_step += DT / args.realtime_factor
            if next_step > now:
                time.sleep(next_step - now)
        deadline = time.monotonic() + 1.5
        while not shutdown_requested and rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(simulator, timeout_sec=.02)
        if args.duration == 0 and viewer is not None and not shutdown_requested and rclpy.ok() and viewer.is_running():
            print("Trajectory complete. Keeping MuJoCo and RViz open at the final pose. "
                  "Close the terminal or MuJoCo window, or press Ctrl+C, to exit and save results.", flush=True)
            while not shutdown_requested and rclpy.ok() and viewer.is_running():
                if mapping.poll() is not None:
                    raise RuntimeError(f"FAST-LIO exited; see {output / 'fast_lio.log'}")
                viewer.sync()
                rclpy.spin_once(simulator, timeout_sec=.05)
    except KeyboardInterrupt:
        pass
    finally:
        if terminal_closed:
            # The terminal no longer accepts output; keep cleanup logs with the run.
            with (output / "shutdown.log").open("a") as shutdown_log:
                os.dup2(shutdown_log.fileno(), 1)
                os.dup2(shutdown_log.fileno(), 2)
        stop(mapping)
        stop(rviz)
        if viewer is not None:
            viewer.close()
        mujoco.set_mjcb_control(None)
        log.close()
        rviz_log.close()
        try:
            summary = simulator.save(output, time.monotonic() - started)
        finally:
            simulator.destroy_node()
            rclpy.try_shutdown()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    print("Saved simulation results to", output, flush=True)
    return 1 if args.check and not summary["validation_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
