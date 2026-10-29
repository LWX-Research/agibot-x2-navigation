#!/usr/bin/env python3
"""Run the installed X2 whole-body simulator and controller with virtual lidar SLAM."""

import argparse
from collections import deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as RosPath
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster
from aimdk_msgs.msg import JointStateArray
import yaml

from native_stack import NativeStack, prepare_native
from prepare_scene import prepare
from run_slam import HERE, DEFAULT_MAP, POINT_DTYPE, RINGS, COLUMNS, SCAN_STEPS, mapping_parameters, stamp, stop
from x2_control import X2Control


class X2Bridge(Node):
    def __init__(self, args, profile, metadata):
        super().__init__("x2_lidar_slam_bridge")
        self.args, self.metadata = args, metadata
        self.model = mujoco.MjModel.from_xml_path(str(profile / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.site_id = self.model.site("imu_1").id
        self.body_id = int(self.model.site_bodyid[self.site_id])
        self.joint_offsets = {self.model.joint(i).name: int(self.model.jnt_qposadr[i]) for i in range(self.model.njnt)}
        self.map_offset = np.asarray(metadata["map_offset_in_native_world"])
        self.last_odometry = self.last_imu = None
        self.imu_window = deque(maxlen=500)
        self.joints = {}
        self.areas = set()
        self.joint_messages = 0
        self.active = False
        self.last_imu_stamp = self.last_scan_sample = 0.
        self.imu_count = 0
        self.scan_parts, self.scan_sizes = [], []
        self.truth, self.estimates = [], []
        self.acceleration_norms, self.body_heights = [], []
        self.cloud_frames = 0
        self.map = o3d.geometry.PointCloud()
        self.imu_publisher = self.create_publisher(Imu, "/sim/imu", 100)
        self.lidar_publisher = self.create_publisher(PointCloud2, "/sim/lidar/points", 10)
        self.truth_publisher = self.create_publisher(Odometry, "/sim/ground_truth", 10)
        self.path_publisher = self.create_publisher(RosPath, "/sim/ground_truth_path", 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.reference_publisher = self.create_publisher(PointCloud2, "/sim/reference_map", qos)
        self.tf = StaticTransformBroadcaster(self)
        self.path = RosPath(header=Header(frame_id="x2_map"))
        self.create_subscription(Odometry, "/aima/hal/odom/state", self.on_native_odometry, qos_profile_sensor_data)
        self.create_subscription(Imu, "/aima/hal/imu/chest/state", self.on_native_imu, qos_profile_sensor_data)
        for area in ("leg", "waist", "head", "arm"):
            self.create_subscription(JointStateArray, f"/aima/hal/joint/{area}/state",
                                     lambda message, area=area: self.on_joints(area, message), qos_profile_sensor_data)
        self.create_subscription(Odometry, "/mujoco_slam/odometry", self.on_estimate, 100)
        self.create_subscription(PointCloud2, "/mujoco_slam/cloud_registered", self.on_cloud, 10)
        self.timer = self.create_timer(.005, self.sample_lidar)
        azimuth, elevation = np.meshgrid(np.arange(COLUMNS)*2*math.pi/COLUMNS,
                                         np.linspace(math.radians(-25), math.radians(25), RINGS), indexing="ij")
        self.directions = np.stack((np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)), axis=-1)
        self.groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.fields = [PointField(name=name, offset=POINT_DTYPE.fields[name][1], count=1,
                                 datatype=PointField.UINT16 if name == "ring" else PointField.FLOAT32)
                       for name in ("x", "y", "z", "intensity", "time", "ring")]

    @property
    def ready(self):
        return self.last_odometry is not None and self.last_imu is not None and len(self.areas) == 4

    def on_joints(self, area, message):
        self.areas.add(area)
        self.joint_messages += 1
        for joint in message.joints:
            self.joints[joint.name] = joint.position

    def on_native_odometry(self, message):
        self.last_odometry = message

    def on_native_imu(self, message):
        self.last_imu = message
        self.imu_window.append(message)
        if not self.active:
            return
        seconds = message.header.stamp.sec + message.header.stamp.nanosec*1e-9
        if seconds <= self.last_imu_stamp:
            return
        self.last_imu_stamp = seconds
        message.header.frame_id = "sim_lidar"
        self.imu_publisher.publish(message)
        self.imu_count += 1
        if seconds - self.time_origin <= self.args.warmup:
            a = message.linear_acceleration
            self.acceleration_norms.append(math.sqrt(a.x*a.x+a.y*a.y+a.z*a.z))

    def update_kinematics(self):
        pose = self.last_odometry.pose.pose
        p, q = pose.position, pose.orientation
        self.data.qpos[:7] = [p.x, p.y, p.z, q.w, q.x, q.y, q.z]
        for name, value in self.joints.items():
            index = self.joint_offsets.get(name)
            if index is not None and index >= 7:
                self.data.qpos[index] = value
        mujoco.mj_kinematics(self.model, self.data)

    def activate(self):
        self.update_kinematics()
        self.origin = self.data.site_xpos[self.site_id].copy() - self.map_offset
        self.initial_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.initial_quaternion = Rotation.from_matrix(self.initial_rotation).as_quat()
        norms = [np.linalg.norm([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
                 for m in self.imu_window]
        self.initial_acceleration_norm = float(np.mean(norms))
        if not 7. < self.initial_acceleration_norm < 13.:
            raise RuntimeError("Chest IMU gravity is invalid; check generated chest sensor definitions")
        imu_q = self.last_imu.orientation
        measured_rotation = Rotation.from_quat([imu_q.x, imu_q.y, imu_q.z, imu_q.w])
        self.imu_fk_orientation_error = float((measured_rotation.inv()*Rotation.from_matrix(self.initial_rotation)).magnitude())
        if self.imu_fk_orientation_error > .15:
            raise RuntimeError("Chest IMU orientation and mirrored robot kinematics disagree")
        self.time_origin = self.get_clock().now().nanoseconds*1e-9
        transform = TransformStamped(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="x2_map"), child_frame_id="sim_map")
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = self.origin.tolist()
        q = self.initial_quaternion
        transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = q.tolist()
        self.tf.sendTransform(transform)
        cloud = o3d.io.read_point_cloud(str(self.args.map_dir / "cloud.pcd")).voxel_down_sample(.20)
        reference = point_cloud2.create_cloud_xyz32(Header(frame_id="x2_map"), np.asarray(cloud.points))
        self.reference_publisher.publish(reference)
        self.active = True

    def on_estimate(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        seconds = message.header.stamp.sec + message.header.stamp.nanosec*1e-9
        self.estimates.append([seconds, p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    def on_cloud(self, message):
        xyz = point_cloud2.read_points_numpy(message, field_names=["x", "y", "z"], skip_nans=True)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=float))
        self.map += cloud
        self.cloud_frames += 1
        if self.cloud_frames % 20 == 0:
            self.map = self.map.voxel_down_sample(.10)

    def sample_lidar(self):
        if not self.active or not self.ready:
            return
        self.update_kinematics()
        seconds = self.last_odometry.header.stamp.sec + self.last_odometry.header.stamp.nanosec*1e-9
        if seconds <= self.last_scan_sample:
            return
        self.last_scan_sample = seconds
        position = self.data.site_xpos[self.site_id].copy()
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        quaternion = Rotation.from_matrix(rotation).as_quat()
        original_position = position - self.map_offset
        self.truth.append([seconds, *original_position.tolist(), *quaternion.tolist()])
        self.body_heights.append(self.last_odometry.pose.pose.position.z)
        batch = len(self.scan_parts)
        if batch == 0:
            self.scan_begin = seconds
        count = COLUMNS // SCAN_STEPS
        local = self.directions[batch*count:(batch+1)*count].reshape(-1, 3)
        world = np.ascontiguousarray(local @ rotation.T)
        distances = np.empty(len(world), dtype=float)
        ids = np.empty(len(world), dtype=np.int32)
        mujoco.mj_multiRay(self.model, self.data, position, world.ravel(), self.groups, True,
                          self.body_id, ids, distances, None, len(distances), 30.)
        valid = (distances >= .3) & (distances <= 30.)
        points = np.zeros(int(valid.sum()), dtype=POINT_DTYPE)
        xyz = local[valid]*distances[valid, None]
        for i, name in enumerate(("x", "y", "z")):
            points[name] = xyz[:, i]
        points["intensity"] = 100.
        points["time"] = seconds-self.scan_begin
        points["ring"] = np.tile(np.arange(RINGS, dtype=np.uint16), count)[valid]
        self.scan_parts.append(points)
        if len(self.scan_parts) != SCAN_STEPS:
            return
        points = np.concatenate(self.scan_parts)
        cloud = PointCloud2(header=Header(stamp=stamp(self.scan_begin), frame_id="sim_lidar"))
        cloud.height, cloud.width = 1, len(points)
        cloud.fields, cloud.point_step, cloud.row_step = self.fields, POINT_DTYPE.itemsize, len(points)*POINT_DTYPE.itemsize
        cloud.is_dense, cloud.data = True, points.tobytes()
        self.lidar_publisher.publish(cloud)
        self.scan_sizes.append(len(points))
        self.scan_parts.clear()
        truth = Odometry(header=Header(stamp=stamp(seconds), frame_id="x2_map"), child_frame_id="x2_ground_truth")
        truth.pose.pose.position.x, truth.pose.pose.position.y, truth.pose.pose.position.z = original_position.tolist()
        q = truth.pose.pose.orientation
        q.x, q.y, q.z, q.w = quaternion.tolist()
        self.truth_publisher.publish(truth)
        self.path.poses.append(PoseStamped(header=truth.header, pose=truth.pose.pose))
        self.path.header.stamp = truth.header.stamp
        self.path_publisher.publish(self.path)

    def save(self, output, elapsed, final_action):
        self.map = self.map.voxel_down_sample(.1)
        if not self.map.is_empty():
            if not o3d.io.write_point_cloud(str(output / "map_merged.pcd"), self.map):
                raise RuntimeError("Failed to write X2 SLAM point cloud")
        truth, estimates = np.asarray(self.truth), np.asarray(self.estimates)
        errors = np.array([])
        if len(truth) and len(estimates):
            estimates = estimates[(estimates[:, 0] >= self.time_origin+self.args.warmup) & (estimates[:, 0] <= truth[-1, 0])]
            if len(estimates):
                expected = np.column_stack([np.interp(estimates[:, 0], truth[:, 0], truth[:, i]) for i in (1, 2, 3)])
                expected = (expected-self.origin) @ self.initial_rotation
                errors = np.linalg.norm(estimates[:, 1:4]-expected, axis=1)
        summary = {
            "map_source": str(self.args.map_dir), "ros_domain_id": self.args.domain,
            "time_origin": self.time_origin, "wall_seconds": elapsed,
            "imu_messages": self.imu_count, "joint_messages": self.joint_messages,
            "lidar_scans": len(self.scan_sizes), "registered_cloud_frames": self.cloud_frames,
            "saved_map_points": len(self.map.points), "estimated_poses_after_warmup": len(estimates),
            "position_rmse_m": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "position_error_max_m": float(errors.max()) if len(errors) else None,
            "ground_truth_travel_m": float(np.linalg.norm(np.diff(truth[:, 1:3], axis=0), axis=1).sum()) if len(truth) else 0.,
            "min_pelvis_height_m": min(self.body_heights) if self.body_heights else None,
            "stationary_acceleration_norm": float(np.mean(self.acceleration_norms)) if self.acceleration_norms else None,
            "initial_acceleration_norm": self.initial_acceleration_norm,
            "imu_fk_orientation_error_rad": self.imu_fk_orientation_error,
            "final_controller_action": final_action,
            "frames": {"saved_map": "sim_map", "original_map": "x2_map",
                       "sim_map_origin_in_x2_map": self.origin.tolist(),
                       "sim_map_rotation_in_x2_map": self.initial_quaternion.tolist()},
            "scope": "Native X2 whole-body physics and MC locomotion; optical scanned map, flat contact floor",
        }
        summary["validation_passed"] = bool(len(errors)>100 and self.cloud_frames>=20 and len(self.map.points)>1000
                                             and summary["ground_truth_travel_m"]>.5
                                             and summary["min_pelvis_height_m"]>.4
                                             and summary["position_rmse_m"]<.5)
        np.savez_compressed(output / "trajectories.npz", ground_truth=truth, estimated=estimates)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
        if len(errors):
            from plot_results import plot
            plot(output)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--sim-root", type=Path, default=Path("/workspace/sim_mujoco"))
    parser.add_argument("--mc-root", type=Path, default=Path("/workspace/mc"))
    parser.add_argument("--domain", type=int, default=75)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--warmup", type=float, default=3.)
    parser.add_argument("--walk-seconds", type=float, default=6.)
    parser.add_argument("--speed", type=float, default=.25)
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--no-walk", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    from aimdk_msgs.msg import JointState
    from aimdk_msgs.srv import SetMcAction
    if not (hasattr(SetMcAction.Request(), "source") and hasattr(JointStateArray(), "state")
            and hasattr(JointState(), "error_code")):
        parser.error("Rebuild aimdk_msgs and py_examples to match the installed AIMDK SDK interfaces")
    if not 0 <= args.domain <= 232:
        parser.error("domain must be between 0 and 232")
    for name in ("duration", "warmup", "walk_seconds", "speed"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name)<=0:
            parser.error(name+" must be positive and finite")
    if args.speed > .4:
        parser.error("This walking verification limits speed to 0.4 m/s")
    if not os.environ.get("DISPLAY"):
        parser.error("The installed native simulator requires a desktop DISPLAY")
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    args.map_dir = args.map_dir.resolve()
    scene_dir = HERE / "generated" / args.map_dir.name
    prepare(args.map_dir, scene_dir, .20)
    profile, metadata = prepare_native(scene_dir, args.map_dir, args.sim_root, args.mc_root)
    output = profile / "runs" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    print("X2 run output:", output, flush=True)
    stack = NativeStack(profile, output, args.sim_root, args.mc_root)
    rclpy.init(args=[])
    node = X2Bridge(args, profile, metadata)
    control = X2Control(node, stack.check)
    mapping = rviz = None
    log = (output / "fast_lio.log").open("w")
    rviz_log = (output / "rviz.log").open("w")
    started = time.monotonic()
    summary = None
    final_action = ""
    try:
        stack.start()
        deadline = time.monotonic()+60
        while not node.ready:
            stack.check()
            if time.monotonic()>deadline:
                raise RuntimeError("X2 sensor startup timed out; inspect simulator/controller logs")
            rclpy.spin_once(node, timeout_sec=.02)
        print(f"Native X2 streams ready; pelvis_z={node.last_odometry.pose.pose.position.z:.3f}", flush=True)
        control.pump(1.)
        control.register()
        control.set_mode("JOINT_DEFAULT")
        control.pump(.5)
        print(f"X2 joint control initialized; pelvis_z={node.last_odometry.pose.pose.position.z:.3f}", flush=True)
        stack.window_key(0xff08)
        reset_deadline = time.monotonic()+1.
        while node.last_odometry.pose.pose.position.z < .5 and time.monotonic()<reset_deadline:
            control.pump(.005)
        if node.last_odometry.pose.pose.position.z < .5:
            raise RuntimeError("The simulator did not acknowledge the initial-pose reset")
        print(f"X2 initial pose reset; pelvis_z={node.last_odometry.pose.pose.position.z:.3f}", flush=True)
        control.set_mode("STAND_DEFAULT")
        control.pump(2.)
        print(f"X2 after balance stand; pelvis_z={node.last_odometry.pose.pose.position.z:.3f}", flush=True)
        if node.last_odometry.pose.pose.position.z < .4:
            raise RuntimeError("X2 did not reach an upright standing pose")
        config = mapping_parameters()
        params = config["/**"]["ros__parameters"]
        params.update({"use_sim_time": False, "mapping.acc_cov": .1, "mapping.gyr_cov": .01,
                       "mapping.b_acc_cov": .0001, "mapping.b_gyr_cov": .0001})
        (output / "mapping.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        mapping = subprocess.Popen(["ros2", "run", "spark_fast_lio", "spark_lio_mapping", "--ros-args",
                                    "--params-file", str(output / "mapping.yaml"), "-r", "__ns:=/mujoco_slam",
                                    "-r", "lidar:=/sim/lidar/points", "-r", "imu:=/sim/imu"],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic()+15
        while not (node.lidar_publisher.get_subscription_count() and node.imu_publisher.get_subscription_count()):
            stack.check()
            if mapping.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError("FAST-LIO startup failed; inspect fast_lio.log")
            rclpy.spin_once(node, timeout_sec=.02)
        node.activate()
        if not args.no_rviz:
            rviz = subprocess.Popen(["rviz2", "-d", str(HERE / "slam.rviz")], stdout=rviz_log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        transitioned = False
        last_status = started
        while time.monotonic()-started<args.duration:
            stack.check()
            if mapping.poll() is not None:
                raise RuntimeError("FAST-LIO exited unexpectedly")
            elapsed = time.monotonic()-started
            if not args.no_walk and not transitioned and elapsed>=args.warmup:
                control.forward = args.speed
                control.publish()
                control.set_mode("LOCOMOTION_DEFAULT")
                transitioned = True
                print("X2 locomotion active", flush=True)
            control.forward = args.speed if transitioned and elapsed<args.warmup+args.walk_seconds else 0.
            rclpy.spin_once(node, timeout_sec=.002)
            if node.last_odometry.pose.pose.position.z < .35:
                control.forward = 0.
                raise RuntimeError("X2 fell; velocity cleared and run stopped")
            if time.monotonic()-last_status>=5:
                print(f"elapsed={elapsed:.1f}s scans={len(node.scan_sizes)} registered={node.cloud_frames} pelvis_z={node.last_odometry.pose.pose.position.z:.3f}", flush=True)
                last_status = time.monotonic()
        control.forward = 0.
        control.pump(1.)
        final_action = control.current_action()
    except KeyboardInterrupt:
        pass
    finally:
        control.forward = 0.
        control.publish()
        stop(mapping)
        stop(rviz)
        stack.close()
        log.close()
        rviz_log.close()
        try:
            if node.active:
                summary = node.save(output, time.monotonic()-started, final_action)
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
    return 1 if args.check and (not summary or not summary["validation_passed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
