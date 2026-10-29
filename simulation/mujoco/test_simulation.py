"""Checks for geometric holes, frame preservation, and physical sensor units."""

import json
import mujoco
import numpy as np
import open3d as o3d

from prepare_scene import prepare
from run_slam import Trajectory, stamp


def test_mesh_preserves_empty_space_and_world_coordinates(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector([[0.25, 0.25, 0.25], [2.25, 0.25, 0.25]])
    assert o3d.io.write_point_cloud(str(source / "cloud.pcd"), cloud)
    (source / "trajectory.json").write_text(json.dumps([[1.5, -2., .5], [1.5, -1.5, .5]]))
    output = tmp_path / "scene"
    info = prepare(source, output, 1.)
    assert info["occupied_voxels"] == 2
    assert prepare(source, output, 1.) == info
    model = mujoco.MjModel.from_binary_path(str(output / "scene.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    # A mesh collision hull would close this gap. Raycasting must use actual faces.
    gap = mujoco.mj_ray(model, data, np.array([1.5, -1., .5]), np.array([0., 1., 0.]), group, True, -1, None)
    wall = mujoco.mj_ray(model, data, np.array([.5, -1., .5]), np.array([0., 1., 0.]), group, True, -1, None)
    assert gap == -1
    assert np.isclose(wall, 1.)
    np.testing.assert_allclose(data.sensor("accelerometer").data, [0., 0., 9.81], atol=1e-10)
    np.testing.assert_allclose(data.sensor("gyroscope").data, [0., 0., 0.], atol=1e-10)


def test_trajectory_starts_stationary_and_respects_endpoints():
    plan = Trajectory(np.array([[2., 3., 0.], [3., 3., 0.], [4., 3., 0.]]), .3, 3.)
    np.testing.assert_allclose(plan.target(0), [2., 3., 0.])
    np.testing.assert_allclose(plan.target(3), [2., 3., 0.])
    assert np.linalg.norm(plan.target(3.005)[:2] - plan.target(3)[:2]) < .00001
    np.testing.assert_allclose(plan.target(100)[:2], [4., 3.])


def test_nanosecond_rounding_carries_into_seconds():
    result = stamp(1.9999999999)
    assert result.sec == 2
    assert result.nanosec == 0


def test_moving_imu_integrates_to_physical_position():
    # A moving position target must update inside RK4 substeps. This catches
    # acceleration aliasing that a stationary gravity check cannot detect.
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.005" integrator="RK4" gravity="0 0 -9.81"/>
      <worldbody><body>
        <joint name="x" type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <site name="imu"/>
      </body></worldbody>
      <actuator><position joint="x" kp="250" kv="35"/></actuator>
      <sensor><accelerometer name="acc" site="imu"/></sensor>
    </mujoco>''')
    data = mujoco.MjData(model)
    plan = Trajectory(np.array([[0.,0.,0.],[3.,0.,0.]]), .3, 1.)
    def callback(model, data):
        data.ctrl[0] = plan.target(data.time)[0]
    mujoco.set_mjcb_control(callback)
    velocity = position = previous_acceleration = 0.
    try:
        mujoco.mj_forward(model, data)
        for _ in range(2000):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            acceleration = float(data.sensor("acc").data[0])
            new_velocity = velocity + .5*(previous_acceleration+acceleration)*.005
            position += .5*(velocity+new_velocity)*.005
            previous_acceleration, velocity = acceleration, new_velocity
        assert abs(position - data.qpos[0]) < .001
    finally:
        mujoco.set_mjcb_control(None)
