"""Prepare isolated X2 simulator/controller profiles and manage their processes."""

import json
import ctypes
import ctypes.util
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
import yaml

from run_slam import stop


ROBOT = "lx2501_3_t2d5"


def prepare_native(scene_dir, map_dir, sim_root, mc_root):
    scene_dir, map_dir = Path(scene_dir), Path(map_dir)
    sim_robot = Path(sim_root) / "configuration/robot" / ROBOT
    mc_robot = Path(mc_root) / "bin/mc_param/src/robot" / ROBOT
    output = scene_dir / "x2"
    output.mkdir(exist_ok=True)
    model = ET.parse(sim_robot / "model_info/x2.xml")
    model.find("compiler").set("meshdir", str(sim_robot / "meshes"))
    sensor = model.find("sensor")
    existing = {item.get("name") for item in sensor}
    for tag, name in (
        ("framequat", "body-orientation-imu_1"),
        ("gyro", "body-angular-velocity-imu_1"),
        ("framepos", "body-linear-pos-imu_1"),
        ("velocimeter", "body-linear-vel-imu_1"),
        ("accelerometer", "body-linear-acceleration-imu_1"),
    ):
        if name in existing:
            continue
        attributes = {"name": name}
        if tag in ("framequat", "framepos"):
            attributes.update(objtype="site", objname="imu_1")
        else:
            attributes["site"] = "imu_1"
        ET.SubElement(sensor, tag, attributes)
    model.write(output / "x2.xml")

    points = np.asarray(o3d.io.read_point_cloud(str(map_dir / "cloud.pcd")).points)
    ground = points[(points[:, 2] > -2.) & (points[:, 2] < -.4), 2]
    if not len(ground):
        raise ValueError("Cannot identify a ground band below the scanned-map origin")
    histogram, edges = np.histogram(ground, bins=np.arange(-2., -.39, .02))
    index = int(histogram.argmax())
    floor_z = float(np.median(ground[(ground >= edges[index]) & (ground < edges[index+1])]))
    offset = np.array([0., 0., -floor_z])
    scene = ET.parse(sim_robot / "model_info/scene.xml")
    scene.find("include").set("file", str(output / "x2.xml"))
    ET.SubElement(scene.find("asset"), "mesh", name="scanned_map",
                  file=str(scene_dir / "map.obj"), inertia="shell", maxhullvert="64")
    ET.SubElement(scene.find("worldbody"), "geom", name="scanned_surfaces", type="mesh", mesh="scanned_map",
                  pos=" ".join(str(value) for value in offset), contype="0", conaffinity="0", group="0",
                  rgba="0.55 0.70 0.80 1")
    scene.write(output / "scene.xml")
    config = yaml.safe_load((sim_robot / "simulator/default.yaml").read_text())
    config["MujocoSimModule"]["simulation_model"] = str(output / "scene.xml")
    config["aimrt"]["configurator"]["temp_cfg_path"] = str(output / "sim_config")
    for backend in config["aimrt"]["log"]["backends"]:
        if backend["type"] == "rotate_file":
            backend["options"]["path"] = str(output / "sim_logs")
    (output / "simulator.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    # Keep generated controller configuration outside the installed MC tree.
    parameter_root = output / "mc_params"
    parameters = parameter_root / ROBOT
    if not parameters.exists():
        shutil.copytree(mc_robot, parameters, ignore=shutil.ignore_patterns("sim", "rl_models", "urdf"))
        for name in ("rl_models", "urdf"):
            if (mc_robot / name).exists():
                (parameters / name).symlink_to(mc_robot / name, target_is_directory=True)
    action_path = parameters / "action_state.yaml"
    actions = yaml.safe_load(action_path.read_text())
    actions["default_action"]["inited"] = "JOINT_DEFAULT"
    action_path.write_text(yaml.safe_dump(actions, sort_keys=False))
    config = yaml.safe_load((mc_robot / "mc_sim.yaml").read_text())
    config["MotionControlModule"]["env_path"] = str(parameter_root)
    config.setdefault("aimrt", {}).setdefault("configurator", {})["temp_cfg_path"] = str(output / "mc_config")
    (output / "controller.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {"map_offset_in_native_world": offset.tolist(), "floor_z_in_original_map": floor_z,
                "robot": ROBOT, "bootstrap_action": "JOINT_DEFAULT",
                "geometry_scope": "Scanned surfaces are optical; native flat floor provides foot contact"}
    (output / "profile.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output, metadata


class NativeStack:
    def __init__(self, profile, output, sim_root, mc_root):
        self.profile, self.output = Path(profile), Path(output)
        self.sim_root, self.mc_root = Path(sim_root), Path(mc_root)
        self.processes = []
        self.logs = []

    def start(self):
        env = os.environ.copy()
        env.pop("AGIBOT_SOFTWARE_HOME", None)
        env.update(RMW_IMPLEMENTATION="rmw_fastrtps_cpp", AGIBOT_ENABLE_HDS_COMPONENT="false",
                   AGIBOT_ENABLE_EVENT_COMPONENT="false", AGIBOT_ENABLE_MONITOR_COMPONENT="false",
                   AGIBOT_ENABLE_MONITOR="true", AGIBOT_ENABLE_AUDIT_COMPONENT="false",
                   AIMRT_PLUGIN_SHM_DEFAULT_WAIT_TIME_US="300")
        sim_env = env.copy()
        sim_env.update(SIM_ROBOT_PATH=str(self.sim_root / "configuration/robot" / ROBOT), EM_APP_NAME="sim")
        sim_env["LD_LIBRARY_PATH"] = f"{self.sim_root}/bin:{self.sim_root}/lib:" + env.get("LD_LIBRARY_PATH", "")
        mc_env = env.copy()
        mc_env.update(MC_ENV_PATH=str(self.profile / "mc_params"), EM_APP_NAME="mc", SUB_NO_SCHEDULER="true")
        mc_env["LD_LIBRARY_PATH"] = f"{self.mc_root}/lib:{self.mc_root}/mc3th/lib:{self.mc_root}/bin:" + env.get("LD_LIBRARY_PATH", "")
        for command, cwd, process_env, log_name in (
            (["./aima-sim-app", "--cfg_file_path="+str(self.profile / "simulator.yaml")], self.sim_root / "bin", sim_env, "x2_simulator.log"),
            (["./aima-mc-app-main", "--cfg_file_path="+str(self.profile / "controller.yaml")], self.mc_root / "bin", mc_env, "x2_controller.log"),
        ):
            stream = (self.output / log_name).open("w")
            self.logs.append(stream)
            self.processes.append(subprocess.Popen(command, cwd=cwd, env=process_env, stdout=stream,
                                                   stderr=subprocess.STDOUT, start_new_session=True))

    def check(self):
        if any(process.poll() is not None for process in self.processes):
            raise RuntimeError(f"X2 simulator/controller exited; see logs in {self.output}")

    def window_key(self, keysym, timeout=5.):
        """Send a viewer shortcut only to the window owned by our simulator PID."""
        deadline = time.monotonic()+timeout
        window = None
        while time.monotonic()<deadline:
            self.check()
            result = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                parts = line.split(None, 1)
                if not parts or not parts[0].startswith("0x") or '"MuJoCo' not in line:
                    continue
                prop = subprocess.run(["xprop", "-id", parts[0], "_NET_WM_PID"], capture_output=True, text=True)
                if prop.stdout.strip().endswith("= "+str(self.processes[0].pid)):
                    window = int(parts[0], 16)
                    break
            if window is not None:
                break
            time.sleep(.05)
        if window is None:
            raise RuntimeError("Cannot find the X2 simulator window owned by this run")

        class KeyEvent(ctypes.Structure):
            _fields_ = [("type",ctypes.c_int),("serial",ctypes.c_ulong),("send_event",ctypes.c_int),
                        ("display",ctypes.c_void_p),("window",ctypes.c_ulong),("root",ctypes.c_ulong),
                        ("subwindow",ctypes.c_ulong),("time",ctypes.c_ulong),
                        ("x",ctypes.c_int),("y",ctypes.c_int),("x_root",ctypes.c_int),("y_root",ctypes.c_int),
                        ("state",ctypes.c_uint),("keycode",ctypes.c_uint),("same_screen",ctypes.c_int)]

        class Event(ctypes.Union):
            _fields_ = [("xkey",KeyEvent),("padding",ctypes.c_long*24)]

        x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x11.XKeysymToKeycode.restype = ctypes.c_uint
        x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
        x11.XSendEvent.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.c_int,ctypes.c_long,ctypes.POINTER(Event)]
        x11.XFlush.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display = x11.XOpenDisplay(None)
        if not display:
            raise RuntimeError("Cannot connect to the native simulator DISPLAY")
        try:
            event = Event()
            event.xkey.display, event.xkey.window = display, window
            event.xkey.root = x11.XDefaultRootWindow(display)
            event.xkey.keycode = x11.XKeysymToKeycode(display, keysym)
            event.xkey.same_screen = 1
            for event_type, mask in ((2,1),(3,2)):
                event.xkey.type = event_type
                if not x11.XSendEvent(display, window, False, mask, ctypes.byref(event)):
                    raise RuntimeError("Could not deliver the simulator viewer shortcut")
            x11.XFlush(display)
        finally:
            x11.XCloseDisplay(display)

    def close(self):
        for process in reversed(self.processes):
            stop(process)
        for stream in self.logs:
            stream.close()
