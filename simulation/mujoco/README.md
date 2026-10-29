# 用扫描地图运行 MuJoCo 与 FAST-LIO

输入是已有的 `cloud.pcd` 和 `trajectory.json`。脚本将点云转换为体素表面网格，
在 MuJoCo 中加载网格，让传感器载体沿轨迹平滑移动，并启动独立的 FAST-LIO。
原始地图目录保持不变，生成文件均写到 `simulation/mujoco/generated/`。

## X2 全身行走与建图

现有 X2 原生仿真和 MC 控制器也已接入，使用：

```bash
bash /workspace/agibot_ws/simulation/mujoco/run_x2.sh
```

这会启动已有 X2 MuJoCo 程序、原有 MC 控制器与 RL 策略、胸部虚拟雷达、
FAST-LIO 和 RViz。机器人先进入位控，再复位初始姿态并切入平衡站立，完成
IMU 初始化后进行短距离行走，最后回到零速度站立。默认记录约 20 秒，自动
行走指令持续 6 秒；这是 X2 的全身物理运动，没有使用载体轨迹替代机器人位姿。

```bash
# 验证行走、姿态稳定、雷达、IMU 和建图结果
bash /workspace/agibot_ws/simulation/mujoco/run_x2.sh --no-rviz --check
```

原生仿真程序需要桌面 `DISPLAY`，并使用系统的 `xwininfo`、`xprop` 和 X11。
脚本仅向本次启动的仿真进程窗口发送一次
[Backspace 复位快捷键](https://mujoco.readthedocs.io/en/stable/programming/samples.html#shortcuts)。
首次加载机器人与扫描地图可能需要约 20 秒。修改 `--walk-seconds`、`--speed`
可调整行走指令，验证程序将速度限制为不超过 0.4 m/s。

X2 模式默认使用 ROS 域 **75**，保留原生程序的系统时间戳，不发布 `/clock`。
它直接转发原生胸部 IMU，并利用原生基座位姿和四个关节状态话题同步一份
MuJoCo 运动学模型，按胸部 IMU 安装位置对场景进行射线扫描。
运动学模型只用于计算虚拟雷达；物理仿真与站立、行走均由原生 X2 和 MC 执行。
真值只参与传感器仿真、显示与误差统计，不作为位姿观测输入 FAST-LIO。

原安装模型缺少胸部 IMU 配置引用的五个传感器，现已在生成的独立模型中补齐。
仓库的 AIMDK 接口已与当前安装的 SDK 对齐：`SetMcAction.source`、
`JointStateArray.state`、`JointState.error_code`。因此需重新编译 `aimdk_msgs`
和 `py_examples`；下面的 `--packages-up-to` 构建命令会包含它们。
若连接不同版本的真机 SDK，需要采用与该机器人一致的接口定义。

X2 的模型、控制器参数与日志写在 `generated/<地图目录名>/x2/`，原安装配置不覆盖。
每次运行结果位于其 `runs/<运行时间>/`，包括新地图、轨迹、误差图、
控制器和仿真器日志。机器人骨盆高度低于 0.35 m 时会清零速度并停止本次运行。

扫描场景目前参与光学扫描，足部接触使用原生平地；墙面和障碍物尚未参与
机器人碰撞，也没有加入 Nav2 避障。该模式验证全身行走下的建图链路，
不代表真实障碍物交互或导航验收。原地图和原生安装配置均保持不变。

## 启动

当前工作区已安装依赖并完成所需 ROS 包的编译。直接运行：

```bash
bash /workspace/agibot_ws/simulation/mujoco/run.sh
```

有桌面环境时会打开 MuJoCo 和 RViz；默认沿整条轨迹运行，跑完后停在末尾，窗口持续保持打开。
关闭运行命令的终端、按 `Ctrl+C` 或关闭 MuJoCo 窗口时，会保存结果并退出 MuJoCo、RViz 和建图进程。
显式指定正数 `--duration` 时，在指定仿真时长后自动保存结果并关闭窗口；`--headless` 默认跑完整条轨迹后退出。
MuJoCo 中橙色球体表示传感器载体。RViz 中绿色为 FAST-LIO 轨迹，橙色为真值轨迹。

无窗口验证 30 秒：

```bash
bash /workspace/agibot_ws/simulation/mujoco/run.sh \
  --headless --duration 30 --check
```

使用其他扫描地图目录：

```bash
bash /workspace/agibot_ws/simulation/mujoco/run.sh \
  --map-dir /path/to/scanned-map --speed 0.30
```

默认 ROS 域为 `73`，用于隔离已有机器人和仿真会话。其他终端查看此仿真时设置
`export ROS_DOMAIN_ID=73`。`--domain` 可指定其他域。`--realtime-factor 2`
请求两倍速度；计算不足时会降低实际运行速度，不跳过物理步骤。

重新安装依赖与编译：

```bash
python3 -m pip install --user -r /workspace/agibot_ws/simulation/mujoco/requirements.txt
source /opt/ros/humble/setup.bash
cd /workspace/agibot_ws
colcon build --symlink-install --packages-up-to py_examples spark_fast_lio
```

## 数据与结果

虚拟雷达通过 MuJoCo 对地图网格逐批射线求交，16 线、360 列、10 Hz，
每帧最多 5,760 点。每个批次按实际采样时刻和传感器姿态计算，点云携带
`x/y/z/intensity/time/ring`，逐点相对时间单位为秒。
IMU 使用 MuJoCo 的加速度计和陀螺仪传感器，200 Hz；控制器在 RK4 的中间
步骤按物理时间更新，避免控制目标变化对加速度采样造成偏差。
雷达与 IMU 共用同一安装位置与方向，外参为单位变换，直接输入仿真专用配置。
所有估计节点使用 `/clock`。真值轨迹只参与显示与误差统计，不输入 FAST-LIO。

| 话题 | 用途 |
| --- | --- |
| `/sim/lidar/points` | 虚拟三维雷达 |
| `/sim/imu` | 虚拟 IMU |
| `/sim/ground_truth` | MuJoCo 真值位姿 |
| `/sim/reference_map` | 原始点云参考图 |
| `/mujoco_slam/odometry` | FAST-LIO 估计位姿 |
| `/mujoco_slam/cloud_registered` | 配准后的点云 |

每次运行的输出目录会在终端打印：

```text
generated/<地图目录名>/
  scene.xml / scene.mjb / map.obj / scene_info.json
  runs/<运行时间>/
    map_merged.pcd       # 仿真重新建立的 XYZ 地图，坐标系 sim_map
    summary.json         # 输入规模、帧数、位置 RMSE、验证结果
    trajectories.npz     # 真值和估计轨迹
    validation.png       # 轨迹与位置误差图
    mapping.yaml         # 本次运行实际采用的 FAST-LIO 参数
    fast_lio.log / rviz.log
```

`sim_map` 原点为原始轨迹起点，方向与 `x2_map` 相同；二者静态变换会发布到 TF，
具体平移也写入 `summary.json`。仿真地图单独保存，不覆盖真机地图或 FAST-LIO
真机流程的 `PCD/scans_*.pcd`。

## 载体模式的验证范围

`run.sh` 提供理想传感器下的载体建图验证。轨迹文件只有位置，没有原始时间和姿态，
因此按指定速度插值 XY 路径，固定传感器高度，并加入平滑的偏航运动。
载体由 MuJoCo 执行平面动力学与位置控制；X2 全身行走使用上文的 `run_x2.sh`。
两种模式均未接 Nav2，地图网格用于射线扫描，未作为全身接触碰撞模型。
未观测区域保持空白，原始扫描的噪声、遮挡和漂移也会影响仿真几何。
默认体素为 20 cm，这是近似场景，不代表原始空间的精确重建。

`--check` 要求产生足够位姿和点云、实际移动超过 0.5 m、静止加速度接近
9.81 m/s²，且位置 RMSE 小于 0.5 m。短于初始化与移动时间的运行可能无法通过。
该标准用于检测链路故障，不代表真机定位精度或导航验收。

几何和运动传感器检查：

```bash
source /workspace/agibot_ws/install/setup.bash
python3 -m pytest -q /workspace/agibot_ws/simulation/mujoco/test_simulation.py
```
