#!/usr/bin/env bash
set -eo pipefail
slam_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
slam_workspace_dir=$(cd -- "$slam_script_dir/../.." && pwd)
source /opt/ros/humble/setup.bash
if [[ ! -f "$slam_workspace_dir/install/setup.bash" ]]; then
  echo "请先编译工作空间：colcon build --symlink-install --packages-up-to py_examples spark_fast_lio" >&2
  exit 1
fi
source "$slam_workspace_dir/install/setup.bash"
exec python3 "$slam_script_dir/run_slam.py" "$@"
