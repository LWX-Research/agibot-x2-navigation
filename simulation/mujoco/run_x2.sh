#!/usr/bin/env bash
set -eo pipefail
x2_slam_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
x2_slam_workspace_dir=$(cd -- "$x2_slam_script_dir/../.." && pwd)
source /opt/ros/humble/setup.bash
source "$x2_slam_workspace_dir/install/setup.bash"
exec python3 "$x2_slam_script_dir/run_x2_slam.py" "$@"
