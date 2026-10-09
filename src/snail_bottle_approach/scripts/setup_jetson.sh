#!/usr/bin/env bash
# Run on the Jetson after copying both packages into this workspace.
set -eo pipefail
workspace="${1:-$HOME/r20_ws}"
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vision_venv="${SNAIL_VISION_VENV:-$HOME/.venvs/snail-bottle}"
if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo 'ROS 2 Humble is required at /opt/ros/humble.' >&2
  exit 1
fi
if [ ! -f "$workspace/src/snail_hardware/package.xml" ] || \
   [ ! -f "$workspace/src/snail_bottle_approach/package.xml" ]; then
  echo 'Copy snail_hardware and snail_bottle_approach into WORKSPACE/src first.' >&2
  exit 1
fi
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Ubuntu 22.04 Python 3.10 with Humble"'
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-opencv python3-serial \
  python3-colcon-common-extensions ros-humble-diagnostic-msgs
python3 -m venv --system-site-packages "$vision_venv"
source "$vision_venv/bin/activate"
python -m pip install -r "$package_dir/requirements-jetson.txt"
source /opt/ros/humble/setup.bash
cd "$workspace"
# Running colcon with the venv interpreter gives console scripts the same
# interpreter as ONNX Runtime. Merely activating a venv before /usr/bin/colcon
# would still build scripts with the system interpreter.
python /usr/bin/colcon build --symlink-install \
  --packages-select snail_hardware snail_bottle_approach
source install/setup.bash
python -c 'import cv2, onnxruntime, rclpy, serial; from snail_bottle_approach.node import BottleApproach; print("PASS: ROS, camera conversion, serial and ONNX Runtime imports")'
printf '\nBuild ready. In each new terminal run:\nsource %s/bin/activate\nsource /opt/ros/humble/setup.bash\nsource %s/install/setup.bash\n' "$vision_venv" "$workspace"
