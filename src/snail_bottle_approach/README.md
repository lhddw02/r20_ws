# Bottle approach on Jetson / ROS 2 Humble

This package uses the forward-facing D435i RGB image to detect bottles, aligned
depth to estimate their distance, and the existing R20 TWD serial bridge to turn
toward a bottle, approach slowly, and remain stopped. The requested default stop
distance is **0.40 m from the camera**. Forward speed is limited to **0.08 m/s**;
rotation is limited to **0.30 rad/s**. It requires three stable observations,
turns in place to center the bottle, then moves straight. Missing detection,
invalid depth or stale images command zero velocity. Reaching the distance
latches the stop until a new approach is explicitly started.

The supplied YOLOv8n COCO model detects **bottles generally**, including bottles
that do not contain water. It cannot identify the contents. If distinguishing
water bottles from other bottles is necessary, collect examples and train a
dedicated detector before autonomous use.

```text
D435i RGB + aligned depth + intrinsics -> detector -> approach controller
                                            |              |
                               /bottle_approach/image       |
                               /bottle_approach/status      v
                                           /bottle_approach/cmd_vel
                                                        |
                                                 R20 serial bridge
                                                        |
                                                      STM32
```

## Copy the packages and prepared model

A ready-to-copy bundle is at `output/r20-bottle-deploy.zip`. It includes both
packages and the exported model. Copy it to the Jetson using SSH or a USB drive.
From this Windows repository in PowerShell:

```powershell
scp .\output\r20-bottle-deploy.zip nano@192.168.1.107:~/
```

Then on the Jetson:

```bash
mkdir -p ~/r20_ws
unzip -o ~/r20-bottle-deploy.zip -d ~/r20_ws
bash ~/r20_ws/src/snail_bottle_approach/scripts/setup_jetson.sh ~/r20_ws
```

`unzip -o` updates the two supplied package folders; preserve your own changes
before replacing an older copy. The archive has `src/` and `models/` at its root.

If copying individual folders instead, use the following layout and commands.
A prepared 640px CPU ONNX model is at `output/models/bottle_yolov8n.onnx` in the
Windows repository, with a `.json` provenance/checksum sidecar. The model is an
ignored generated artifact; include it explicitly when copying to another
computer. No model download happens while driving.

From PowerShell in `C:\Users\Luka\Documents\Project\snail-robot`:

```powershell
ssh nano@192.168.1.107 'mkdir -p ~/r20_ws/src ~/r20_ws/models'
scp -r .\ws\src\snail_hardware .\ws\src\snail_bottle_approach nano@192.168.1.107:~/r20_ws/src/
scp .\output\models\bottle_yolov8n.onnx .\output\models\bottle_yolov8n.json nano@192.168.1.107:~/r20_ws/models/
```

If SSH authentication is unavailable, copy these same two package folders and
model files using a USB drive. The destination layout must contain
`~/r20_ws/src/snail_hardware`, `~/r20_ws/src/snail_bottle_approach` and
`~/r20_ws/models/bottle_yolov8n.onnx`.

On the Jetson, run the setup script once:

```bash
bash ~/r20_ws/src/snail_bottle_approach/scripts/setup_jetson.sh ~/r20_ws
```

It installs OpenCV and serial dependencies, creates a Python 3.10 virtual
environment inheriting the ROS system packages, installs pinned CPU ONNX Runtime
and NumPy, and builds just the two hardware packages. It uses the venv Python
to run colcon so the installed detector executable can import ONNX Runtime.
It does not install PyTorch or change the Jetson CUDA stack. Rebuild this way
after updating the Python environment; a previous system-Python build will not
see packages installed only in the venv.

In each Jetson terminal used below:

```bash
source ~/.venvs/snail-bottle/bin/activate
source /opt/ros/humble/setup.bash
source ~/r20_ws/install/setup.bash
```

## 1. Enable aligned camera depth

The existing RealSense driver can remain running if these topics exist:

```bash
ros2 topic list | grep -E 'color/image_raw|aligned_depth_to_color'
```

Expected:

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/aligned_depth_to_color/camera_info
```

If aligned depth is absent, stop the old camera launch and restart it with:

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true \
  align_depth.enable:=true enable_sync:=true
```

Alternatively stop the old camera launch and pass `start_camera:=true` to the
bottle launch below. Only one RealSense driver should own the camera. Topic
prefixes vary with the installed wrapper: pass `color_topic:=...`,
`depth_topic:=...` and `camera_info_topic:=...` if yours differ. Depth must be
aligned to the color pixels and have the same optical frame and resolution;
ordinary `/depth/image_rect_raw` is not a substitute. Supported depth encodings
are RealSense `16UC1` millimeters and `32FC1` meters.

## 2. Check bottles and distance without motion

Stop the keyboard launch and any other R20 serial program first. The bottle
launch owns the same serial port and connects the bridge to an exclusive command
topic. Replace `/dev/ttyUSB0` with the port that worked for keyboard control.

```bash
ros2 launch snail_bottle_approach bottle.launch.py \
  model_path:=$HOME/r20_ws/models/bottle_yolov8n.onnx \
  port:=/dev/ttyUSB0 enable_motion:=false stop_distance:=0.40
```

In another sourced terminal on the Jetson desktop:

```bash
ros2 run rqt_image_view rqt_image_view
```

Select `/bottle_approach/image` to see boxes and the measured range. If necessary,
install `ros-humble-rqt-image-view`. Inspect status separately:

```bash
ros2 topic echo /bottle_approach/status
```

Expected: `motion_allowed: false`, `armed: false`, increasing `frames_processed`,
and a plausible `distance_m` for a bottle. Compare it with a tape measure at
several positions, especially near 0.40 m. An empty scene should report no valid
target (`distance_m: null`). Moving the bottle to image right gives a positive
`bearing_rad`; moving it to image left gives a negative value. Detection-only
launch sends only zero commands to the STM32 even if the target is confirmed.

Clear plastic can produce missing depth or apparently valid depth on the wall
behind the bottle. The controller rejects insufficient or inconsistent depth,
but a uniform background return can still pass. **Validate your actual bottle's
depth before movement.** Use a clearly labeled or opaque bottle for initial
trials. This prototype has no general obstacle avoidance or path planning; use
a short, clear path and supervise the test.

## 3. Approach and stop at 0.40 m

Keep the camera fixed and forward-facing. First verify rotation signs with
the wheels raised: a bottle on camera right should turn the chassis right.
Stop the detection-only launch with Ctrl+C, then start:

```bash
ros2 launch snail_bottle_approach bottle.launch.py \
  model_path:=$HOME/r20_ws/models/bottle_yolov8n.onnx \
  port:=/dev/ttyUSB0 enable_motion:=true stop_distance:=0.40
```

This arms the approach when fresh base telemetry and confirmed RGB-D detections
are available. After the raised-wheel check, stop the launch, place the base on
the clear floor with the bottle about 1 m ahead, and launch again. Hold access
to motor power throughout the trial. It approaches the nearest valid bottle,
slows toward 0.40 m, then reports `REACHED` and keeps publishing zero. It never
reverses to back away from a bottle already closer than the stop threshold.
Near the threshold it retains a minimum 0.01 m/s request (or a smaller configured
maximum) to avoid serial rounding to zero before reaching the target. Actual
braking, depth error, inference delay and motor stiction affect the final gap;
the physical stopping accuracy has to be measured on your prototype.

To stop and disarm from another sourced terminal:

```bash
ros2 service call /bottle_approach/stop std_srvs/srv/Trigger '{}'
```

To start a new approach after stopping or reaching the bottle:

```bash
ros2 service call /bottle_approach/start std_srvs/srv/Trigger '{}'
```

After a base connection loss, restored telemetry does not rearm the approach;
call `start` explicitly. Detection-only mode refuses `start` requests. A normal
camera/detection outage stops movement; fresh observations must be confirmed
again before it resumes an armed approach. Ctrl+C shuts down the controller and
bridge and sends final zeros while serial is usable.

The supplied STM32 source has **no MCU velocity watchdog**. Host stop commands
cannot protect against cable loss, Jetson power failure or a killed process
that cannot send zeros. Keep motor power accessible and add/verify an MCU
command timeout before unattended operation. This is the same firmware limitation
documented by the working keyboard package.

## Distance and integration parameters

With `camera_to_front:=0.0`, the 0.40 m threshold is optical forward depth from
the camera. If the camera sits 0.10 m behind the robot's front and you want a
0.40 m **front clearance**, use `camera_to_front:=0.10`; the node subtracts that
offset from measured depth. This assumes a level, forward-facing camera and
straight final approach. Mounting pitch/yaw or arbitrary obstacle geometry
requires proper camera-to-base transforms rather than a single offset.

| Launch argument | Default | Meaning |
| --- | --- | --- |
| `stop_distance` | `0.40` | Target optical/front clearance in meters |
| `camera_to_front` | `0.0` | Forward offset from camera to front in meters |
| `max_linear` | `0.08` | Forward speed cap in m/s |
| `max_angular` | `0.30` | Rotation cap in rad/s |
| `confidence` | `0.55` | Minimum generic bottle confidence |
| `frame_timeout` | `0.50` | Maximum original frame/result age in seconds |
| `start_camera` | `false` | Include aligned RealSense launch |
| `start_base` | `true` | Own serial bridge on exclusive approach command topic |
| `cmd_topic` | `/bottle_approach/cmd_vel` | Approach velocity requests |
| `connected_topic` | `/r20/connected` | Bridge telemetry status |

If keeping an externally launched bridge, pass `start_base:=false` and set
`cmd_topic:=/r20/cmd_vel` (or its actual input). Its motion setting must match
your test, and all keyboard/other velocity publishers must be stopped. Prefer
the default launch with its own bridge. The controller disarms when it sees a
second publisher on its command topic; this check is not a command multiplexer
and cannot block commands sent directly by another node.

Diagnostics: no processed frames usually means wrong topics, absent alignment,
camera frame mismatch or old timestamps. `WAITING_BASE` means missing/faded
telemetry; inspect `/r20/connected` and `/r20/diagnostics`.
`INFERENCE_TOO_SLOW_OR_STALE` means the original camera frame expired during
processing. Do not mask slow inference by increasing the timeout during driving;
measure it in detection-only mode and optimize inference first. CPU inference
does not assume a particular JetPack/TensorRT version.

### Detection works but commands remain zero

Inspect both statuses while the launch is running:

```bash
ros2 topic echo /bottle_approach/status --once
ros2 topic echo /r20/connected --once
```

`WAITING_FRESH_DETECTION` with `armed: true`, `motion_allowed: true`, a previous
valid distance and a connected base means the previous camera result has
expired. A detection box and a remembered distance do not make that result
current. Check the added timing fields:

| Status field | Meaning |
| --- | --- |
| `result_fresh` | Original receive and capture times both pass the frame deadline |
| `result_age_s` / `capture_age_s` | Age of the last accepted observation |
| `inference_s` | Duration of the latest completed image/detection/depth work |
| `completed_frame_age_s` | Original receive-to-completion age, including pending time |
| `result_interval_s` | Interval between completed worker results |
| `frames_rejected` / `last_rejection` | Expired-result count and most recent rejection reason |
| `confirmation_count` | Distinct associated RGB-D observations confirmed so far |
| `base_ready` / `base_status_age_s` | Base connection readiness and telemetry-status age |

The original controller cleared confirmation history on every expired-command
interval. At a 0.50 s deadline, approximately 0.28–0.35 s CPU inference could
therefore prevent three confirmations despite every individual result being
fresh when it arrived. The updated controller keeps history across those
zero-command intervals and checks continuity using the original observation
times when the next result arrives. A genuine long observation gap, missing
bottle, invalid depth, invalid timestamps or a rejected stale inference result
still restarts confirmation. **Every expired observation still commands zero;
the 0.50 s deadline has not been increased.**

The update also publishes preview images using the ROS unsigned-byte array fast
path. Assigning raw bytes made the ROS Python setter validate every pixel in
Python on the control thread; large images could delay the 20 Hz velocity timer
and camera callbacks. The optimized preview carries the same image data without
that validation loop.

If `completed_frame_age_s` exceeds `frame_timeout`, results remain rejected and
the robot stays stopped. Measure performance with `enable_motion:=false` before
optimizing inference; do not substitute completion timestamps or widen the
driving timeout to make an old observation appear fresh.

To install the timing update on an already configured Jetson, stop the old
bottle launch, extract `output/r20-bottle-timing-update.zip` into `~/r20_ws`, then:

```bash
source ~/.venvs/snail-bottle/bin/activate
source /opt/ros/humble/setup.bash
cd ~/r20_ws
python /usr/bin/colcon build --symlink-install --packages-select snail_bottle_approach
source install/setup.bash
```

Restart the same bottle launch. The small update archive contains only the
bottle package; it uses the existing model and serial package. Replacing a
package directory updates its supplied files, so preserve any local changes.

## Model regeneration and verification

To regenerate the model on a development computer, create a separate Python
environment, install CPU `torch==2.5.1`, `torchvision==0.20.1`,
`ultralytics==8.3.40`, `onnx==1.17.0`, `numpy==1.26.4`, then run:

```bash
python ws/src/snail_bottle_approach/scripts/prepare_bottle_model.py \
  --output output/models/bottle_yolov8n.onnx
```

The script downloads the official pinned weights, checks class 39 (`bottle`),
exports static `[1,3,640,640] -> [1,84,8400]` ONNX without NMS, and writes
source/license/checksum metadata. It refuses to overwrite existing outputs.
The model/software are subject to Ultralytics AGPL-3.0 or an applicable commercial
license; the repository's own package license does not replace the model license.

Host verification uses a temporary ROS Humble container, synthetic RGB-D ROS
messages, a serial pseudo-terminal and actual ONNX inference. It verifies host
behavior, not the installed Jetson runtime, physical camera or motors. See
`tests/results/bottle-host-verification.json` and `tests/verify_bottle_host.sh`.

Primary implementation references: [RealSense ROS wrapper](https://github.com/realsenseai/realsense-ros),
[COCO detection classes](https://docs.ultralytics.com/datasets/detect/coco/),
[Ultralytics ONNX export](https://docs.ultralytics.com/modes/export/), and
[ONNX Runtime installation](https://onnxruntime.ai/docs/install/).
