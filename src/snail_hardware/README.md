# R20 TWD keyboard test on Jetson / ROS 2 Humble

This package controls the **two-wheel differential R20 prototype** through its
existing STM32F407 serial protocol. It is independent of the simulation packages.
The STM32 handles encoders, motor PID and TWD kinematics; the Jetson sends body
velocity commands. The already running D435i can remain running separately.

```text
keyboard -> /r20/cmd_vel -> r20_base_driver -> serial -> STM32 -> motors
                              ^                         |
                              +------ telemetry --------+
```

## Check the STM32 firmware first

The supplied firmware source is
`prototype/R20/OpenCTR_H60V32_R20_1024_V3.01/OpenCTR_H60V32_R20_1024_V3.01`.
In `Robot/ax_robot.h`, line 160 currently selects `ROBOT_MEC`.
Your installed firmware must instead have been built with:

```c
#define ROBOT_TYPE ROBOT_TWD
```

If the base already has working TWD firmware, keep that firmware. If rebuilding
the supplied Keil project, select TWD and verify its wheel diameter, track width,
encoder resolution and motor wiring before flashing. The source's TWD values are
72.4 mm wheel diameter, 206 mm track width and 122880 encoder counts per wheel
revolution. These are vendor defaults, not measurements of your prototype.
The bundled `xproject.hex` has not been verified as TWD; do not assume it matches.
This package does not flash the STM32.

For the forward-facing D435i bottle approach and 0.40 m stopping trial, see
the separate [bottle approach guide](../snail_bottle_approach/README.md).

The host sends model-check command `0x5A` with TWD value `4` after receiving valid
telemetry. **A short beep (about 200 ms) means the firmware model matches; a long
beep (about 1 s) means it does not.** This command only checks the compiled model;
it does not select or change the firmware's kinematics. The model is not reported
over serial, so software cannot automatically confirm the match.

**First test with the drive wheels raised and motor power readily accessible.**
The supplied firmware has no velocity command watchdog: an unplugged cable,
Jetson power loss or a killed bridge can leave the last motor command running.
The host's timeouts and final zero commands only work while it can transmit.
Do not test cable removal while moving. Before robust on-floor operation, add and
verify an MCU-side velocity timeout that clears targets when valid velocity
packets cease (for example, 300 ms at this package's 20 Hz command rate).

## Connect and identify the serial port

Prefer the R20 board's USB serial interface with a data-capable USB cable. In the
source, it is USART3 (PD8 TX / PD9 RX). The other motion interface is USART2 TTL
(PA2 TX / PA3 RX). Both run **230400 baud, 8N1, no flow control**. USART1 is the
debug interface and is not the motion interface. For an external TTL adapter,
use compatible 3.3 V UART levels, cross TX/RX and share GND; check the board
connector pinout before wiring. The MCU pin names do not identify connector order.
Power the wheel base using its normal supply; do not infer motor power from USB.

On the Jetson:

```bash
ls -l /dev/serial/by-id/
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

Use the actual device found. `/dev/ttyUSB0` below is an example. A
`/dev/serial/by-id/...` path is preferable because USB numbers can change.
For permissions, add your Jetson account to `dialout` if necessary:

```bash
sudo usermod -aG dialout "$USER"
```

Then log out and back in. Run only one serial program at a time: close the vendor
driver, a serial monitor or the probe before starting the bridge. Direct Jetson
UART device names depend on the carrier board and configuration; this package
does not assume a Jetson header pinout or disable a serial console.

## Copy and build only this package

Copy `ws/src/snail_hardware` into `~/r20_ws/src/snail_hardware` on the Jetson.
For example, from this Windows repository in PowerShell, replace both placeholders:

```powershell
ssh <JETSON_USER>@<JETSON_IP> 'mkdir -p ~/r20_ws/src'
scp -r .\ws\src\snail_hardware <JETSON_USER>@<JETSON_IP>:~/r20_ws/src/
```

On the Jetson, with the existing Humble installation:

```bash
sudo apt update
sudo apt install python3-serial python3-tk python3-pytest \
  python3-colcon-common-extensions ros-humble-diagnostic-msgs
source /opt/ros/humble/setup.bash
cd ~/r20_ws
colcon build --symlink-install --packages-select snail_hardware
source install/setup.bash
colcon test --packages-select snail_hardware --event-handlers console_direct+
colcon test-result --verbose
```

## 1. Test communication with zero motion

Keep the wheels raised. The probe sends only zero velocity at 20 Hz, receives
and validates telemetry, requests the TWD model-check beep once, and sends final
zeros before closing the port:

```bash
ros2 run snail_hardware r20_serial_probe --port /dev/ttyUSB0 --duration 10
```

Expected: `PASS`, many valid telemetry packets and a plausible battery voltage.
Telemetry is normally 50 Hz; startup can take several seconds. Listen for the
short TWD match beep. A long beep means stop and correct the firmware model.
No beep means command reception/model match remains unverified even if telemetry
arrives. The periodic telemetry is not a velocity command acknowledgement.

You can then check the ROS bridge without permitting motion:

```bash
ros2 launch snail_hardware keyboard.launch.py \
  port:=/dev/ttyUSB0 keyboard:=false enable_motion:=false
```

In another sourced terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/r20_ws/install/setup.bash
ros2 topic hz /r20/feedback_velocity
ros2 topic echo /r20/battery_voltage
ros2 topic echo /r20/diagnostics
```

Run those commands separately, using Ctrl+C between them. Stop the test launch
before starting the motion launch.

## 2. Drive with the Jetson keyboard and desktop

With the port confirmed and a short TWD match beep:

```bash
ros2 launch snail_hardware keyboard.launch.py port:=/dev/ttyUSB0
```

Click the keyboard window, then hold a key. Initial speeds are **0.10 m/s** and
**0.30 rad/s**. Release the key or change window focus to stop.

| Keys | Movement |
| --- | --- |
| W / I / Up | Forward |
| S / comma / Down | Reverse |
| A / J / Left | Rotate left |
| D / L / Right | Rotate right |
| U / O | Forward with left / right turn |
| M / period | Reverse with left / right turn |
| Space / K | Stop |
| Q | Quit and stop the launch |

The bridge caps commands at 0.20 m/s and 0.60 rad/s, rejects lateral/non-planar
and non-finite commands, sends zeros while no command arrives for 0.30 s, and
sends zeros when telemetry is missing for 0.50 s. It initially permits motion
only after valid telemetry. Opening the GUI does not itself move the base.
Closing the GUI or ending the launch sends final stop packets. A serial I/O
failure ends the launch; it does not reconnect and replay commands.

For lower initial speed:

```bash
ros2 launch snail_hardware keyboard.launch.py \
  port:=/dev/ttyUSB0 speed:=0.05 turn:=0.15
```

## Terminal keyboard / SSH

ROS launch does not provide interactive stdin to a child keyboard process.
For a terminal-only Jetson or SSH connection, use two terminals.

Terminal 1:

```bash
source /opt/ros/humble/setup.bash
source ~/r20_ws/install/setup.bash
ros2 launch snail_hardware keyboard.launch.py port:=/dev/ttyUSB0 keyboard:=false
```

Terminal 2 (use `ssh -t` if needed to allocate a TTY):

```bash
source /opt/ros/humble/setup.bash
source ~/r20_ws/install/setup.bash
ros2 run snail_hardware r20_keyboard --ros-args \
  -r __ns:=/r20 -p mode:=terminal
```

Terminal mode cannot detect key release or focus loss. Each movement keypress
permits a short pulse; it stops after **0.35 s** without another movement
keypress. Keyboard repeat may start later than that, causing an initial pause
when holding a key. Space/K stops immediately, Q quits. Prefer the desktop GUI
for accurate hold-and-release driving. Do not pipe this terminal controller or
start it as a background service.

## Topics and verification

| Topic | Message | Meaning |
| --- | --- | --- |
| `/r20/cmd_vel` | `geometry_msgs/Twist` | Keyboard body velocity request |
| `/r20/applied_cmd_vel` | `geometry_msgs/Twist` | Limited velocity successfully written to serial, not an MCU acknowledgement |
| `/r20/feedback_velocity` | `geometry_msgs/Twist` | Encoder-derived body velocity received from STM32 |
| `/r20/imu/data_raw` | `sensor_msgs/Imu` | Base MPU6050 acceleration and angular velocity; no orientation estimate |
| `/r20/battery_voltage` | `std_msgs/Float32` | STM32 battery voltage in volts |
| `/r20/connected` | `std_msgs/Bool` | Fresh valid telemetry, published once per second |
| `/r20/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Link state and received/sent packet counts |

With wheels raised, briefly test forward/reverse and rotation at low speed.
Confirm both wheel directions, positive forward feedback and positive yaw for
left rotation. Release the key and verify both wheels stop and feedback tends
to zero. Compare applied commands and feedback; a zero command alone cannot
prove that the STM32 executed it. Keep the RealSense topics separate: this test
does not add depth obstacle avoidance, camera transforms, a new URDF, odometry
integration or autonomous simulation behaviors.

Useful diagnosis: no device usually means cable/enumeration; permission denied
means device access; no valid telemetry means wrong interface/baud/power/firmware;
long model beep means wrong compiled chassis; telemetry with no motion suggests
`enable_motion:=false`, wiring, power, or firmware/motor setup. Inspect diagnostics
and `/r20/applied_cmd_vel` before changing motor or PID settings.

## Protocol and host tests

Packets are `AA 55 total_length id payload checksum`. Checksum is the sum of all
preceding bytes modulo 256. Velocity ID `0x50` has three big-endian signed int16
values (vx, vy, wz), scaled by 1000. TWD always sends vy=0. Telemetry ID `0x10`
has 20 bytes: nine signed int16 values (acceleration xyz, gyro xyz, velocity xyz)
and one unsigned uint16 battery voltage scaled by 100. Telemetry length is 25.

```text
Stop:       AA 55 0B 50 00 00 00 00 00 00 5A
Forward .1: AA 55 0B 50 00 64 00 00 00 00 BE
TWD check:  AA 55 06 5A 04 63
```

The test suite exercises exact protocol bytes, signed telemetry/scales, corrupt
stream recovery, key handling, watchdogs and Linux pseudo-terminal ROS serial
integration. A pseudo-terminal verifies host behavior; it does not verify the
physical serial adapter, installed firmware, motor polarity or mechanical motion.
See `tests/results/r20-host-verification.json` at the repository root for the
recorded local verification and its limits.

Implementation references: [ROS 2 Humble launch documentation](https://docs.ros.org/en/ros2_documentation/humble/How-To-Guides/Launch-file-different-formats.html)
and [pySerial API](https://pyserial.readthedocs.io/en/latest/pyserial_api.html).
