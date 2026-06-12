# Obstacle Avoidance Controller Design — Lane Following + Right-Side Wall Following

## Overview

This project implements an autonomous vehicle controller (BMW X5) in Webots that combines **lane-following** via computer vision with a **right-side wall-following obstacle avoidance** algorithm. The vehicle detects stationary buses in its path using a Recognition node, measures distance with LiDAR, and executes an evasion maneuver using lateral distance sensors and the gyroscope to recover its original orientation.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VEHICLE SENSORS                            │
├──────────┬──────────┬───────────┬────────────┬──────────────┤
│  Camera  │  LiDAR   │ Gyroscope │  3 Distance│    GPS       │
│  128x64  │ SICK 291 │  (z-axis) │  Sensors   │  (available) │
│  + Recog │          │           │  (right)   │              │
└────┬─────┴────┬─────┴─────┬─────┴──────┬─────┴──────────────┘
     │          │           │            │
     ▼          ▼           ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│              FINITE STATE MACHINE (FSM)                       │
├─────────────────────────────────────────────────────────────┤
│  STATE 0: LANE_FOLLOWING     (line tracking with PID)        │
│  STATE 1: STEER_LEFT         (dodge left to pass bus)        │
│  STATE 2: WALL_FOLLOWING     (follow bus wall on right)      │
│  STATE 3: REGAIN_ORIENTATION (gyro-based heading recovery)   │
│  STATE 4: REENTER_LANE      (drift back into original lane) │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│              VEHICLE ACTUATORS                                │
├─────────────────────────────────────────────────────────────┤
│  setSteeringAngle()  │  setCruisingSpeed()                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Sensors Used

### 1. Camera with Recognition Node

**World file (.wbt) definition:**
```
Camera {
  translation 0.72 0 -0.05
  fieldOfView 1
  width 128
  recognition Recognition {
    maxRange 80
    maxObjects 10
    segmentation TRUE
  }
}
```

**Python initialization:**
```python
camera = driver.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)
```

**Functionality:**
- Captures 128×64 images for yellow line detection
- Recognition node identifies objects with defined `recognitionColors` (the buses)
- Retrieves model (`getModel()`) and color (`getColors()`) for each recognized object
- Color is mapped to a specific bus ID (vehicle(1) through vehicle(4))

### 2. LiDAR (SICK LMS 291)

**Location:** Front sensor slot of the vehicle

**Initialization:**
```python
lidar = driver.getDevice("Sick LMS 291")
lidar.enable(timestep)
lidar.enablePointCloud()
```

**Functionality:**
- Generates a 3D point cloud of the environment
- Points are filtered within a ±15° forward cone (~30° beamwidth)
- Computes minimum distance to the closest frontal obstacle
- Activation threshold: **18 meters** — when a bus is detected at this distance, evasion begins

### 3. Gyroscope

**Location:** Center slot of the vehicle (already included in the world)

**Initialization:**
```python
gyro = driver.getDevice("gyro")
gyro.enable(timestep)
```

**Functionality:**
- Reads angular velocity on the Z-axis (`gyro.getValues()[2]`)
- Integrated every timestep: `current_yaw += gyro_z * dt`
- Reset to 0 when avoidance starts (used as relative reference)
- During recovery, serves as stop condition: when `current_yaw <= 0`, the original orientation has been recovered

### 4. Right-Side Distance Sensors

**World file (.wbt) definition:** Placed inside `sensorsSlotCenter` with Y=-1.0 offset (right side)
```
DistanceSensor {
  translation 0.8 -1.0 0     # ds_right_front (front-right)
  rotation 0 0 1 -1.5708      # pointing right
  name "ds_right_front"
  lookupTable [
    0 0 0
    0.5 50 0
    3.0 300 0.01
    7.0 700 0.02
  ]
  type "sonar"
}
```

**3 sensors included:**
| Sensor | X Translation | Purpose |
|--------|--------------|---------|
| `ds_right_front` | +0.8 | Detect start of obstacle |
| `ds_right_mid` | 0.0 | PD control for wall-following |
| `ds_right_rear` | -0.8 | Detect end of obstacle |

**Lookup Table:**
- 0m → value 0
- 0.5m → value 50
- 3.0m → value 300
- 7.0m → value 700
- Conversion: `distance_meters = sensor_value / 100`

---

## Bus Identification

Each bus has a unique color defined as `recognitionColors` in the PROTO:

| Bus ID | RGB Color | Description |
|--------|-----------|-------------|
| vehicle(1) | (0.031, 0.122, 0.420) | Dark blue |
| vehicle(2) | (1.000, 0.000, 0.000) | Red |
| vehicle(3) | (0.863, 0.541, 0.867) | Pink/Lilac |
| vehicle(4) | (0.180, 0.761, 0.494) | Green |

Identification is done by comparing the recognized color against the table using Euclidean distance in RGB space.

---

## Finite State Machine — Complete Flow

### State 0: LANE_FOLLOWING (Line Tracking)

**Vision pipeline:**
1. Capture BGR image from camera
2. Convert to HSV color space
3. Yellow color mask (H: 18-40, S: 30-255, V: 80-255)
4. Dilation with 5×5 kernel to thicken the line
5. Canny edge detection
6. Region of Interest (ROI) — bottom 45% of the image
7. Hough Transform (HoughLinesP) for line segment detection
8. Angle filtering (only lines >30° from horizontal)
9. Lane center calculation (average of midpoints in X)

**PID Control:**
- Error = (detected_center - setpoint) / setpoint
- Setpoint = 35% of image width (yellow line should appear on the left)
- Output clamped to ±0.5 radians
- Cruising speed: 55 km/h

**Transition:** When Recognition detects a bus AND LiDAR < 18m → `STATE_STEER_LEFT`

---

### State 1: STEER_LEFT (Initial Dodge Maneuver)

- **Steering angle:** -0.35 rad (turn left)
- **Speed:** 25 km/h
- **Gyroscope action:** Reset `current_yaw = 0` as reference for original orientation

**Transition:**
- If `ds_right_mid` detects the bus (< 5.25m) after 10 steps → `STATE_WALL_FOLLOWING`
- Timeout after 60 steps → `STATE_WALL_FOLLOWING`

---

### State 2: WALL_FOLLOWING (Right-Side Wall Tracking)

**PD Controller:**
```
error = desired_distance - measured_distance_mid
steering = -(Kp * error + Kd * derivative)
```

- **Desired distance:** 3.5 meters from the bus
- **Kp:** 0.15, **Kd:** 0.05
- Steering clamped to ±0.3 rad
- Too close → steer left (move away)
- Too far → steer right (move closer)

**Transition:** When `ds_right_rear > 6.5m` (rear sensor no longer sees the bus) after 40 steps → `STATE_REGAIN_ORIENTATION`

---

### State 3: REGAIN_ORIENTATION (Gyro-Based Heading Recovery)

- **Steering angle:** +0.2 rad (constant right turn)
- **Stop condition:** `current_yaw <= 0.05 rad` — gyroscope indicates the vehicle has returned to its saved orientation
- Progress printed to console: `[GYRO] Recovering: yaw_offset=X.XXX rad`

**Transition:** When yaw crosses 0 → `STATE_REENTER_LANE`

---

### State 4: REENTER_LANE (Return to Original Lane)

After recovering orientation, the vehicle is still **displaced to the left**. This state moves it back into the right lane.

- **Steering angle:** +0.15 rad (gentle right turn)
- **Duration:** 50 simulation steps

**Transition:** After 50 steps → `STATE_LANE_FOLLOWING`

---

## State Transition Diagram

```
                    ┌──────────────────┐
                    │  LANE_FOLLOWING   │◄─────────────────────┐
                    │  (PID + Vision)   │                      │
                    └────────┬─────────┘                      │
                             │                                │
                    Bus recognized +                          │
                    LiDAR < 18m                               │
                             │                                │
                             ▼                                │
                    ┌──────────────────┐                      │
                    │   STEER_LEFT     │                      │
                    │  (steer = -0.35) │                      │
                    │  Gyro: yaw = 0   │                      │
                    └────────┬─────────┘                      │
                             │                                │
                    ds_right_mid < 5.25m                      │
                             │                                │
                             ▼                                │
                    ┌──────────────────┐                      │
                    │  WALL_FOLLOWING   │                      │
                    │  (PD on ds_mid)   │                      │
                    └────────┬─────────┘                      │
                             │                                │
                    ds_right_rear > 6.5m                      │
                             │                                │
                             ▼                                │
                    ┌──────────────────┐                      │
                    │REGAIN_ORIENTATION │                      │
                    │ (steer = +0.2)   │                      │
                    │ until yaw <= 0   │                      │
                    └────────┬─────────┘                      │
                             │                                │
                    Gyro: yaw <= 0.05                         │
                             │                                │
                             ▼                                │
                    ┌──────────────────┐                      │
                    │  REENTER_LANE    │                      │
                    │ (steer = +0.15)  │──── 50 steps ────────┘
                    │  (50 steps)      │
                    └──────────────────┘
```

---

## Real-Time Visualization

The controller displays two OpenCV windows:

1. **"Camera (Hough + Canny)"** — Camera image with:
   - Detected Hough lines (green)
   - Red bounding box around recognized buses
   - Bus model name label
   - Current FSM state (top-left corner, yellow text)

2. **"ROI"** — Binary edge image with the region of interest applied

Both windows are scaled ×4 for better visibility.

---

## Console Output

The controller prints informative messages at each key step:

```
[INIT] Camera: 128x64, Recognition enabled
[INIT] LIDAR threshold: 18.0m
[INIT] Gyroscope + 3 right-side distance sensors ready
[RECOGNITION] Bus detected: vehicle(2) | LIDAR distance: 15.32m
[AVOIDANCE] Begin. Gyro orientation saved (yaw=0 reference).
[AVOIDANCE] Bus on right side. Wall-following.
[WALL] front:3.2 mid:3.5 rear:4.1 steer:0.000
[AVOIDANCE] Bus cleared. Regaining orientation (current yaw offset: 0.524 rad, target: 0)
[GYRO] Recovering: yaw_offset=0.312 rad
[AVOIDANCE] Orientation recovered (yaw=0.043). Re-entering lane...
[AVOIDANCE] Done. Resuming lane-following.
```

---

## Configurable Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CRUISING_SPEED` | 55 km/h | Speed during lane-following |
| `WALL_FOLLOW_SPEED` | 25 km/h | Speed during avoidance |
| `LIDAR_OBSTACLE_THRESHOLD` | 18 m | Distance to trigger avoidance |
| `STEER_LEFT_ANGLE` | -0.35 rad | Initial dodge steering angle |
| `WALL_DESIRED_DISTANCE` | 3.5 m | Desired distance from bus |
| `WALL_KP` | 0.15 | Proportional gain for wall-follow |
| `WALL_KD` | 0.05 | Derivative gain for wall-follow |
| `DS_NO_OBSTACLE_THRESHOLD` | 6.5 m | Distance to consider "clear" |
| `ANGLE_TOLERANCE` | 0.03 rad | Gyro recovery tolerance |
| `KP, KI, KD` | 1.0, 0.005, 0.2 | PID gains for lane-following |

---

## Modified Files

| File | Changes |
|------|---------|
| `worlds/city_2025a.wbt` | Added Recognition node to camera; added 3 DistanceSensors in sensorsSlotCenter pointing right |
| `controllers/lane_controller/lane_controller.py` | Rewritten with 5-state FSM, full sensor initialization, wall-following, and gyro-based orientation recovery |
