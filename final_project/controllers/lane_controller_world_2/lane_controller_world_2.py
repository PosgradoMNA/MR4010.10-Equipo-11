"""
CIL Autonomous Driving Controller.

This controller implements an autonomous driving system for a Webots
vehicle using a Conditional Imitation Learning (CIL) model for steering
prediction. The controller combines camera images, LiDAR obstacle
detection, and Webots object recognition to safely navigate a simulated
environment.

Features
--------
* Conditional Imitation Learning (TensorFlow Lite)
* LiDAR-based obstacle detection
* Pedestrian emergency braking
* Finite State Machine (FSM) obstacle avoidance
* Keyboard-controlled navigation commands
* Cruise speed control

Keyboard Controls
-----------------
W
    Follow lane.

A
    Turn left.

D
    Turn right.

S
    Continue straight.

E
    Enable/disable obstacle avoidance.

Author
------
<Your Name>

"""

from vehicle import Driver
import numpy as np
import cv2
import os
import sys

# =============================================================================
# TensorFlow Lite Import
# =============================================================================
#
# TensorFlow Lite is loaded from the local virtual environment rather than
# assuming it is installed globally.
#

VENV_SITE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "venv",
    "lib",
)

for d in os.listdir(VENV_SITE) if os.path.isdir(VENV_SITE) else []:
    sp = os.path.join(VENV_SITE, d, "site-packages")
    if os.path.isdir(sp) and sp not in sys.path:
        sys.path.insert(0, sp)

import tensorflow.lite as tflite

# =============================================================================
# Model Configuration
# =============================================================================

#: Path to the TensorFlow Lite CIL model.
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "models",
    "cil_model.tflite",
)

#: Default cruise speed (km/h).
CRUISING_SPEED = 30.0

#: Maximum steering angle sent to the vehicle.
MAX_STEERING_ANGLE = 0.5

#: Small steering bias to keep the vehicle centred in the lane.
RIGHT_LANE_BIAS = 0.01

#: Number of high-level driving commands.
NUM_COMMANDS = 4

#: Optional display scaling factor.
DISPLAY_SCALE = 2

# =============================================================================
# LiDAR Configuration
# =============================================================================

#: Webots LiDAR device name.
LIDAR_DEVICE_NAME = "Sick LMS 291"

#: Angular region used for obstacle detection.
LIDAR_DETECTION_ANGLE_DEG = 20

#: Emergency braking distance.
LIDAR_BRAKE_DISTANCE = 8.0

#: Distance where the vehicle slows down.
LIDAR_SLOW_DISTANCE = 15.0

#: Distance that triggers obstacle avoidance.
LIDAR_AVOIDANCE_TRIGGER = 12.0

#: Reduced cruising speed when obstacles are nearby.
SLOW_SPEED = 12.0

# =============================================================================
# Finite State Machine (FSM)
# =============================================================================

#: Normal Conditional Imitation Learning driving mode.
STATE_CIL = 0

#: Steering left to avoid an obstacle.
STATE_STEER_LEFT = 1

#: Driving past an obstacle.
STATE_PASS = 2

#: Returning to the original lane.
STATE_STEER_RIGHT = 3

#: Vehicle speed during avoidance.
AVOIDANCE_SPEED = 18.0

#: Steering angle while moving left.
STEER_LEFT_ANGLE = -0.25

#: Steering angle while moving right.
STEER_RIGHT_ANGLE = 0.15

#: Simulation steps spent steering left.
STEER_LEFT_STEPS = 35

#: Simulation steps spent driving beside the obstacle.
PASS_STEPS = 60

#: Simulation steps spent steering back.
STEER_RIGHT_STEPS = 40

# =============================================================================
# Driving Commands
# =============================================================================

#: Follow the detected lane.
CMD_FOLLOW_LANE = 0

#: Turn left.
CMD_TURN_LEFT = 1

#: Turn right.
CMD_TURN_RIGHT = 2

#: Continue straight.
CMD_GO_STRAIGHT = 3


def get_image(camera):
    """
    Capture an image from the Webots camera.

    The Webots camera returns a BGRA image buffer. This function converts
    the raw image into an OpenCV-compatible BGR image.

    Parameters
    ----------
    camera : Camera
        Webots camera device.

    Returns
    -------
    numpy.ndarray
        BGR image with shape (height, width, 3).
    """
    raw = camera.getImage()

    img = np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def preprocess(frame):
    """
    Preprocess an image before TensorFlow Lite inference.

    The preprocessing pipeline performs the following operations:

    1. Converts the image from BGR to RGB.
    2. Normalises pixel values to the range [0, 1].
    3. Adds a batch dimension required by TensorFlow Lite.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image captured from the front camera.

    Returns
    -------
    numpy.ndarray
        Tensor with shape (1, H, W, 3).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(rgb, axis=0)


def get_lidar_min_distance(lidar):
    """
    Determine the nearest obstacle directly in front of the vehicle.

    Rather than using every LiDAR point, only measurements within a
    configurable angular window centred on the vehicle are considered.
    This reduces the influence of objects beside the vehicle.

    Parameters
    ----------
    lidar : Lidar
        Webots LiDAR sensor.

    Returns
    -------
    float
        Minimum detected distance (metres). Returns ``float('inf')`` if
        no valid measurements are available.
    """
    range_image = lidar.getRangeImage()

    if not range_image:
        return float("inf")

    num_points = len(range_image)

    angle_per_point = 180.0 / num_points

    center = num_points // 2

    half_window = int(
        (LIDAR_DETECTION_ANGLE_DEG / 2.0) / angle_per_point
    )

    start = max(0, center - half_window)
    end = min(num_points, center + half_window)

    min_dist = float("inf")

    for i in range(start, end):
        if range_image[i] < min_dist:
            min_dist = range_image[i]

    return min_dist
    
def main():
    """
    Main execution loop for the autonomous driving controller.

    This function initializes the Webots driver, sensors, TensorFlow Lite
    model, and runtime state variables. It then enters the control loop
    where perception, prediction, and control are executed at every timestep.
    """

    # =========================================================================
    # Driver initialization
    # =========================================================================
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    # =========================================================================
    # Camera setup
    # =========================================================================
    camera = driver.getDevice("camera")
    camera.enable(timestep)
    camera.recognitionEnable(timestep)

    # =========================================================================
    # LiDAR setup
    # =========================================================================
    lidar = driver.getDevice(LIDAR_DEVICE_NAME)
    lidar.enable(timestep)
    lidar.enablePointCloud()

    # =========================================================================
    # Keyboard setup
    # =========================================================================
    keyboard = driver.getKeyboard()
    keyboard.enable(timestep)

    # =========================================================================
    # Load TensorFlow Lite model
    # =========================================================================
    print(f"Loading CIL model from: {MODEL_PATH}")

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # -------------------------------------------------------------------------
    # Identify model input tensors
    # -------------------------------------------------------------------------
    # One input corresponds to the image (width 160), the other to the command
    img_input_idx = next(
        i for i, d in enumerate(input_details)
        if d["shape"][1] == 160
    )

    cmd_input_idx = next(
        i for i, d in enumerate(input_details)
        if d["shape"][1] == 4
    )

    print("Model loaded successfully!")

    # =========================================================================
    # Driving state and command initialization
    # =========================================================================

    #: Current high-level driving command
    current_command = CMD_FOLLOW_LANE

    #: Human-readable command labels
    command_names = {
        0: "FOLLOW",
        1: "LEFT",
        2: "RIGHT",
        3: "STRAIGHT",
    }

    #: FSM state (CIL / avoidance modes)
    state = STATE_CIL

    #: Counter used for timed FSM transitions
    state_counter = 0

    #: Toggle for obstacle avoidance behaviour
    avoidance_enabled = False

    #: Human-readable FSM state names
    state_names = {
        0: "CIL",
        1: "AVOID_LEFT",
        2: "PASSING",
        3: "AVOID_RIGHT",
    }

    # =========================================================================
    # Precomputed command tensors (one-hot vectors)
    # =========================================================================
    cmd_tensors = [
        np.eye(NUM_COMMANDS, dtype=np.float32)[[i]]
        for i in range(NUM_COMMANDS)
    ]

    #: Cached steering value from last inference
    cached_steering = 0.0

    #: Frame counter for logging
    frame_count = 0

    # =========================================================================
    # Vehicle initial configuration
    # =========================================================================
    driver.setCruisingSpeed(CRUISING_SPEED)

    # =========================================================================
    # Console UI
    # =========================================================================
    print("=" * 50)
    print("  CIL AUTONOMOUS DRIVING - WORLD 2")
    print("=" * 50)
    print("  W=Follow | A=Left | D=Right | S=Straight")
    print("  E=Toggle avoidance")
    print("=" * 50)