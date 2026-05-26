# This controller implements a lane detection pipeline using computer vision
# techniques and a PID controller to steer the vehicle along the yellow center
# line of roads. The vehicle maintains a constant speed of at least 50 km/h.

from vehicle import Driver
from ultralytics import YOLO
import numpy as np
import cv2
import joblib
import os

# =============================================================================
# MODIFICATION: Added LiDAR-based obstacle detection and emergency braking.
# The SICK LMS 291 LiDAR is enabled and only the central 20° arc is read.
# If any point in that arc is closer than 20m, emergency braking is applied.
# Reference: Webots SICK LiDAR documentation
# https://www.cyberbotics.com/doc/guide/lidar-sensors
# =============================================================================

# =============================================================================
# MODIFICATION: Added HOG + SVM pedestrian classification.
# When LiDAR detects an obstacle, the camera image is scanned using a sliding
# window approach with HOG features fed to a trained Linear SVM to classify
# whether the obstacle is a pedestrian or a barrel.
# Reference: Dalal & Triggs, "HOG for Human Detection", CVPR 2005
# Reference: https://medium.com/@ricardo.zuccolo/self-driving-cars-opencv-and-svm
# =============================================================================

# SVM Model paths (relative to this controller file)
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CONTROLLER_DIR, "..", "..", "svm_model")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "pedestrian_svm_model.pkl")
SVM_SCALER_PATH = os.path.join(MODEL_DIR, "pedestrian_svm_scaler.pkl")
YOLO_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")

# Switch between YOLO and custom SVM model
USE_YOLO = True  # True = YOLO, False = custom HOG+SVM model

# Sliding window parameters for 256x128 camera image
SVM_WINDOW_SIZE = (64, 128)  # HOG detection window (width, height)
SVM_STEP_SIZE = 16           # Pixels to slide the window
SVM_SCALES = [0.5]           # 0.5 = 2x upscale only (image becomes 512x256)
SVM_CONFIDENCE_THRESHOLD = 0.0  # 0 = SVM decision boundary (positive = pedestrian)

# LiDAR detection parameters
LIDAR_DETECTION_ANGLE_DEG = 30  # Only read central 30° of the 180° scan
LIDAR_MAX_RANGE_M = 20.0  # Obstacle detection limited to 20 meters
LIDAR_DEVICE_NAME = "Sick LMS 291"  # Default name from SickLms291 proto

# Proportional gain, main corrective force
KP = 1.0

# Integral gain, small to avoid windup oscillations
KI = 0.005

# Derivative gain, reduced to minimize zigzag from noise
KD = 0.2

# Maximum steering angle in radians (limits the PID output)
MAX_STEERING_ANGLE = 0.5

# Constant cruising speed in km/h
CRUISING_SPEED = 40

# Default steering angle when no lines are detected (drive straight)
DEFAULT_ANGLE = 0.0

# Lower and upper thresholds for the Canny edge detector.
CANNY_LOW = 40
CANNY_HIGH = 120

# Yellow lane marking color in Webots city: #dbce7a → OpenCV HSV ≈ (26, 113, 219).
# The S and V lower bounds are widened (S=30, V=80) to capture the yellow stripe
# at various distances and lighting conditions where saturation and brightness drop.
# The hue range [18-40] is centered around 26 and wide enough for slight color shifts.
YELLOW_HSV_LOW = np.array([18, 30, 80])
YELLOW_HSV_HIGH = np.array([40, 255, 255])

# Distance resolution of the accumulator in pixels (1 pixel).
HOUGH_RHO = 1

# Angular resolution of the accumulator in radians (1 degree).
HOUGH_THETA = np.pi / 180

# Minimum number of intersections (votes) to detect a line.
HOUGH_THRESHOLD = 10

# Minimum length of a line segment to be considered valid
# (must be shorter than ROI height: bottom 30% of 64px = ~19px).
HOUGH_MIN_LINE_LENGTH = 8

# Maximum gap between two points to be considered part of same line.
HOUGH_MAX_LINE_GAP = 15

# Minimum angle from horizontal (in degrees) for a line to be considered
# a valid lane marking. Lines more horizontal than this are rejected
# (e.g., crossing road markings, pedestrian crossings). 45° means only lines
# between 45°-90° from horizontal are kept (mostly vertical = lane markings).
MIN_LINE_ANGLE_DEG = 50

# Multiplier to make the steering smoother.
SMOOTHING_ALPHA = 0.3

# Percentage of the lane to calculate the setpoint.
LANE_SETPOINT_PERCENTAGE = 0.35

# Max number of lines for the algorithm to detect a lane separator.
MAX_LINES_FOR_LANE_THRESHOLD = 4

# Percentage of image width to use for ROI (centered). 0.7 = center 70%.
ROI_WIDTH_PERCENTAGE = 0.70

# Percentage of image height for ROI top boundary. 0.55 = bottom 45% of image.
ROI_TOP_PERCENTAGE = 0.55

# Multiplier for the UI display windows at runtime.
DISPLAY_UI_SCALE_MULTIPLIER = 2


class PIDController:
    """
    A PID (Proportional-Integral-Derivative) controller for steering.

    The PID controller calculates a correction value based on:
    - Proportional term: proportional to the current error
    - Integral term: proportional to the accumulated error over time
    - Derivative term: proportional to the rate of change of error
    """

    def __init__(self, kp, ki, kd, setpoint):
        """
        Initialize the PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: The desired target value (midpoint of image width)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, measurement):
        """
        Compute the PID control output.

        The error is calculated as a normalized value:
            error = (measurement - setpoint) / setpoint
        This normalizes the error to the range [-1, 1].

        A positive error means the lane center is to the right of image center,
        so the steering output is negative (steer right to follow it).
        A negative error means the lane is to the left, steer left.

        Args:
            measurement: The current measured value (horizontal center of
                        detected lane lines)

        Returns:
            control_output: The steering angle (clamped to max limits)
        """
        error = (measurement - self.setpoint) / self.setpoint

        p_term = self.kp * error

        self.integral += error

        self.integral = np.clip(self.integral, -5.0, 5.0)
        i_term = self.ki * self.integral

        derivative = error - self.previous_error
        d_term = self.kd * derivative

        self.previous_error = error

        steer_output = p_term + i_term + d_term

        steer_output = np.clip(steer_output, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

        return steer_output


def get_image(camera):
    """
    Obtains the image from the onboard camera.

    The camera returns a raw byte buffer which is reshaped into a
    numpy array with dimensions (height, width, 4) - BGRA format.
    We convert it to BGR for OpenCV compatibility.

    Args:
        camera: Webots Camera device

    Returns:
        frame: numpy array of shape (height, width, 3) in BGR format
    """
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    # Convert from BGRA to BGR for OpenCV processing
    frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return frame


def convert_to_grayscale(frame):
    """
    Converts the camera image to grayscale.

    Grayscale conversion reduces the image from 3 color channels to 1,
    simplifying subsequent processing steps (edge detection works on
    single-channel images). This also reduces computational cost.

    Args:
        frame: BGR color image

    Returns:
        gray_image: Single-channel grayscale image
    """
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_image


def get_yellow_mask(frame):
    """
    Creates a binary mask that isolates yellow-colored pixels in the image.

    This filters out white dashed lane dividers and only keeps the yellow
    center line that we want to follow. The mask is created by converting
    the image to HSV color space and thresholding for yellow hues.

    Args:
        frame: BGR color image from camera

    Returns:
        yellow_mask: Binary mask where white = yellow pixel, black = non-yellow
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH)

    return yellow_mask


def detect_edges(gray_image):
    """
    Applies Canny edge detection algorithm.
    Applies a Gaussian blur before Canny to further reduce noise in the image.

    Args:
        gray_image: Grayscale input image

    Returns:
        edges: Binary image with detected edges (white = edge, black = no edge)
    """
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    return edges


def apply_region_of_interest(edges):
    """
    Defines and applies a Region of Interest (ROI) using fillPoly.

    The ROI is a rectangular mask that isolates the lower portion of the image
    where lane lines are expected to appear. This eliminates irrelevant
    edges from the sky, trees, buildings, etc.

    Uses cv2.fillPoly to create a filled polygon mask, then applies it
    to the edge image using a bitwise AND operation.

    Args:
        edges: Binary edge image from Canny detection

    Returns:
        masked_edges: Edge image with only the ROI region preserved
    """
    height, width = edges.shape

    mask = np.zeros_like(edges)

    # In order: bottom-leftm top-left, top-right, bottom-right.
    # Only look at the bottom 45% of the image and center ROI_WIDTH_PERCENTAGE
    # This prevents reacting to pedestrians, cones, and objects on the sides
    margin = (1.0 - ROI_WIDTH_PERCENTAGE) / 2.0
    left = int(width * margin)
    right = int(width * (1.0 - margin))
    roi_vertices = np.array(
        [
            [
                (left, height),
                (left, int(height * ROI_TOP_PERCENTAGE)),
                (right, int(height * ROI_TOP_PERCENTAGE)),
                (right, height),
            ]
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(mask, roi_vertices, 255)

    masked_edges = cv2.bitwise_and(edges, mask)

    return masked_edges


def detect_lines(masked_edges):
    """
    Detect straight lines using the Hough Transform (HoughLinesP).

    The Probabilistic Hough Line Transform (HoughLinesP) detects line segments
    in the edge image. It returns line segments defined by their two endpoints
    (x1, y1, x2, y2).

    Args:
        masked_edges: Edge image with ROI applied

    Returns:
        lines: Array of detected line segments or None if no lines are detected
    """
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    return lines


def calculate_lane_center(lines, frame):
    """
    Convert detected lines into a PID controller input by calculating the
    lane center from non-horizontal line midpoints.
    If no valid lines are found, return None to signal that the default
    straight-driving angle should be used.

    Args:
        lines: Array of detected line segments from HoughLinesP
        frame: BGR image for drawing debug lines

    Returns:
        lane_center: The x-coordinate of the average lane center,
                    or None if no valid lines exist
    """
    if lines is None:
        return None

    x_values = []
    min_angle_rad = np.radians(MIN_LINE_ANGLE_DEG)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate angle from horizontal using atan2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        angle = np.arctan2(dy, dx)

        # Reject lines that are too horizontal — this filters out the wide
        # yellow bands at crossing roads / intersections in Webots city.
        # Crossing road markings appear as horizontal lines (angle ≈ 0°),
        # while the lane stripe we want to follow is mostly vertical (≈ 90°).
        if angle < min_angle_rad:
            continue

        midpoint_x = (x1 + x2) / 2.0
        x_values.append(midpoint_x)

        # Visualy show the lines in the camera.
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if len(x_values) > 0:
        lane_center = np.mean(x_values)
        return lane_center

    return None


def display(driver, image_width, image_height, frame, gray, masked_edges):
    frame_big = cv2.resize(
        frame,
        (
            image_width * DISPLAY_UI_SCALE_MULTIPLIER,
            image_height * DISPLAY_UI_SCALE_MULTIPLIER,
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    gray_big = cv2.resize(
        gray,
        (
            image_width * DISPLAY_UI_SCALE_MULTIPLIER,
            image_height * DISPLAY_UI_SCALE_MULTIPLIER,
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    roi_big = cv2.resize(
        masked_edges,
        (
            image_width * DISPLAY_UI_SCALE_MULTIPLIER,
            image_height * DISPLAY_UI_SCALE_MULTIPLIER,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.imshow("Camera (Hough + Canny)", frame_big)
    cv2.imshow("Gray mask", gray_big)
    cv2.imshow("ROI", roi_big)

    if driver.getTime() < 0.1:
        cv2.moveWindow("Camera (Hough + Canny)", 10, 50)
        cv2.moveWindow(
            "Gray mask", 10, 50 + image_height * DISPLAY_UI_SCALE_MULTIPLIER + 40
        )
        cv2.moveWindow(
            "ROI", 10, 50 + (image_height * DISPLAY_UI_SCALE_MULTIPLIER + 40) * 2
        )

    cv2.waitKey(1)


def detect_pedestrian_svm(frame, svm_model, svm_scaler, hog):
    """
    MODIFICATION: Pedestrian detection using either YOLOv8 or HOG+SVM.
    Controlled by USE_YOLO flag at the top of the file.

    Reference: Sliding Window + HOG + SVM technique
    https://medium.com/@ricardo.zuccolo/self-driving-cars-opencv-and-svm
    Reference: Ultralytics YOLOv8 - https://docs.ultralytics.com/
    """
    if USE_YOLO:
        if not hasattr(detect_pedestrian_svm, 'yolo_model'):
            detect_pedestrian_svm.yolo_model = YOLO(YOLO_PATH)

        results = detect_pedestrian_svm.yolo_model(frame, verbose=False, conf=0.3)

        detected = False
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, f"PED {conf:.2f}", (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    detected = True

        print(f"[YOLO] pedestrian={detected}")
        return detected
    else:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = False
        max_score = -999

        x_start = int(w * 0.25)
        x_end = int(w * 0.75)
        y_start = int(h * 0.1)
        y_end = int(h * 0.85)

        for y in range(y_start, max(y_start + 1, y_end - 128 + 1), 32):
            for x in range(x_start, max(x_start + 1, x_end - 64 + 1), 32):
                window = gray[y:y + 128, x:x + 64]
                if window.shape[0] != 128 or window.shape[1] != 64:
                    continue
                features = hog.compute(window).flatten().reshape(1, -1)
                features_scaled = svm_scaler.transform(features)
                score = svm_model.decision_function(features_scaled)[0]

                if score > max_score:
                    max_score = score

                if score > SVM_CONFIDENCE_THRESHOLD:
                    cv2.rectangle(frame, (x, y), (x + 64, y + 128), (0, 0, 255), 1)
                    detected = True

        print(f"[SVM] max_score={max_score:.2f} pedestrian={detected}")
        return detected


def main():
    """
    Main function that initializes the vehicle and runs the autonomous
    lane-following control loop.

    The vehicle is positioned over the yellow center line of roads.
    At each timestep:
    1. Capture camera image
    2. Process image through the lane detection pipeline
    3. Calculate lane center from detected lines
    4. Feed lane center into PID controller to compute steering angle
    5. Apply steering angle and maintain constant speed
    """

    driver = Driver()

    timestep = int(driver.getBasicTimeStep())

    camera = driver.getDevice("camera")
    camera.enable(timestep)

    # MODIFICATION: Enable SICK LiDAR sensor for obstacle detection
    lidar = driver.getDevice(LIDAR_DEVICE_NAME)
    lidar.enable(timestep)
    lidar.enablePointCloud()  # needed to read range data

    # MODIFICATION: Load trained SVM model and scaler for pedestrian detection
    print(f"Loading SVM model from: {SVM_MODEL_PATH}")
    svm_model = joblib.load(SVM_MODEL_PATH)
    svm_scaler = joblib.load(SVM_SCALER_PATH)
    hog = cv2.HOGDescriptor(SVM_WINDOW_SIZE, (16, 16), (8, 8), (8, 8), 9)
    print("SVM model loaded successfully")

    image_width = camera.getWidth()
    image_height = camera.getHeight()

    # Setpoint is where the yellow line SHOULD appear in the camera image
    # when the car is correctly in the right lane.
    setpoint = image_width * LANE_SETPOINT_PERCENTAGE

    print(f"Camera resolution: {image_width}x{image_height}")
    print(f"PID Setpoint (yellow line target x): {setpoint}")
    print(f"Cruising speed: {CRUISING_SPEED} km/h")
    print(f"PID Gains - Kp: {KP}, Ki: {KI}, Kd: {KD}")

    pid = PIDController(kp=KP, ki=KI, kd=KD, setpoint=setpoint)

    smoothed_lane_center = setpoint

    driver.setCruisingSpeed(CRUISING_SPEED)

    was_braking = False
    hazard_on = False
    last_is_pedestrian = False
    brake_frame_count = 0

    while driver.step() != -1:
        frame = get_image(camera)
        gray = convert_to_grayscale(frame)
        yellow_mask = get_yellow_mask(frame)

        # Dilate the mask with a 3x3 kernel to make the yellow stripe thicker
        # but not so much that it bleeds into nearby white crossing stripes.
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)

        edges = detect_edges(yellow_mask)
        masked_edges = apply_region_of_interest(edges)
        lines = detect_lines(masked_edges)

        # If too many edges generate unstable line detections,
        # use the yellow mask directly for a cleaner Hough transform.
        if lines is not None and len(lines) > MAX_LINES_FOR_LANE_THRESHOLD:
            masked_edges = apply_region_of_interest(yellow_mask)
            lines = detect_lines(masked_edges)

        lane_center = calculate_lane_center(lines, frame)

        if lane_center is not None:
            smoothed_lane_center = (
                SMOOTHING_ALPHA * lane_center
                + (1 - SMOOTHING_ALPHA) * smoothed_lane_center
            )
            steering_angle = pid.compute(smoothed_lane_center)
        else:
            steering_angle = DEFAULT_ANGLE

        # MODIFICATION: LiDAR obstacle detection - read central 30° arc
        obstacle_detected = False
        min_distance = float('inf')
        range_image = lidar.getRangeImage()
        if range_image:
            num_points = len(range_image)
            # SICK LMS 291 has 180° FOV; calculate indices for central 30°
            angle_per_point = 180.0 / num_points
            center = num_points // 2
            half_window = int((LIDAR_DETECTION_ANGLE_DEG / 2.0) / angle_per_point)
            start_idx = center - half_window
            end_idx = center + half_window

            for i in range(start_idx, end_idx):
                distance = range_image[i]
                if distance < min_distance:
                    min_distance = distance
                if distance < LIDAR_MAX_RANGE_M:
                    obstacle_detected = True

        # MODIFICATION: Emergency braking when obstacle detected
        if obstacle_detected:
            driver.setThrottle(0.0)
            driver.setBrakeIntensity(1.0)
            driver.setSteeringAngle(steering_angle)  # Keep steering to maintain lane

            # MODIFICATION: Use YOLO to classify obstacle every 5 frames
            if not was_braking or brake_frame_count % 5 == 0:
                is_pedestrian = detect_pedestrian_svm(frame, svm_model, svm_scaler, hog)
                last_is_pedestrian = is_pedestrian
            else:
                is_pedestrian = last_is_pedestrian
            brake_frame_count += 1

            if is_pedestrian:
                # Pedestrian: brake only, NO hazard lights
                if hazard_on:
                    driver.setHazardFlashers(False)
                    hazard_on = False
                print(f"[BRAKE] PEDESTRIAN at {min_distance:.1f}m - STOPPING!")
                cv2.rectangle(frame, (0, 0), (image_width, 14), (0, 0, 200), -1)
                cv2.putText(frame, "PEDESTRIAN", (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # Not a pedestrian: brake + hazard lights ON
                if not hazard_on:
                    driver.setHazardFlashers(True)
                    hazard_on = True
                print(f"[BRAKE] OBSTACLE at {min_distance:.1f}m - STOPPING!")
                cv2.rectangle(frame, (0, 0), (image_width, 14), (0, 100, 200), -1)
                cv2.putText(frame, "OBSTACLE", (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            was_braking = True
        else:
            driver.setBrakeIntensity(0.0)
            if was_braking:
                driver.setHazardFlashers(False)
                hazard_on = False
            driver.setSteeringAngle(steering_angle)
            driver.setCruisingSpeed(CRUISING_SPEED)
            cv2.rectangle(frame, (0, 0), (image_width, 14), (0, 150, 0), -1)
            cv2.putText(frame, "OK", (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            was_braking = False
            brake_frame_count = 0

        if steering_angle > 0:
            # print(f"lane_center: {lane_center}, steering: {steering_angle:.4f}")
            pass

        display(driver, image_width, image_height, frame, gray, masked_edges)


if __name__ == "__main__":
    main()
