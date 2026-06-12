# Lane detection + right-side wall-following obstacle avoidance controller.
# When a bus is recognized ahead and LIDAR confirms proximity, the vehicle
# steers left gently to pass, follows the bus wall on its right, then
# regains original orientation and resumes lane-following.

from vehicle import Driver
import numpy as np
import cv2

# =============================================================================
# Lane-Following Constants
# =============================================================================
KP = 1.0
KI = 0.005
KD = 0.2
MAX_STEERING_ANGLE = 0.5
CRUISING_SPEED = 55.0
DEFAULT_ANGLE = 0.0
CANNY_LOW = 40
CANNY_HIGH = 120
YELLOW_HSV_LOW = np.array([18, 30, 80])
YELLOW_HSV_HIGH = np.array([40, 255, 255])
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 10
HOUGH_MIN_LINE_LENGTH = 8
HOUGH_MAX_LINE_GAP = 15
MIN_LINE_ANGLE_DEG = 30
SMOOTHING_ALPHA = 0.3
LANE_SETPOINT_PERCENTAGE = 0.35
MAX_LINES_FOR_LANE_THRESHOLD = 4
DISPLAY_UI_SCALE_MULTIPLIER = 4

# =============================================================================
# Obstacle Avoidance Constants
# =============================================================================
LIDAR_OBSTACLE_THRESHOLD = 18.0
WALL_FOLLOW_SPEED = 25.0
WALL_DESIRED_DISTANCE = 3.5
WALL_KP = 0.15
WALL_KD = 0.05
DS_NO_OBSTACLE_THRESHOLD = 6.5
ANGLE_TOLERANCE = 0.03

# Gentle steer-left angle for initial lane change
STEER_LEFT_ANGLE = -0.35

# States
STATE_LANE_FOLLOWING = 0
STATE_STEER_LEFT = 1
STATE_WALL_FOLLOWING = 2
STATE_REGAIN_ORIENTATION = 3
STATE_REENTER_LANE = 4


class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, measurement):
        error = (measurement - self.setpoint) / self.setpoint
        self.integral = np.clip(self.integral + error, -5.0, 5.0)
        derivative = error - self.previous_error
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)


def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def detect_lane_center(frame):
    """Full lane detection pipeline. Returns (lane_center_x, debug_frame) or (None, frame)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH)
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)

    edges = cv2.Canny(cv2.GaussianBlur(yellow_mask, (5, 5), 0), CANNY_LOW, CANNY_HIGH)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi_vertices = np.array(
        [[(0, height), (0, int(height * 0.55)),
          (width, int(height * 0.55)), (width, height)]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if lines is not None and len(lines) > MAX_LINES_FOR_LANE_THRESHOLD:
        masked_edges = cv2.bitwise_and(yellow_mask, mask)
        lines = cv2.HoughLinesP(
            masked_edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP,
        )

    if lines is None:
        return None, masked_edges

    min_angle_rad = np.radians(MIN_LINE_ANGLE_DEG)
    x_values = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1))
        if angle >= min_angle_rad:
            x_values.append((x1 + x2) / 2.0)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if x_values:
        return np.mean(x_values), masked_edges
    return None, masked_edges


def get_lidar_front_distance(lidar):
    """Get minimum distance from LIDAR point cloud in forward direction."""
    point_cloud = lidar.getPointCloud()
    if not point_cloud:
        return float('inf')

    min_dist = float('inf')
    half_beam = np.radians(15)

    for point in point_cloud:
        x, y, z = point.x, point.y, point.z
        if x <= 0:
            continue
        angle = abs(np.arctan2(y, x))
        if angle < half_beam:
            dist = np.sqrt(x * x + y * y)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def get_recognized_bus(camera):
    """Check if a bus is recognized. Returns (object, bus_id) or (None, None)."""
    num_objects = camera.getRecognitionNumberOfObjects()
    if num_objects == 0:
        return None, None
    objects = camera.getRecognitionObjects()
    for obj in objects:
        model = obj.getModel()
        if "bus" in model.lower() or "Bus" in model:
            colors = obj.getColors()
            # Colors is a ctypes pointer: [r, g, b] per color
            r, g, b = colors[0], colors[1], colors[2]
            bus_id = identify_bus(r, g, b)
            return obj, bus_id
    return None, None


def identify_bus(r, g, b):
    """Identify bus by its recognition color. Returns bus ID string."""
    bus_colors = {
        "vehicle(1)": (0.031, 0.122, 0.420),
        "vehicle(2)": (1.000, 0.000, 0.000),
        "vehicle(3)": (0.863, 0.541, 0.867),
        "vehicle(4)": (0.180, 0.761, 0.494),
    }
    best_id = "unknown"
    best_dist = float('inf')
    for name, (cr, cg, cb) in bus_colors.items():
        dist = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
        if dist < best_dist:
            best_dist = dist
            best_id = name
    return best_id if best_dist < 0.1 else "unknown"


def display_debug(driver, camera, image_width, image_height, frame, masked_edges, state):
    """Show camera view with recognition overlay and ROI."""
    # Draw recognition bounding boxes on frame
    num_objects = camera.getRecognitionNumberOfObjects()
    if num_objects > 0:
        objects = camera.getRecognitionObjects()
        for obj in objects:
            pos = obj.getPositionOnImage()
            size = obj.getSizeOnImage()
            model = obj.getModel()
            if "bus" not in model.lower() and "Bus" not in model:
                continue
            x = int(pos[0] - size[0] / 2)
            y = int(pos[1] - size[1] / 2)
            w = int(size[0])
            h = int(size[1])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(frame, model, (x, max(y - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    frame_big = cv2.resize(
        frame,
        (image_width * DISPLAY_UI_SCALE_MULTIPLIER,
         image_height * DISPLAY_UI_SCALE_MULTIPLIER),
        interpolation=cv2.INTER_NEAREST,
    )
    roi_big = cv2.resize(
        masked_edges,
        (image_width * DISPLAY_UI_SCALE_MULTIPLIER,
         image_height * DISPLAY_UI_SCALE_MULTIPLIER),
        interpolation=cv2.INTER_NEAREST,
    )

    state_names = ["LANE_FOLLOW", "STEER_LEFT", "WALL_FOLLOW", "REGAIN_YAW", "REENTER"]
    cv2.putText(frame_big, state_names[state], (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Camera (Hough + Canny)", frame_big)
    cv2.imshow("ROI", roi_big)

    if driver.getTime() < 0.1:
        cv2.moveWindow("Camera (Hough + Canny)", 10, 50)
        cv2.moveWindow("ROI", 10, 50 + image_height * DISPLAY_UI_SCALE_MULTIPLIER + 40)

    cv2.waitKey(1)


def main():
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    # --- Initialize Camera with Recognition ---
    camera = driver.getDevice("camera")
    camera.enable(timestep)
    camera.recognitionEnable(timestep)

    image_width = camera.getWidth()
    image_height = camera.getHeight()

    # --- Initialize LIDAR ---
    lidar = driver.getDevice("Sick LMS 291")
    lidar.enable(timestep)
    lidar.enablePointCloud()

    # --- Initialize Gyroscope ---
    gyro = driver.getDevice("gyro")
    gyro.enable(timestep)

    # --- Initialize Right-Side Distance Sensors ---
    ds_right_front = driver.getDevice("ds_right_front")
    ds_right_mid = driver.getDevice("ds_right_mid")
    ds_right_rear = driver.getDevice("ds_right_rear")
    ds_right_front.enable(timestep)
    ds_right_mid.enable(timestep)
    ds_right_rear.enable(timestep)

    # --- PID for lane following ---
    setpoint = image_width * LANE_SETPOINT_PERCENTAGE
    pid = PIDController(kp=KP, ki=KI, kd=KD, setpoint=setpoint)
    smoothed_lane_center = setpoint

    # --- State Machine ---
    state = STATE_LANE_FOLLOWING
    saved_yaw = 0.0
    current_yaw = 0.0
    prev_wall_error = 0.0
    step_counter = 0

    print(f"[INIT] Camera: {image_width}x{image_height}, Recognition enabled")
    print(f"[INIT] LIDAR threshold: {LIDAR_OBSTACLE_THRESHOLD}m")
    print(f"[INIT] Gyroscope + 3 right-side distance sensors ready")

    driver.setCruisingSpeed(CRUISING_SPEED)

    while driver.step() != -1:
        # --- Integrate gyro yaw ---
        gyro_values = gyro.getValues()
        dt = timestep / 1000.0
        current_yaw += gyro_values[2] * dt

        # --- Read distance sensors (value / 100 = meters) ---
        ds_front_m = ds_right_front.getValue() / 100.0
        ds_mid_m = ds_right_mid.getValue() / 100.0
        ds_rear_m = ds_right_rear.getValue() / 100.0

        # --- Get camera image ---
        frame = get_image(camera)

        # ==============================================================
        # STATE: LANE FOLLOWING
        # ==============================================================
        if state == STATE_LANE_FOLLOWING:
            bus_obj, bus_id = get_recognized_bus(camera)
            if bus_obj:
                lidar_dist = get_lidar_front_distance(lidar)
                print(f"[RECOGNITION] Bus detected: {bus_id} "
                      f"| LIDAR distance: {lidar_dist:.2f}m")

                if lidar_dist < LIDAR_OBSTACLE_THRESHOLD:
                    # Reset yaw integrator — from now we track relative rotation
                    current_yaw = 0.0
                    saved_yaw = 0.0
                    state = STATE_STEER_LEFT
                    step_counter = 0
                    driver.setCruisingSpeed(WALL_FOLLOW_SPEED)
                    print(f"[AVOIDANCE] Begin. Gyro orientation saved (yaw=0 reference).")

            if state == STATE_LANE_FOLLOWING:
                lane_center, masked_edges = detect_lane_center(frame)
                if lane_center is not None:
                    smoothed_lane_center = (
                        SMOOTHING_ALPHA * lane_center
                        + (1 - SMOOTHING_ALPHA) * smoothed_lane_center
                    )
                    steering_angle = pid.compute(smoothed_lane_center)
                else:
                    steering_angle = DEFAULT_ANGLE
                driver.setSteeringAngle(steering_angle)
                driver.setCruisingSpeed(CRUISING_SPEED)
                display_debug(driver, camera, image_width, image_height, frame, masked_edges, state)
                continue

        # ==============================================================
        # STATE: STEER LEFT (gentle lane change to pass on left side)
        # ==============================================================
        elif state == STATE_STEER_LEFT:
            driver.setCruisingSpeed(WALL_FOLLOW_SPEED)
            driver.setSteeringAngle(STEER_LEFT_ANGLE)
            step_counter += 1

            # Transition: once the right-mid sensor picks up the bus,
            # or after enough steps to have cleared the front
            if ds_mid_m < WALL_DESIRED_DISTANCE * 1.5 and step_counter > 10:
                state = STATE_WALL_FOLLOWING
                prev_wall_error = 0.0
                step_counter = 0
                print("[AVOIDANCE] Bus on right side. Wall-following.")
            elif step_counter > 60:
                # Timeout fallback
                state = STATE_WALL_FOLLOWING
                prev_wall_error = 0.0
                step_counter = 0
                print("[AVOIDANCE] Timeout -> wall-following.")

        # ==============================================================
        # STATE: WALL FOLLOWING (keep bus on right at desired distance)
        # ==============================================================
        elif state == STATE_WALL_FOLLOWING:
            driver.setCruisingSpeed(WALL_FOLLOW_SPEED)
            step_counter += 1

            # PD control: positive error = too close -> steer left (negative angle)
            wall_error = WALL_DESIRED_DISTANCE - ds_mid_m
            wall_derivative = wall_error - prev_wall_error
            prev_wall_error = wall_error

            # Negative steering = steer left, positive = steer right
            # If wall_error > 0 (too close), steer left (negative)
            # If wall_error < 0 (too far), steer right (positive) toward wall
            steering = -(WALL_KP * wall_error + WALL_KD * wall_derivative)
            steering = np.clip(steering, -0.3, 0.3)
            driver.setSteeringAngle(steering)

            print(f"[WALL] front:{ds_front_m:.1f} mid:{ds_mid_m:.1f} "
                  f"rear:{ds_rear_m:.1f} steer:{steering:.3f}")

            # End: rear sensor clears the obstacle
            if ds_rear_m > DS_NO_OBSTACLE_THRESHOLD and step_counter > 40:
                state = STATE_REGAIN_ORIENTATION
                step_counter = 0
                print(f"[AVOIDANCE] Bus cleared. Regaining orientation "
                      f"(current yaw offset: {current_yaw:.3f} rad, target: 0)")

        # ==============================================================
        # STATE: REGAIN ORIENTATION (use gyro to steer back to saved yaw)
        # ==============================================================
        elif state == STATE_REGAIN_ORIENTATION:
            driver.setCruisingSpeed(WALL_FOLLOW_SPEED)
            step_counter += 1

            # Steer right gently until gyro shows we've returned to original orientation
            driver.setSteeringAngle(0.2)

            if current_yaw <= 0.05 and step_counter > 15:
                state = STATE_REENTER_LANE
                step_counter = 0
                print(f"[AVOIDANCE] Orientation recovered (yaw={current_yaw:.3f}). "
                      f"Re-entering lane...")
            elif step_counter % 20 == 0:
                print(f"[GYRO] Recovering: yaw_offset={current_yaw:.3f} rad")

        # ==============================================================
        # STATE: REENTER LANE (drift right to get back into original lane)
        # ==============================================================
        elif state == STATE_REENTER_LANE:
            driver.setCruisingSpeed(WALL_FOLLOW_SPEED)
            driver.setSteeringAngle(0.15)
            step_counter += 1

            if step_counter > 50:
                driver.setSteeringAngle(0.0)
                state = STATE_LANE_FOLLOWING
                smoothed_lane_center = setpoint
                pid.integral = 0.0
                pid.previous_error = 0.0
                print("[AVOIDANCE] Done. Resuming lane-following.")
                if step_counter % 20 == 0:
                    print(f"[GYRO] yaw_offset={current_yaw:.3f} "
                          f"error={yaw_error:.3f} steer={steering:.3f}")

        # Display for non-lane-following states
        if state != STATE_LANE_FOLLOWING:
            _, masked_edges = detect_lane_center(frame)
            display_debug(driver, camera, image_width, image_height, frame, masked_edges, state)


if __name__ == "__main__":
    main()
