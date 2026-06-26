from vehicle import Driver
import numpy as np
import cv2
import os
import sys

VENV_SITE = os.path.join(os.path.dirname(__file__), "..", "..", "venv", "lib")
for d in os.listdir(VENV_SITE) if os.path.isdir(VENV_SITE) else []:
    sp = os.path.join(VENV_SITE, d, "site-packages")
    if os.path.isdir(sp) and sp not in sys.path:
        sys.path.insert(0, sp)

import tensorflow.lite as tflite

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "cil_model.tflite")
CRUISING_SPEED = 30.0
MAX_STEERING_ANGLE = 0.5
RIGHT_LANE_BIAS = 0.01
NUM_COMMANDS = 4
DISPLAY_SCALE = 2

LIDAR_DEVICE_NAME = "Sick LMS 291"
LIDAR_DETECTION_ANGLE_DEG = 20
LIDAR_BRAKE_DISTANCE = 8.0
LIDAR_SLOW_DISTANCE = 15.0
LIDAR_AVOIDANCE_TRIGGER = 12.0
SLOW_SPEED = 12.0

STATE_CIL = 0
STATE_STEER_LEFT = 1
STATE_PASS = 2
STATE_STEER_RIGHT = 3
AVOIDANCE_SPEED = 18.0
STEER_LEFT_ANGLE = -0.25
STEER_RIGHT_ANGLE = 0.15
STEER_LEFT_STEPS = 35
PASS_STEPS = 60
STEER_RIGHT_STEPS = 40

CMD_FOLLOW_LANE = 0
CMD_TURN_LEFT = 1
CMD_TURN_RIGHT = 2
CMD_GO_STRAIGHT = 3


def get_image(camera):
    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(rgb, axis=0)


def get_lidar_min_distance(lidar):
    range_image = lidar.getRangeImage()
    if not range_image:
        return float('inf')
    num_points = len(range_image)
    angle_per_point = 180.0 / num_points
    center = num_points // 2
    half_window = int((LIDAR_DETECTION_ANGLE_DEG / 2.0) / angle_per_point)
    start = max(0, center - half_window)
    end = min(num_points, center + half_window)
    min_dist = float('inf')
    for i in range(start, end):
        if range_image[i] < min_dist:
            min_dist = range_image[i]
    return min_dist


def main():
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    camera = driver.getDevice("camera")
    camera.enable(timestep)
    camera.recognitionEnable(timestep)

    lidar = driver.getDevice(LIDAR_DEVICE_NAME)
    lidar.enable(timestep)
    lidar.enablePointCloud()

    keyboard = driver.getKeyboard()
    keyboard.enable(timestep)

    print(f"Loading CIL model from: {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_input_idx = next(i for i, d in enumerate(input_details) if d['shape'][1] == 160)
    cmd_input_idx = next(i for i, d in enumerate(input_details) if d['shape'][1] == 4)
    print("Model loaded successfully!")

    current_command = CMD_FOLLOW_LANE
    command_names = {0: "FOLLOW", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}

    state = STATE_CIL
    state_counter = 0
    avoidance_enabled = False
    state_names = {0: "CIL", 1: "AVOID_LEFT", 2: "PASSING", 3: "AVOID_RIGHT"}

    cmd_tensors = [np.eye(NUM_COMMANDS, dtype=np.float32)[[i]] for i in range(NUM_COMMANDS)]
    cached_steering = 0.0
    frame_count = 0

    driver.setCruisingSpeed(CRUISING_SPEED)

    print("=" * 50)
    print("  CIL AUTONOMOUS DRIVING - WORLD 2")
    print("=" * 50)
    print("  W=Follow | A=Left | D=Right | S=Straight")
    print("  E=Toggle avoidance")
    print("=" * 50)

    while driver.step() != -1:
        key = keyboard.getKey()
        while key > 0:
            if key == ord('W') or key == ord('w'):
                current_command = CMD_FOLLOW_LANE
            elif key == ord('A') or key == ord('a'):
                current_command = CMD_TURN_LEFT
            elif key == ord('D') or key == ord('d'):
                current_command = CMD_TURN_RIGHT
            elif key == ord('S') or key == ord('s'):
                current_command = CMD_GO_STRAIGHT
            elif key == ord('E') or key == ord('e'):
                avoidance_enabled = not avoidance_enabled
                print(f"  Avoidance: {'ON' if avoidance_enabled else 'OFF'}")
            key = keyboard.getKey()

        min_dist = get_lidar_min_distance(lidar)
        frame = get_image(camera)
        frame_count += 1

        if state == STATE_CIL:
            if current_command == CMD_GO_STRAIGHT:
                cached_steering = 0.0
            else:
                img_input = preprocess(frame)
                pred = None
                interpreter.set_tensor(input_details[img_input_idx]['index'], img_input)
                interpreter.set_tensor(input_details[cmd_input_idx]['index'], cmd_tensors[current_command])
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                cached_steering = float(pred[0][0]) * MAX_STEERING_ANGLE + RIGHT_LANE_BIAS

            steering_angle = np.clip(cached_steering, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

            speed = CRUISING_SPEED
            braking = False

            pedestrian_detected = False
            num_objects = camera.getRecognitionNumberOfObjects()
            if num_objects > 0:
                for obj in camera.getRecognitionObjects():
                    if "pedestrian" in obj.getModel().lower():
                        pedestrian_detected = True
                        break

            if pedestrian_detected and min_dist < LIDAR_SLOW_DISTANCE:
                braking = True
                driver.setBrakeIntensity(1.0)
                if frame_count % 10 == 0:
                    print(f"  [PEDESTRIAN] Detected at {min_dist:.1f}m — BRAKING")
            elif avoidance_enabled and min_dist < LIDAR_AVOIDANCE_TRIGGER:
                state = STATE_STEER_LEFT
                state_counter = 0
                print(f"  [AVOIDANCE] Obstacle at {min_dist:.1f}m — steering left")
            elif min_dist < LIDAR_BRAKE_DISTANCE:
                braking = True
                driver.setBrakeIntensity(1.0)
            elif min_dist < LIDAR_SLOW_DISTANCE:
                speed = SLOW_SPEED
                driver.setBrakeIntensity(0.0)
            else:
                driver.setBrakeIntensity(0.0)

            driver.setSteeringAngle(steering_angle)
            if not braking:
                driver.setCruisingSpeed(speed)

        elif state == STATE_STEER_LEFT:
            driver.setBrakeIntensity(0.0)
            driver.setCruisingSpeed(AVOIDANCE_SPEED)
            driver.setSteeringAngle(STEER_LEFT_ANGLE)
            state_counter += 1
            if state_counter >= STEER_LEFT_STEPS:
                state = STATE_PASS
                state_counter = 0
                print("  [AVOIDANCE] Passing obstacle...")

        elif state == STATE_PASS:
            driver.setCruisingSpeed(AVOIDANCE_SPEED)
            driver.setSteeringAngle(0.0)
            state_counter += 1
            if state_counter >= PASS_STEPS:
                state = STATE_STEER_RIGHT
                state_counter = 0
                print("  [AVOIDANCE] Returning to lane...")

        elif state == STATE_STEER_RIGHT:
            driver.setCruisingSpeed(AVOIDANCE_SPEED)
            driver.setSteeringAngle(STEER_RIGHT_ANGLE)
            state_counter += 1
            if state_counter >= STEER_RIGHT_STEPS:
                state = STATE_CIL
                state_counter = 0
                print("  [AVOIDANCE] Done — resuming CIL")

        if frame_count % 10 == 0:
            cur_steer = driver.getSteeringAngle()
            avd_str = "AVD:ON" if avoidance_enabled else "AVD:OFF"
            print(f"  {state_names[state]} | CMD:{command_names[current_command]} | Steer:{cur_steer:.3f} | Dist:{min_dist:.1f}m | {avd_str}")


if __name__ == "__main__":
    main()
