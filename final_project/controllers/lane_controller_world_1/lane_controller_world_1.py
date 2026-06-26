from vehicle import Driver
import numpy as np
import cv2
import os
import csv

CRUISING_SPEED = 25.0
STEERING_INCREMENT = 0.02
MAX_STEERING_ANGLE = 0.5
STEERING_DECAY = 0.92
CAPTURE_EVERY_N_STEPS = 2
DISPLAY_SCALE = 2

CMD_FOLLOW_LANE = 0
CMD_TURN_LEFT = 1
CMD_TURN_RIGHT = 2
CMD_GO_STRAIGHT = 3

KEY_LEFT = 314
KEY_RIGHT = 316
KEY_UP = 315
KEY_DOWN = 317

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CONTROLLER_DIR, "..", "..", "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH = os.path.join(DATASET_DIR, "driving_log.csv")


def setup_dataset():
    """Create dataset directories and CSV file if they don't exist."""
    os.makedirs(IMAGES_DIR, exist_ok=True)

    existing = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')]
    start_idx = len(existing)

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'steering_angle', 'command'])

    return start_idx


def get_image(camera):
    """Get BGR image from Webots camera."""
    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def main():
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    camera = driver.getDevice("camera")
    camera.enable(timestep)

    keyboard = driver.getKeyboard()
    keyboard.enable(timestep)

    image_width = camera.getWidth()
    image_height = camera.getHeight()
    img_idx = setup_dataset()
    step_count = 0
    recording = True

    steering_angle = 0.0
    current_command = CMD_FOLLOW_LANE
    command_names = {0: "FOLLOW", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}
    space_was_pressed = False

    driver.setCruisingSpeed(CRUISING_SPEED)

    print("=" * 60)
    print("  CIL DATA COLLECTION - MANUAL DRIVING")
    print("=" * 60)
    print(f"  Camera: {image_width}x{image_height}")
    print(f"  Speed: {CRUISING_SPEED} km/h (fixed)")
    print(f"  Starting at image index: {img_idx}")
    print("-" * 60)
    print("  CONTROLS:")
    print("    LEFT/RIGHT arrows = Steer")
    print("    W = Follow lane | A = Turn left")
    print("    D = Turn right  | S = Go straight")
    print("    SPACE = Pause/Resume recording")
    print("=" * 60)

    while driver.step() != -1:
        step_count += 1

        key = keyboard.getKey()
        steering_pressed = False
        any_space = False

        while key > 0:
            if key == KEY_LEFT:
                steering_angle -= STEERING_INCREMENT
                steering_pressed = True
            elif key == KEY_RIGHT:
                steering_angle += STEERING_INCREMENT
                steering_pressed = True
            elif key == ord('W') or key == ord('w'):
                current_command = CMD_FOLLOW_LANE
            elif key == ord('A') or key == ord('a'):
                current_command = CMD_TURN_LEFT
            elif key == ord('D') or key == ord('d'):
                current_command = CMD_TURN_RIGHT
            elif key == ord('S') or key == ord('s'):
                current_command = CMD_GO_STRAIGHT
            elif key == ord(' '):
                any_space = True
                if not space_was_pressed:
                    recording = not recording
                    print(f"  [{'RECORDING' if recording else 'PAUSED'}]")
                    space_was_pressed = True
            key = keyboard.getKey()

        if not any_space:
            space_was_pressed = False

        if not steering_pressed:
            steering_angle *= STEERING_DECAY

        steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(CRUISING_SPEED)

        if recording and step_count % CAPTURE_EVERY_N_STEPS == 0:
            frame = get_image(camera)
            img_name = f"img_{img_idx:06d}.png"
            img_path = os.path.join(IMAGES_DIR, img_name)
            cv2.imwrite(img_path, frame)

            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([img_name, f"{steering_angle:.6f}", current_command])

            img_idx += 1

        frame = get_image(camera)
        display_frame = cv2.resize(
            frame,
            (image_width * DISPLAY_SCALE, image_height * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST
        )

        color = (0, 255, 0) if recording else (0, 0, 255)
        status = "REC" if recording else "PAUSED"
        cv2.putText(display_frame, f"{status} | imgs: {img_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_frame, f"CMD: {command_names[current_command]}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Steer: {steering_angle:.3f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        bar_center = display_frame.shape[1] // 2
        bar_y = display_frame.shape[0] - 20
        bar_offset = int((steering_angle / MAX_STEERING_ANGLE) * 150)
        cv2.line(display_frame, (bar_center - 150, bar_y), (bar_center + 150, bar_y), (100, 100, 100), 3)
        cv2.circle(display_frame, (bar_center + bar_offset, bar_y), 8, (0, 255, 255), -1)

        cv2.imshow("CIL Data Collection", display_frame)
        if driver.getTime() < 0.1:
            cv2.moveWindow("CIL Data Collection", 10, 50)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
