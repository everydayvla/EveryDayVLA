import argparse
import json
import time
from math import pi

import cv2
import ikpy.chain
import ikpy.inverse_kinematics
import json_numpy
import numpy as np
import requests
import serial
from scipy.spatial.transform import Rotation as R


def post_act(image, instruction, unnorm_key, server_url):
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a NumPy array")

    payload = {
        "full_image": image.tolist(),
        "instruction": instruction,
        "unnorm_key": unnorm_key,
    }
    encoded_payload = json.dumps({"encoded": json.dumps(payload)})

    try:
        response = requests.post(f"{server_url.rstrip('/')}/act", json=json.loads(encoded_payload))
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        print(f"HTTP request failed: {e}")
        return None


def send_message(ser, message):
    if isinstance(message, np.ndarray):
        message_str = ",".join(map(str, message.tolist()))
    else:
        message_str = str(message)

    ser.write((message_str + "\n").encode("utf-8"))
    print(f"Sent to Arduino: {message_str}")


def parse_video_source(video_source):
    try:
        return int(video_source)
    except ValueError:
        return video_source


def load_robot_chain(urdf_path):
    return ikpy.chain.Chain.from_urdf_file(
        urdf_path,
        active_links_mask=[False, False, True, True, True, True, True, True, False],
    )


def action_to_joint_angles(chain, eef_pose):
    target_position = eef_pose[:3]
    target_orientation = eef_pose[3:-1]
    gripper_open = eef_pose[-1]

    frame = np.eye(4)
    frame[:3, 3] = target_position
    frame[:3, :3] = R.from_euler("ZYX", target_orientation).as_matrix()
    angles = ikpy.inverse_kinematics.inverse_kinematic_optimization(
        chain=chain,
        target_frame=frame,
        starting_nodes_angles=[0, 0, 0, 0, 0, 0, 0, 0, 0],
        max_iter=50,
        orientation_mode="all",
        no_position=False,
    )
    print("position: ", target_position, target_orientation, gripper_open)
    return np.append(angles[2:-1], gripper_open).round(3)


def run_robot_client(args):
    ikpy.inverse_kinematics.ORIENTATION_COEFF = args.orientation_coeff

    ser = serial.Serial(args.arduino_port, args.baud_rate, timeout=1)
    time.sleep(args.serial_warmup)

    cap = cv2.VideoCapture(parse_video_source(args.video))
    if not cap.isOpened():
        ser.close()
        raise RuntimeError(f"Could not open video source: {args.video}")

    chain = load_robot_chain(args.urdf)
    frame_num = 0
    action_num = 1
    actions = []
    ack = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video stream ended.")
                break

            if frame_num < args.warmup_frames or frame_num % args.frame_interval != 0:
                frame_num += 1
                continue
            frame_num += 1

            downsampled_frame = cv2.resize(frame, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
            cv2.imshow("EveryDayVLA camera", downsampled_frame)

            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8").strip()
                print(f"Arduino: {line}")
                if line in {"Action Done", "Done with setup() function!"}:
                    ack = True

            if action_num >= len(actions):
                response = post_act(downsampled_frame, args.instruction, args.unnorm_key, args.server_url)
                if response is None:
                    continue
                actions = json_numpy.loads(response.json())
                action_num = 0
                print("Generated action chunk:")
                print(actions)

            if ack and action_num < len(actions):
                send_message(ser, action_to_joint_angles(chain, actions[action_num]))
                action_num += 1
                ack = False

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        ser.close()
        cap.release()
        cv2.destroyAllWindows()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run EveryDayVLA real-robot inference from a laptop.")
    parser.add_argument("--instruction", default="pick up the block and move it away from the robot")
    parser.add_argument("--unnorm-key", default="consolidated_dataset")
    parser.add_argument("--arduino-port", default="COM3")
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--video", default="0", help="Camera index, video file, or IP camera URL.")
    parser.add_argument("--server-url", default="http://localhost:9000")
    parser.add_argument("--urdf", default="seal.urdf")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--frame-interval", type=int, default=6)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--serial-warmup", type=float, default=5.0)
    parser.add_argument("--orientation-coeff", type=float, default=0.001)
    return parser


def main():
    run_robot_client(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
