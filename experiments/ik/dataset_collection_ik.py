import os
import ikpy, ikpy.chain
import ikpy.inverse_kinematics
import numpy as np
import serial
import time
import cv2
from scipy.spatial.transform import Rotation as R
from math import pi
import os

deg2rad = pi / 180
rad2deg = 180 / pi

# Replace 'COM3' with your Arduino's port (use '/dev/ttyUSB0' on Linux/macOS)
arduino_port = 'COM3'
baud_rate = 9600  # Must match the Arduino's baud rate

'''
Camera Parameters
69cm away from robot
'''

'''
Known coordinates
0.18,0,0.18,0.1,0,-0.095,1
0.18,0,0,0.01,0,-0.2,0

'''
modify_coords = lambda coords, x, y, z: [x, y, z, *coords[3:]] 

known_coordinates = {
    "center_mid_grounded": [0.18,0.0,0.02],
    "center_mid_raised": [0.18,0,0.1],
    "center_mid_raised_lower": [0.2,0,0.08],
    "center_close_grounded": [0.1,0,0.02],
    "center_close_raised": [0.1, 0, 0.1],
    "center_far_grounded": [0.30, 0, 0],
    "center_far_grounded_side": [0.30, 0, 0.02],
    "center_mid_high": [0.18, 0, 0.16],
    "left_close_grounded":[0.07, 0.07, 0.02,],
    "left_close_raised": [0.07, 0.07, 0.15],
    "left_mid_grounded": [0.14, 0.14, 0.02],
    "left_mid_raised": [0.14, 0.14, 0.1],
    "left_mid_high": [0.14, 0.14, 0.3],
    "left_extreme_grounded": [0.0001, 0.35, 0.1],
    "left_extreme_high": [0.02, 0.3, 0.3],
    "right_close_grounded":[0.07, -0.07, 0.02],
    "right_close_raised": [0.07, -0.07, 0.15],
    "right_mid_grounded": [0.14, -0.14, 0.02],
    "right_mid_raised": [0.14, -0.14, 0.1],
    "right_mid_q3_high": [0.14, -0.14, 0.2],
    "right_mid_high": [0.14, -0.14, 0.3],
    "right_extreme_grounded": [0.0001, -0.35, 0.1],
    "right_extreme_high": [0.02, 0.3, 0.3],
}

known_orientations = {
    "vertical_forward": [0, 180, 0],
    "vertical_left": [90, 180, 0],
    "vertical_right": [-90, 180, 0],
    "side_forward": [0, 90, 0],
}
for n in known_orientations:
    known_orientations[n] = [x*deg2rad for x in known_orientations[n]]

gripper_state = {
    "hold":[0],
    "1cm": [0.25],
    "3cm": [0.3],
    "5cm": [0.35],
    "release":[1]
}

datasets = [
    # {"Coordinates": np.array([
    #     known_coordinates["center_mid_raised"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + gripper_state["hold"],
    #     known_coordinates["center_mid_raised"] + gripper_state["hold"],
    # ])},
    # {"Coordinates": np.array([
    #     modify_coords(known_coordinates["center_mid_raised"], 0.18, -0.07, 0.1) + gripper_state["release"],
    #     modify_coords(known_coordinates["center_mid_grounded"], 0.1, -0.07, 0.02) + gripper_state["release"],
    #     modify_coords(known_coordinates["center_mid_grounded"], 0.1, -0.07, 0.02) + gripper_state["hold"],
    #     modify_coords(known_coordinates["center_mid_raised"], 0.18, -0.07, 0.05) + gripper_state["hold"],
    #     modify_coords(known_coordinates["center_mid_raised"], 0.18, 0.07, 0.05) + gripper_state["hold"],
    #     modify_coords(known_coordinates["center_mid_raised"], 0.1, 0.07, 0.05) + gripper_state["release"]
    # ])},
    # {"Coordinates": np.array([
    #     known_coordinates["center_mid_raised"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + gripper_state["hold"],
    #     known_coordinates["center_mid_raised"] + gripper_state["hold"],
    #     known_coordinates["center_close_raised"] + gripper_state["hold"],
    #     known_coordinates["center_close_grounded"] + gripper_state["hold"],
    #     known_coordinates["center_close_grounded"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_close_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_close_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_close_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the yellow block and put it closer to the robot",
    # "Folder": "block_center_mid_to_center_close_ios"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    # ]),
    # "Instruction": "pick up the yellow block",
    # "Folder": "pick_up_center_block_ios"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the yellow block and move it to the right",
    # "Folder": "block_left_mid_to_right_mid"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the yellow block and move to the middle",
    # "Folder": "block_left_mid_to_center_mid"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    #     "Instruction": "stack the right yellow block on the center yellow block",
    #     "Folder": "stack_block_right_mid_to_center_mid"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    #     "Instruction": "stack the right yellow block on the tower of center yellow blocks",
    #     "Folder": "stack_block_tower_left_mid_to_center_mid"
    # },
    {"Coordinates": np.array([
        known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
        known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
        known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
        known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
        known_coordinates["center_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
        known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
        known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],

        known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
        known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
        known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
        known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
        known_coordinates["center_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
        known_coordinates["center_mid_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    ]),
        "Instruction": "stack the yellow blocks to create a block tower",
        "Folder": "stack_block_tower"
    },
    # {"Coordinates": np.array([
    #         known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #         known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #         known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     ]),
    #     "Instruction": "pick up the flask and move it to the left",
    #     "Folder": "flask_right_mid_to_left_mid"
    #     },
    # {"Coordinates": np.array([
    #     known_coordinates["right_mid_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["right_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the flask and move it to the left",
    # "Folder": "flask_right_mid_to_left_mid"
    # },
    # {"Coordinates": np.array([
    #         known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #         known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #         known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["right_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["right_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     ]),
    #     "Instruction": "pick up the blue flask and move it to the right",
    #     "Folder": "flask_left_mid_to_right_mid"
    #     },
    # {"Coordinates": np.array([
    #     known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["left_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["left_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["right_extreme_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["right_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the blue flask and move it to the extreme right",
    # "Folder": "flask_left_extreme_to_right_extreme"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["right_mid_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["right_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["right_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     known_coordinates["left_extreme_high"] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     known_coordinates["left_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     known_coordinates["left_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the blue flask and move it to the extreme left",
    # "Folder": "flask_right_extreme_to_left_extreme"
    # },
    # {"Coordinates": np.array([
    #         known_coordinates["right_mid_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #         known_coordinates["right_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #         known_coordinates["right_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #         known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #         known_coordinates["left_extreme_high"] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #         known_coordinates["left_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #         known_coordinates["left_extreme_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     ]),
    #     "Instruction": "pick up the blue flask and move it to the extreme left",
    #     "Folder": "flask_right_extreme_to_left_extreme"
    #     },
    # {"Coordinates": np.array([
    #         known_coordinates["right_mid_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #         known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #         known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     ]),
    #     "Instruction": "pick up the blue ball and put it in the green cup",
    #     "Folder": "ball_cup_right_mid_to_left_mid"
    #     },

    # {"Coordinates": np.array([
    #         known_coordinates["left_mid_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #         known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #         known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["right_mid_q3_high"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #         known_coordinates["right_mid_q3_high"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     ]),
    #     "Instruction": "pick up the blue ball and put it in the green cup",
    #     "Folder": "ball_cup_left_mid_to_right_mid"
    #     },
]

def send_message(ser, message):
    if isinstance(message, np.ndarray):
        # Convert NumPy array to a space-separated string
        message_str = ",".join(map(str, message.tolist()))  # ✅ Correct (Space-Separated)
    else:
        message_str = str(message)

    # Append newline for Arduino parsing
    message_bytes = (message_str + "\n").encode('utf-8')

    ser.write(message_bytes)  # Send message
    print(f"Sent to Arduino: {message_str}")  # Debugging



def main(record_video=False):
    ikpy.inverse_kinematics.ORIENTATION_COEFF = 0.001
    ack = False
    i = 0
    j = 0
    url = 0
    if record_video:
        cap = cv2.VideoCapture(url)
        fps = 30
        ret, frame = cap.read()
        frame_size = (frame.shape[1], frame.shape[0])
    else:
        cap = cv2.VideoCapture(0)
    
    # Create folder structure
    dataset_dir = './datasets/6dof'
    os.makedirs(dataset_dir, exist_ok=True)
    try:
        video_dir = dataset_dir + f'/{datasets[i]["Folder"]}'
    except:
        video_dir = dataset_dir + '/unnamed'
    os.makedirs(video_dir, exist_ok=True)
    angles_file = open(video_dir + "/end_effector_angles.txt", "w")
    instruction_file = open(video_dir + "/language_instruction.txt", "w")
    if record_video:
        video = cv2.VideoWriter(os.path.join(video_dir, f"traj_{i}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    
    seal = ikpy.chain.Chain.from_urdf_file("seal.urdf",
                                           active_links_mask=[False, False, True, True, True, True, True, True, False])

    try:
        # Open serial connection
        ser = serial.Serial(arduino_port, baud_rate, timeout=1)
        time.sleep(5)  # Allow Arduino to initialize

        print(f"Listening on {arduino_port}... (Press Ctrl+C to stop)")

        while True:
            ret, frame = cap.read()
            if ret and record_video:
                video.write(frame)

            ack = False
            # Check if there is a message from Arduino
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()  # Read and decode
                print(f"Arduino: {line}")  # Print Arduino message
                if line == "Action Done" or line == "Done with setup() function!":
                    ack = True

            # Ask user for input (only if setup is done or timed out)
            if ack:

                position = np.array([0.01, 0, 0.28, 0, 0, 0, 1])
                if j < datasets[i]["Coordinates"].shape[0]:
                    position = datasets[i]["Coordinates"][j]
                j += 1
                if j > datasets[i]["Coordinates"].shape[0]:
                    angles_file.write(np.array2string(datasets[i]["Coordinates"], separator=",",).replace(" ", ""))
                    angles_file.close()
                    instruction_file.write(datasets[i]["Instruction"])
                    instruction_file.close()
                    
                    # make new video
                    if record_video:
                        video.release()
                    i += 1
                    j = 0
                    if i >= len(datasets):
                        print("Finished all trajectories!")
                        break

                    # Gives some time to set up next object, including unpowering robot and grabbing object from its maw
                    print("---------------Set up the next object position!--------------")
                    time.sleep(20)


                    if i < len(datasets):
                        video_dir = dataset_dir + f"/{datasets[i]['Instruction']}"
                        os.makedirs(video_dir, exist_ok=True)
                        angles_file = open(video_dir + f"/end_effector_angles_{i}.txt", "w")
                        instruction_file = open(video_dir + f"/language_instruction_{i}.txt", "w")
                        if record_video:
                            video = cv2.VideoWriter(os.path.join(video_dir, f"traj_{i}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

                try:
                    # Convert user input into a NumPy array
                    if len(position) != 7:
                        raise ValueError  # Force error handling if length is incorrect
                except ValueError:
                    print("Error: Invalid input! Using default values.")
                    position = np.array([0.1, 0.1, 0.17, 0, 135, 0, 0.9])
                
                # inverse kinematics
                frame = np.eye(4)
                frame[:3,3] = position[0:3]
                frame[:3, :3] = R.from_euler("ZYX", position[3:6]).as_matrix()
                angles = ikpy.inverse_kinematics.inverse_kinematic_optimization(chain=seal,
                                                                                target_frame=frame,
                                                                                starting_nodes_angles=[0,0,0,0,0,0,0,0,0],
                                                                                max_iter=50,
                                                                                orientation_mode="all",
                                                                                no_position=False)
                print("user_angles:", angles[2:-1])

                # Send angles to Arduino
                angles = np.append(angles[2:-1], position[-1]).round(3)

                send_message(ser, angles)

                ack = False
                
                if i >= len(datasets):
                    break

            # saving video

    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nExiting program.")
    except TimeoutError:
        print("Datasets collected!")
    finally:
        # print("Done!")
        ser.close()
        if record_video:
            video.release()
        cap.release()
        angles_file.close()


if __name__ == '__main__':
    main(record_video=True)
