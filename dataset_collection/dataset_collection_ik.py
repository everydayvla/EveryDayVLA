import os
import ikpy, ikpy.chain
import ikpy.inverse_kinematics
import numpy as np
import scipy.spatial
import serial
import time
import cv2
from scipy.spatial.transform import Rotation as R
from math import pi, pow
import math
import os
from dataset_collection_constants import *
from dataset_collection_scripts import *

directory_path = "path/to/your/directory"

# Replace 'COM3' with your Arduino's port (use '/dev/ttyUSB0' on Linux/macOS)
arduino_port = 'COM10'
baud_rate = 9600  # Must match the Arduino's baud rate

'''
Camera Parameters
70cm away from robot
Lens 13cm above base of the robot
'''



# datasets = create_random_dataset_2(50, "block", 0.02)
# print(datasets)
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
    # url = "https://10.6.0.17:8080/video"
    url = "http://10.5.105.157:4747/video"
    if record_video:
        cap = cv2.VideoCapture(url)
        fps = 30
        ret, frame = cap.read()
        frame_size = (frame.shape[1], frame.shape[0])
    else:
        cap = cv2.VideoCapture(0)
    
    # Create folder structure
    dataset_dir = './datasets/test'

    # datasets = create_random_block_stack_dataset(5, 3, 16)
    datasets = create_cup_drop_dataset(5, folder_start=0, cup_position=[0.001, 0.25, 0.27], drop_off_position=[0.13,0, 0.01], radial_bounds=np.array([[0.13, -pi/2], [0.35, pi/2]]), object_name="pen")
    # datasets = create_random_dataset_2(100, 100, "flask", 0.085)
    # datasets = create_uniform_tasks(0, "golf ball", 0.2, 0.02)
    # datasets = create_random_bottle_pour_dataset(10, [0.075, -0.25], radial_bounds=full_radial_bounds, orientation1=[-pi/2, pi/2, 0], orientation2=[-pi * 5, -0.3, pi * 0.6], start_position=[0.25,0,0.12+height_adjust([0.25,0])])
    # datasets = create_random_drawer_dataset(3, 3, 3)
    print(datasets)

    os.makedirs(dataset_dir, exist_ok=True)
    try:
        video_dir = dataset_dir + f'/{datasets[i]["Folder"]}'
    except:
        video_dir = dataset_dir + '/unnamed'

    
    
    seal = ikpy.chain.Chain.from_urdf_file("seal.urdf",
                                           active_links_mask=[False, False, True, True, True, True, True, True, False])

    try:
        # Open serial connection
        ser = serial.Serial(arduino_port, baud_rate, timeout=1)
        time.sleep(5)  # Allow Arduino to initialize
        
        os.makedirs(video_dir, exist_ok=True)
        angles_file = open(video_dir + "/end_effector_angles.txt", "w")
        instruction_file = open(video_dir + "/language_instruction.txt", "w")
        if record_video:
            video = cv2.VideoWriter(os.path.join(video_dir, f"traj.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

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
                    print(f"Sample Number: {i}")
                    angles_file.write(np.array2string(datasets[i]["Coordinates"], separator=",",max_line_width=10000).replace(" ", ""))
                    angles_file.close()
                    instruction_file.write(datasets[i]["Instruction"])
                    instruction_file.close()
                    
                    # make new video
                    if record_video:
                        video.release()
                    send_message(ser, [0,0,0,0,0,0,1])
                    i += 1
                    j = 0
                    if i >= len(datasets):
                        print("Finished all trajectories!")
                        break

                    # Gives some time to set up next object, including unpowering robot and grabbing object from its maw
                    print("---------------Set up the next object position!--------------")
                    
                    time.sleep(10)


                    if i < len(datasets):
                        video_dir = dataset_dir + f"/{datasets[i]['Folder']}"
                        os.makedirs(video_dir, exist_ok=True)
                        angles_file = open(video_dir + f"/end_effector_angles.txt", "w")
                        instruction_file = open(video_dir + f"/language_instruction.txt", "w")
                        if record_video:
                            video = cv2.VideoWriter(os.path.join(video_dir, f"traj.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
                        continue

                try:
                    # Convert user input into a NumPy array
                    if len(position) != 7:
                        raise ValueError  # Force error handling if length is incorrect
                except ValueError:
                    print("Error: Invalid input! Using default values.")
                    position = np.array([0.1, 0.1, 0.17, 0, 135, 0, 0.9])
                
                # inverse kinematics
                print("Desired location: " + str(position))
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
    main(record_video=False)
    pass