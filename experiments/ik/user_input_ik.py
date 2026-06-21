import numpy as np
import serial
import time
import ikpy
from math import pi
import ikpy.chain
import ikpy.inverse_kinematics
from scipy.spatial.transform import Rotation as R


deg2rad = pi / 180
rad2deg = 180 / pi
# Replace 'COM3' with your Arduino's port (use '/dev/ttyUSB0' on Linux/macOS)
arduino_port = 'COM3'
baud_rate = 9600  # Must match the Arduino's baud rate

seal = ikpy.chain.Chain.from_urdf_file("seal.urdf",
                                       active_links_mask=[False, True, True, True, True, True, True, True, False])

ikpy.inverse_kinematics.ORIENTATION_COEFF = 0.001


def send_message(ser, message):
    if isinstance(message, np.ndarray):
        # Convert NumPy array to a space-separated string
        # message_str = ",".join(map(str, message.tolist()))
        message_str = ",".join(map(str, message.tolist()))  # ✅ Correct (Space-Separated)
    else:
        message_str = str(message)

    # Append newline for Arduino parsing
    message_bytes = (message_str + "\n").encode('utf-8')

    ser.write(message_bytes)  # Send message
    print(f"Sent to Arduino: {message_str}")  # Debugging

ack = False

try:
    # Open serial connection
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)  # Allow Arduino to initialize

    print(f"Listening on {arduino_port}... (Press Ctrl+C to stop)")

    # timeout_start = time.time()  # Start timeout counter
    # timeout_limit = 10  # 10 seconds timeout to prevent indefinite waiting
    # while True:
    #     if ser.in_waiting > 0:
    #         line = ser.readline().decode('utf-8').strip()  # Read and decode
    #         print(f"Arduino: {line}")  # Print Arduino message

    #         # if line == "In arduino loop function()":
    #         #     cond = True
    #         if line == "Action Done" or line == "Done with setup() function":
    #             break
    while True:
        cond = False
        ack = False
        # Check if there is a message from Arduino
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()  # Read and decode
            print(f"Arduino: {line}")  # Print Arduino message

            # if line == "In arduino loop function()":
            #     cond = True
            if line == "Action Done" or line == "Done with setup() function!":
                ack = True
        # Ask user for input (only if setup is done or timed out)
        if ack:
            print(ack, line)
            user_input = input("Enter position (space-separated) or press Enter to use default: ").strip()

            try:
                # Convert user input into a NumPy array
                # user_angles = np.array([float(x) for x in user_input.split()])
                user_input = np.array([np.float64(x) for x in user_input.replace(',', ' ').split()])
                if len(user_input) != 7:
                    raise ValueError  # Force error handling if length is incorrect
            except ValueError:
                print("Error: Invalid input! Using default values.")
                user_input = np.array([0.1, 0.1, 0.17, 0.4, 0.5, 0.6, 0.9])

            print("user_input:", user_input)
            frame = np.eye(4)
            frame[:3,3] = user_input[0:3]
            frame[:3, :3] = R.from_euler("ZYX", user_input[3:6], degrees=True).as_matrix()
            seal_ik = ikpy.inverse_kinematics.inverse_kinematic_optimization(chain=seal,
                                                                            target_frame=frame,
                                                                            starting_nodes_angles=[0,0,0,0,0,0,0,0,0],
                                                                            max_iter=100,
                                                                            orientation_mode="all",
                                                                            no_position=False)
            # seal_ik = seal.inverse_kinematics(target_position=user_input[0:3], target_orientation=user_input[3:6],
            #                       orientation_mode="Z", max_iter=500).round(3)
            user_angles = seal_ik[2:-1]
            user_angles = np.append(user_angles, user_input[-1])

            # Send angles to Arduino
            send_message(ser, user_angles)
            ack = False

except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("\nExiting program.")
finally:
    ser.close()
