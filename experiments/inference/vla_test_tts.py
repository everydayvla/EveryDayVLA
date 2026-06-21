import serial
import requests
import numpy as np
import json_numpy
import json
import time
import base64
import cv2
import tensorflow as tf
from math import pi
import ikpy
import ikpy.chain
import ikpy.inverse_kinematics
from scipy.spatial.transform import Rotation as R
import os
import threading
import keyboard
from tts.speech_to_text import SpeechToText
from tts.emotive_tts import EmotiveTextToSpeech

# Constants
deg2rad = pi / 180
rad2deg = 180 / pi


# Create a JSON Encoder class
class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj
        return json.JSONEncoder.default(self, obj)


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def post_act(image, instruction, unnorm_key):
    if not isinstance(image, np.ndarray):
        raise ValueError("Error: Image must be a NumPy array")

    payload = {
        "full_image": image.tolist(),
        "instruction": instruction,
        "unnorm_key": unnorm_key
    }

    encoded_payload = json.dumps({"encoded": json.dumps(payload)})  # Double-encode for JSON compliance

    try:
        response = requests.post("http://localhost:9000/act", json=json.loads(encoded_payload))
        response.raise_for_status()  # Raise error if request fails
        return response
    except requests.RequestException as e:
        print(f"HTTP Request Failed: {e}")
        return None  # Return None if the request fails


def send_message(ser, message):
    if isinstance(message, np.ndarray):
        message_str = ",".join(map(str, message.tolist()))
    else:
        message_str = str(message)

    # Append newline for Arduino parsing
    message_bytes = (message_str + "\n").encode('utf-8')

    ser.write(message_bytes)  # Send message
    print(f"Sent to Arduino: {message_str}")  # Debugging


class RobotController:
    def __init__(self, arduino_port, baud_rate, unnorm_key='six_dof_new_base_dataset', api_key=None):
        # Initialize speech components
        self.stt = SpeechToText(api_key)
        self.tts = EmotiveTextToSpeech(api_key)
        self.tts.set_voice("nova")  # Use more expressive voice
        self.tts.set_model("tts-1-hd")  # Use high-definition model for better quality

        # Robot control parameters
        self.arduino_port = arduino_port
        self.baud_rate = baud_rate
        self.unnorm_key = unnorm_key
        self.instruction = None
        self.ser = None
        self.running = False

        # Initialize IK settings
        ikpy.inverse_kinematics.ORIENTATION_COEFF = 0.001

        # Initialize robot chain
        self.seal = ikpy.chain.Chain.from_urdf_file("seal.urdf",
                                                    active_links_mask=[False, False, True, True, True, True, True, True,
                                                                       False])

    def get_voice_instruction(self):
        """Get instruction from user's voice using speech-to-text"""
        self.tts.speak("Please provide your instruction for the robot.", emotion="gentle")
        print("Listening for instruction...")

        # Record and transcribe audio (7 seconds should be enough for most instructions)
        instruction = self.stt.transcribe(
            audio_file=None if self.stt.record_seconds else self.stt.record_audio(seconds=7))

        print(f"Instruction received: {instruction}")

        # Confirm the instruction
        self.tts.speak(f"I understood: {instruction}. Is that correct?", emotion="uncertain")

        # In a real implementation, you would wait for confirmation here
        # For simplicity, we'll just proceed with the instruction
        self.tts.speak("I'll execute that instruction now.", emotion="confident")

        return instruction

    def connect_to_arduino(self):
        """Establish connection with Arduino"""
        try:
            self.ser = serial.Serial(self.arduino_port, self.baud_rate, timeout=1)
            print(f"Connected to Arduino on {self.arduino_port}")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            self.tts.speak("I couldn't connect to the robot's control system. Please check the connections.",
                           emotion="sad")
            return False

    def wait_for_arduino_ready(self):
        """Wait for Arduino to signal it's ready"""
        line = ''
        print("READY")
        while line != "Action Done" and line != "Done with setup() function!":
            if self.ser.in_waiting > 0:  # Check if data is available
                line = self.ser.readline().decode('utf-8').strip()  # Read and decode
                print(f"Arduino: {line}")  # Print to console
        return True

    def process_frame(self, frame):
        """Process a video frame and send commands to Arduino"""
        # Resize frame for processing
        downsampled_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)

        # Display the frame
        cv2.imshow('Robot Vision', downsampled_frame)

        print(downsampled_frame)
        print("self.instruction: ", self.instruction)

        # Wait for Arduino to be ready
        if self.wait_for_arduino_ready():
            # Send the frame and instruction to the model
            response = post_act(downsampled_frame, self.instruction, self.unnorm_key)

            if response:
                # Process the response
                end_effector_pose_and_gripper_open = json_numpy.loads(response.json())
                print(f"Received pose data: {end_effector_pose_and_gripper_open}")

                # Process each pose in the response
                for action_idx, eef_pose in enumerate(end_effector_pose_and_gripper_open):
                    target_position = eef_pose[:3]
                    target_orientation = eef_pose[3:-1]
                    gripper_open = eef_pose[-1]

                    # Calculate inverse kinematics
                    frame = np.eye(4)
                    frame[:3, 3] = target_position
                    frame[:3, :3] = R.from_euler("ZYX", target_orientation).as_matrix()
                    angles = ikpy.inverse_kinematics.inverse_kinematic_optimization(
                        chain=self.seal,
                        target_frame=frame,
                        starting_nodes_angles=[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        max_iter=50,
                        orientation_mode="all",
                        no_position=False
                    )

                    print("Joint angles:", angles[2:-1])
                    print("Target position:", target_position, target_orientation, gripper_open)

                    # Send angles to Arduino
                    angles = np.append(angles[2:-1], gripper_open).round(3)
                    send_message(self.ser, angles)

                    if action_idx < len(end_effector_pose_and_gripper_open) - 1:
                        print("Started ready")
                        self.wait_for_arduino_ready()
                        print("Finished ready")
            else:
                print("Failed to get a response from the model")

    def run(self, video_source):
        """Main execution loop"""
        self.running = True

        # Get instruction from voice if not already set
        if not self.instruction:
            self.instruction = self.get_voice_instruction()

        # Connect to Arduino
        if not self.connect_to_arduino():
            return

        # Open video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            self.tts.speak("I couldn't access the camera. Please check the connection.", emotion="sad")
            return

        # Announce start
        self.tts.speak(f"Starting to execute: {self.instruction}", emotion="confident")

        frame_interval = 6  # Process every 6th frame
        frame_num = 0

        # Start keyboard listener in a separate thread
        # keyboard_thread = threading.Thread(target=self.keyboard_listener)
        # keyboard_thread.daemon = True
        # keyboard_thread.start()

        try:
            while self.running:
                # Capture frame
                ret, frame = cap.read()

                # If frame reading failed, break
                if not ret:
                    print("Error: Can't receive frame. Exiting...")
                    break

                # Process at reduced frame rate
                if not frame_num % frame_interval == 0 or frame_num < 8:
                    frame_num += 1
                    continue

                print(f"Processing frame: {frame_num}")
                frame_num += 1

                # Process the frame
                self.process_frame(frame)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except:
            self.ser.close()

        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            if self.ser:
                self.ser.close()

            # Announce completion
            self.tts.speak("Task completed successfully.", emotion="happy")

    def keyboard_listener(self):
        """Listen for keyboard events"""
        print("Press 'space' to speak to the robot, 'e' to end the task, or 'q' to quit")

        while self.running:
            try:
                # Check for space key (speak)
                if keyboard.is_pressed('space'):
                    print("Space pressed - listening for new instruction")
                    self.instruction = self.get_voice_instruction()
                    time.sleep(0.5)  # Debounce

                # Check for 'e' key (end task)
                if keyboard.is_pressed('e'):
                    print("E pressed - ending task")
                    self.running = False
                    self.tts.speak("Thank you for working with me today.", emotion="gentle")
                    time.sleep(0.5)  # Wait for speech to start
                    self.tts.speak("No problem. Let me know if you need anything else!", emotion="happy")
                    time.sleep(0.5)  # Debounce

                # Check for 'q' key (quit)
                if keyboard.is_pressed('q'):
                    print("Q pressed - quitting")
                    self.running = False
                    time.sleep(0.5)  # Debounce

                time.sleep(0.1)  # Reduce CPU usage

            except Exception as e:
                print(f"Error in keyboard listener: {e}")
                time.sleep(0.5)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Legacy VLA voice-control test client")
    parser.add_argument("--port", default="COM3", help="Arduino serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Arduino baud rate")
    parser.add_argument("--dataset", default="random_dataset", help="Action normalization key")
    parser.add_argument("--video", default="0", help="Camera index, video file, or IP camera URL")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key using:")
        print("export OPENAI_API_KEY='your-api-key'")
        return

    robot = RobotController(args.port, args.baud, args.dataset, api_key=api_key)

    video_source = args.video
    try:
        video_source = int(video_source)
    except ValueError:
        pass
    robot.run(video_source)


if __name__ == '__main__':
    ikpy.inverse_kinematics.ORIENTATION_COEFF = 0.001
    main()
