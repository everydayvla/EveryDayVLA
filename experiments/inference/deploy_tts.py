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
        message_str = ",".join(map(str, message.tolist()))  # Correct (Space-Separated)
    else:
        message_str = str(message)

    # Append newline for Arduino parsing
    message_bytes = (message_str + "\n").encode('utf-8')

    ser.write(message_bytes)  # Send message
    print(f"Sent to Arduino: {message_str}")  # Debugging


# For testing purposes
import random


def create_spoofed_pose():
    result = []
    for i in range(8):
        result.append(
            [random.random() * 0.1 + 0.1, random.random() * 0.1 + 0.1, random.random() * 0.1 + 0.01, 0, pi, 0, 1])
    return result


class VLAVoiceController:
    """
    Voice-enabled controller for VLA robot manipulation.
    Integrates speech-to-text and text-to-speech with robot control.
    """

    def __init__(self, arduino_port, baud_rate, unnorm_key='random_dataset', api_key=None):
        # Initialize speech components
        self.stt = SpeechToText(api_key=api_key)
        self.tts = EmotiveTextToSpeech(api_key=api_key)
        self.tts.set_voice("nova")  # Use more expressive voice
        self.tts.set_model("tts-1-hd")  # Use high-definition model for better quality

        # Robot control parameters
        self.arduino_port = arduino_port
        self.baud_rate = baud_rate
        self.unnorm_key = unnorm_key
        self.instruction = None
        self.ser = None
        self.running = False
        self.action_num = 1
        self.response = []
        self.ack = False

        # Initialize IK settings
        ikpy.inverse_kinematics.ORIENTATION_COEFF = 0.001

        # Initialize robot chain
        self.seal = None

        # Keyboard listener active flag
        self.keyboard_active = False

    def get_voice_instruction(self):
        """Get instruction from user's voice using speech-to-text"""
        self.tts.speak("Please provide your instruction for the robot.", emotion="gentle")
        print("Listening for instruction...")

        # Record and transcribe audio (7 seconds should be enough for most instructions)
        instruction = self.stt.transcribe(
            audio_file=None if self.stt.record_seconds else self.stt.record_audio(seconds=10))

        print(f"Instruction received: {instruction}")

        # Strip trailing full stop if it exists
        if instruction.endswith('.'):
            instruction = instruction[:-1]

        # Confirm the instruction
        self.tts.speak(f"I understood: {instruction}. I'll execute that instruction now.", emotion="confident")

        return instruction

    def connect_to_arduino(self):
        """Establish connection with Arduino"""
        try:
            self.ser = serial.Serial(self.arduino_port, self.baud_rate, timeout=1)
            print(f"Connected to Arduino on {self.arduino_port}")
            self.tts.speak("Connected to robot control system.", emotion="happy")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            self.tts.speak("I couldn't connect to the robot's control system. Please check the connections.",
                           emotion="sad")
            return False

    def start_keyboard_listener(self):
        """Start keyboard listener in a separate thread"""
        if not self.keyboard_active:
            import threading
            self.keyboard_active = True
            keyboard_thread = threading.Thread(target=self.keyboard_listener)
            keyboard_thread.daemon = True
            keyboard_thread.start()
            print("Keyboard listener started. Press 'space' to speak, 'e' to end task, 'q' to quit.")

    def keyboard_listener(self):
        """Listen for keyboard events"""
        while self.running and self.keyboard_active:
            try:
                # Check for space key (speak)
                if keyboard.is_pressed('space'):
                    print("Space pressed - listening for new instruction")
                    self.instruction = self.get_voice_instruction()
                    # Reset action counter to process new instruction
                    self.action_num = len(self.response)  # Force new response generation
                    time.sleep(0.5)  # Debounce

                # Check for 'e' key (end task)
                if keyboard.is_pressed('e'):
                    print("E pressed - ending task")
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

    def run(self, video_source, instruction=None):
        """Main execution loop"""
        self.running = True

        # Get instruction from voice if not already set
        if not instruction and not self.instruction:
            self.instruction = self.get_voice_instruction()
        elif instruction:
            self.instruction = instruction
            self.tts.speak(f"I'll execute the instruction: {self.instruction}", emotion="confident")

        # Connect to Arduino
        if not self.connect_to_arduino():
            return

        # Initialize robot chain
        self.seal = ikpy.chain.Chain.from_urdf_file("seal.urdf",
                                                    active_links_mask=[False, False, True, True, True, True, True, True,
                                                                       False])

        # Open video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            self.tts.speak("I couldn't access the camera. Please check the connection.", emotion="sad")
            return

        # Start keyboard listener
        self.start_keyboard_listener()

        # Processing parameters
        frame_interval = 6  # Process every 6th frame
        frame_num = 0

        try:
            while self.running:
                # Capture frame
                ret, frame = cap.read()

                # If frame reading failed, break
                if not ret:
                    print("Error: Can't receive frame. Exiting...")
                    self.tts.speak("Video feed lost. Stopping operation.", emotion="serious")
                    break

                # Process at reduced frame rate
                if not frame_num % frame_interval == 0 or frame_num < 8:
                    frame_num += 1
                    continue

                frame_num += 1

                # Resize frame for processing
                downsampled_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)

                # Display the frame
                cv2.imshow('Robot Vision', downsampled_frame)

                print(f"self.instruction: {self.instruction}")

                # Check for Arduino messages
                line = ''
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()  # Read and decode
                    print(f"Arduino: {line}")  # Print Arduino message
                    if line == "Action Done" or line == "Done with setup() function!":
                        self.ack = True

                # Generate new response if needed
                if self.action_num >= len(self.response):
                    print("Generating new actions based on instruction...")
                    response_obj = post_act(downsampled_frame, self.instruction, self.unnorm_key)
                    if response_obj:
                        self.response = json_numpy.loads(response_obj.json())
                        print(" ------------- Action being generated  ------------")
                        print(self.response)
                        # self.tts.speak("I've planned the next sequence of actions.", emotion="confident")
                    else:
                        print("Failed to get response from model, using spoofed pose")
                        self.response = create_spoofed_pose()
                        self.tts.speak("I couldn't connect to the planning system. Using backup plan.",
                                       emotion="uncertain")
                    self.action_num = 0

                # Execute action if acknowledged and available
                if self.ack and self.action_num < len(self.response):
                    eef_pose = self.response[self.action_num]
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

                    # Announce gripper action if it's changing
                    # if self.action_num > 0 and gripper_open != self.response[self.action_num - 1][-1]:
                    #     if gripper_open > 0.5:
                    #         self.tts.speak("Opening gripper.", emotion="gentle")
                    #     else:
                    #         self.tts.speak("Closing gripper to grasp object.", emotion="gentle")

                    # Send angles to Arduino
                    angles = np.append(angles[2:-1], gripper_open).round(3)
                    send_message(self.ser, angles)

                    self.action_num += 1
                    self.ack = False

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

        except Exception as e:
            print(f"Error during execution: {e}")
            self.tts.speak("An error occurred during operation. Stopping for safety.", emotion="serious")
        finally:
            # Clean up
            if self.ser:
                self.ser.close()
            cap.release()
            cv2.destroyAllWindows()
            self.keyboard_active = False

            # Announce completion
            self.tts.speak("Task execution has ended. Let me know if you need anything else!", emotion="gentle")


def main():
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key using:")
        print("export OPENAI_API_KEY='your-api-key'")
        return

    # Default parameters - can be overridden with command line arguments
    instruction = None  # Will be obtained through voice if not provided
    # instruction = "pick up the ball and move it to a random location towards the robot"
    # unnorm_key = 'random_dataset'
    unnorm_key = 'consolidated_dataset'
    arduino_port = 'COM3'  # Change as needed for your system
    baud_rate = 115200

    # Parse command line arguments if provided
    import argparse
    parser = argparse.ArgumentParser(description='VLA Voice Control')
    parser.add_argument('--instruction', type=str, help='Text instruction (if not provided, will use voice input)')
    parser.add_argument('--port', type=str, default=arduino_port, help='Arduino port')
    parser.add_argument('--baud', type=int, default=baud_rate, help='Baud rate')
    parser.add_argument('--video', type=str, default="0",
                        help='Video source (URL, file path, or camera index)')
    parser.add_argument('--dataset', type=str, default=unnorm_key, help='Dataset key for normalization')

    args = parser.parse_args()

    # Update parameters from arguments
    if args.instruction:
        instruction = args.instruction
    arduino_port = args.port
    baud_rate = args.baud
    video_source = args.video
    unnorm_key = args.dataset

    # Try to convert video source to integer if it's a camera index
    try:
        video_source = int(video_source)
    except ValueError:
        # Keep as string (URL or file path)
        pass

    # Create and run VLA Voice Controller
    controller = VLAVoiceController(arduino_port, baud_rate, unnorm_key, api_key=api_key)
    controller.run(video_source, instruction)


if __name__ == '__main__':
    ikpy.inverse_kinematics.ORIENTATION_COEFF = 0.001
    main()
