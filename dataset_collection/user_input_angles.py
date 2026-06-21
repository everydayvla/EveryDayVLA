import numpy as np
import serial
import time

# Replace 'COM3' with your Arduino's port (use '/dev/ttyUSB0' on Linux/macOS)
arduino_port = 'COM3'
baud_rate = 9600  # Must match the Arduino's baud rate


def send_message(ser, message):
    if isinstance(message, np.ndarray):
        # Convert NumPy array to a space-separated string
        message_str = ",".join(map(str, message.tolist())) 
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

    while True:
        cond = False
        ack = False
        # Check if there is a message from Arduino
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()  # Read and decode
            print(f"Arduino: {line}")  # Print Arduino message

            if line == "Action Done" or line == "Done with setup() function!":
                ack = True
        

        # Ask user for input (only if setup is done or timed out)
        if ack:
            print(ack, line)
            user_input = input("Enter 7 angles (space-separated) or press Enter to use default: ").strip()

            try:
                # Convert user input into a NumPy array
                user_angles = np.array([np.float64(x) for x in user_input.replace(',', ' ').split()])
                if len(user_angles) != 7:
                    raise ValueError  # Force error handling if length is incorrect
            except ValueError:
                print("Error: Invalid input! Using default values.")
                user_angles = np.array([0.1, 0.1, 0.17, 0.4, 0.5, 0.6, 0.9])

            print("user_angles:", user_angles)

            # Send angles to Arduino
            send_message(ser, user_angles)
            ack = False

except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("\nExiting program.")
finally:
    ser.close()
