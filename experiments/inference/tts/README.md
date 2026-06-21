# VLA Voice Interface Documentation

## Overview

The VLA Voice Interface is a Python package that integrates OpenAI's speech-to-text and text-to-speech capabilities to create a bidirectional voice communication interface for Vision Language Action (VLA) models used in robot manipulation tasks.

This interface allows:
- Converting spoken commands to text using OpenAI's Whisper model
- Processing these commands through a VLA model
- Responding with synthesized speech using OpenAI's TTS models

## Requirements

- Python 3.6+
- OpenAI API key
- PyAudio
- pydub
- ffmpeg (for audio playback)

## Installation

1. Install the required system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio ffmpeg
```

2. Install the required Python packages:
```bash
pip install openai pyaudio pydub
```

3. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Project Structure

The project consists of three main Python modules:

1. `speech_to_text.py`: Handles audio recording and transcription using OpenAI's Whisper model
2. `text_to_speech.py`: Handles text-to-speech conversion using OpenAI's TTS models
3. `vla_voice_interface.py`: Integrates both components into a complete interface

Additionally, there's a test script:
- `test_vla_interface.py`: Provides command-line options to test different components

## Usage

### Basic Usage

```python
from vla_voice_interface import VLAVoiceInterface

# Initialize the interface
vla_interface = VLAVoiceInterface(api_key='your-api-key')

# Listen for a command
command = vla_interface.listen()

# Respond with a voice message
vla_interface.respond(message="I'll execute that command now.")

# Process a command through your VLA model
def my_vla_model(command):
    # Process the command and return a response
    return "Task completed successfully."

command, response = vla_interface.process_command(vla_model_function=my_vla_model)
```

### Running the Demo

The package includes a demo mode that simulates interactions with a VLA model:

```python
# Run the demo with default settings
vla_interface.demo_loop()

# Run the demo with custom settings
vla_interface.demo_loop(num_interactions=3, recording_seconds=7)
```

### Testing Individual Components

You can use the test script to test individual components:

```bash
# Test speech-to-text
python test_vla_interface.py --test-stt --duration 5

# Test text-to-speech
python test_vla_interface.py --test-tts --message "Hello, I am a robot assistant."

# Test with different voice
python test_vla_interface.py --test-tts --voice nova --message "This is the nova voice."

# Run the full demo
python test_vla_interface.py --demo --interactions 3
```

## Customization

### Voice Options

The interface supports all of OpenAI's available voices:
- `alloy`: Neutral, versatile voice
- `echo`: Soft, warm voice
- `fable`: Expressive, youthful voice
- `onyx`: Deep, authoritative voice
- `nova`: Warm, natural voice
- `shimmer`: Clear, bright voice

```python
vla_interface.set_voice("nova")
```

### TTS Model Options

Two TTS models are available:
- `tts-1`: Standard quality model
- `tts-1-hd`: Higher quality model

```python
vla_interface.set_model("tts-1-hd")
```

### Recording Duration

You can adjust the recording duration for speech input:

```python
vla_interface.set_recording_duration(7)  # 7 seconds
```

## Integration with VLA Models

To integrate with your VLA model, create a function that processes text commands and returns responses:

```python
def vla_model_function(command):
    # Here you would integrate with your actual VLA model
    # For example:
    # result = my_vla_model.process(command)
    # return result.response
    
    # Simple example:
    if "pick up" in command.lower():
        return "Picking up the object."
    elif "move" in command.lower():
        return "Moving the object to the specified location."
    else:
        return "Command received."

# Use this function with the interface
vla_interface.process_command(vla_model_function=vla_model_function)
```

## Error Handling

The interface includes basic error handling for API issues and audio recording problems. For production use, you may want to enhance error handling based on your specific requirements.

## Limitations

- Requires an active internet connection for API calls
- Audio quality depends on your microphone setup
- API costs apply based on OpenAI's pricing

## License

This project is provided as-is for educational and development purposes.
