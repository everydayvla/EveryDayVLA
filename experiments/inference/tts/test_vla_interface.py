import os
import argparse
from vla_voice_interface import VLAVoiceInterface

def main():
    """
    Test script for the VLA Voice Interface.
    
    This script allows testing different components of the VLA Voice Interface
    with command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Test VLA Voice Interface')
    parser.add_argument('--test-stt', action='store_true', help='Test speech-to-text only')
    parser.add_argument('--test-tts', action='store_true', help='Test text-to-speech only')
    parser.add_argument('--message', type=str, help='Message for text-to-speech test')
    parser.add_argument('--voice', type=str, default='nova', help='Voice to use for TTS (alloy, echo, fable, onyx, nova, shimmer)')
    parser.add_argument('--model', type=str, default='tts-1', help='Model to use for TTS (tts-1, tts-1-hd)')
    parser.add_argument('--duration', type=int, default=5, help='Recording duration in seconds')
    parser.add_argument('--demo', action='store_true', help='Run the full demo')
    parser.add_argument('--interactions', type=int, default=2, help='Number of interactions for demo')
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key using:")
        print("export OPENAI_API_KEY='your-api-key'")
        return
    
    # Create the VLA Voice Interface
    vla_interface = VLAVoiceInterface(api_key=api_key)
    vla_interface.set_voice(args.voice)
    vla_interface.set_model(args.model)
    vla_interface.set_recording_duration(args.duration)
    
    # Test speech-to-text
    if args.test_stt:
        print(f"Testing Speech-to-Text (recording for {args.duration} seconds)...")
        command = vla_interface.listen(seconds=args.duration)
        print(f"Transcribed text: {command}")
    
    # Test text-to-speech
    if args.test_tts:
        message = args.message or "This is a test of the text to speech system using OpenAI's API."
        print(f"Testing Text-to-Speech with message: '{message}'")
        vla_interface.respond(custom_response=message)
    
    # Run the demo
    if args.demo:
        vla_interface.demo_loop(num_interactions=args.interactions, recording_seconds=args.duration)
    
    # If no specific test is selected, show help
    if not (args.test_stt or args.test_tts or args.demo):
        parser.print_help()

if __name__ == "__main__":
    main()
