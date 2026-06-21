import os
import time
from speech_to_text import SpeechToText
from text_to_speech import TextToSpeech

class VLAVoiceInterface:
    """
    Voice interface for Vision Language Action (VLA) models.
    
    This class integrates speech-to-text and text-to-speech capabilities to create
    a bidirectional voice communication interface for robot manipulation tasks.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the VLA voice interface.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, it will look for OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key
        self.stt = SpeechToText(api_key=self.api_key)
        self.tts = TextToSpeech(api_key=self.api_key)
        
        # Default acknowledgment responses
        self.acknowledgments = [
            "Acknowledged, I will execute that command.",
            "I understand. Will perform that action now.",
            "Command received. Executing task.",
            "Processing your instruction now.",
            "Got it. Will do that right away."
        ]
        
        # Default voice settings
        self.set_voice("nova")  # More natural-sounding voice
        
    def set_voice(self, voice):
        """
        Set the voice for text-to-speech responses.
        
        Args:
            voice (str): Voice to use. Options include 'alloy', 'echo', 'fable', 'onyx', 'nova', and 'shimmer'.
        """
        self.tts.set_voice(voice)
        
    def set_model(self, model):
        """
        Set the model for text-to-speech synthesis.
        
        Args:
            model (str): Model to use. Options include 'tts-1' and 'tts-1-hd'.
        """
        self.tts.set_model(model)
        
    def set_recording_duration(self, seconds):
        """
        Set the duration for audio recording.
        
        Args:
            seconds (int): Duration in seconds.
        """
        self.stt.record_seconds = seconds
        
    def listen(self, seconds=None):
        """
        Listen for voice commands and convert to text.
        
        Args:
            seconds (int, optional): Duration of recording in seconds.
            
        Returns:
            str: Transcribed command.
        """
        print("Listening for command...")
        
        # Record and transcribe audio
        command = self.stt.transcribe(audio_file=None if seconds is None else self.stt.record_audio(seconds=seconds))
        
        print(f"Command received: {command}")
        return command
        
    def respond(self, message=None, custom_response=None):
        """
        Respond to the user with voice.
        
        Args:
            message (str, optional): Message to respond with. If not provided, a random acknowledgment will be used.
            custom_response (str, optional): Custom response to use instead of default acknowledgments.
            
        Returns:
            str: Path to the generated audio file.
        """
        import random
        
        # Determine the response message
        if custom_response:
            response = custom_response
        elif message:
            response = message
        else:
            response = random.choice(self.acknowledgments)
            
        print(f"Responding: {response}")
        
        # Convert to speech and play
        return self.tts.speak(response)
    
    def process_command(self, command=None, vla_model_function=None):
        """
        Process a voice command through the VLA model.
        
        Args:
            command (str, optional): Command to process. If not provided, it will listen for a command.
            vla_model_function (callable, optional): Function to call with the command.
                This function should take a string command and return a response.
                
        Returns:
            tuple: (command, response)
        """
        # Listen for command if not provided
        if command is None:
            command = self.listen()
            
        # Acknowledge the command
        self.respond()
        
        # Process command with VLA model if provided
        if vla_model_function:
            print("Processing command with VLA model...")
            response = vla_model_function(command)
            
            # Respond with the VLA model's response
            self.respond(custom_response=response)
            return command, response
            
        return command, None
        
    def demo_loop(self, num_interactions=3, recording_seconds=5):
        """
        Run a demonstration loop of the voice interface.
        
        Args:
            num_interactions (int, optional): Number of interactions to demonstrate.
            recording_seconds (int, optional): Duration of recording in seconds.
        """
        print(f"Starting VLA Voice Interface demo with {num_interactions} interactions")
        self.set_recording_duration(recording_seconds)
        
        # Simulate VLA model function (in a real scenario, this would be your actual VLA model)
        def mock_vla_model(command):
            # Simple mock responses based on command keywords
            if "pick" in command.lower() or "grab" in command.lower():
                return "I will pick up the object now."
            elif "move" in command.lower() or "place" in command.lower():
                return "Moving the object to the specified location."
            elif "stop" in command.lower():
                return "Stopping all actions immediately."
            elif "thanks" in command.lower() or "thank" in command.lower():
                return "No problem. Let me know if you need anything else."
            else:
                return "I will process that instruction now."
        
        # Welcome message
        self.tts.speak("Hello, I am your VLA voice interface. Please give me a command.")
        
        # Interaction loop
        for i in range(num_interactions):
            print(f"\nInteraction {i+1}/{num_interactions}")
            time.sleep(1)  # Brief pause between interactions
            
            command, response = self.process_command(vla_model_function=mock_vla_model)
            
            # Brief pause between interactions
            time.sleep(2)
            
        # Farewell message
        self.tts.speak("Demo complete. Thank you for using the VLA voice interface.")


def main():
    """
    Main function to demonstrate the VLA Voice Interface.
    """
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key using:")
        print("export OPENAI_API_KEY='your-api-key'")
        return
    
    # Create and run the VLA Voice Interface
    vla_interface = VLAVoiceInterface(api_key=api_key)
    
    # Optional: Configure the interface
    vla_interface.set_voice("nova")  # More natural-sounding voice
    vla_interface.set_model("tts-1")  # Standard TTS model (use tts-1-hd for higher quality)
    
    # Run the demo
    vla_interface.demo_loop(num_interactions=2, recording_seconds=5)


if __name__ == "__main__":
    main()
