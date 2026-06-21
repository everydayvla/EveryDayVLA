import os
import tempfile
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

class TextToSpeech:
    def __init__(self, api_key=None):
        """
        Initialize the TextToSpeech component.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, it will look for OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.voice = "alloy"  # Default voice
        self.model = "tts-1"  # Default model
        
    def set_voice(self, voice):
        """
        Set the voice to use for text-to-speech.
        
        Args:
            voice (str): Voice to use. Options include 'alloy', 'echo', 'fable', 'onyx', 'nova', and 'shimmer'.
        """
        valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        if voice in valid_voices:
            self.voice = voice
        else:
            print(f"Invalid voice: {voice}. Using default voice: {self.voice}")
    
    def set_model(self, model):
        """
        Set the model to use for text-to-speech.
        
        Args:
            model (str): Model to use. Options include 'tts-1' and 'tts-1-hd'.
        """
        valid_models = ['tts-1', 'tts-1-hd']
        if model in valid_models:
            self.model = model
        else:
            print(f"Invalid model: {model}. Using default model: {self.model}")
    
    def synthesize(self, text, output_file=None):
        """
        Convert text to speech using OpenAI's TTS model.
        
        Args:
            text (str): Text to convert to speech.
            output_file (str, optional): Path to save the audio file. If not provided, a temporary file will be used.
            
        Returns:
            str: Path to the generated audio file.
        """
        if output_file is None:
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, "synthesized_speech.mp3")
        
        print(f"Converting text to speech using voice: {self.voice}, model: {self.model}")
        
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text
        )
        
        response.stream_to_file(output_file)
        
        return output_file
    
    def speak(self, text, output_file=None):
        """
        Convert text to speech and play it immediately.
        
        Args:
            text (str): Text to convert to speech and play.
            output_file (str, optional): Path to save the audio file. If not provided, a temporary file will be used.
            
        Returns:
            str: Path to the generated audio file.
        """
        audio_file = self.synthesize(text, output_file)
        
        print(f"Playing audio: {audio_file}")
        
        # Load and play the audio
        audio = AudioSegment.from_file(audio_file)
        play(audio)
        
        return audio_file
