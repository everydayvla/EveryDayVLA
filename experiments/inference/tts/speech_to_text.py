import os
import tempfile
import pyaudio
import wave
from openai import OpenAI

class SpeechToText:
    def __init__(self, api_key=None):
        """
        Initialize the SpeechToText component.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, it will look for OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.record_seconds = 5  # Default recording time
        
    def record_audio(self, output_file=None, seconds=None):
        """
        Record audio from the microphone.
        
        Args:
            output_file (str, optional): Path to save the recorded audio. If not provided, a temporary file will be used.
            seconds (int, optional): Duration of recording in seconds. Defaults to self.record_seconds.
            
        Returns:
            str: Path to the recorded audio file.
        """
        if seconds is not None:
            self.record_seconds = seconds
            
        if output_file is None:
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, "recorded_audio.wav")
            
        audio = pyaudio.PyAudio()
        
        # Open stream
        stream = audio.open(format=self.format,
                           channels=self.channels,
                           rate=self.rate,
                           input=True,
                           frames_per_buffer=self.chunk)
        
        print("Recording...")
        
        frames = []
        
        for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)
            
        print("Recording finished.")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save the recorded audio to a WAV file
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            
        return output_file
    
    def transcribe(self, audio_file=None):
        """
        Transcribe audio to text using OpenAI's Whisper model.
        
        Args:
            audio_file (str, optional): Path to the audio file to transcribe. 
                                       If not provided, it will record audio first.
                                       
        Returns:
            str: Transcribed text.
        """
        if audio_file is None:
            audio_file = self.record_audio()
            
        print(f"Transcribing audio file: {audio_file}")
        
        with open(audio_file, "rb") as audio:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
            
        return transcript.text
