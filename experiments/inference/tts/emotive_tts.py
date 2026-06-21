import os
import tempfile
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play


class EmotiveTextToSpeech:
    def __init__(self, api_key=None):
        """
        Initialize the EmotiveTextToSpeech component with enhanced emotional capabilities.

        Args:
            api_key (str, optional): OpenAI API key. If not provided, it will look for OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.voice = "nova"  # Default voice (more expressive)
        self.model = "tts-1"  # Default model

        # Emotion presets using SSML patterns
        self.emotion_presets = {
            "happy": {
                "prefix": '<speak><prosody rate="medium" pitch="+15%">',
                "suffix": '</prosody></speak>',
                "description": "Upbeat, cheerful tone with higher pitch"
            },
            "excited": {
                "prefix": '<speak><prosody rate="fast" pitch="+20%">',
                "suffix": '</prosody></speak>',
                "description": "Fast-paced, high energy with raised pitch"
            },
            "sad": {
                "prefix": '<speak><prosody rate="slow" pitch="-10%">',
                "suffix": '</prosody></speak>',
                "description": "Slower pace with lower pitch"
            },
            "serious": {
                "prefix": '<speak><prosody rate="medium" pitch="-5%">',
                "suffix": '</prosody></speak>',
                "description": "Measured pace with slightly lower pitch"
            },
            "gentle": {
                "prefix": '<speak><prosody volume="soft" rate="medium">',
                "suffix": '</prosody></speak>',
                "description": "Soft, gentle tone"
            },
            "confident": {
                "prefix": '<speak><prosody volume="loud" rate="medium" pitch="+5%">',
                "suffix": '</prosody></speak>',
                "description": "Strong, assured tone"
            },
            "uncertain": {
                "prefix": '<speak>',
                "suffix": '</speak>',
                "description": "Hesitant tone with pauses",
                "transform": lambda text: text.replace('. ', '... ').replace('? ', '...? ')
            },
            "emphatic": {
                "prefix": '<speak>',
                "suffix": '</speak>',
                "description": "Emphasizes key words",
                "transform": lambda text: self._add_emphasis_to_keywords(text)
            }
        }

    def _add_emphasis_to_keywords(self, text):
        """Add emphasis to likely important words in the text."""
        # Simple heuristic to identify potentially important words
        important_words = ["must", "critical", "important", "urgent", "immediately",
                           "danger", "warning", "caution", "success", "failed",
                           "completed", "error", "crucial"]

        for word in important_words:
            if word in text.lower():
                # Replace with emphasized version, preserving case
                text = text.replace(word, f'<emphasis level="strong">{word}</emphasis>')
                text = text.replace(word.capitalize(), f'<emphasis level="strong">{word.capitalize()}</emphasis>')

        return text

    def set_voice(self, voice):
        """
        Set the voice to use for text-to-speech.

        Args:
            voice (str): Voice to use. Options include 'alloy', 'echo', 'fable', 'onyx', 'nova', and 'shimmer'.
                         For more emotive speech, 'nova', 'shimmer', and 'fable' are recommended.
        """
        valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        if voice in valid_voices:
            self.voice = voice
        else:
            print(f"Invalid voice: {voice}. Using default voice: {self.voice}")

        # Provide recommendation for emotive speech
        if voice in ['alloy', 'onyx', 'echo']:
            print(f"Note: {voice} is less expressive. For more emotive speech, consider 'nova', 'shimmer', or 'fable'.")

    def set_model(self, model):
        """
        Set the model to use for text-to-speech.

        Args:
            model (str): Model to use. Options include 'tts-1' and 'tts-1-hd'.
                         'tts-1-hd' provides higher quality audio with more nuanced emotion.
        """
        valid_models = ['tts-1', 'tts-1-hd']
        if model in valid_models:
            self.model = model
        else:
            print(f"Invalid model: {model}. Using default model: {self.model}")

        # Provide recommendation for emotive speech
        if model == 'tts-1':
            print("Note: For more nuanced emotional expression, consider using 'tts-1-hd'.")

    def list_emotions(self):
        """
        List all available emotion presets with descriptions.

        Returns:
            dict: Dictionary of emotion presets with descriptions.
        """
        return {emotion: preset["description"] for emotion, preset in self.emotion_presets.items()}

    def add_custom_emotion(self, name, prefix, suffix, description, transform_func=None):
        """
        Add a custom emotion preset.

        Args:
            name (str): Name of the emotion preset.
            prefix (str): SSML prefix to apply.
            suffix (str): SSML suffix to apply.
            description (str): Description of the emotion.
            transform_func (callable, optional): Function to transform the text before applying SSML.
        """
        self.emotion_presets[name] = {
            "prefix": prefix,
            "suffix": suffix,
            "description": description
        }

        if transform_func:
            self.emotion_presets[name]["transform"] = transform_func

        print(f"Added custom emotion preset: {name} - {description}")

    def apply_emotion(self, text, emotion):
        """
        Apply an emotion preset to text.

        Args:
            text (str): Text to apply emotion to.
            emotion (str): Emotion preset to apply.

        Returns:
            str: Text with SSML tags for the specified emotion.
        """
        if emotion not in self.emotion_presets:
            print(f"Unknown emotion: {emotion}. Using default (no emotion).")
            return text

        preset = self.emotion_presets[emotion]

        # Clean the text of any system tags or unwanted content
        # This will remove anything between < and > that isn't valid SSML
        import re
        text = re.sub(r'<(?!speak|\/speak|prosody|\/prosody|emphasis|\/emphasis|break)[^>]*>', '', text)

        # Apply text transformation if specified
        if "transform" in preset:
            text = preset["transform"](text)

        # Apply SSML tags
        # return f"{preset['prefix']}{text}{preset['suffix']}"
        return text

    def add_pause(self, text, duration_ms=500):
        """
        Add a pause to the text.

        Args:
            text (str): Text to add pause to.
            duration_ms (int): Duration of pause in milliseconds.

        Returns:
            str: Text with pause SSML tag.
        """
        # Convert milliseconds to seconds for SSML
        duration_sec = duration_ms / 1000

        # Ensure the text is wrapped in speak tags
        if not text.startswith('<speak>'):
            text = f'<speak>{text}</speak>'

        # Insert pause before the closing speak tag
        return text.replace('</speak>', f'<break time="{duration_sec}s"/></speak>')

    def emphasize(self, text, word, level="moderate"):
        """
        Emphasize a specific word in the text.

        Args:
            text (str): Text containing the word to emphasize.
            word (str): Word to emphasize.
            level (str): Emphasis level. Options: "moderate", "strong", "reduced".

        Returns:
            str: Text with emphasis SSML tag.
        """
        valid_levels = ["moderate", "strong", "reduced"]
        if level not in valid_levels:
            print(f"Invalid emphasis level: {level}. Using 'moderate'.")
            level = "moderate"

        # Ensure the text is wrapped in speak tags
        if not text.startswith('<speak>'):
            text = f'<speak>{text}</speak>'

        # Replace the word with emphasized version, preserving case
        emphasized_word = f'<emphasis level="{level}">{word}</emphasis>'

        # Replace all occurrences while preserving case
        result = text
        result = result.replace(f' {word} ', f' {emphasized_word} ')
        result = result.replace(f' {word.capitalize()} ', f' <emphasis level="{level}">{word.capitalize()}</emphasis> ')

        # Handle word at beginning or end of text
        if result.startswith(f'{word} '):
            result = f'<speak><emphasis level="{level}">{word}</emphasis> {result[len(word) + 1:]}'
        if result.endswith(f' {word}</speak>'):
            result = f'{result[:-len(word) - 8]} <emphasis level="{level}">{word}</emphasis></speak>'

        return result

    def synthesize(self, text, output_file=None, emotion=None):
        """
        Convert text to speech using OpenAI's TTS model with emotional expression.

        Args:
            text (str): Text to convert to speech.
            output_file (str, optional): Path to save the audio file. If not provided, a temporary file will be used.
            emotion (str, optional): Emotion preset to apply.

        Returns:
            str: Path to the generated audio file.
        """
        if output_file is None:
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, "synthesized_speech.mp3")

        # Apply emotion if specified
        if emotion:
            text = self.apply_emotion(text, emotion)
            print(f"Applied emotion: {emotion}")
        elif not text.startswith('<speak>'):
            # If no emotion and no SSML, wrap in speak tags for consistency
            text = f'<speak>{text}</speak>'

        print(f"Converting text to speech using voice: {self.voice}, model: {self.model}")
        print(f"SSML: {text}")

        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="mp3"
        )

        response.stream_to_file(output_file)

        return output_file

    def speak(self, text, output_file=None, emotion=None):
        """
        Convert text to speech with emotion and play it immediately.

        Args:
            text (str): Text to convert to speech and play.
            output_file (str, optional): Path to save the audio file. If not provided, a temporary file will be used.
            emotion (str, optional): Emotion preset to apply.

        Returns:
            str: Path to the generated audio file.
        """
        audio_file = self.synthesize(text, output_file, emotion)

        print(f"Playing audio: {audio_file}")

        # Load and play the audio
        audio = AudioSegment.from_file(audio_file)
        play(audio)

        return audio_file
