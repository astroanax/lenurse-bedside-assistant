"""
Voice Processing Module for JARVIS
Handles: Speech-to-Text → Translation → LLM → Translation → Text-to-Speech

Uses Google Cloud Speech-to-Text, Translate, and Text-to-Speech APIs
"""

import os
import io
import base64
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Check if Google Cloud credentials are available
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
USE_GOOGLE_CLOUD = GOOGLE_CREDENTIALS_PATH and os.path.exists(GOOGLE_CREDENTIALS_PATH)

if USE_GOOGLE_CLOUD:
    try:
        from google.cloud import speech_v1 as speech
        from google.cloud import texttospeech_v1 as texttospeech
        from google.cloud import translate_v2 as translate

        GOOGLE_AVAILABLE = True
    except ImportError:
        GOOGLE_AVAILABLE = False
        print(
            "Google Cloud libraries not installed. Install with: pip install google-cloud-speech google-cloud-texttospeech google-cloud-translate"
        )
else:
    GOOGLE_AVAILABLE = False
    print(
        "Google Cloud credentials not configured. Set GOOGLE_APPLICATION_CREDENTIALS in .env"
    )


class VoiceProcessor:
    """
    Handles voice processing pipeline:
    1. Speech-to-Text (transcribe user audio)
    2. Language detection and translation to English
    3. Process with LLM (handled externally)
    4. Translate response back to user's language
    5. Text-to-Speech (generate audio response)
    """

    # Supported languages
    SUPPORTED_LANGUAGES = {
        "en": {"name": "English", "code": "en-US", "tts_code": "en-US"},
        "ml": {"name": "Malayalam", "code": "ml-IN", "tts_code": "ml-IN"},
    }

    def __init__(self):
        self.speech_client = None
        self.tts_client = None
        self.translate_client = None

        if GOOGLE_AVAILABLE:
            try:
                self.speech_client = speech.SpeechClient()
                self.tts_client = texttospeech.TextToSpeechClient()
                self.translate_client = translate.Client()
                print("[VoiceProcessor] Google Cloud clients initialized")
            except Exception as e:
                print(
                    f"[VoiceProcessor] Failed to initialize Google Cloud clients: {e}"
                )

    def is_available(self) -> bool:
        """Check if voice processing is available"""
        return self.speech_client is not None

    def transcribe_audio(
        self, audio_data: bytes, language: str = "ml"
    ) -> Tuple[str, str]:
        """
        Transcribe audio to text using Google Speech-to-Text

        Args:
            audio_data: Raw audio bytes (WAV or WEBM format)
            language: Expected language code ("en" or "ml")

        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        if not self.speech_client:
            return "", "en"

        try:
            # Get language config
            lang_config = self.SUPPORTED_LANGUAGES.get(
                language, self.SUPPORTED_LANGUAGES["en"]
            )

            # Configure recognition
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                sample_rate_hertz=48000,
                language_code=lang_config["code"],
                alternative_language_codes=["en-US", "ml-IN"],  # Also detect these
                enable_automatic_punctuation=True,
            )

            audio = speech.RecognitionAudio(content=audio_data)

            # Perform transcription
            response = self.speech_client.recognize(config=config, audio=audio)

            if response.results:
                result = response.results[0]
                transcript = result.alternatives[0].transcript
                # Get detected language if available
                detected_lang = getattr(result, "language_code", lang_config["code"])
                # Normalize language code
                if detected_lang.startswith("ml"):
                    detected_lang = "ml"
                else:
                    detected_lang = "en"

                return transcript, detected_lang

            return "", language

        except Exception as e:
            print(f"[VoiceProcessor] Transcription error: {e}")
            return "", language

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text between languages

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        if not self.translate_client or source_lang == target_lang or not text:
            return text

        try:
            result = self.translate_client.translate(
                text, source_language=source_lang, target_language=target_lang
            )
            return result["translatedText"]
        except Exception as e:
            print(f"[VoiceProcessor] Translation error: {e}")
            return text

    def text_to_speech(self, text: str, language: str = "ml") -> Optional[bytes]:
        """
        Convert text to speech audio

        Args:
            text: Text to speak
            language: Language code for speech

        Returns:
            MP3 audio bytes or None on error
        """
        if not self.tts_client or not text:
            return None

        try:
            lang_config = self.SUPPORTED_LANGUAGES.get(
                language, self.SUPPORTED_LANGUAGES["en"]
            )

            # Set up synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Configure voice
            voice = texttospeech.VoiceSelectionParams(
                language_code=lang_config["tts_code"],
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
            )

            # Configure audio output
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
            )

            # Generate speech
            response = self.tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            return response.audio_content

        except Exception as e:
            print(f"[VoiceProcessor] TTS error: {e}")
            return None

    def process_voice_input(self, audio_data: bytes, user_language: str = "ml") -> dict:
        """
        Process voice input: transcribe and translate to English

        Args:
            audio_data: Raw audio bytes
            user_language: User's preferred language

        Returns:
            Dict with transcription results
        """
        # Transcribe audio
        transcript, detected_lang = self.transcribe_audio(audio_data, user_language)

        if not transcript:
            return {
                "success": False,
                "error": "Could not transcribe audio",
                "original_text": "",
                "english_text": "",
                "detected_language": user_language,
            }

        # Translate to English if not already English
        english_text = transcript
        if detected_lang != "en":
            english_text = self.translate_text(transcript, detected_lang, "en")

        return {
            "success": True,
            "original_text": transcript,
            "english_text": english_text,
            "detected_language": detected_lang,
        }

    def process_response(
        self, english_response: str, target_language: str = "ml"
    ) -> dict:
        """
        Process LLM response: translate and generate speech

        Args:
            english_response: Response text in English
            target_language: User's preferred language

        Returns:
            Dict with translated text and audio
        """
        # Translate response if needed
        translated_text = english_response
        if target_language != "en":
            translated_text = self.translate_text(
                english_response, "en", target_language
            )

        # Generate speech audio
        audio_data = self.text_to_speech(translated_text, target_language)
        audio_base64 = None
        if audio_data:
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return {
            "success": True,
            "english_text": english_response,
            "translated_text": translated_text,
            "target_language": target_language,
            "audio_base64": audio_base64,
        }


# Singleton instance
_voice_processor = None


def get_voice_processor() -> VoiceProcessor:
    """Get or create the voice processor singleton"""
    global _voice_processor
    if _voice_processor is None:
        _voice_processor = VoiceProcessor()
    return _voice_processor


# =============================================================================
# FALLBACK: Browser-based STT with server-side translation
# =============================================================================


class FallbackVoiceProcessor:
    """
    Fallback voice processor that uses:
    - Browser's Web Speech API for STT (handled on frontend)
    - Google Translate API for translation
    - Google TTS or browser TTS for speech output
    """

    def __init__(self):
        self.translate_client = None
        self.tts_client = None

        if GOOGLE_AVAILABLE:
            try:
                self.translate_client = translate.Client()
                self.tts_client = texttospeech.TextToSpeechClient()
            except Exception as e:
                print(f"[FallbackVoiceProcessor] Init error: {e}")

    def translate_to_english(self, text: str, source_lang: str = "ml") -> str:
        """Translate text to English"""
        if not self.translate_client or source_lang == "en" or not text:
            return text

        try:
            result = self.translate_client.translate(
                text, source_language=source_lang, target_language="en"
            )
            return result["translatedText"]
        except Exception as e:
            print(f"[FallbackVoiceProcessor] Translation error: {e}")
            return text

    def translate_from_english(self, text: str, target_lang: str = "ml") -> str:
        """Translate text from English to target language"""
        if not self.translate_client or target_lang == "en" or not text:
            return text

        try:
            result = self.translate_client.translate(
                text, source_language="en", target_language=target_lang
            )
            return result["translatedText"]
        except Exception as e:
            print(f"[FallbackVoiceProcessor] Translation error: {e}")
            return text

    def generate_speech(self, text: str, language: str = "ml") -> Optional[str]:
        """Generate speech audio and return as base64"""
        if not self.tts_client or not text:
            return None

        try:
            lang_codes = {"en": "en-US", "ml": "ml-IN"}
            lang_code = lang_codes.get(language, "en-US")

            synthesis_input = texttospeech.SynthesisInput(text=text)

            voice = texttospeech.VoiceSelectionParams(
                language_code=lang_code,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            response = self.tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            return base64.b64encode(response.audio_content).decode("utf-8")

        except Exception as e:
            print(f"[FallbackVoiceProcessor] TTS error: {e}")
            return None


_fallback_processor = None


def get_fallback_processor() -> FallbackVoiceProcessor:
    """Get or create fallback processor"""
    global _fallback_processor
    if _fallback_processor is None:
        _fallback_processor = FallbackVoiceProcessor()
    return _fallback_processor
