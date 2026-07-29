from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from character_chat_service import CharacterChatService
from conversation_engine import CharacterProfile, ConversationTurn
from llm_config import LLMConfig
from openai_conversation_engine import LLMReply
from voice_engine import VoiceGenerationResult


class CharacterChatServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeConversationEngine.instances = []

    def test_generate_reply_passes_profile_config_and_initial_history(self) -> None:
        profile = self._profile()
        config = self._config()
        history = [
            ConversationTurn(
                user_text="こんにちは",
                character_text="こんにちは。",
            )
        ]
        service = CharacterChatService(
            llm_config=config,
            voice_engine=FakeVoiceEngine(),
        )

        with patch(
            "character_chat_service.OpenAIConversationEngine",
            FakeConversationEngine,
        ):
            service.generate_reply(
                "次の入力",
                profile,
                initial_history=history,
            )

        engine = FakeConversationEngine.instances[0]
        self.assertIs(engine.profile, profile)
        self.assertIs(engine.config, config)
        self.assertEqual(engine.initial_history, tuple(history))
        self.assertEqual(engine.user_text, "次の入力")

    def test_generate_reply_returns_llm_reply_and_updated_history(self) -> None:
        service = CharacterChatService(
            llm_config=self._config(),
            voice_engine=FakeVoiceEngine(),
        )

        with patch(
            "character_chat_service.OpenAIConversationEngine",
            FakeConversationEngine,
        ):
            result = service.generate_reply(
                "次の入力",
                self._profile(),
                initial_history=[],
            )

        self.assertEqual(result.reply, FakeConversationEngine.reply)
        self.assertEqual(result.history, FakeConversationEngine.updated_history)

    def test_generate_voice_passes_text_and_reference_audio_to_voice_engine(self) -> None:
        voice_engine = FakeVoiceEngine()
        service = CharacterChatService(
            llm_config=self._config(),
            voice_engine=voice_engine,
        )

        result = service.generate_voice(
            "返答本文",
            reference_audio="reference.wav",
        )

        self.assertEqual(
            voice_engine.calls,
            [
                {
                    "text": "返答本文",
                    "reference_audio": "reference.wav",
                }
            ],
        )
        self.assertEqual(result, voice_engine.result)

    def test_service_does_not_depend_on_gradio(self) -> None:
        service_source = Path("character_chat_service.py").read_text(encoding="utf-8")

        self.assertNotIn("import gradio", service_source)
        self.assertNotIn("gr.update", service_source)

    def _profile(self) -> CharacterProfile:
        return CharacterProfile(
            name="テストキャラクター",
            first_person="私",
            personality="明るい",
            speaking_style="自然に話す",
        )

    def _config(self) -> LLMConfig:
        return LLMConfig(
            api_key="test-api-key",
            model="test-model",
            transcription_model="test-transcribe",
            max_history_turns=8,
            max_output_tokens=200,
        )


class FakeConversationEngine:
    reply = LLMReply(
        text="サービスからの返答",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        request_id="request-id",
    )
    updated_history = (
        ConversationTurn(
            user_text="次の入力",
            character_text="サービスからの返答",
        ),
    )
    instances = []

    def __init__(
        self,
        *,
        profile: CharacterProfile,
        config: LLMConfig,
        initial_history,
    ) -> None:
        self.profile = profile
        self.config = config
        self.initial_history = tuple(initial_history)
        self.user_text = None
        self.history = self.updated_history
        self.instances.append(self)

    def generate_reply(self, user_text: str) -> LLMReply:
        self.user_text = user_text
        return self.reply


class FakeVoiceEngine:
    def __init__(self) -> None:
        self.result = VoiceGenerationResult(
            output_path=Path("voice.wav"),
            used_seed=1234,
            generation_seconds=0.125,
        )
        self.calls = []

    def generate(
        self,
        text: str,
        reference_audio=None,
    ) -> VoiceGenerationResult:
        self.calls.append(
            {
                "text": text,
                "reference_audio": reference_audio,
            }
        )
        return self.result


if __name__ == "__main__":
    unittest.main()
