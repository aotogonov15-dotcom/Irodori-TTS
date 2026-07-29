from __future__ import annotations

import unittest

from conversation_engine import CharacterProfile, ConversationTurn
from llm_config import LLMConfig
from openai_conversation_engine import OpenAIConversationEngine


class OpenAIConversationEngineTest(unittest.TestCase):
    def test_initial_history_defaults_to_empty(self) -> None:
        engine = OpenAIConversationEngine(
            profile=self._profile(),
            config=self._config(),
        )

        self.assertEqual(engine.history, ())

    def test_initial_history_is_stored(self) -> None:
        history = [
            ConversationTurn(
                user_text="こんにちは",
                character_text="こんにちは。",
            ),
            ConversationTurn(
                user_text="元気？",
                character_text="元気だよ。",
            ),
        ]

        engine = OpenAIConversationEngine(
            profile=self._profile(),
            config=self._config(),
            initial_history=history,
        )

        self.assertEqual(engine.history, tuple(history))

    def test_initial_history_is_copied_from_caller_list(self) -> None:
        history = [
            ConversationTurn(
                user_text="最初の発話",
                character_text="最初の返答",
            ),
        ]
        engine = OpenAIConversationEngine(
            profile=self._profile(),
            config=self._config(),
            initial_history=history,
        )

        history.append(
            ConversationTurn(
                user_text="後から追加",
                character_text="後から追加された返答",
            )
        )

        self.assertEqual(
            engine.history,
            (
                ConversationTurn(
                    user_text="最初の発話",
                    character_text="最初の返答",
                ),
            ),
        )

    def test_initial_history_is_trimmed_to_max_history_turns(self) -> None:
        history = [
            ConversationTurn(user_text="user 1", character_text="reply 1"),
            ConversationTurn(user_text="user 2", character_text="reply 2"),
            ConversationTurn(user_text="user 3", character_text="reply 3"),
        ]

        engine = OpenAIConversationEngine(
            profile=self._profile(),
            config=self._config(max_history_turns=2),
            initial_history=history,
        )

        self.assertEqual(engine.history, tuple(history[-2:]))

    def _profile(self) -> CharacterProfile:
        return CharacterProfile(
            name="テストキャラクター",
            first_person="私",
            personality="明るい",
            speaking_style="自然に話す",
        )

    def _config(self, *, max_history_turns: int = 8) -> LLMConfig:
        return LLMConfig(
            api_key="test-api-key",
            model="test-model",
            transcription_model="test-transcribe",
            max_history_turns=max_history_turns,
            max_output_tokens=200,
        )


if __name__ == "__main__":
    unittest.main()
