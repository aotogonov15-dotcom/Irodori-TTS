from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from conversation_engine import CharacterProfile, ConversationTurn
from llm_config import LLMConfig
from openai_conversation_engine import LLMReply, OpenAIConversationEngine
from voice_engine import VoiceEngine, VoiceGenerationResult, VoiceGenerationSettings


@dataclass(frozen=True)
class CharacterReplyResult:
    reply: LLMReply
    history: tuple[ConversationTurn, ...]


class CharacterChatService:
    def __init__(
        self,
        llm_config: LLMConfig,
        voice_engine: VoiceEngine,
    ) -> None:
        self.llm_config = llm_config
        self.voice_engine = voice_engine

    def generate_reply(
        self,
        user_text: str,
        profile: CharacterProfile,
        initial_history: Iterable[ConversationTurn] | None = None,
    ) -> CharacterReplyResult:
        conversation_engine = OpenAIConversationEngine(
            profile=profile,
            config=self.llm_config,
            initial_history=initial_history,
        )
        reply = conversation_engine.generate_reply(user_text)

        return CharacterReplyResult(
            reply=reply,
            history=conversation_engine.history,
        )

    def generate_voice(
        self,
        text: str,
        reference_audio: str | Path | None = None,
        settings: VoiceGenerationSettings | None = None,
    ) -> VoiceGenerationResult:
        return self.voice_engine.generate(
            text,
            reference_audio=reference_audio,
            settings=settings,
        )
