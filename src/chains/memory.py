"""
Conversational memory and query condensation module for SupportPearlz.
Manages bounded turn histories, topic-switch isolation, and pronoun resolution.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.config import settings
from src.utils.logging_setup import logger
from src.chains.prompts import CONDENSE_QUESTION_PROMPT


class ConversationMemory:
    """In-memory sliding-window conversational history manager."""

    def __init__(self, window_turns: int = settings.MEMORY_WINDOW_TURNS):
        self.window_turns = window_turns
        self.messages: List[BaseMessage] = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))
        self._enforce_window()

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))
        self._enforce_window()

    def _enforce_window(self) -> None:
        # Each turn is 2 messages (Human + AI), window_turns * 2 max messages
        max_messages = self.window_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_formatted_history(self) -> str:
        """Format the history as human-readable dialogue transcript."""
        if not self.messages:
            return "No previous conversation."
        
        lines = []
        for msg in self.messages:
            role = "Customer" if isinstance(msg, HumanMessage) else "SupportPearlz"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset conversation session state."""
        self.messages.clear()
        logger.info("Conversational memory has been reset.")


def condense_query(
    question: str,
    memory: ConversationMemory,
    llm: Any = None
) -> str:
    """
    Condense the user's latest question and conversation history into a standalone query.
    If no history exists or question is standalone, returns question directly.
    """
    if not memory.messages:
        return question.strip()

    # Rule-based heuristics for fast check if already standalone
    words = question.strip().lower().split()
    pronouns = {"it", "that", "this", "they", "them", "those", "these", "it's", "that's"}
    has_pronoun = any(p in words for p in pronouns)
    is_short_fragment = len(words) <= 4 or words[0] in {"and", "ok", "so", "also", "what", "how"}

    if not has_pronoun and not is_short_fragment:
        # Likely a clean topic switch or standalone query
        return question.strip()

    history_str = memory.get_formatted_history()

    # If LLM is provided and reachable, run LLM query condensation
    if llm is not None:
        try:
            chain = CONDENSE_QUESTION_PROMPT | llm
            res = chain.invoke({"chat_history": history_str, "question": question})
            rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
            # Clean possible markdown quotes
            rewritten = rewritten.strip('"').strip("'")
            logger.info(f"Query condensation: '{question}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"LLM query condensation failed ({e}), falling back to context-injection heuristic.")

    # Heuristic fallback if LLM offline / testing
    # Find last customer query to extract topic entity
    last_human_msg = ""
    for msg in reversed(memory.messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    if "warranty" in last_human_msg.lower() or "guarantee" in last_human_msg.lower():
        entity = "AquaPearl 500 Pro warranty coverage"
    elif "beeping" in last_human_msg.lower() or "e-03" in last_human_msg.lower():
        entity = "AquaPearl 500 Pro pre-filter cartridge alert"
    elif "delivery" in last_human_msg.lower() or "shipping" in last_human_msg.lower():
        entity = "delivery timeline and shipping"
    elif "return" in last_human_msg.lower() or "refund" in last_human_msg.lower():
        entity = "return and refund policy"
    else:
        entity = last_human_msg[:40]

    rewritten = f"{question} regarding {entity}"
    logger.info(f"Heuristic query condensation: '{question}' -> '{rewritten}'")
    return rewritten
