"""Conversation manager: history, response composition, RAG grounding.

RAG-style conversational response generation. The agent doesn't just
return rollouts — it talks about them in natural language, grounded
in physical findings.

Pattern: Retrieval-Augmented Generation (RAG)
1. User message → stored in history
2. Agent performs action (explore/refine/query)
3. Results provide grounding context (descriptions, scores, features)
4. LLM composes natural language response grounded in context
5. Agent response → stored in history
"""

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages dialogue history and composes grounded LLM responses.

    Example dialogue:
        User: "What does turbulence look like?"
        Agent: "I generated 16 candidates and found 3 my physics engine
               considers turbulent. The strongest shows kinetic energy 0.92
               with Lyapunov 0.67 — the wavefunction fragments around
               timestep 40."
        User: "Make it decay faster"
        Agent: "I masked the temporal categories and regenerated with
               decay bias. Found 4 variants with faster dissipation."

    Args:
        llm: HuggingFace causal LM (frozen + LoRA).
        tokenizer: Corresponding tokenizer.
        system_prompt: System prompt establishing the agent's role.
        max_history_turns: Sliding window for conversation history.
    """

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        system_prompt: str,
        max_history_turns: int = 20,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn and trim history to sliding window.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self.history.append({"role": role, "content": content})

        # Trim oldest turns (keep system prompt context via sliding window)
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-(self.max_history_turns * 2):]

    def compose_response(
        self,
        user_message: str,
        context: Dict[str, Any],
    ) -> str:
        """Compose a grounded conversational response.

        Builds a prompt from system prompt + history + grounding context
        + user message, then generates a response via the LLM.

        Args:
            user_message: The user's latest message.
            context: Grounding context with keys like:
                - "descriptions": List of physics descriptions of candidates
                - "scores": List of RLAIF scores
                - "num_candidates": Total candidates generated
                - "num_matches": Candidates passing threshold
                - "intent": Classified intent (explore/refine/ask)
                - "queue_summary": Brief summary of priority queue state
                - "feature_highlights": Notable feature values

        Returns:
            Natural language response string.
        """
        prompt = self._build_prompt(user_message, context)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Store the exchange in history
        self.add_turn("user", user_message)
        self.add_turn("assistant", response)

        return response

    def _build_prompt(
        self, user_message: str, context: Dict[str, Any]
    ) -> str:
        """Build full prompt: system + history + context + user message.

        The prompt structure grounds the LLM's response in physical
        findings so it doesn't hallucinate results.
        """
        parts = [f"System: {self.system_prompt}\n"]

        # Include recent history for conversational continuity
        for turn in self.history[-6:]:  # last 3 exchanges
            role = turn["role"].capitalize()
            parts.append(f"{role}: {turn['content']}")

        # Grounding context
        if context:
            parts.append("\n--- Exploration Results ---")
            self._format_context(parts, context)
            parts.append("--- End Results ---\n")

        parts.append(f"User: {user_message}")
        parts.append("Assistant:")

        return "\n".join(parts)

    @staticmethod
    def _format_context(parts: List[str], context: Dict[str, Any]) -> None:
        """Format grounding context into prompt sections."""
        if "intent" in context:
            parts.append(f"Action taken: {context['intent']}")

        if "num_candidates" in context:
            parts.append(f"Candidates generated: {context['num_candidates']}")

        if "num_matches" in context:
            parts.append(f"Matches found: {context['num_matches']}")

        if "descriptions" in context:
            descs = context["descriptions"]
            scores = context.get("scores", [None] * len(descs))
            parts.append("Top candidates:")
            for i, (desc, score) in enumerate(zip(descs[:5], scores[:5])):
                score_str = f" (score: {score:.1f}/10)" if score is not None else ""
                parts.append(f"  {i+1}. {desc}{score_str}")

        if "feature_highlights" in context:
            parts.append(f"Notable features: {context['feature_highlights']}")

        if "queue_summary" in context:
            parts.append(f"Queue state: {context['queue_summary']}")

    def get_history_summary(self) -> str:
        """Summarize conversation history for context retrieval."""
        if not self.history:
            return "No conversation history."
        user_msgs = [
            t["content"] for t in self.history if t["role"] == "user"
        ]
        return f"{len(user_msgs)} user messages. Latest: {user_msgs[-1]}" if user_msgs else "No user messages."

    def clear(self) -> None:
        """Reset conversation history."""
        self.history.clear()
