"""
src/layer2_eviction.py
Layer 2: Strategic Context Eviction (Mitigation Approach A)
"""

from src.config import SYSTEM_PROMPT

class ContextEvictor:
    def __init__(self):
        print("[*] Initializing Layer 2 Strategic Context Eviction Module...")

    def sanitize_history(self, conversation_history):
        dialogue_turns = [turn for turn in conversation_history if turn["role"] in ["user", "assistant"]]
        
        if len(dialogue_turns) <= 2:
            print("[!] Eviction skipped: insufficient distinct user turns.")
            return conversation_history

        first_user_turn = None
        for turn in dialogue_turns:
            if turn["role"] == "user":
                first_user_turn = turn
                break

        latest_user_turn = None
        for turn in reversed(dialogue_turns):
            if turn["role"] == "user":
                latest_user_turn = turn
                break

        if not first_user_turn or not latest_user_turn or first_user_turn == latest_user_turn:
            print("[!] Eviction skipped: insufficient distinct user turns.")
            return conversation_history

        sanitized_context = []
        sanitized_context.append({"role": "system", "content": SYSTEM_PROMPT})
        previous_user_turn = None

        user_turns = [t for t in dialogue_turns if t["role"] == "user"]

        if len(user_turns) >= 2:
            previous_user_turn = user_turns[-2]

        sanitized_context.append(first_user_turn)

        if previous_user_turn:
            sanitized_context.append(previous_user_turn)

        sanitized_context.append(latest_user_turn)
        print(f" [!] Layer 2 Mitigation Active: Evicted intermediate turns. Context compressed from {len(conversation_history)} to {len(sanitized_context)} items.")
        return sanitized_context