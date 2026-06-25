"""
Belief Agent — core agent class for the Agentic Diffusion Simulator.

Each agent has a belief state (0.0–1.0) representing how strongly they
hold the "new behavior". Belief updating uses either:
  - Pure math mode (fast, no API key needed)
  - LLM mode (Groq, for richer narrative output)

States follow Krishnan (2025) Chapter 5 four-stage model:
  0 = Unaware
  1 = Aware
  2 = Action / Trial
  3 = Persistent Habit
"""

import random
import os
from typing import Optional


STAGE_LABELS = {0: "Unaware", 1: "Aware", 2: "Action", 3: "Habit"}


class BeliefAgent:
    def __init__(
        self,
        agent_id: int,
        persona: str = "neutral",
        intrinsic_benefit: Optional[float] = None,
        use_llm: bool = False,
    ):
        self.id = agent_id
        self.persona = persona  # "champion", "skeptic", "neutral", "laggard"
        self.intrinsic_benefit = intrinsic_benefit or random.uniform(0.1, 0.9)
        self.belief = 0.0          # 0.0 = fully old, 1.0 = fully new
        self.stage = 0             # Krishnan 4-stage state
        self.behavior = -1         # -1 old, +1 new
        self.memory = []           # last N neighbor messages
        self.use_llm = use_llm
        self._client = None

        # Persona modifiers
        persona_params = {
            "champion":  {"openness": 0.9, "conformity": 0.7},
            "skeptic":   {"openness": 0.2, "conformity": 0.2},
            "neutral":   {"openness": 0.5, "conformity": 0.5},
            "laggard":   {"openness": 0.1, "conformity": 0.4},
        }
        p = persona_params.get(persona, persona_params["neutral"])
        self.openness = p["openness"]
        self.conformity = p["conformity"]

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        return self._client

    def receive_message(self, sender_id: int, sender_belief: float, sender_stage: int):
        """Store a neighbor's signal in short-term memory (last 5)."""
        self.memory.append({
            "sender": sender_id,
            "belief": sender_belief,
            "stage": sender_stage,
        })
        if len(self.memory) > 5:
            self.memory.pop(0)

    def _math_update(self, cost: float = 0.3):
        """Krishnan myopic utility rule: (benefit - cost) + conformity * peer_pressure."""
        if not self.memory:
            return

        avg_neighbor_belief = sum(m["belief"] for m in self.memory) / len(self.memory)
        peer_pressure = self.conformity * (avg_neighbor_belief - self.belief)
        net_utility = (self.intrinsic_benefit - cost) + peer_pressure

        # Belief shifts proportionally, capped at [0,1]
        delta = self.openness * net_utility * 0.2
        self.belief = max(0.0, min(1.0, self.belief + delta))

        # Stage transitions
        if self.belief > 0.75 and self.stage < 3:
            self.stage = 3
        elif self.belief > 0.5 and self.stage < 2:
            self.stage = 2
        elif self.belief > 0.25 and self.stage < 1:
            self.stage = 1

        self.behavior = 1 if self.belief > 0.5 else -1

    def _llm_update(self, cost: float = 0.3) -> str:
        """LLM-powered belief update — returns narrative reasoning."""
        if not self.memory:
            return ""

        avg_nb = sum(m["belief"] for m in self.memory) / len(self.memory)
        nb_stages = [STAGE_LABELS[m["stage"]] for m in self.memory]

        prompt = f"""You are agent {self.id}, persona: {self.persona}.
Your current belief in adopting the new behavior: {self.belief:.2f} (0=strongly against, 1=strongly for).
Your current stage: {STAGE_LABELS[self.stage]}.
Your openness to change: {self.openness:.2f}. Conformity tendency: {self.conformity:.2f}.

Your neighbors have beliefs averaging {avg_nb:.2f}. Their stages: {nb_stages}.
The perceived cost of switching is {cost:.2f}.

Based on this, decide:
1. Your new belief value (0.0-1.0).
2. Your new stage (0=Unaware, 1=Aware, 2=Action, 3=Habit).
3. One sentence of your internal reasoning.

Respond ONLY as JSON: {{"belief": 0.XX, "stage": X, "reason": "..."}}"""

        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.4,
            )
            import json
            text = resp.choices[0].message.content.strip()
            data = json.loads(text)
            self.belief = max(0.0, min(1.0, float(data["belief"])))
            self.stage = max(0, min(3, int(data["stage"])))
            self.behavior = 1 if self.belief > 0.5 else -1
            return data.get("reason", "")
        except Exception as e:
            # Fallback to math
            self._math_update(cost)
            return f"[LLM error, used math fallback: {e}]"

    def update(self, cost: float = 0.3) -> str:
        """Update belief and return narrative (empty string in math mode)."""
        if self.use_llm:
            return self._llm_update(cost)
        else:
            self._math_update(cost)
            return ""

    def __repr__(self):
        return (
            f"Agent({self.id}, {self.persona}, "
            f"belief={self.belief:.2f}, stage={STAGE_LABELS[self.stage]})"
        )
