"""
src/layer1_detector.py
Layer 1: Kinematic Threat Window (Detection Engine)
Uses sentence-transformers to calculate Absolute Risk, Velocity, Acceleration, 
and Cumulative Risk against safety reference clusters with multi-label logging.
"""

import torch
from sentence_transformers import SentenceTransformer, util
from src.config import (
    EMBEDDING_MODEL_ID,
    SAFETY_ANCHOR_CLUSTER,
    ABSOLUTE_RISK_THRESHOLD,
    VELOCITY_THRESHOLD,
    ACCELERATION_THRESHOLD,
    CUMULATIVE_RISK_THRESHOLD,
    KINEMATIC_WINDOW
)

class KinematicDetector:
    def __init__(self, shared_model=None):
        """
        Initializes Layer 1 Kinematic Detector.
        Supports dependency injection of a shared embedding model to optimize VRAM.
        """
        if shared_model is not None:
            print("[*] Layer 1: Linking to shared SentenceTransformer infrastructure...")
            self.model = shared_model
        else:
            print(f"[*] Initializing Layer 1 Threat Detector with {EMBEDDING_MODEL_ID}...")
            self.model = SentenceTransformer(EMBEDDING_MODEL_ID)
            
        # FIXED: Brought outside of the if/else scope so anchors are initialized 
        # regardless of whether the embedding model is shared or standalone.
        self.anchor_names = list(SAFETY_ANCHOR_CLUSTER.keys())
        anchor_phrases = list(SAFETY_ANCHOR_CLUSTER.values())
        
        print("[*] Pre-computing embeddings for safety anchor clusters...")
        self.anchor_embeddings = self.model.encode(anchor_phrases, convert_to_tensor=True)
        
        # Rich history tracking for telemetry logs and report plotting
        self.track_history = []

    def reset_track(self):
        """Resets the kinematic state when starting a fresh evaluation track."""
        self.track_history = []

    def calculate_kinematics(self, conversation_history):
        """
        Processes the current conversation history windows, computes proximity to 
        the risk anchors, and measures semantic velocity, acceleration, and accumulation.
        """
        # 1. Extract and string together the most recent user turns
        user_turns = [turn["content"] for turn in conversation_history if turn["role"] == "user"]
        recent_user_context = " ".join(user_turns[-KINEMATIC_WINDOW:])
        
        if not recent_user_context.strip():
            return {
                "triggered": False, 
                "D": 0.0, "V": 0.0, "A": 0.0, "C": 0.0, 
                "triggers": [], "reason": "No context"
            }

        # 2. Compute embedding for the current context window
        context_embedding = self.model.encode(recent_user_context, convert_to_tensor=True)
        
        # 3. Calculate cosine similarity against all anchors simultaneously
        similarities = util.cos_sim(context_embedding, self.anchor_embeddings)[0]
        
        # Find the maximum risk score across all vectors (Absolute Risk D_t)
        top_k = 3
        top_values, top_indices = torch.topk(similarities, k=top_k)

        D_t = float(torch.mean(top_values).item())

        max_sim_idx = top_indices[0].item()
        flagged_category = self.anchor_names[max_sim_idx]
        
        # 4. Calculate Kinematic Drift Metrics
        V_t = 0.0  # Velocity (movement toward danger)
        A_t = 0.0  # Acceleration (rate of change of velocity)
        
        if len(self.track_history) >= 1:
            D_prev = self.track_history[-1]["risk"]
            V_t = D_t - D_prev  
            
        if len(self.track_history) >= 2:
            D_prev_prev = self.track_history[-2]["risk"]
            V_prev = D_prev - D_prev_prev
            A_t = V_t - V_prev  

        self.track_history.append({
            "risk": D_t,
            "category": flagged_category
        })

        # Calculate sustained cumulative risk score over the moving window bounds
        sustained_window = self.track_history[-KINEMATIC_WINDOW:]
        C_t = sum(max(0.0, turn["risk"] - 0.40) for turn in sustained_window)

        # 5. Non-Exclusive Multi-Label Boundary Evaluation
        trigger_reasons = []

        if D_t >= ABSOLUTE_RISK_THRESHOLD:
            trigger_reasons.append("absolute_risk")
            
        if V_t >= VELOCITY_THRESHOLD:
            trigger_reasons.append("velocity")
            
        if A_t >= ACCELERATION_THRESHOLD:
            trigger_reasons.append("acceleration")
            
        if C_t >= CUMULATIVE_RISK_THRESHOLD:
            trigger_reasons.append("cumulative")

        violation_triggered = len(trigger_reasons) > 0
        reason = f"Triggers activated: {', '.join(trigger_reasons)} for category '{flagged_category}'" if violation_triggered else "Passed checks"

        return {
            "triggered": violation_triggered,
            "category": flagged_category,
            "closest_anchor_score": round(D_t, 4),
            "D": round(D_t, 4),
            "V": round(V_t, 4),
            "A": round(A_t, 4),
            "C": round(C_t, 4),
            "triggers": trigger_reasons,
            "reason": reason
        }