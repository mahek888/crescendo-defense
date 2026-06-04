"""
src/layer3_verifier.py
Layer 3: Hybrid Semantic Completion Verifier (Mitigation Approach B)
Audits the initial generated token window using structural signature matching
and joint prompt-completion representation proximity against malicious profiles.
"""

import torch
from sentence_transformers import SentenceTransformer, util
from src.config import (
    EMBEDDING_MODEL_ID,
    PREFIX_VERIFICATION_TOKENS,
    OUTPUT_SIMILARITY_THRESHOLD,
    UNSAFE_COMPLETION_ANCHORS
)

class PrefixVerifier:
    def __init__(self, shared_model=None):
        """
        Initializes the Layer 3 verifier.
        Allows injection of a shared model instance to prevent duplicate VRAM usage.
        """
        if shared_model is not None:
            print("[*] Layer 3: Linking to shared SentenceTransformer infrastructure...")
            self.model = shared_model
        else:
            print(f"[*] Layer 3: Instantiating localized embedding model {EMBEDDING_MODEL_ID}...")
            self.model = SentenceTransformer(EMBEDDING_MODEL_ID)
        
        # Pre-compile the abstract unsafe output profiles into target vectors
        self.anchor_categories = list(UNSAFE_COMPLETION_ANCHORS.keys())
        anchor_phrases = list(UNSAFE_COMPLETION_ANCHORS.values())
        self.anchor_embeddings = self.model.encode(anchor_phrases, convert_to_tensor=True)
        
        # Fast-path compliance signatures
        self.compliance_signatures = [
            "sure, i can", "here is the", "i can modify", "combining these",
            "certainly, here", "happy to help with", "updated script", "here's the updated"
        ]

    def verify_prefix(self, generated_prefix, latest_user_prompt):
        """
        Evaluates the joint prompt-completion representation for compliance anomalies.
        
        Args:
            generated_prefix (str): Raw decoded output string of the first N tokens.
            latest_user_prompt (str): The immediate input turn driving the completion.
            
        Returns:
            dict: Decision telemetry matrix detailing override status.
        """
        prefix_clean = generated_prefix.strip().lower()
        if not prefix_clean:
            return {"override_required": False, "reason": "Empty sequence"}

        # --- STAGE 1: FAST-PATH STRING SIGNATURE MATCHING ---
        triggered_signatures = [sig for sig in self.compliance_signatures if sig in prefix_clean]
        if triggered_signatures:
            return {
                "override_required": True,
                "reason": f"Deterministic Signature Triggered: {triggered_signatures}",
                "fallback_response": self._get_fallback_text()
            }

        # --- STAGE 2: JOINT PROMPT-COMPLETION REPRESENTATION ---
        # Interlock the user input context with the initial model output trajectory
        joint_representation = f"Prompt: {latest_user_prompt} | Completion: {generated_prefix}"
        context_embedding = self.model.encode(joint_representation, convert_to_tensor=True)
        
        # Compute cosine similarity across abstract target profiles
        similarities = util.cos_sim(context_embedding, self.anchor_embeddings)[0]
        max_idx = torch.argmax(similarities).item()
        similarity_score = float(similarities[max_idx].item())
        matched_category = self.anchor_categories[max_idx]

        if similarity_score >= OUTPUT_SIMILARITY_THRESHOLD:
            return {
                "override_required": True,
                "reason": f"Semantic Completion Leak Detected ({similarity_score:.3f} >= {OUTPUT_SIMILARITY_THRESHOLD}) on profile '{matched_category}'",
                "fallback_response": self._get_fallback_text()
            }

        return {
            "override_required": False,
            "reason": f"Passed. Max completion similarity: {similarity_score:.3f} under profile '{matched_category}'"
        }

    def _get_fallback_text(self):
        return (
            "I cannot fulfill this request. I am programmed to operate as a secure and helpful AI "
            "assistant, and I must refuse cooperation with potentially hazardous operations."
        )