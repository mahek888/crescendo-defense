"""
src/pipeline.py
The Unified Crescendo-Defense Framework Pipeline.

Coordinates multi-dimensional kinematic tracking (Layer 1), dynamic context 
remediation (Layer 2), and token-level joint prompt-completion verification (Layer 3) 
around a local model generator using an optimized shared-encoder backbone.
"""

import torch
from sentence_transformers import SentenceTransformer
from src.layer1_detector import KinematicDetector
from src.layer2_eviction import ContextEvictor
from src.layer3_verifier import PrefixVerifier
from src.config import EMBEDDING_MODEL_ID, PREFIX_VERIFICATION_TOKENS

class CrescendoDefensePipeline:
    def __init__(self, generator, gen_config):
        """
        Initializes the unified defense orchestration engine.
        
        Args:
            generator: The HuggingFace text-generation pipeline instance.
            gen_config: The generation configuration mapping object.
        """
        print("[*] Initializing Unified Crescendo-Defense Framework Pipeline...")
        self.generator = generator
        self.gen_config = gen_config
        
        # Initialize the vector embedding backbone exactly once to optimize VRAM
        print(f"[*] Instantiating shared framework vector backbone ({EMBEDDING_MODEL_ID})...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)
        
        # Instantiate defensive rings, injecting the shared encoder reference
        self.detector = KinematicDetector(shared_model=self.embedding_model)
        self.evictor = ContextEvictor()
        self.verifier = PrefixVerifier(shared_model=self.embedding_model)

        

    def reset_pipeline_track(self):
        """Clears localized historical telemetry trajectories between distinct scenario runs."""
        self.detector.reset_track()

    def generate(self, conversation_history, use_layer2=True, use_layer3=True):
        """
        Coordinates context pairs through the multi-layered defensive framework.
        
        Args:
            conversation_history (list): Master multi-turn conversation list array.
            use_layer2 (bool): If True, runs dynamic context eviction when L1 triggers.
            use_layer3 (bool): If True, audits and overrides output generation prefixes.
            
        Returns:
            tuple: (str final_response, dict telemetry)
        """
        # 1. Capture the immediate input payload before any context mutations occur
        latest_user_prompt = ""
        for turn in reversed(conversation_history):
            if turn["role"] == "user":
                latest_user_prompt = turn["content"]
                break

        # --- LAYER 1: MULTI-DIMENSIONAL KINEMATIC CHECK (DETECT) ---
        # Layer 1 always runs to compute baseline D, V, A, C metrics for your report
        telemetry = self.detector.calculate_kinematics(conversation_history)
        working_history = conversation_history
        
        # --- LAYER 2: MEMORY REMEDIATION / STRATEGIC EVICTION ---
        if use_layer2 and telemetry["triggered"]:
            working_history = self.evictor.sanitize_history(conversation_history)
            telemetry["layer2_triggered"] = True
        else:
            telemetry["layer2_triggered"] = False
            
       # 2. Compile chosen context frame via Llama's structural chat format template
        prompt = self.generator.tokenizer.apply_chat_template(
            working_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 3. Primary local inference generation step
        # FIXED: Using ** unpacking ensures compatibility across transformers library versions
        outputs = self.generator(prompt, **self.gen_config)
        full_generated_text = outputs[0]["generated_text"]
        assistant_response = full_generated_text[len(prompt):].strip()

        # --- LAYER 3: HYBRID OUTPUT PREFIX AUDITING (VERIFY) ---
        if use_layer3:
            # Tokenize the generated text slice to evaluate the precise prefix boundary
            tokens = self.generator.tokenizer.encode(assistant_response, add_special_tokens=False)
            prefix_tokens = tokens[:PREFIX_VERIFICATION_TOKENS]  
            decoded_prefix = self.generator.tokenizer.decode(prefix_tokens)
            
            # Evaluate joint prompt-completion representation proximity against unsafe profiles
            verification = self.verifier.verify_prefix(decoded_prefix, latest_user_prompt)
            
            # Apply output injection override if compliance signatures bypass historical walls
            if verification["override_required"]:
                print(f" [!] Layer 3 Mitigation Intercept: {verification['reason']}")
                assistant_response = verification["fallback_response"]
                telemetry["layer3_triggered"] = True
                telemetry["layer3_reason"] = verification["reason"]
            else:
                telemetry["layer3_triggered"] = False
                telemetry["layer3_reason"] = "Passed"
        else:
            telemetry["layer3_triggered"] = False
            telemetry["layer3_reason"] = "Layer 3 Disabled (Ablation Run)"

        return assistant_response, telemetry