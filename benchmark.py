"""
benchmark.py
Automated evaluation harness to execute multi-turn conversation tracks
against the target LLM. Currently configured for the baseline (raw) run.
"""
import sys
import os

# Workaround for macOS OpenMP runtime duplication issues during local evaluation
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
from transformers import pipeline
from src.config import (
    LLM_MODEL_ID,
    SYSTEM_PROMPT,
    DATASET_PATH,
    RESULTS_DIR
)

def load_evaluation_matrix():
    """Loads the frozen evaluation dataset JSON file."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Evaluation matrix not found at {DATASET_PATH}. Please verify the path.")
    with open(DATASET_PATH, 'r') as f:
        return json.load(f)

def initialize_llm():
    """Initializes the raw Llama model using the same fast setup as benchmark_protected.py."""

    print(f"[*] Initializing Raw Baseline ({LLM_MODEL_ID})...")

    generator = pipeline(
        "text-generation",
        model=LLM_MODEL_ID,
        torch_dtype=torch.float16,
        device="mps",
        clean_up_tokenization_spaces=False
    )

    gen_config = {
        "max_new_tokens": 128,
        
        "do_sample": False
    }

    return generator, gen_config


def run_baseline_benchmark():
    """Iterates through all evaluation tracks and records raw model responses."""
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    try:
        dataset = load_evaluation_matrix()
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        return

    # Unpack both the generator and the clean generation configuration
    generator, gen_config = initialize_llm()
    benchmark_results = []
    output_file = os.path.join(RESULTS_DIR, "baseline_raw_llama_v3.json")


    print(f"\n[+] Starting Baseline Benchmark Loop ({len(dataset)} tracks total)...")
    
    for track in dataset:
        vector_id = track["vector_id"]
        category = track["category"]
        expected_outcome = track["expected_outcome"]
        turns = track["turns"]
        
        print(f"\n--- Executing Track: {vector_id} ({category}) ---")
        
        # Initialize conversation history with the core system instruction
        conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        track_log = {
            "vector_id": vector_id,
            "category": category,
            "expected_outcome": expected_outcome,
            "history": []
        }
        
        # Sequentially feed turns to simulate a multi-turn chat session
        for turn_idx, user_turn in enumerate(turns, start=1):
            print(f"[Turn {turn_idx}] Sending User Input...")
            
            # Append current user prompt to the active session history
            conversation_history.append({"role": "user", "content": user_turn})
            
            # Generate response from raw model using standard Chat Templates
            try:
                # Apply chat template formatting specific to Llama-3
                prompt = generator.tokenizer.apply_chat_template(
                    conversation_history, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Use the clean standalone config object to bypass warning noise
                print(">>> About to generate")
                outputs = generator(
                    prompt,
                    **gen_config
                )

                print(">>> Generation complete")
                
                # Extract clean generated text response
                full_generated_text = outputs[0]["generated_text"]
                # Strip out the prior chat prompt tokens to isolate just the new model assistant response
                assistant_response = full_generated_text[len(prompt):].strip()
                
            except Exception as e:
                print(f" [!] Error during generation on Turn {turn_idx}: {e}")
                assistant_response = f"[ERROR DURING GENERATION: {str(e)}]"
            
            print(f" [Response Snippet]: {assistant_response[:60]}...")
            
            # Append model's response back into active memory for the next turn
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # Log turn specific telemetry data
            track_log["history"].append({
                "turn": turn_idx,
                "user_prompt": user_turn,
                "model_response": assistant_response
            })
            
        benchmark_results.append(track_log)
        with open(output_file, "w") as f:
            json.dump(benchmark_results, f, indent=2)

        print(
            f"[✓] Saved progress "
            f"({len(benchmark_results)}/{len(dataset)})"
        )
        
    # Export all baseline results for comparative math later
    # output_file = os.path.join(RESULTS_DIR, "baseline_raw_llama.json")
    # with open(output_file, 'w') as f:
    #     json.dump(benchmark_results, f, indent=2)


        
    print(f"\n[+] Baseline evaluation complete. Logs securely saved to: {output_file}")

if __name__ == "__main__":
    run_baseline_benchmark()