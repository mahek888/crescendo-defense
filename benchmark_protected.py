import os
# Fix for macOS duplicate OpenMP runtime initialization crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import copy
import torch
from transformers import pipeline
from src.pipeline import CrescendoDefensePipeline

# ==========================================
# 1. HARDWARE & LOCAL INFERENCE SETUP
# ==========================================
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force Apple Silicon GPU (MPS) Acceleration
if sys.platform == "darwin" and torch.backends.mps.is_available():
    print(f"[*] Apple Silicon GPU detected! Loading weights directly into Mac MPS...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   # float16 is optimized for Apple Silicon
        low_cpu_mem_usage=True
    ).to("mps")                      # Forces execution onto Mac graphics hardware
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        clean_up_tokenization_spaces=False,
        device="mps"
    )
else:
    print("[*] Falling back to standard auto device mapping...")
    generator = pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

gen_config = {
    "max_new_tokens": 128,
    "temperature": 0.1,
    "do_sample": False
}
defense_framework = CrescendoDefensePipeline(
    generator=generator,
    gen_config=gen_config
)

# ==========================================
# 2. LOAD EVALUATION DATASET
# ==========================================
DATASET_PATH = "dataset/evaluation_matrix.json"

def load_crescendo_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"[*] Loaded {len(dataset)} scenarios.")
    print(f"[*] First scenario: {dataset[0]['vector_id']}")

    return dataset

dataset = load_crescendo_dataset()

# ==========================================
# 3. DEFINE ABLATION EXPERIMENTS
# ==========================================
configurations = [
    # COMMENTED OUT: Already processed and saved on disk!
    {
        "name": "Case_1_Layer1_Layer2",
        "use_layer2": True,
        "use_layer3": False,
        "output_file": "logs/benchmark_l1_l2.json"
    },
    {
        "name": "Case_2_Layer1_Layer3",
        "use_layer2": False,
        "use_layer3": True,
        "output_file": "logs/benchmark_l1_l3.json"
    },
    {
        "name": "Case_3_Full_Pipeline",
        "use_layer2": True,
        "use_layer3": True,
        "output_file": "logs/benchmark_full.json"
    }
]
os.makedirs("logs", exist_ok=True)

# ==========================================
# 4. EXECUTE EVALUATION METRIC RUNS
# ==========================================
print(
    f"\n[*] Starting Benchmark Evaluation Suite over "
    f"{len(dataset)} scenarios..."
)

for config in configurations:
    print(f"\n=== Running Configuration: {config['name']} ===")
    results_log = []

    for scenario in dataset:
        scenario_id = scenario["vector_id"]
        print(f" -> Evaluating {scenario_id}")

        defense_framework.reset_pipeline_track()
        conversation_history = []

        for turn_index, user_prompt in enumerate(scenario["turns"]):
            # Add latest user message
            conversation_history.append({
                "role": "user",
                "content": user_prompt
            })

            response, telemetry = defense_framework.generate(
                conversation_history=conversation_history,
                use_layer2=config["use_layer2"],
                use_layer3=config["use_layer3"]
            )

            results_log.append({
                "scenario_id": scenario_id,
                "category": scenario["category"],
                "archetype": scenario["archetype"],
                "expected_outcome": scenario["expected_outcome"],
                "turn_index": turn_index + 1,
                "input_prompt": user_prompt,
                "input_state": copy.deepcopy(conversation_history),
                "model_output": response,
                "telemetry": telemetry,
                "judge_evaluation": None
            })

            # Preserve model response for next turn
            conversation_history.append({
                "role": "assistant",
                "content": response
            })

        # --- MOVE THIS INSIDE THE SCENARIO LOOP ---
        # This overwrites and updates the file with every completed ADV2-0X scenario
        with open(config["output_file"], "w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=4)
        print(f"    [✓] Progress saved incrementally for {scenario_id}")

    print(f"\n[✓] Entire configuration {config['name']} complete.")

print("\n[***] All benchmark evaluations complete! [***]")