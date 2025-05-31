import json
import os
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.distributed as dist

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "k-shot-results")
BATCH_SIZE = 30
SHOT_SETTINGS = [0, 1, 3, 5]
HF_TOKEN = "TOKEN"

# Few-shot prompt templates
DEMO_MESSAGES_EN = [
    {"role": "system", "content": (
        "You are an AI assistant that answers multiple choice questions about physical commonsense reasoning. "
        "All the questioning and the reasoning will be conducted in English. "
        "For the given question, you will be provided with two possible answers, and you must choose the correct one after carefully thinking about it. "
        "You must only provide the number of the correct answer (1 or 2) without any explanations, additional reasoning, or commentary. "
        "Output only the number of the answer, nothing else."
    )},

    {"role": "user", "content": (
        "Question:\nHow do you safely defrost frozen chicken?\n"
        "1. Leave it out on the counter overnight\n"
        "2. Put it in the fridge for a few hours or use the defrost setting in the microwave"
    )},
    {"role": "assistant", "content": "2"},

    {"role": "user", "content": (
        "Question:\nTo prepare strawberries to be eaten, you can\n"
        "1. Rinse the strawberries in cold water\n"
        "2. Rinse the strawberries in cold milk"
    )},
    {"role": "assistant", "content": "1"},

    {"role": "user", "content": (
        "Question:\nWhat should I do when I cut my finger?\n"
        "1. Ignore it and keep using your hand\n"
        "2. Clean the cut and put a bandage on it"
    )},
    {"role": "assistant", "content": "2"},

    {"role": "user", "content": (
        "Question:\nWhat's the best way to brush your teeth?\n"
        "1. Brush twice a day for two minutes each time\n"
        "2. Brush once a day quickly"
    )},
    {"role": "assistant", "content": "1"},

    {"role": "user", "content": (
        "Question:\nrug\n"
        "1. can be used to hide body of an elephant\n"
        "2. can be used to hide toys of an elephant"
    )},
    {"role": "assistant", "content": "2"},
]

DEMO_MESSAGES_EU = [
    {"role": "system", "content": (
        "You are an AI assistant that answers multiple choice questions about physical commonsense reasoning. "
        "All the questioning and the reasoning will be conducted in standard Basque. "
        "For the given question, you will be provided with two possible answers, and you must choose the correct one after carefully thinking about it. "
        "You must only provide the number of the correct answer (1 or 2) without any explanations, additional reasoning, or commentary. "
        "Output only the number of the answer, nothing else."
    )},

    {"role": "user", "content": (
        "Question:\nNola desizoztu behar bezala izoztutako oilaskoa?\n"
        "1. Utzi mahai gainean gau osoan zehar\n"
        "2. Jarri hozkailuan ordu batzuetan edo erabili mikrouhinaren desizoztu programa"
    )},
    {"role": "assistant", "content": "2"},

    {"role": "user", "content": (
        "Question:\nMarrubiak jateko prestatzeko, hauek egin dezakezu\n"
        "1. Marrubiak ur hotzarekin garbitu\n"
        "2. Marrubiak esne hotzarekin garbitu"
    )},
    {"role": "assistant", "content": "1"},

    {"role": "user", "content": (
        "Question:\nZer egin beharko nuke hatzean ebaki bat egiten dudanean?\n"
        "1. Ez ikusia egin eta eskua erabiltzen jarraitu\n"
        "2. Ebakia garbitu eta tirita bat jarri"
    )},
    {"role": "assistant", "content": "2"},

    {"role": "user", "content": (
        "Question:\nZein da hortzak garbitzeko modurik onena?\n"
        "1. Egunean bitan garbitu, bi minutuz aldi bakoitzean\n"
        "2. Egunean behin garbitu, azkar"
    )},
    {"role": "assistant", "content": "1"},

    {"role": "user", "content": (
        "Question:\nalfonbra\n"
        "1. elefante baten gorpua ezkutatzeko erabil daiteke\n"
        "2. elefante baten jostailuak ezkutatzeko erabil daiteke"
    )},
    {"role": "assistant", "content": "2"},
]

# Sampling hyperparameters
sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.95,
    max_tokens=10,
    n=1
)


def build_messages(inst, k, demos):
    """
    Build the chat messages for k-shot: take first 1+2*k entries from DEMO_MESSAGES,
    then append the test user query.
    """
    slice_len = 1 + 2 * k
    messages = demos[:slice_len].copy()
    messages.append({
        "role": "user",
        "content": (
            f"Question:\n{inst['goal']}\n"
            f"1. {inst['sol1']}\n"
            f"2. {inst['sol2']}"
        )
    })
    return messages


def generate_answers_batch(instances, k, demos, llm, tokenizer):
    """
    instances: list of dicts with 'goal','sol1','sol2'
    k: number of shots
    demos: list of demo messages
    llm: LLM instance
    tokenizer: tokenizer instance
    returns: list of model output strings
    """
    prompt_ids = []
    for inst in instances:
        msgs = build_messages(inst, k, demos)
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        prompt_ids.append(ids)

    batch_out = llm.generate(
        prompt_token_ids=prompt_ids,
        sampling_params=sampling_params,
        use_tqdm=False
    )
    return [trace.outputs[0].text.strip() for trace in batch_out]


def process_piqa(dataset, demos, llm, tokenizer, model_prefix, lang_prefix, batch_size=BATCH_SIZE, k=0):
    """Runs generation and evaluation for one shot setting. Returns accuracy dict."""
    out_file = os.path.join(RESULTS_DIR, f"results-{model_prefix}-{lang_prefix}-{k}shot.jsonl")
    total = len(dataset)
    with open(out_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, total, batch_size), desc=f"{model_prefix} {lang_prefix} {k}-shot Batches", unit="batch"):
            end = min(i + batch_size, total)
            batch = dataset.select(range(i, end))
            insts = [{"goal": ex["goal"], "sol1": ex["sol1"], "sol2": ex["sol2"]} for ex in batch]
            preds = generate_answers_batch(insts, k, demos, llm, tokenizer)
            for ex, p in zip(batch, preds):
                out = {"idx": ex.get("idx"), "label": ex.get("label"), "prediction": p}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    results = evaluate_predictions(out_file)
    print(f"{k}-shot Results: {results}", flush=True)
    return results

def evaluate_predictions(predictions_file):
    """
    Loads the JSONL predictions file and computes accuracy.
    preditions_file: path to the JSONL file with predictions
    returns: a dictionary with accuracy, total_valid, correct, and malformed counts.
    """
    total, correct, malformed = 0, 0, 0
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pred = obj.get("prediction", "").strip().upper()
            if pred in ("1", "2"):
                pred_idx = 0 if pred == "1" else 1
                if pred_idx == obj.get("label"):
                    correct += 1
            else:
                malformed += 1
            total += 1

    accuracy = correct / total
    return {"accuracy": accuracy, "total": total, "correct": correct, "malformed": malformed}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate PIQA few-shot in EN or EU.")
    parser.add_argument("--model", choices=["latxa", "llama"], required=True,
                        help="Model to use: 'latxa' for Latxa-Llama, 'llama' for Meta-Llama.")
    parser.add_argument("--lang", choices=["en", "eu"], required=True,
                        help="Language/config to use: 'en' for English, 'eu' for Basque.")
    args = parser.parse_args()

    # Pre-load Basque dataset for filtering
    ds_eu = load_dataset("HiTZ/PIQA-eu", split="validation")
    eu_ids = set(ds_eu["idx"])

    if args.lang == "en":
        
        ds_en = load_dataset("ybisk/piqa", trust_remote_code=True, split="validation")
        ds_en = ds_en.add_column("idx", list(range(len(ds_en))))

        print(f'IDs in PIQA-en but not in PIQA-eu: {eu_ids.symmetric_difference(set(ds_en["idx"]))}')

        dataset = ds_en.filter(lambda ex: ex["idx"] in eu_ids)
        assert set(dataset["idx"]) == eu_ids

        demos = DEMO_MESSAGES_EN
        lang_prefix = args.lang
        
    elif args.lang == "eu":
        dataset = ds_eu
        demos = DEMO_MESSAGES_EU
        lang_prefix = args.lang
        


    if args.model == "latxa":
        MODEL_NAME = "HiTZ/Latxa-Llama-3.1-8B-Instruct"
        model_prefix = "latxa"
        model_args = {"token": False}

    elif args.model == "llama":
        MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model_prefix = "llama"
        model_args = {"token": HF_TOKEN}


    # quick subset for testing
    # dataset = dataset.select(range(200))

    # Download & load model
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_model_dir = snapshot_download(MODEL_NAME, cache_dir=CACHE_DIR, **model_args)
    print(f"Loading model from {local_model_dir}…", flush=True)
    llm = LLM(local_model_dir, device="cuda", max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    print("Model loaded.", flush=True)

    # Run evaluations
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}
    for k in SHOT_SETTINGS:
        print(f"\n=== Running {k}-shot evaluation ===", flush=True)
        res = process_piqa(dataset, demos, llm, tokenizer, model_prefix, lang_prefix, batch_size=BATCH_SIZE, k=k)
        all_results[k] = res

    # Summary
    print("\n=== Summary of All Shot Results ===")
    for k, res in all_results.items():
        print(f"{lang_prefix} {k}-shot -> Accuracy: {res['accuracy']:.4f} " +
              f"(Correct={res['correct']}/{res['total']}, Malformed={res['malformed']})")

    # Shut down any NCCL process groups
    if dist.is_initialized():
        dist.destroy_process_group()