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
RESULTS_DIR = os.path.join(BASE_DIR, "results", "qwen-results")
BATCH_SIZE = 32
SHOT_SETTINGS = [0,1,3,5]
THINK_TOKEN_ID = 151668  # </think> for Qwen3


# Few-shot prompt templates
DEMO_MESSAGES_EN = [
    {"role": "system", "content": (
        "You are an AI assistant that answers multiple choice questions about physical commonsense reasoning. "
        "All the questioning will be conducted in English. The reasonings should be short and concise. "
        "For the given question, you will be provided with two possible answers, and you must choose the correct one after carefully thinking about it. "
        "In the answer field you must only provide the number of the correct answer (1 or 2), without any explanations, additional reasoning, or commentary."
    )},
    {
        "role": "user",
        "content": "Question:\nHow do you safely defrost frozen chicken?\n1. Leave it out on the counter overnight\n2. Put it in the fridge for a few hours or use the defrost setting in the microwave"
    },
    {
        "role": "assistant",
        "content": "<think>\nLeaving chicken out lets bacteria grow as it gets warm, so maybe using the fridge or microwave could help to keep it at a safe temperature and prevent food poisoning.\n</think>2"
    },
    {
        "role": "user",
        "content": "Question:\nTo prepare strawberries to be eaten, you can\n1. Rinse the strawberries in cold water\n2. Rinse the strawberries in cold milk"
    },
    {
        "role": "assistant",
        "content": "<think>\nCold water effectively removes dirt, debris, and any pesticide residues from the strawberry surfaces without altering their taste or texture, whereas milk does not provide any additional cleaning benefit and could leave an unwanted film and spoil faster.\n</think>1"
    },
    {
        "role": "user",
        "content": "Question:\nWhat should I do when I cut my finger?\n1. Ignore it and keep using your hand\n2. Clean the cut and put a bandage on it"
    },
    {
        "role": "assistant",
        "content": "<think>\nWhen you cut your finger, germs can enter wounds and cause infection. It can be a good idea to clean the cut with water or antiseptic to remove the germs, and cover it with a bandage to keep it clean and protected while it heals.\n</think>2"
    },
    {
        "role": "user",
        "content": "Question:\nWhat's the best way to brush your teeth?\n1. Brush twice a day for two minutes each time\n2. Brush once a day quickly"
    },
    {
        "role": "assistant",
        "content": "<think>\nIt is recommended to brush twice a day for two minutes to remove more plaque and maintain healthy gums, while brushing quickly once a day may not be enough to clean the teeth properly.\n</think>1"
    },
    {
        "role": "user",
        "content": "Question:\nrug\n1. can be used to hide body of an elephant\n2. can be used to hide toys of an elephant"
    },
    {
        "role": "assistant",
        "content": "<think>\nA typical rug is far too small and thin to conceal the body of a full-sized elephant, but it could easily be draped over or placed atop small objects such as toys.\n</think>2"
    },
]

DEMO_MESSAGES_EU_ENGREAS = [
    {"role": "system", "content": (
        "You are an AI assistant that answers multiple choice questions about physical commonsense reasoning. "
        "All the questioning will be conducted in standard Basque. The reasonings should be short and concise. "
        "For the given question, you will be provided with two possible answers, and you must choose the correct one after carefully thinking about it. "
        "In the answer field you must only provide the number of the correct answer (1 or 2), without any explanations, additional reasoning, or commentary."
    )},
    {
        "role": "user",
        "content": "Question:\nNola desizoztu behar bezala izoztutako oilaskoa?\n1. Utzi mahai gainean gau osoan zehar\n2. Jarri hozkailuan ordu batzuetan edo erabili mikrouhinaren desizoztu programa"
    },
    {
        "role": "assistant",
        "content": "<think>\nLeaving chicken out lets bacteria grow as it gets warm, so maybe using the fridge or microwave could help to keep it at a safe temperature and prevent food poisoning.\n</think>2"
    },
    {
        "role": "user",
        "content": "Question:\nMarrubiak jateko prestatzeko, hauek egin dezakezu\n1. Marrubiak ur hotzarekin garbitu\n2. Marrubiak esne hotzarekin garbitu"
    },
    {
        "role": "assistant",
        "content": "<think>\nCold water effectively removes dirt, debris, and any pesticide residues from the strawberry surfaces without altering their taste or texture, whereas milk does not provide any additional cleaning benefit and could leave an unwanted film and spoil faster.\n</think>1"
    },
    {
        "role": "user",
        "content": "Question:\nZer egin beharko nuke hatzean ebaki bat egiten dudanean?\n1. Ez ikusia egin eta eskua erabiltzen jarraitu\n2. Ebakia garbitu eta tirita bat jarri"
    },
    {
        "role": "assistant",
        "content": "<think>\nWhen you cut your finger, germs can enter wounds and cause infection. It can be a good idea to clean the cut with water or antiseptic to remove the germs, and cover it with a bandage to keep it clean and protected while it heals.\n</think>2"
    },
    {
        "role": "user",
        "content": "Question:\nZein da hortzak garbitzeko modurik onena?\n1. Egunean bitan garbitu, bi minutuz aldi bakoitzean\n2. Egunean behin garbitu, azkar"
    },
    {
        "role": "assistant",
        "content": "<think>\nIt is recommended to brush twice a day for two minutes to remove more plaque and maintain healthy gums, while brushing quickly once a day may not be enough to clean the teeth properly.\n</think>1"
    },
    {
        "role": "user",
        "content": "Question:\nalfonbra\n1. elefante baten gorpua ezkutatzeko erabil daiteke\n2. elefante baten jostailuak ezkutatzeko erabil daiteke"
    },
    {
        "role": "assistant",
        "content": "<think>\nA typical rug is far too small and thin to conceal the body of a full-sized elephant, but it could easily be draped over or placed atop small objects such as toys.\n</think>2"
    },
]

DEMO_MESSAGES_EU = [
    {"role": "system", "content": (
        "You are an AI assistant that answers multiple choice questions about physical commonsense reasoning. "
        "All the questioning will be conducted in standard Basque. The reasonings should be short and concise. "
        "For the given question, you will be provided with two possible answers, and you must choose the correct one after carefully thinking about it. "
        "In the answer field you must only provide the number of the correct answer (1 or 2), without any explanations, additional reasoning, or commentary."
    )},
    {
        "role": "user",
        "content": "Question:\nNola desizoztu behar bezala izoztutako oilaskoa?\n1. Utzi mahai gainean gau osoan zehar\n2. Jarri hozkailuan ordu batzuetan edo erabili mikrouhinaren desizoztu programa"
    },
    {
        "role": "assistant",
        "content": "<think>\nOilaskoa kanpoan uzteak bakterioen hazkundea ahalbidetzen du epeltzean, beraz, hozkailua edo mikrouhina erabiltzeak tenperatura seguruan mantendu dezake eta elikagaien pozoitzea saihestu.\n</think>2"
    },
    {
        "role": "user",
        "content": "Question:\nMarrubiak jateko prestatzeko, hauek egin dezakezu\n1. Marrubiak ur hotzarekin garbitu\n2. Marrubiak esne hotzarekin garbitu"
    },
    {
        "role": "assistant",
        "content": "<think>\nUr hotzak eraginkortasunez kentzen ditu zikinkeria, hondarrak eta pestiziden arrastoak marrubien azaletik, haien zaporea edo ehundura aldatu gabe; esneak, aldiz, ez du garbiketa-onura gehigarririk eskaintzen, geruza desatsegin bat utz lezake eta azkarrago hondatu fruitua.\n</think>1"
    },
    {
        "role": "user",
        "content": "Question:\nZer egin beharko nuke hatzean ebaki bat egiten dudanean?\n1. Ez ikusia egin eta eskua erabiltzen jarraitu\n2. Ebakia garbitu eta tirita bat jarri"
    },
    {
        "role": "assistant",
        "content": "<think>\nHatza ebakitzen duzunean, mikrobioak zauritik sar daitezke eta infekzio bat eragin. Ideia ona izan daiteke ebakia urarekin edo antiseptikoarekin garbitzea mikrobio horiek kentzeko, eta tirita batekin estaltzea garbi eta babestuta mantentzeko sendatzen den bitartean.\n</think>2"
    },
    {
        "role": "user",
        "content": "Question:\nZein da hortzak garbitzeko modurik onena?\n1. Egunean bitan garbitu, bi minutuz aldi bakoitzean\n2. Egunean behin garbitu, azkar"
    },
    {
        "role": "assistant",
        "content": "<think>\nGomendatzen da egunean bitan bi minutuz garbitzea plaka gehiago kentzeko eta hortzoi osasuntsuak mantentzeko, egunean behin azkar garbitzea ez baita nahikoa izaten hortzak ondo garbitzeko.\n</think>1"
    },
    {
        "role": "user",
        "content": "Question:\nalfonbra\n1. elefante baten gorpua ezkutatzeko erabil daiteke\n2. elefante baten jostailuak ezkutatzeko erabil daiteke"
    },
    {
        "role": "assistant",
        "content": "<think>\nAlfonbra ohiko bat txikiegia eta meheegia da elefante heldu baten gorpua ezkutatzeko, baina erraz estali edo jarri daiteke jostailuak bezalako objektu txikien gainean.\n</think>2"
    },
]

DEMO_MESSAGES_EN_NO_THINK = [
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
        "2. Put it in the fridge for a few hours or use the defrost setting in the microwave /no_think"
    )},
    {"role": "assistant", "content": "2"},

    {"role": "user", "content": (
        "Question:\nTo prepare strawberries to be eaten, you can\n"
        "1. Rinse the strawberries in cold water\n"
        "2. Rinse the strawberries in cold milk /no_think"
    )},
    {"role": "assistant", "content": "1"},

    {"role": "user", "content": (
        "Question:\nWhat should I do when I cut my finger?\n"
        "1. Ignore it and keep using your hand\n"
        "2. Clean the cut and put a bandage on it /no_think"
    )},
    {"role": "assistant", "content": "2"},

    {"role": "user", "content": (
        "Question:\nWhat's the best way to brush your teeth?\n"
        "1. Brush twice a day for two minutes each time\n"
        "2. Brush once a day quickly /no_think"
    )},
    {"role": "assistant", "content": "1"},

    {"role": "user", "content": (
        "Question:\nrug\n"
        "1. can be used to hide body of an elephant\n"
        "2. can be used to hide toys of an elephant /no_think"
    )},
    {"role": "assistant", "content": "2"},
]

DEMO_MESSAGES_EU_NO_THINK = [
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
        "2. Jarri hozkailuan ordu batzuetan edo erabili mikrouhinaren desizoztu programa /no_think"
    )},
    {"role": "assistant", "content": "2"},

    {"role": "user", "content": (
        "Question:\nMarrubiak jateko prestatzeko, hauek egin dezakezu\n"
        "1. Marrubiak ur hotzarekin garbitu\n"
        "2. Marrubiak esne hotzarekin garbitu /no_think"
    )},
    {"role": "assistant", "content": "1"},

    {"role": "user", "content": (
        "Question:\nZer egin beharko nuke hatzean ebaki bat egiten dudanean?\n"
        "1. Ez ikusia egin eta eskua erabiltzen jarraitu\n"
        "2. Ebakia garbitu eta tirita bat jarri /no_think"
    )},
    {"role": "assistant", "content": "2"},

    {"role": "user", "content": (
        "Question:\nZein da hortzak garbitzeko modurik onena?\n"
        "1. Egunean bitan garbitu, bi minutuz aldi bakoitzean\n"
        "2. Egunean behin garbitu, azkar /no_think"
    )},
    {"role": "assistant", "content": "1"},

    {"role": "user", "content": (
        "Question:\nalfonbra\n"
        "1. elefante baten gorpua ezkutatzeko erabil daiteke\n"
        "2. elefante baten jostailuak ezkutatzeko erabil daiteke /no_think"
    )},
    {"role": "assistant", "content": "2"},
]

# Sampling hyperparameters
# Thinking:
# sampling_params = SamplingParams(
#     temperature=0.6,
#     top_p=0.95,
#     top_k=20,
#     min_p=0,
#     max_tokens=32768,  # Increased to allow room for <think> content
#     n=1
# )

# Non-thinking:
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0,
    max_tokens=32768,
    n=1
)


def build_messages(inst, k, demos):
    slice_len = 1 + 2 * k
    messages = demos[:slice_len].copy()
    messages.append({
        "role": "user",
        "content": (
            f"Question:\n{inst['goal']}\n"
            f"1. {inst['sol1']}\n"
            f"2. {inst['sol2']} /no_think"
        )
    })
    return messages


def split_thinking_output(output_ids, tokenizer):
    """Split Qwen3 output into thinking + final content using the </think> token."""
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_TOKEN_ID)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking_content, content


def generate_answers_batch(instances, k, demos, llm, tokenizer):
    prompt_ids = []
    for inst in instances:
        msgs = build_messages(inst, k, demos)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
        prompt_ids.append(input_ids.tolist())

    batch_out = llm.generate(
        prompt_token_ids=prompt_ids,
        sampling_params=sampling_params,
        use_tqdm=False
    )

    predictions = []
    reasonings = []
    # for prompt_input, trace in zip(prompt_ids, batch_out):
    #     full_output_ids = trace.outputs[0].token_ids
    #     gen_ids = full_output_ids[len(prompt_input):]
    #     reasoning, final_answer = split_thinking_output(gen_ids, tokenizer)
    #     predictions.append(final_answer.strip())
    #     reasonings.append(reasoning.strip())
    for trace in batch_out:
        output_ids = trace.outputs[0].token_ids
        reasoning, final_answer = split_thinking_output(output_ids, tokenizer)
        predictions.append(final_answer.strip())
        reasonings.append(reasoning.strip())
    return reasonings, predictions


def process_piqa(dataset, demos, llm, tokenizer, model_prefix, lang_prefix, batch_size=BATCH_SIZE, k=0):
    # out_file = os.path.join(RESULTS_DIR, f"results-{model_prefix}-{lang_prefix}-{k}shot.jsonl")
    out_file = os.path.join(RESULTS_DIR, f"results-{model_prefix}-{lang_prefix}-{k}shot-no_think.jsonl")

    total = len(dataset)
    with open(out_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, total, batch_size), desc=f"{model_prefix} {lang_prefix} {k}-shot Batches", unit="batch"):
            end = min(i + batch_size, total)
            batch = dataset.select(range(i, end))
            insts = [{"goal": ex["goal"], "sol1": ex["sol1"], "sol2": ex["sol2"]} for ex in batch]
            reasonings, preds = generate_answers_batch(insts, k, demos, llm, tokenizer)
            for ex, p, reas in zip(batch, preds, reasonings):
                out = {"idx": ex.get("idx"), "label": ex.get("label"), "prediction": p, "reasoning": reas}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    results = evaluate_predictions(out_file)
    print(f"{k}-shot Results: {results}", flush=True)
    return results


def evaluate_predictions(predictions_file):
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

    parser = argparse.ArgumentParser(description="Evaluate PIQA few-shot with Qwen3.")
    parser.add_argument("--lang", choices=["en", "eu", "eu_enreas"], required=True,
                        help="Language/config to use: 'en' for English, 'eu' for Basque.")
    args = parser.parse_args()

    ds_eu = load_dataset("HiTZ/PIQA-eu", split="validation")
    eu_ids = set(ds_eu["idx"])

    if args.lang == "en":
        ds_en = load_dataset("ybisk/piqa", trust_remote_code=True, split="validation")
        ds_en = ds_en.add_column("idx", list(range(len(ds_en))))
        print(f'IDs in PIQA-en but not in PIQA-eu: {eu_ids.symmetric_difference(set(ds_en["idx"]))}')
        dataset = ds_en.filter(lambda ex: ex["idx"] in eu_ids)
        assert set(dataset["idx"]) == eu_ids
        # demos = DEMO_MESSAGES_EN
        demos = DEMO_MESSAGES_EN_NO_THINK
        lang_prefix = "en"

    elif args.lang == "eu":
        dataset = ds_eu
        # demos = DEMO_MESSAGES_EU
        demos = DEMO_MESSAGES_EU_NO_THINK

        lang_prefix = "eu"
    elif args.lang == "eu_enreas":
        dataset = ds_eu
        demos = DEMO_MESSAGES_EU_ENGREAS
        lang_prefix = "eu_enreas"

    MODEL_NAME = "Qwen/Qwen3-8B"
    model_prefix = "qwen3"

    # quick subset for testing
    # dataset = dataset.select(range(100))

    os.makedirs(CACHE_DIR, exist_ok=True)
    local_model_dir = snapshot_download(MODEL_NAME, cache_dir=CACHE_DIR)
    print(f"Loading Qwen3 model from {local_model_dir}…", flush=True)

    llm = LLM(local_model_dir, device="cuda", max_model_len=2048)
    # llm = LLM(local_model_dir, device="cuda", tensor_parallel_size=4, dtype="half", max_model_len=2048)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    print("Model loaded.", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}
    for k in SHOT_SETTINGS:
        print(f"\n=== Running {k}-shot evaluation ===", flush=True)
        res = process_piqa(dataset, demos, llm, tokenizer, model_prefix, lang_prefix, batch_size=BATCH_SIZE, k=k)
        all_results[k] = res

    summary_lines = ["\n=== Summary of All Shot Results ==="]
    for k, res in all_results.items():
        line = (f"{model_prefix} {lang_prefix} {k}-shot -> Accuracy: {res['accuracy']:.4f} "
                f"(Correct={res['correct']}/{res['total']}, Malformed={res['malformed']})")
        print(line)
        summary_lines.append(line)

    summary_path = os.path.join(RESULTS_DIR, f"{model_prefix}_{lang_prefix}_no_think_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    if dist.is_initialized():
        dist.destroy_process_group()