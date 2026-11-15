import os
import json
import torch
import numpy as np
from openai import OpenAI
import argparse
import sys
import time
import torch.nn as nn
import re
from utils_demo import load_demo_model
from utils.tokenizer import SimpleTokenizer

MY_API_KEY = "sk-o"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'XXX')
LOG_FILE = 'XXX'
CANDIDATES_FILE = 'XXX'

client = OpenAI(
    api_key=MY_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

def encode_text_baseline_manual(clip_model, text):
    if text.dim() == 1:
        text = text.unsqueeze(0)

    cast_dtype = clip_model.transformer.get_cast_dtype()

    x = clip_model.token_embedding(text).to(cast_dtype)
    x = x + clip_model.positional_embedding.to(cast_dtype)

    mask = clip_model.attn_mask
    if mask is not None and mask.device != x.device:
        mask = mask.to(x.device)

    x = clip_model.transformer(x, attn_mask=mask)

    x = clip_model.ln_final(x)

    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

    if clip_model.text_projection is not None:
        if isinstance(clip_model.text_projection, nn.Linear):
            x = clip_model.text_projection(x)
        else:
            x = x @ clip_model.text_projection

    return x

class MRPDAgent:
    def __init__(self, device='cuda:0'):
        print("\n>>> [System] Initializing robust 3D perception agent (comparison mode)...")

        base_path = os.path.join(DB_PATH, 'XXX')
        mrpd_path = os.path.join(DB_PATH, 'XXX')
        meta_path = os.path.join(DB_PATH, 'XXX')

        if not os.path.exists(base_path):
            raise FileNotFoundError("Please rebuild the feature databases.")

        self.db_base = torch.tensor(np.load(base_path)).to(device)
        self.db_mrpd = torch.tensor(np.load(mrpd_path)).to(device)

        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)

        self.paths = self.metadata['paths']
        self.labels = self.metadata['labels']
        self.classnames = self.metadata['classnames']

        self.candidates = {}
        if os.path.exists(CANDIDATES_FILE):
            with open(CANDIDATES_FILE, 'r') as f:
                c_list = json.load(f)
                for item in c_list:
                    if item['true_class'] not in self.candidates:
                        self.candidates[item['true_class']] = item
            print(f">>> [System] Loaded demo candidates: covering {len(self.candidates)} categories")
        else:
            print(f">>> [Warning] Not found {CANDIDATES_FILE}, will use normal search mode.")

        parser = argparse.ArgumentParser()
        parser.add_argument("--embed_dim", type=int, default=1280)
        parser.add_argument("--pc_encoder_dim", type=int, default=512)
        parser.add_argument("--evaluate_3d", action="store_true", default=True)
        parser.add_argument("--num_group", type=int, default=512)
        parser.add_argument("--group_size", type=int, default=64)
        args, _ = parser.parse_known_args()

        self.model, _ = load_demo_model(args, device)
        self.tokenizer = SimpleTokenizer()
        self.device = device

        try:
            self.logit_scale = self.model.clip_model_text.logit_scale.exp().item()
        except:
            self.logit_scale = 100.0

        print(">>> [System] Pre-computing full category text features (for 4 zero-shot experiments)...")
        self.text_feats_base_all = self._precompute_all_text_feats(use_prompt=False)
        self.text_feats_mrpd_all = self._precompute_all_text_feats(use_prompt=True)

        print(f">>> [System] Ready. Logit Scale: {self.logit_scale:.2f}")

    def _precompute_all_text_feats(self, use_prompt=False):
        if use_prompt:
            with torch.no_grad():
                feats = self.model.get_text_feat()
        else:
            with torch.no_grad():
                prompts = [f"A point cloud of a {c}." for c in self.classnames]
                tokens = self.tokenizer(prompts).to(self.device)
                feats = encode_text_baseline_manual(self.model.clip_model_text, tokens)

        return feats / feats.norm(dim=-1, keepdim=True)

    def llm_parse_intent(self, user_query):
        print(f"    ...thinking (LLM Parsing)...")

        class_list_str = ", ".join(self.classnames)
        system_prompt = f"""
You are a helpful assistant for a 3D point cloud retrieval system.
Your task is to extract the target object class from the user's instruction.
You MUST map the user's request to one of the following valid class names:
[{class_list_str}]

If the user asks for "car", you should map it to "car" (or the closest match in the list).
If the user asks for something not in the list or irrelevant, return "None".
Return ONLY a JSON object in this format: {{"class": "detected_class_name"}}
Do not output any other text or explanation.
"""

        try:
            response = client.chat.completions.create(
                model="z-ai/glm-4.5-air:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()

            if "```" in content:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if match:
                    content = match.group(1)

            try:
                result = json.loads(content)
                parsed_class = result.get("class", "None")
            except json.JSONDecodeError:
                print(f"    [Warning] LLM JSON parse failed: {content}")
                return "None"

            if parsed_class not in self.classnames and parsed_class != "None":
                print(f"    [Warning] LLM returned invalid class '{parsed_class}' (not in database)")
                return "None"

            return parsed_class

        except Exception as e:
            print(f"    [Error] LLM Call Failed: {e}")
            return "None"

    def run_comparison(self, category):
        if category not in self.classnames:
            return None, "Category not found"

        target_sample = self.candidates.get(category)

        if target_sample:
            idx = target_sample['data_index']
            print(f"\n>>> [Demo] Index: {idx} (Target: {category})")

            pc_feat_base = self.db_base[idx].unsqueeze(0)
            pc_feat_mrpd = self.db_mrpd[idx].unsqueeze(0)

            file_path = self.paths[idx]

        else:
            print(f"\n>>> [Search] Searching full database for the best match '{category}' sample...")

            text_feat_base_one = self._precompute_all_text_feats(use_prompt=False)[self.classnames.index(category)].unsqueeze(0)
            sim = torch.mm(self.db_base, text_feat_base_one.t())
            idx = sim.argmax().item()

            pc_feat_base = self.db_base[idx].unsqueeze(0)
            pc_feat_mrpd = self.db_mrpd[idx].unsqueeze(0)

            file_path = self.paths[idx]


        experiments = [
            {"name": "1. Baseline (No Prompt)", "pc_feat": pc_feat_base, "text_feats": self.text_feats_base_all},
            {"name": "2. Point Prompt Only",    "pc_feat": pc_feat_mrpd, "text_feats": self.text_feats_base_all},
            {"name": "3. Text Prompt Only",     "pc_feat": pc_feat_base, "text_feats": self.text_feats_mrpd_all},
            {"name": "4. MRPD (Full Ours)",     "pc_feat": pc_feat_mrpd, "text_feats": self.text_feats_mrpd_all},
        ]

        print(f"\n{'='*20} Perception Results: '{category}' {'='*20}")
        print(f"{'Experiment Setup':<25} | {'Predicted':<15} | {'Conf':<10} | {'Status'}")
        print("-" * 68)

        results_summary = []

        for exp in experiments:
            logits = torch.mm(exp["pc_feat"], exp["text_feats"].t()) * self.logit_scale
            probs = logits.softmax(dim=-1)

            val, pred_idx = torch.topk(probs, 1)
            score = val.item()
            pred_class = self.classnames[pred_idx.item()]

            is_correct = (pred_class == category)

            if is_correct:
                status = "Success"
            else:
                status = f"-> {pred_class}"

            print(f"{exp['name']:<25} | {pred_class:<15} | {score:.4f}     | {status}")

            results_summary.append({
                "setup": exp['name'],
                "score": score,
                "prediction": pred_class,
                "correct": is_correct
            })

        print("=" * 68 + "\n")
        return None, results_summary

def main():
    agent = MRPDAgent()

    history = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f: history = json.load(f)
        except: pass

    print("\n" + "="*50)
    print(" 3D Perception Chat Demo (type 'exit' to quit)")
    print("="*50)

    while True:
        try:
            print("\n[User]: ", end='', flush=True)
            try:
                raw_input = sys.stdin.buffer.readline()
                user_input = raw_input.decode('utf-8', errors='ignore').strip()
            except: user_input = ""

            if not user_input: continue
            if user_input.lower() in ['exit', 'quit']:
                with open(LOG_FILE, 'w') as f:
                    json.dump(history, f, indent=4)
                print("\n>>> Saving conversation history to", LOG_FILE)
                break

            parsed_class = agent.llm_parse_intent(user_input)
            print(f" -> Recognized intent: [{parsed_class}]")

            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "user_input": user_input,
                "parsed_intent": parsed_class,
                "results": None
            }

            if parsed_class != "None":
                _, results = agent.run_comparison(parsed_class)
                log_entry["results"] = results
            else:
                print(" -> Cannot recognize category or category not in database.")

            history.append(log_entry)
            with open(LOG_FILE, 'w') as f:
                json.dump(history, f, indent=4)

        except KeyboardInterrupt:
            with open(LOG_FILE, 'w') as f:
                json.dump(history, f, indent=4)
            print("\n>>> Saving conversation history to", LOG_FILE)
            print("\n>>> Interrupted. Bye!")
            break

if __name__ == "__main__":
    main()
