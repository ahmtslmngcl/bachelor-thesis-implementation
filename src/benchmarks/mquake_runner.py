import os
import sys
import json
import random
import time
import collections
from tqdm import tqdm
import importlib
from collections import defaultdict
import time
import importlib
from datetime import datetime
import io
import yaml
import argparse



def run_mquake_benchmark(cfg: dict):

    version      = int(cfg.get("version"))
    model_name   = cfg.get("model_name") # i.e. which lm
    editor_name  = f"kedkg_v{version}"
    llm_cfg      = cfg.get("llm_cfg")
    dataset_name = cfg.get("dataset_name")
    dataset_file = f"{dataset_name}.json"
    batch_edits  = int(cfg.get("num_of_edits"))
    extra_prints = bool(cfg.get("extra_prints"))

    REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    KEDKG_ROOT   = os.path.join(REPO_ROOT, "external", "kedkg")
    KEDKG_SRC    = os.path.join(KEDKG_ROOT, "src")
    DATASET_PATH = os.path.join(KEDKG_ROOT, "datasets", dataset_file)
    RESULTS_DIR  = os.path.join(REPO_ROOT, "results", "mquake")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if KEDKG_ROOT not in sys.path:
        sys.path.insert(0, KEDKG_ROOT)

    if KEDKG_SRC not in sys.path:
        sys.path.insert(0, KEDKG_SRC)
    
    with open(DATASET_PATH, "r") as input:
        data = json.load(input)

    #os.environ.setdefault("KEDKG_ROOT", KEDKG_ROOT)

    kedkg_mod = importlib.import_module(f"src.kedkg_v{version}")
    kedkg = kedkg_mod.KEDKG(model_name, editor_name, llm_cfg, extra_prints)
    kedkg.prepare(cfg)

    #### Writing preparation ####
    experiment_name = f"kedkg_v{version}_{model_name}_{dataset_name}_mquake"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(RESULTS_DIR, f"{experiment_name}.log")
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    log_capture = io.StringIO()
    sys_stdout_original = sys.stdout
    if extra_prints:
        sys.stdout = Tee(sys_stdout_original, log_capture)

    print(f"=== RUNNING MQuAKE BENCHMARK: {experiment_name} ===")
    print(f"Timestamp: {timestamp}\nConfig:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("-" * 60)

    cor = tot = 0
    ver_cor = 0
    cor_list = [0,0,0]
    ver_cor_list = [0,0,0]
    tot_list = [0,0,0]

    random.shuffle(data)

    batch_size = 1868 // batch_edits
    batch_data = []
    idx = 0
    for _ in range(batch_size):
        b = []
        for i in range(batch_edits):
            if idx < len(data):
                b.append(data[idx])
            idx += 1
        batch_data.append(b)

    for batch in batch_data:
        """
        kedkg.qid2name = collections.defaultdict(str)
        kedkg.triples = collections.defaultdict(set)
        kedkg.qid_relations = collections.defaultdict(set)
        """
        kedkg.kedkg_restore_kb()
        # construct knowledge graph
        kedkg.construct_graph(batch, kedkg.qid2name, kedkg.triples, kedkg.qid_relations, kedkg_mod.nlp)

        for i in tqdm(range(len(batch))):
            d = batch[i]
            tot += 1
            tot_list[len(d["single_hops"]) - 2] += 1
            # modify knowledge graph
            kedkg.modify_graph(d, kedkg.qid2name, kedkg.triples, kedkg.qid_relations, kedkg_mod.nlp)

            found_ans = gold_path = False
            entity_list = []
            for q in d['questions']:
                entity_list = kedkg.kedkg_answer_question(q, return_all=True)
                ans = entity_list[-1]
                
                # if the  answer is correct -> positive instance for Acc
                if ans.lower() == d["new_answer"].lower() or d["new_answer"].lower() in ans.lower() or any(ans.lower() in k.lower() for k in d["new_answer_alias"]):
                    if not found_ans:
                        cor += 1
                        cor_list[len(d["new_single_hops"]) - 2] += 1
                        found_ans = True

                    
                    gold_path = kedkg.verify_gold_path(entity_list, len(d["single_hops"]), d["new_single_hops"])

                    if gold_path:
                        ver_cor += 1
                        ver_cor_list[len(d["single_hops"]) - 2] += 1
                        break

                
            """
            print(f"Total: {total}, Correct: {correct},  Total: {correct / total}")

            print(f'Acc = {cor / tot} ({cor} / {tot})')
            print(f'Hop-Acc = {ver_cor / tot} ({ver_cor} / {tot})')
            print(f'2-hop-tot = {tot_list[0]} 2-hop-cor = {cor_list[0]} 2-hop-acc = {ver_cor_list[0]}')
            print(f'3-hop-tot = {tot_list[1]} 3-hop-cor = {cor_list[1]} 3-hop-acc = {ver_cor_list[1]}')
            print(f'4-hop-tot = {tot_list[2]} 4-hop-cor = {cor_list[2]} 4-hop-acc = {ver_cor_list[2]}')
            """

    def safe_div(n, d): return 0.0 if d == 0 else n / d

    results = {
        "M-Acc": safe_div(cor, tot),
        "H-Acc": safe_div(ver_cor, tot),
        "2-hop": {"M-Acc": safe_div(cor_list[0], tot_list[0]), "H-Acc": safe_div(ver_cor_list[0], tot_list[0])},
        "3-hop": {"M-Acc": safe_div(cor_list[1], tot_list[1]), "H-Acc": safe_div(ver_cor_list[1], tot_list[1])},
        "4-hop": {"M-Acc": safe_div(cor_list[2], tot_list[2]), "H-Acc": safe_div(ver_cor_list[2], tot_list[2])},
    }

    print("\n================ FINAL RESULTS ================")
    print(json.dumps(results, indent=2))
    print("==============================================")

    #### Write results ####
    results_dict = {
        "timestamp": timestamp,
        "config": cfg,
        "results": results
    }
    json_path = os.path.join(RESULTS_DIR, f"{experiment_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    #### Write run log ####
    if extra_prints:
        sys.stdout = sys_stdout_original
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(log_capture.getvalue())

    print(f"Saved results to: {RESULTS_DIR}")
        
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/mquake.yaml",
        help="path to config file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    """
    ex_config = {
        "version" : "0",
        "model_name" : "qwen3:4b-instruct",
        "dataset_name" : "MQuAKE-CF-3k",
        "num_of_edits" : 2,
        "extra_prints" : True,
        "llm_cfg": {
            "answer_model" : "qwen3:4b-instruct", # same as model_name
            "divide_model" : "qwen3:4b-instruct",
            "base_url"     : "http://127.0.0.1:11434/v1",
            "api_key"      : "dummy"
        },
        "prompt_cfg": {
            "divide_prompt": "v0/divide.txt",
            "answer_prompt": "v0/answer.txt"
        }
    }
    """

    run_mquake_benchmark(config)

