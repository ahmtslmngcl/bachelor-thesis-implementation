import os
import sys
import importlib
from collections import defaultdict
from datetime import datetime
import json
import io
import argparse
import yaml 

def run_ripple_benchmark(cfg: dict):

    version      = int(cfg.get("version"))
    model_name   = cfg.get("model_name") # which llm is being edited or doing the QA
    editor_name  = f"kedkg_v{version}"
    llm_cfg = cfg.get("llm_cfg")
    dataset_name = cfg.get("dataset_name")
    dataset_file = f"{dataset_name}.json"
    num_of_examples = cfg.get("num_examples", 3)
    extra_prints = bool(cfg.get("extra_prints"))

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    RIPPLE_ROOT = os.path.join(REPO_ROOT, "external", "rippleedits")
    RIPPLE_SRC  = os.path.join(RIPPLE_ROOT, "src")
    DATASET_PATH = os.path.join(RIPPLE_ROOT, "data", "benchmark", dataset_file)
    RESULTS_DIR = os.path.join(REPO_ROOT, "results", "ripple")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if RIPPLE_ROOT not in sys.path:
        sys.path.insert(0, RIPPLE_ROOT)

    if RIPPLE_SRC not in sys.path:
        sys.path.insert(0, RIPPLE_SRC)

    os.environ.setdefault("RIPPLEEDIT_ROOT", RIPPLE_ROOT)

    from benchmark import Dataset, TestsAxis
    from evaluation import Evaluator
    from wikidata.utils import write_json

    kedkg_mod = importlib.import_module(f"src.kedkg_v{version}")
    kedkg = kedkg_mod.KEDKG(model_name, editor_name, llm_cfg, extra_prints)
    kedkg.prepare(cfg)

    query_executor, model_editor = kedkg.make_ripple_adapters()

    if dataset_name == "recent":
        dataset_name = 'recently_modified'
    if dataset_name == "random":
        dataset_name = 'fake_facts'
    if dataset_name == "popular":
        dataset_name = 'top_views'

    #### Writing preparation ####
    experiment_name = f"kedkg_v{version}_{model_name}_{dataset_name}_rippleedits"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    log_path = os.path.join(RESULTS_DIR, f"{experiment_name}_log.txt")
    log_capture = io.StringIO()
    sys_stdout_original = sys.stdout
    sys.stdout = Tee(sys_stdout_original, log_capture)

    print(f"=== RippleEdits Run Log ===")
    print(f"Timestamp: {timestamp}")
    print("Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("\nRunning evaluation...\n")

    evaluator = Evaluator(query_executor=query_executor, model_editor=model_editor)
    dataset = Dataset.from_file(DATASET_PATH)

    precisions_json = dict()

    examples_for_eval = dataset.sample(num_of_examples)
    eval_size = len(examples_for_eval)

    succeeded_edits = defaultdict(lambda: 0)
    average_precision = defaultdict(lambda: 0)
    average_executed = defaultdict(lambda: 0)
    average_size = defaultdict(lambda: 0)
    total_checked_examples = defaultdict(lambda: 0)
    executed_portion_dict = defaultdict(lambda: 0)

    for i, example in enumerate(examples_for_eval):
        if (i + 1) % 10 == 0:
            print(f'{i + 1}/{eval_size}')

        if example.fact.get_subject_label() == '' or example.fact.get_target_label() == '':
            print(f'Skipping example: {example.to_dict()}')
            continue

        evaluation_results = evaluator.evaluate(example)

        res_dict_for_json = dict()
        for axis, results in evaluation_results.items():
            precision, executed, size, edit_succeeded = results
            if executed == 0.0:
                continue
            if edit_succeeded:
                succeeded_edits[axis] += 1
                average_precision[axis] += precision
                res_dict_for_json[axis.name] = precision
                average_executed[axis] += executed
                average_size[axis] += size
                # precisions_json[str(example.fact)] = precision
            total_checked_examples[axis] += 1

        precisions_json[str(example.fact)] = res_dict_for_json

        for axis in TestsAxis:
            if axis in evaluation_results:
                executed_portion_dict[axis] += evaluation_results[axis][1]

    res_str = ''
    for axis in TestsAxis:
        print(f'Results of axis {axis}:')
        res_str += f'Results of axis {axis}:\n'

        if total_checked_examples[axis] == 0:
            print(f'No checked tests for this axis')
            res_str += f'No checked tests for this axis\n'
            continue
        
        if succeeded_edits[axis] != 0:
            average_precision[axis] /= succeeded_edits[axis]
            average_executed[axis] /= succeeded_edits[axis]
            average_size[axis] /= succeeded_edits[axis]
        else:
            average_precision[axis] = 0
            average_executed[axis] = 0 
            average_size[axis] = 0


        print(f'{(succeeded_edits[axis] / eval_size) * 100} successful edits (out of {eval_size})')
        res_str += f'{(succeeded_edits[axis] / eval_size) * 100} successful edits (out of {eval_size})\n'
        print(f'Average accuracy is {average_precision[axis]}')
        res_str += f'Average accuracy is {average_precision[axis]}\n'
        print(f'Average portion of executed_tests is {average_executed[axis]}')
        res_str += f'Average portion of executed_tests is {average_executed[axis]}\n'
        print(f'Average total number of tests is {average_size[axis]}')
        res_str += f'Average total number of tests is {average_size[axis]}\n'

    #### Write results ####
    json_output = {
        "timestamp": timestamp,
        "config": cfg,
        "results": precisions_json
    }
    write_json(json_output, os.path.join(RESULTS_DIR, f"{experiment_name}_res_2.json"))

    txt_path = os.path.join(RESULTS_DIR, f"{experiment_name}_2.txt")
    with open(txt_path, "w+", encoding="utf-8") as f:
        f.write("=== RippleEdits Benchmark ===\n")
        f.write(f"Timestamp: {timestamp}\n\nConfig:\n")
        for k, v in cfg.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n------------------------------------\n\n")
        f.write(res_str)

    #### Write run log ####
    if extra_prints:
        sys.stdout = sys_stdout_original
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(log_capture.getvalue())

    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/ripple.yaml",
        help="path to config file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    """
    ex_config = {
        "version" : "0",
        "model_name" : "qwen3:4b-instruct",
        "dataset_name" : "popular",
        "num_examples" : 1,
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

    run_ripple_benchmark(config)