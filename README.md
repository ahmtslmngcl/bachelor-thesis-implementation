# Bachelor-Thesis-Implementation

This repository contains the implementation of my Bachelor Thesis. It consists of a refactored and extended implementation of the **KEDKG (Knowledge Editing via Dynamic Graphs)** framework. The framework has been adapted to support both the **MQuAKE** and **RippleEdits** benchmarks for evaluating LLM knowledge editing.

# Project Run Guide

## 1. Environment Setup

```
git clone https://github.com/ahmtslmngcl/bachelor-thesis-implementation.git
conda create -n Kedkg python=3.9
conda activate Kedkg
pip install -r requirements.txt
cd external/kedkg
python -m spacy_entity_linker "download_knowledge_base"
```

## 2. Repository Structure

```
.
├── external				# External KEDKG & RippleEdits repositories
│   ├── kedkg
│   │   ├── ...
│   │   ├── model
│   │   │   ├── distilbert-base-cased	# Download from HuggingFace
│   │   │   └── rebel-large		# Download from HuggingFace
│   │   ├── ...
│   │   └── train
│   │       ├── ...
│   │       ├── results			# Download from Google Drive
│   │       │   └── best_model
│   │       ├── results_entity_judge	# Download from Google Drive
│   │       │   └── best_model_entity_judge
│   │       └── ...
│   └── rippleedits
│       └── ...
├── results				# Output folder for experiment logs and results
├── src				 	# Core source files for this implementation
│   ├── benchmarks
│   │   ├── mquake_runner.py		# Runner for MQuAKE evaluation
│   │   └── ripple_runner.py		# Runner for RippleEdits evaluation
│   ├── configs				# YAML configuration files
│   │   ├── mquake.yaml
│   │   └── ripple.yaml
│   ├── prompts				# Prompt templates
│   │   └── v0
│   │       ├── answer.txt
│   │       ├── divide.txt
│   │       └── test.txt
│   ├── kedkg_v0.py			# Corresponding KEDKG versions
│   ├── ...
│   └── kedkg_vX.py
├── requirements.txt
└── README.md
```

## 3. Config Files

The configuration files for experiments are located under `src/configs`. You should review and adjust the configuration parameters according to your experiment setup.

> **Important:** If you are running on local LLMs, make sure the server is accessible at the URL specified in the config (usually http://127.0.0.1:11434/v1)

## 4. Running Experiments

From the repository root, run the following commands:

#### Run on the MQuAKE benchmark

```
python -m src.benchmarks.mquake_runner --config src/configs/mquake.yaml
```

#### Run on the RippleEdits benchmark

```
python -m src.benchmarks.ripple_runner --config src/configs/ripple.yaml
```

Results and detailed logs will be saved under `results/`

# External Dependencies & Adjustments

This repository includes the [KEDKG](https://github.com/ABi-dot/Kedkg) and [RippleEdits](https://github.com/edenbiran/RippleEdits) repositories under `external/`.
All code of `external/` remains consistent with the original repositories, except:

* **`external/rippleedits/src/wikidata/utils.py`**
  Updated `get_label()` and `get_aliases()` to use API-based retrieval for Wikidata entities.
* **`external/rippleedits/src/testrunner.py`**
  Added optional `extra_prints` for enhanced interpretability during evaluation runs.

The repositories were last accessed on 05.11.2025, 7 pm

The following components must be manually downloaded and placed inside the project as printed above before running experiments (see [KEDKG](https://github.com/ABi-dot/Kedkg)):

| Component                                | Source                                                                                            | Destination              |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------ |
| **distilbert-base-cased**          | [Hugging Face](https://huggingface.co/distilbert/distilbert-base-cased)                              | `external/kedkg/model` |
| **rebel-large**                    | [Hugging Face](https://huggingface.co/Babelscape/rebel-large)                                        | `external/kedkg/model` |
| **entity & relation judge models** | [Google Drive](https://drive.google.com/drive/folders/14xr7ruFZdmqCJ6_thbgirmTIVeP1QWHk?usp=sharing) | `external/kedkg/train` |
