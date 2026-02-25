# CS 5744 — Assignment 1 (Perceptron + MLP)

**Name:** Zexun Yao  

This repository contains two classifiers for text classification:
1) A **Perceptron** (stdlib-only) with multiple feature templates  
2) A **Multi-Layer Perceptron (MLP)** in **PyTorch** using an embedding + mean-pooling (NBOW-style) encoder

Supported datasets:
- `sst2`
- `newsgroups`

---

## Repository Structure

- `perceptron.py` — Perceptron training / evaluation / test prediction
- `features.py` — Feature functions (BoW + custom features)
- `multilayer_perceptron.py` — PyTorch MLP training / evaluation / test prediction + optional batching benchmark
- `stopwords.txt` — Stopword list used by some features/tokenizer
- `results/` — Output predictions and any optional plots

---

### Python
- Tested with **Python 3.10+** (should work with 3.9+).

### Perceptron (Model 1)
- **No third-party packages required.**
- Uses only **Python standard library** + the course starter utilities:
  - `utils.py` (course-provided)
- No NumPy / SciPy / sklearn / pandas / torch needed for the perceptron.

### MLP (Model 2)
- Requires **PyTorch**:
  - `torch`
  - `torch.utils.data`
  - `torch.nn`

**If you are using Apple Silicon:**
- Use `--device mps` (requires a PyTorch build with MPS support).

**If you are using NVIDIA GPU:**
- Use `--device cuda` (requires a CUDA-enabled PyTorch install).

To comply:
- Stopwords should be loaded via standard Python file I/O (`open(...)`) instead of pandas.
- Progress bars should be removed (plain `for batch in dataloader:` loop).
- Plots should be omitted or generated outside the required training script.

# Running the Perceptron
    Train + Dev Predictions + Test Predictions
        python perceptron.py -d sst2 -f bow -e 3 -l 0.1

        Common feature combinations:

        bow (baseline)
        bow+len
        bow+polarity
        bow+punct
        bow+neg
        bow+shape
        bow_ng (unigram + bigram)

Example (newsgroups with multiple features):
    python perceptron.py -d newsgroups -f bow+shape+len -e 5 -l 0.1

Outputs (Perceptron)
    Dev predictions:
        results/perceptron_<DATA>_<FEATURES>_dev_predictions.csv

    Test predictions:
        results/perceptron_<DATA>_test_predictions.csv

    Model weights:
        results/perceptron_<DATA>_<FEATURES>_model.json

#Running the MLP (PyTorch)
    Train + Dev Accuracy + Test Predictions
        python multilayer_perceptron.py -d sst2 -e 10 -l 0.001 -a relu --device cpu
    On Apple Silicon (MPS):
        python multilayer_perceptron.py -d sst2 -e 10 -l 0.001 -a relu --device mps
    On CUDA (if available):
        python multilayer_perceptron.py -d newsgroups -e 10 -l 0.001 -a relu --device cuda

    Activation Function

        You can select the activation function with --activation (or -a):
        Supported options:
            relu
            tanh
            sigmoid
        Example:
            python multilayer_perceptron.py -d sst2 -e 10 -l 0.001 --activation tanh --device cpu


    Outputs (MLP)

        Test predictions:
        results/mlp_<DATA>_test_predictions.csv

        Training loss plot:
        results/mlp_<DATA>_<ACTIVATION>_train_loss.png

        Validation accuracy plot:
        results/mlp_<DATA>_<ACTIVATION>_val_accuracy.png 

#Batching / Inference Benchmark (MLP)

The code includes an optional benchmark to measure inference time per 1,000 examples for different batch sizes, repeated multiple times.

Example:
    python multilayer_perceptron.py -d sst2 -e 5 -l 0.001 -a relu --device mps --benchmark --repeats 10 --num_examples 1000

It prints a table:

    Batch size
    Mean milliseconds per 1,000 examples
    Std dev milliseconds per 1,000 examples

#Reproducibility
    Random seeds are set in both scripts (Python + Torch) for reproducibility.
    Perceptron uses deterministic shuffling with random.seed(0).
    MLP sets random.seed(0) and torch.manual_seed(0).

#Notes on Features (Perceptron)

    Implemented feature templates in features.py:
    bow: bag-of-words (stopword filtered)
    bow_ng: unigram + bigram
    len: sentence length bucket
    polarity: small sentiment lexicon flags
    punct: punctuation + all-caps emphasis cues
    neg: negation scope features (windowed)
    shape: URL/email/header/digit/caps/length fraction cues (useful for newsgroups)
    Use -f to combine: bow+shape+len, etc.

# Commands

## Virtual environment creation

It's highly recommended to use a virtual environment for each assignment.
You may use environment manager like uv, conda, venv etc.
Here is how you can create an environment with uv and install dependencies.

To install uv
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To download required dependencies using uv
```sh
cd a1/
uv sync
```

To run python programs using the new environment
```sh
source .venv/bin/activate
python perceptron.py
```

## Train and predict commands

Example command for the original code (subject to change, if additional arguments are added):

```sh
python perceptron.py -d newsgroups -f bow
python perceptron.py -d sst2 -f bow
python multilayer_perceptron.py -d newsgroups
```

## Commands to run unittests

It's recommended to ensure that your code passes the unittests before submitting it.
The commands can be run from the root directory of the project.

```sh
pytest
pytest tests/test_perceptron.py
pytest tests/test_multilayer_perceptron.py
```

Please do NOT commit any code that changes the following files and directories:

- tests/
- .github/
- pytest.ini

Otherwise, your submission may be flagged by GitHub Classroom autograder.

Please DO commit your output labels in results/ following the same name and content format.
Our leaderboard periodically pulls your outputs and computes accuracy against hidden test labels.
The leaderboard is available here: <https://github.com/Cornell-Tech-CS5744-Spring-2026/leaderboards/>.
