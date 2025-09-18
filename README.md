# MuTIS: Multi-Turn Intervention Sampling for Reinforcement Learning

MuTIS (Multi-Turn Intervention Sampling) is a reinforcement learning framework that enhances reasoning efficiency through multi-turn interactions. Built on top of veRL (Volcano Engine Reinforcement Learning) and Search-R1, MuTIS enables training language models with improved reasoning capabilities through structured multi-turn dialogue and intervention sampling.

## Features

- **Multi-Turn Reasoning**: Supports multi-turn conversations with configurable maximum turns
- **PPO Training**: Implements Proximal Policy Optimization for language model training
- **Flexible Data Processing**: Handles various mathematical reasoning datasets (LIMO, R1-220K)
- **Comprehensive Evaluation**: Built-in evaluation tools for model performance assessment

## Architecture

The project consists of several key components:

- **`verl/`**: Core reinforcement learning framework built on veRL
  - `trainer/`: Training modules including PPO trainer and evaluation scripts
  - `models/`: Model implementations with support for Llama and other architectures
  - `single_controller/`: Single-node training controller
- **`mutis/`**: MuTIS-specific implementations
  - `llm_agent/`: LLM agent for multi-turn generation and interaction
- **`scripts/`**: Data processing and utility scripts

## Installation

### Prerequisites
- Python 3.9+
- CUDA 12.1+ (for GPU support)
- Conda or similar environment manager

### Environment Setup

```bash
# Create and activate conda environment
conda create -n mutis python=3.9
conda activate mutis

# Install PyTorch (optional - vllm will install compatible version)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM for efficient inference
pip3 install vllm==0.6.3 

# Install the package in development mode
pip install -e .

# Install Flash Attention for memory-efficient attention
pip3 install flash-attn --no-build-isolation

# Install Weights & Biases for experiment tracking
pip install wandb
```

### Additional Dependencies

Install additional requirements:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Process your training data using the provided scripts:

```bash
# Process LIMO dataset
python scripts/data_process/limo_process.py \
  --input_file data/limo.jsonl \
  --local_dir data/processed_limo \
  --template_type base

```

### 2. Training

Configure your training parameters and run:

```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='data/processed_limo'
export BASE_MODEL='agentica-org/DeepScaleR-1.5B-Preview'
export EXPERIMENT_NAME='mutis_experiment'
export WANDB_API_KEY='your_api_key'

# Run training
bash training_example.sh
```

Or use the Python training script directly:

```bash
python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    actor_rollout_ref.model.path=$BASE_MODEL \
    trainer.experiment_name=$EXPERIMENT_NAME \
    max_turns=3
```

### 3. Evaluation

Evaluate trained models:

```bash
python3 -m verl.trainer.main_eval \
    --config-path config \
    --config-name evaluation \
    data.path=path/to/generated_results.parquet
```

## Configuration

### Key Training Parameters

- **`max_turns`**: Maximum number of turns in multi-turn reasoning (default: 3)
- **`data.max_prompt_length`**: Maximum input prompt length (default: 4096)
- **`data.max_response_length`**: Maximum response length (default: 1000)
- **`actor_rollout_ref.actor.optim.lr`**: Actor learning rate (default: 1e-6)
- **`critic.optim.lr`**: Critic learning rate (default: 1e-5)

### Model Configuration

The framework supports various model architectures through the `verl.models` registry:
- Llama family models with Megatron parallelization
- Custom model implementations via model registry

### Data Format

Training data should be in Parquet format with the following columns:
- `prompt`: Input question or problem
- `response`: Expected response or ground truth
- `data_source`: Dataset identifier (e.g., 'limo', 'r1_220k')

## Project Structure

```
MuTIS/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── pyproject.toml           # Modern Python packaging
├── training_example.sh      # Example training script
├── mutis/                   # MuTIS-specific code
│   └── llm_agent/          # Multi-turn generation logic
├── verl/                    # Core RL framework
│   ├── trainer/            # Training modules
│   ├── models/             # Model implementations
│   ├── single_controller/  # Training controller
│   └── utils/              # Utility functions
└── scripts/                # Data processing scripts
    └── data_process/       # Dataset preprocessing
```

## Advanced Usage

### Multi-Turn Generation

The core multi-turn generation logic is implemented in `mutis/llm_agent/generation.py`. Key features:

- **Timeout Handling**: Robust timeout mechanisms for long-running operations
- **Action Processing**: Supports multiple response formats (tags, LaTeX, etc.)
- **GPU Padding**: Automatic batch padding for multi-GPU training

### Custom Reward Functions

Implement custom reward functions in `verl/utils/reward_score/`:

```python
def custom_reward_function(predictions, references, data_source):
    # Your reward logic here
    return scores
```

Register your function in the reward manager:

```python
def _select_rm_score_fn(data_source):
    if data_source == 'custom_dataset':
        return custom_reward_function
    # ... other cases
```



### Performance Tips

- Use gradient checkpointing for memory efficiency
- Enable FSDP offloading for large models
- Optimize batch sizes based on available GPU memory
- Use Flash Attention for improved performance


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use MuTIS in your research, please cite:

```bibtex
@article{mutis2024,
  title={MuTIS: Enhancing Reasoning Efficiency through Multi-Turn Intervention Sampling in Reinforcement Learning},
  author={Your Authors},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

This implementation is built upon:
- [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning framework
- [Search-R1](https://github.com/petergriffinjin/search-r1) - Search-based reasoning framework

The evaluation phase uses the [LIMO](https://github.com/GAIR-NLP/LIMO) framework

We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.
