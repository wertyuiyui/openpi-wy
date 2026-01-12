export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/qiyuan_research_vepfs_001/weiyi/huggingface
export export https_proxy=http://100.68.164.44:3128 http_proxy=http://100.68.164.44:3128
export WANDB_API_KEY=21a8bda930a645b08f2834efd21ba21e98cd83cf
# export WANDB_MODE=disabled

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_dexora_test --exp-name=test --overwrite