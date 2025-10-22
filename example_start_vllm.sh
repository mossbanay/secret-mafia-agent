CUDA_VISIBLE_DEVICES=0 uv run vllm serve Qwen/Qwen3-8B --gpu-memory-utilization 0.9 --port 8005 --max-model-len 16384 --served-model-name qwen3-8b
