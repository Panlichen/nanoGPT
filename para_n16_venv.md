```shell
conda create -n nanogpt310 python=3.10
conda activate nanogpt310
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets tiktoken wandb tqdm -y
```

```shell
sbatch train_shakespeare_char.sh
sbatch infer_shakespeare_char.sh
```