# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from pathlib import Path
from llama import LLaMA
from model import Transformer, ModelArgs
from tokenizer import Tokenizer

# model
dim = 4096
n_layers = 8
n_heads = 8
multiple_of = 256
dropout = 0.0
max_seq_len: int = 1024
max_batch_size: int = 32

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_heads,
    vocab_size=32000,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
    max_batch_size=max_batch_size
)  # start with model_args from command line



def load(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        
) -> LLaMA:
    print("Creating model...")
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    gptconf = ModelArgs(**model_args)
    
    tokenizer = Tokenizer()

    
    model = Transformer(gptconf)


    # Original copyright by tloen
    # https://github.com/tloen/llama-int8/blob/main/example.py
    key_to_dim = {
        "w1": 0,
        "w2": -1,
        "w3": 0,
        "wo": -1,
        "wq": 0,
        "wk": 0,
        "wv": 0,
        "output": 0,
        "tok_embeddings": -1,
        "ffn_norm": None,
        "attention_norm": None,
        "norm": None,
        "rope": None,
    }
    
    for i, ckpt in enumerate(checkpoints):
        print(f"Loading checkpoint {i}")
        checkpoint = torch.load(ckpt, map_location="cpu")
        for parameter_name, parameter in model.named_parameters():
            short_name = parameter_name.split(".")[-2]
            if key_to_dim[short_name] is None and i == 0:
                parameter.data = checkpoint[parameter_name]
            elif key_to_dim[short_name] == 0:
                size = checkpoint[parameter_name].size(0)
                parameter.data[size * i: size * (i + 1), :] = checkpoint[
                    parameter_name
                ]
            elif key_to_dim[short_name] == -1:
                size = checkpoint[parameter_name].size(-1)
                parameter.data[:, size * i: size * (i + 1)] = checkpoint[
                    parameter_name
                ]
            del checkpoint[parameter_name]
        del checkpoint

    model.to("cpu")

    generator = LLaMA(model, tokenizer)
    print(f"Loaded model in {time.time() - start_time:.2f} seconds")
    return generator


def main(
        ckpt_dir: str = '../../../code/python/llama/llama/llama-2-7b-chat',
        tokenizer_path: str = './tokenizer.model',
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_seq_len: int = 512,
        max_batch_size: int = 32,
):
    # torch.manual_seed(1)
    # torch.set_default_dtype(torch.bfloat16)

    generator = load(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)

    while True:
        prompt = input(f'prompt> ')
        if len(prompt.strip()) > 0:
            prompts = [prompt]
            results = generator.generate(
                prompts, max_gen_len=256, temperature=temperature, top_p=top_p
            )

            for result in results:
                print(result)


if __name__ == "__main__":
    fire.Fire(main)