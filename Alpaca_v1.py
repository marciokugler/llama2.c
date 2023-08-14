



"""
Download, preprocess and serve the OpenAssistant dataset as a DataLoader.
"""
import gzip
import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer
from pathlib import Path

DATA_CACHE_DIR = "data"
DATASET_DIR = "out"

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
import pandas as pd

from treelib import Tree

#ds = load_dataset("OpenAssistant/oasst1")
#print(ds)

def download_alpaca():
    data_url = 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json'
    input_file_path = os.path.join(DATA_CACHE_DIR, "out/alpaca/", "alpaca_data.json") 
    download_file(data_url, input_file_path)
   

                    
def process_file_alpaca(args):
    input_file_path = args
    tokenized_filename = input_file_path.replace(".json", ".bin")
    enc = Tokenizer()
    input_file_path = Path(input_file_path)
    if input_file_path.suffix == ".gz":
        file_in = gzip.open(str(input_file_path), mode="tr", encoding="UTF-8")
        print (file_in)
    else:
        file_in = input_file_path.open("r", encoding="UTF-8")
        #print (file_in)
 
    #print (input_file_path)
    json_object = json.load(file_in)
    all_tokens = []
    #print (json_object)
    for i in json_object:
        #print (i)
        for k in i:
            h = json.loads(json.dumps(i))
            instruction = (h['instruction'])
            output = (h['output'])
            #print ('instruction')
            #print (instruction)
            #print ('output')
            #print (output)
            text = "instruction: " + instruction + " output: " + output
            print (text)
            tokens = enc.encode(text, bos=True, eos=False)
            print (tokens)
            all_tokens.extend(tokens)
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    print(f"Saved {tokenized_filename}")


def tokenize_alpaca():
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "out/alpaca")
    #data_dir = os.path.join(DATA_CACHE_DIR, "out")
    
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    
    # process all the shards in a process pool
    with ProcessPoolExecutor() as executor:
        executor.map(process_file_alpaca, shard_filenames)
    print("Done.")



def pretokenize():
    enc = Tokenizer()
    def process_shard(shard):
        with open(shard, "r") as f:
            #data = json.load(f)
            df = pd.read_json(f)
            sample_story=df.sample(1).transpose().to_dict()
            print(f"Example story:\n{sample_story}")
        all_tokens = []
        for example in tqdm(data):
            text = example["text"]
            text = text.strip() # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        # write to disk
        tokenized_filename = shard.replace(".json", ".bin")
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        print(f"Saved {tokenized_filename}")

    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "oasst_all.messages")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # process all the shards in a threadpool
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_shard, shard_filenames)

    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        data_dir = os.path.join(DATA_CACHE_DIR, "out/alpaca/")
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        #shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        print (shard_filenames)
        print("Shuffling")
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                print("Sharding - " + shard )
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:

    @staticmethod
    def iter_batches(split, batch_size, max_seq_len, device, num_workers=0):
        ds = PretokDataset(split, max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

class AlpacaTask:

    @staticmethod
    def iter_batches(split, batch_size, max_seq_len, device, num_workers=0):
        ds = PretokDataset(split, max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download_alpaca", "tokenize_alpaca"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    fun = {
            "download_alpaca": download_alpaca,
            "tokenize_alpaca": tokenize_alpaca,
    }
    fun[args.stage]()

