# datasets/llm_benchmark.py
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class CausalLMDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        # Temporarily increase the max_length to silence the HF warning
        # Since we manually chunk it below, this is perfectly safe.
        original_max_length = tokenizer.model_max_length
        tokenizer.model_max_length = int(1e9)
        
        # Tokenize everything into one massive stream
        tokenized = tokenizer(" ".join(texts), return_tensors="pt", truncation=False)["input_ids"][0]
        
        # Restore the tokenizer's original safety limits
        tokenizer.model_max_length = original_max_length
        
        # Drop the last chunk to ensure uniform block sizes
        self.block_size = block_size
        self.num_blocks = len(tokenized) // block_size
        self.tokens = tokenized[:self.num_blocks * block_size].view(self.num_blocks, block_size)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        return tokens, tokens

class LLMDomainSplit:
    def __init__(self, model_name="gpt2", block_size=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.block_size = block_size

    def get_task_loaders(self, batch_size=8):
        print("Loading Task 1: WikiText-2 (General Domain)...")
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        wiki_texts = [t for t in wiki["text"] if len(t.strip()) > 10]
        t1_ds = CausalLMDataset(wiki_texts, self.tokenizer, self.block_size)
        
        print("Loading Task 2: IMDB (Subjective Domain Shift)...")
        imdb = load_dataset("imdb", split="train")
        imdb_texts = [t for t in imdb["text"] if len(t.strip()) > 10]
        t2_ds = CausalLMDataset(imdb_texts, self.tokenizer, self.block_size)

        t1_loader = DataLoader(t1_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        t2_loader = DataLoader(t2_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        
        return t1_loader, t2_loader