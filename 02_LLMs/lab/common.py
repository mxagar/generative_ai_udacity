import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # cross_attention = False

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class MultiHeadSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x):
        # batch size, sequence length (in tokens), embedding dimensionality (n_embd per token)
        B, T, C = x.size()
        hs = C // self.n_head  # head size

        # # print some debug information
        # print(f"batch size: {B}")
        # print(f"sequence length: {T}")
        # print(f"embedding dimensionality: {C}")
        # print(f"number of heads: {self.n_head}")
        # print(f"head size: {hs}")

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # resulting dims for k, q, and v are (B, n_head, T, hs)
        k = self.key(x).view(B, T, self.n_head, hs).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, hs).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, hs).transpose(1, 2)

        # === EXERCISE START: IMPLEMENT THE MULTI-HEAD ATTENTION ===

        #######################################################################
        # TODO: multiply q and k_t matrices, then divide by the square root of d_k
        # print("=== Calculate MatrixMultiplication(Q, K_T) / sqrt(d_k) ===")

        k_t = k.transpose(-2, -1)  # what is the shape of k_t?
        d_k = k.size(-1)

        # Matrix multiplication (hint: not "*")
        # att = <TODO>
        att = q @ k_t / math.sqrt(d_k)

        # print(f"q.shape: {q.shape}")
        # print(f"k_t.shape: {k_t.shape}")
        # print(f"d_k: {d_k}")
        # print(f"att.shape: {att.shape}")

        #######################################################################
        # TODO: set the mask fill value to negative infinity
        # print("=== Apply the attention mask ===")

        # masked_fill_value = <TODO>
        masked_fill_value = float("-inf")

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, masked_fill_value)

        # Show the result of applying the mask
        # print(f"att: {att}")

        #######################################################################
        # TODO: apply softmax
        # print("=== Softmax ===")

        # att = F.softmax(att, dim=<TODO>)
        att = F.softmax(att, dim=-1)

        att = self.attn_drop(att)

        # Show the result of applying the softmax and check that
        # the sum of the attention weights in each row is 1
        # print(f"att.shape: {att.shape}")
        # print(f"att: {att}")
        # print(f"att.sum(dim=-1): {att.sum(dim=-1)}")
        # if not (all(((att.sum(dim=-1) - 1.0) ** 2 < 1e-6).tolist())):
        #     raise ValueError("Attention weight rows do not sum to 1")

        ######################################################################
        # TODO: multiply att and v matrices
        # (B, n_head, T, T) x (B, n_head, T, hs) -> (B, n_head, T, hs)
        # print("=== Calculate final attention ===")

        # y = <TODO>
        y = att @ v

        # print(f"y.shape: {y.shape}")

        ######################################################################

        # === EXERCISE END: IMPLEMENT THE MULTI-HEAD ATTENTION ===

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, **kwargs):
        """Character tokenizer for Hugging Face transformers.

        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.

            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

    def get_vocab(self):
        return self._vocab_str_to_int


class AdditionDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        block_size,
        numbers=range(100, 200),
        include_intermediate_steps=True,
    ):
        self.numbers = numbers
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.include_intermediate_steps = include_intermediate_steps

    def __len__(self):
        return len(self.numbers) ** 2

    def __getitem__(self, idx):
        a = self.numbers[idx // len(self.numbers)]
        b = self.numbers[idx % len(self.numbers)]

        # calculate the "label" of the addition problem a + b
        c = a + b

        # Convert c its ones, tens, hundreds, etc. parts
        c_parts = [(int(s) * 10**i) for i, s in enumerate(str(c)[::-1])]

        x = f"{a}+{b}="  # e.g.  345+678=
        if self.include_intermediate_steps:
            x += "+".join([f"{p}" for p in c_parts]) + "="  # e.g.  345+678=3+20+0+1000=
        x += f"{c}"  # e.g.  345+678=3+20+0+1000=1023

        # Ensure that the length of x is less than or equal to block_size
        if len(x) > self.block_size:
            raise ValueError(
                f"Length of x is {len(x)} which is greater than block_size {self.block_size}"
            )

        # predict the next token in the sequence
        y = x[1:]  # e.g.  45+678=3+20+0+1000=1023

        # tokenize the input and output strings
        x = self.tokenizer.encode(x)
        y = self.tokenizer.encode(y)

        # pad x and y to the block_size
        x = x + [self.tokenizer.pad_token_id] * (self.block_size - len(x))
        y = y + [self.tokenizer.pad_token_id] * (self.block_size - len(y))

        # convert to torch tensors
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        # mask the output tokens with -1 to only calculate loss on the other tokens
        y[: len(f"{a}+{b}=")] = -1

        return x, y


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset, config, test_dataset=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            print("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(
            optim_groups, lr=config.learning_rate, betas=config.betas
        )

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle=True,
            )

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y) in pbar:
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = (
                        loss.mean()
                    )  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()

                    lr = config.learning_rate

                    # report progress
                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                    )

            if not is_train:
                print(f"test loss: {np.mean(losses):.5f}")

        for epoch in range(config.max_epochs):
            run_epoch("train")
            if self.test_dataset is not None:
                run_epoch("test")

            self.save_checkpoint()


def show_examples(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    tokenizer: CharacterTokenizer,
    device: torch.device,
    max_num: int = 10,
    seed: int = 42,
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    top_k: int = 3,
):
    correct, total = 0, 0

    # Get num samples from the dataset
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[:max_num]

    for i in indices:
        x, y = dataset[i]

        # extract a and b, e.g. [CLS]90+26= -> 90, 26
        a, b = tokenizer.decode(x).split("[CLS]")[1].split("=")[0].split("+")
        a = int(a)
        b = int(b)

        # remove -1's from y
        y = y[y != -1]

        # cut off x to the first `=`` token
        x = x.tolist()
        x = x[: x.index(tokenizer._convert_token_to_id("=")) + 1]
        input_string = tokenizer.decode(x)
        x = torch.tensor(x).unsqueeze(0)

        x = x.to(device)

        y_hat = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=[tokenizer._convert_token_to_id(x) for x in ("[SEP]", "[CLS]")],
        )

        # extract the answer from the generated sequence
        output_string = tokenizer.decode(y_hat[0])[len(input_string) :]

        matches = re.findall(r"=\d+", output_string)
        if len(matches) > 0:
            c_pred = int(matches[-1].split("=")[-1])
        else:
            c_pred = None

        if c_pred == a + b:
            correct += 1

            print(
                f"✅ input->output: {input_string.strip()} 👉 {output_string.strip()}"
            )
        else:
            print(
                f"❌ input->output: {input_string.strip()} 👉 {output_string.strip()} expected: {a+b}, got: {c_pred}"
            )

        total += 1
    print()
    print(f"Correct: {correct} out of {total}: {round(correct / total * 100, 1)}%")
