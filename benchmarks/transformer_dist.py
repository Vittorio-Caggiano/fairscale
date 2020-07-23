# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import os
import threading
import time
import warnings

import torch
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.utils import get_tokenizer

# Lot of code has been borrowed from https://github.com/pytorch/tutorials/pull/948/files

#########################################################
#                   helper functions                    #
#########################################################


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_on_rref(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and return an RRef
    of the result.
    """
    return rpc.remote(rref.owner(), _call_method, args=[method, rref] + list(args), kwargs=kwargs)


def _async_on_rref(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    return rpc.rpc_async(rref.owner(), _call_method, args=[method, rref] + list(args), kwargs=kwargs)


def _parameter_rrefs(module):
    r"""
    Create one RRef for each parameter in the given local module, and return a
    list of RRefs.
    """
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs


#########################################################


class EmbeddingLayer(nn.Embedding):
    def __init__(self, ntoken, ninp, initrange):
        super().__init__(ntoken, ninp)
        self.ninp = ninp
        self.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        return super().forward(src) * math.sqrt(self.ninp)


class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerDecoderLayer(nn.TransformerEncoderLayer):
    """Though this class inherits from torch.nn.TransformerEncoderLayer,
        it functions as a decoder in this model"""

    def __init__(self, ninp, nhead, nhid, droupout):
        super().__init__(ninp, nhead, nhid, droupout)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        return super().forward(src, self.src_mask)


class LinearLayer(nn.Linear):
    def __init__(self, ninp, ntoken, initrange):
        super().__init__(ninp, ntoken)
        self.bias.data.zero_()
        self.weight.data.uniform_(-initrange, initrange)


class TransformerLMSequntial(nn.Sequential):
    """A small language model based on the design of GPT-2 using nn.Sequeitnal
       for compatability with Pipe"""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange):
        super(TransformerLMSequntial, self).__init__(
            EmbeddingLayer(ntokens, ninp, initrange),
            PositionalEncodingLayer(ninp, dropout),
            TransformerDecoderLayer(ninp, nhead, nhid, dropout),
            LinearLayer(ninp, ntokens, initrange),
        )


class LMPart1(nn.Module):
    def __init__(self, device, ntokens, ninp, dropout, initrange):
        super(LMPart1, self).__init__()
        self._lock = threading.Lock()
        self.device = device
        self.arch = nn.Sequential(EmbeddingLayer(ntokens, ninp, initrange), PositionalEncodingLayer(ninp, dropout)).to(
            self.device
        )

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.arch(x)
        return out.cpu()


class LMPart2(nn.Module):
    def __init__(self, device, ntokens, ninp, nhid, nhead, dropout, initrange):
        super(LMPart2, self).__init__()
        self.device = device
        self._lock = threading.Lock()
        self.arch = nn.Sequential(
            TransformerDecoderLayer(ninp, nhead, nhid, dropout), LinearLayer(ninp, ntokens, initrange)
        ).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.arch(x)
        return out.cpu()


class DistTransformerLM(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """

    def __init__(self, split_size, workers, ntokens, ninp, nhead, nhid, dropout, initrange, *args, **kwargs):
        super(DistTransformerLM, self).__init__()

        self.split_size = split_size  # number of microbatches

        # Put the first part of the LM on workers[0]
        self.p1_rref = rpc.remote(
            workers[0], LMPart1, args=("cuda:0", ntokens, ninp, dropout, initrange) + args, kwargs=kwargs
        )

        # Put the second part of the LM on workers[1]
        self.p2_rref = rpc.remote(
            workers[1], LMPart2, args=("cuda:1", ntokens, ninp, nhid, nhead, dropout, initrange) + args, kwargs=kwargs
        )

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = _remote_on_rref(LMPart1.forward, self.p1_rref, x_rref)
            z_fut = _async_on_rref(LMPart2.forward, self.p2_rref, y_rref)
            out_futures.append(z_fut)

        # wait for all RPC to finish
        outs = [fut.wait() for fut in out_futures]
        # cat all tensors into one tensor.
        out = torch.cat(outs)
        return out

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(_remote_on_rref(_parameter_rrefs, self.p1_rref).to_here())
        remote_params.extend(_remote_on_rref(_parameter_rrefs, self.p2_rref).to_here())
        return remote_params


def get_train_data(device):
    with warnings.catch_warnings(record=True) as fjldska:
        TEXT = torchtext.data.Field(
            tokenize=get_tokenizer("basic_english"), init_token="<sos>", eos_token="<eos>", lower=True
        )
        train_txt, val_txt, test_txt = torchtext.datasets.WikiText103.splits(TEXT)
        TEXT.build_vocab(train_txt)
        ntokens = len(TEXT.vocab.stoi)

        batch_size = 100
        train_data = batchify(train_txt, batch_size, TEXT, device)

        return ntokens, train_data


def batchify(data, bsz, TEXT, device):
    data = TEXT.numericalize([data.examples[0].text])
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def get_number_of_words(data):
    return data.size()[0] * data.size()[1]


def run_master(split_size):
    bptt = 35
    ninp = 50  # embedding dimension
    nhid = 50  # the dimension of the feedforward network model in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0
    initrange = 0.1
    ntokens, train_data = get_train_data("cpu")

    # put the two model parts on worker1 and worker2 respectively
    model = DistTransformerLM(split_size, ["worker1", "worker2"], ntokens, ninp, nhead, nhid, dropout, initrange)
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.01
    opt = DistributedOptimizer(optim.Adam, model.parameter_rrefs(), lr=lr,)

    start_time = time.time()
    total_loss = 0.0
    for batch, i in enumerate(range(0, bptt + 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)

        with dist_autograd.context() as context_id:
            outputs = model(data)
            outputs = outputs.view(-1, ntokens)
            loss = loss_fn(outputs, targets)
            dist_autograd.backward(context_id, [loss])
            opt.step(context_id)

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print(
                    "| {:5d}/{:5d} batches | ms/batch {:5.2f} | "
                    "loss {:5.2f} | ppl {:8.2f}".format(
                        batch, len(train_data) // bptt, elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)
                    )
                )
                total_loss = 0
    print(f"elapsed = {time.time() - start_time}")


def run_worker(rank, world_size, num_split):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    options = rpc.ProcessGroupRpcBackendOptions(num_send_recv_threads=256)

    if rank == 0:
        rpc.init_rpc("master", rank=rank, world_size=world_size, rpc_backend_options=options)
        run_master(num_split)
    else:
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 3
    for num_split in [1, 2, 4, 8]:
        tik = time.time()
        mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
        tok = time.time()
        print(f"number of splits = {num_split}, execution time = {tok - tik}")
