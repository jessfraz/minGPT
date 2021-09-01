#!/usr/bin/env python
# coding: utf-8

# ## Train GPT on addition
#
# Train a GPT model on a dedicated addition dataset to see if a Transformer can learn to add.

# In[1]:


# set up logging
import logging


# In[2]:


# make deterministic
from mingpt.utils import set_seed


# In[3]:


import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# In[4]:


from torch.utils.data import Dataset

class AdditionDataset(Dataset):
    """
    Returns addition problems of up to some number of digits in the inputs. Recall
    that all GPT cares about are sequences of integers, and completing them according to
    patterns in the data. Therefore, we have to somehow encode addition problems
    as a sequence of integers.

    The sum of two n-digit numbers gives a third up to (n+1)-digit number. So our
    encoding will simply be the n-digit first number, n-digit second number,
    and (n+1)-digit result, all simply concatenated together. Because each addition
    problem is so structured, there is no need to bother the model with encoding
    +, =, or other tokens. Each possible sequence has the same length, and simply
    contains the raw digits of the addition problem.

    As a few examples, the 2-digit problems:
    - 85 + 50 = 135 becomes the sequence [8, 5, 5, 0, 1, 3, 5]
    - 6 + 39 = 45 becomes the sequence [0, 6, 3, 9, 0, 4, 5]
    etc.

    We will also only train GPT on the final (n+1)-digits because the first
    two n-digits are always assumed to be given. So when we give GPT an exam later,
    we will e.g. feed it the sequence [0, 6, 3, 9], which encodes that we'd like
    to add 6 + 39, and hope that the model completes the integer sequence with [0, 4, 5]
    in 3 sequential steps.

    fun exercise: does it help if the result is asked to be produced in reverse order?
    """

    def __init__(self, ndigit, split):
        self.split = split # train/test
        self.ndigit = ndigit
        self.vocab_size = 10 # 10 possible digits 0..9
        # +1 due to potential carry overflow, but then -1 because very last digit doesn't plug back
        self.block_size = ndigit + ndigit + ndigit + 1 - 1

        # split up all addition problems into either training data or test data
        num = (10**self.ndigit)**2 # total number of possible combinations
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 1000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def __len__(self):
        return self.ixes.size

    def __getitem__(self, idx):
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx]
        nd = 10**self.ndigit
        a = idx // nd
        b = idx %  nd
        c = a + b
        render = f'%0{self.ndigit}d%0{self.ndigit}d%0{self.ndigit+1}d' % (a,b,c) # e.g. 03+25=28 becomes "0325028"
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:self.ndigit*2-1] = -100 # we will only train in the output locations. -100 will mask loss to zero
        return x, y


# In[5]:
# now let's give the trained model an addition exam
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import sample

def give_exam(trainer, dataset, batch_size=32, max_batches=-1):

    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        d1d2 = x[:, :ndigit*2]
        d1d2d3 = sample(model, d1d2, ndigit+1)
        d3 = d1d2d3[:, -(ndigit+1):]
        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)
        # decode the integers from individual digits
        d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
        d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
        d3i_pred = (d3 * factors).sum(1)
        d3i_gt = d1i + d2i
        correct = (d3i_pred == d3i_gt).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line, lol
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            judge = 'YEP!!!' if correct[i] else 'NOPE'
            if not correct[i]:
                print("GPT claims that %03d + %03d = %03d (gt is %03d; %s)"
                      % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i], judge))

        if max_batches >= 0 and b+1 >= max_batches:
            break

    print("final score: %d/%d = %.2f%% correct" % (np.sum(results), len(results), 100*np.mean(results)))





# In[7]:


from mingpt.model import GPT, GPTConfig, GPT1Config



# In[8]:


from mingpt.trainer import Trainer, TrainerConfig
from multiprocessing import freeze_support
import torch_xla.distributed.xla_multiprocessing as xmp

def do_device(index):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(42)
    # create a dataset for e.g. 2-digit addition
    ndigit = 2
    train_dataset = AdditionDataset(ndigit=ndigit, split='train')
    test_dataset = AdditionDataset(ndigit=ndigit, split='test')


    # In[6]:


    train_dataset[0] # sample a training instance just to see what one raw example looks like
    # initialize a baby GPT model
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=2, n_head=4, n_embd=128)
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=50, batch_size=512, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=1024, final_tokens=50*len(train_dataset)*(ndigit+1),
                      num_workers=4)
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()


    # training set: how well did we memorize?
    give_exam(trainer, train_dataset, batch_size=1024, max_batches=10)


    # test set: how well did we generalize?
    give_exam(trainer, test_dataset, batch_size=1024, max_batches=-1)


# In[ ]:


# well that's amusing... our model learned everything except 55 + 45


if __name__ == '__main__':
    #freeze_support()
    xmp.spawn(do_device, args=())

