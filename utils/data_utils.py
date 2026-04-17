import random
from re import split

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer
from torch.utils.data import DataLoader


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('/raid2/DATA/llm_model/dataset/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('/raid2/DATA/llm_model/dataset/wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    # import pdb
    # pdb.set_trace()
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        attention_mask = torch.ones_like(inp)

        # Append the dictionary
        trainloader.append({
            'input_ids': inp,
            'labels': tar,
            'attention_mask': attention_mask
        })

    new_trainloader = []
    num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
    for i in range(0, num_batches):
        start = i * batch_size
        end = min(start + batch_size, nsamples)
        
        batched_inp = []
        batched_tar = []
        batched_attention_mask = []
        
        for j in range(start, end):
            batched_inp.append(trainloader[j]['input_ids'])
            batched_tar.append(trainloader[j]['labels'])
            batched_attention_mask.append(trainloader[j]['attention_mask'])
        
        # Concatenate along batch dimension
        batched_inp = torch.cat(batched_inp)
        batched_tar = torch.cat(batched_tar)
        batched_attention_mask = torch.cat(batched_attention_mask)

        # Append batched data as a dictionary
        new_trainloader.append({
            'input_ids': batched_inp.flatten(),
            'labels': batched_tar.flatten(),
            'attention_mask': batched_attention_mask.flatten()
        })
    
    # Clean up the old trainloader
    del trainloader
    trainloader=new_trainloader
    del new_trainloader
    # Prepare validation set (optional, but assuming you need to process it similarly)
    valenc = tokenizer(' '.join(testdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_c4(nsamples, seed, seqlen, model, tokenizer, batch_size):
    traindata = load_dataset('json', data_files={'train': '/raid2/DATA/llm_model/dataset/c4/c4-train.00000-of-01024.json.gz'},split='train')
    valdata = load_dataset('json', data_files={'validation': '/raid2/DATA/llm_model/dataset/c4/c4-validation.00000-of-00008.json.gz'},split='validation')

    random.seed(seed)
    trainloader = []
    
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        
        # Randomly select a sub-sequence of length `seqlen`
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]  # input_ids
        tar = inp.clone()  # labels
        tar[:, :-1] = -100  # Set all but the last token in `labels` to -100 (ignoring those tokens during loss computation)

        # Create attention_mask (1s for all positions, since padding isn't involved here)
        attention_mask = torch.ones_like(inp)

        # Append the dictionary
        trainloader.append({
            'input_ids': inp,
            'labels': tar,
            'attention_mask': attention_mask
        })
    
    # Batch the data
    new_trainloader = []
    num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
    for i in range(0, num_batches):
        start = i * batch_size
        end = min(start + batch_size, nsamples)
        
        batched_inp = []
        batched_tar = []
        batched_attention_mask = []
        
        for j in range(start, end):
            batched_inp.append(trainloader[j]['input_ids'])
            batched_tar.append(trainloader[j]['labels'])
            batched_attention_mask.append(trainloader[j]['attention_mask'])
        
        # Concatenate along batch dimension
        batched_inp = torch.cat(batched_inp)
        batched_tar = torch.cat(batched_tar)
        batched_attention_mask = torch.cat(batched_attention_mask)

        # Append batched data as a dictionary
        new_trainloader.append({
            'input_ids': batched_inp.flatten(),
            'labels': batched_tar.flatten(),
            'attention_mask': batched_attention_mask.flatten()
        })
    
    # Clean up the old trainloader
    del trainloader
    trainloader=new_trainloader
    del new_trainloader
    # Prepare validation set (optional, but assuming you need to process it similarly)
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=1024, seed=0, seqlen=2048, tokenizer=None, model='', batch_size=1):
    if tokenizer is None:
        tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer, batch_size)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer, batch_size)

def get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    # import pdb
    # pdb.set_trace()
    traindata = load_dataset('/home/fmk/dataset/wikitext', 'wikitext-2-raw-v1', split='train')
    traindata = traindata.shuffle(seed=seed)
    # trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')
    train_dataset = traindata.select(range(nsamples))  # 限制样本数量
    def preprocess_function(examples):
        # 使用 tokenizer 对文本进行编码，限制最大长度
        encoding = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=seqlen)
        encoding['labels'] = encoding['input_ids']  # 在自回归任务中，labels = input_ids
        return encoding 
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    # 将训练集转换为 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    return train_dataloader
  
def get_c4_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    # import pdb
    # pdb.set_trace()
    traindata = load_dataset('json', data_files={'train': '/home/fmk/dataset/c4/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('json', data_files={'validation': '/home/fmk/dataset/c4/c4-validation.00000-of-00008.json.gz'}, split='validation')
    traindata = traindata.shuffle(seed=seed)
    
    train_dataset = traindata.select(range(nsamples))  # 限制样本数量
    def preprocess_function(examples):
        # 使用 tokenizer 对文本进行编码，限制最大长度
        encoding = tokenizer(examples['text'], truncation=True, padding=True, max_length=seqlen)
        encoding['labels'] = encoding['input_ids']  # 在自回归任务中，labels = input_ids
        return encoding 
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    # 将训练集转换为 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    return train_dataloader
    # trainenc = tokenizer(' '.join(traindata[:nsamples]['text']), return_tensors='pt')
    # trainenc = trainenc.input_ids

    # class TokenizerWrapper:
    #     def __init__(self, input_ids):
    #         self.input_ids = input_ids
    # trainenc = TokenizerWrapper(trainenc)

    return trainenc

def get_trainloaders(name, nsamples=128, seed=0, seqlen=2048, model='', batch_size=1):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'c4' in name:
        return get_c4_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)