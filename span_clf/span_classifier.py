from clearml import Task, StorageManager, Dataset as ds
import argparse, json, os, random, math, ipdb, jsonlines
 
# config = json.load(open('config.json'))
config={
    "gamma": 10,
    "alpha": 0.1,
    "dice_smooth": 1,
    "dice_ohem": 0.8,
    "dice_alpha": 0.01,
    "width_embed_len": 300,
    "lr": 3e-5,
    "num_epochs":50,
    "train_batch_size":12,
    "eval_batch_size":6,
    "max_length": 2048, # be mindful underlength will cause device cuda side error
    "max_span_len": 8,
    "max_num_spans":1024*3,
    "dataset": "muc4",
    "train": True
}

args = argparse.Namespace(**config)

Task.add_requirements("transformers", package_version="4.1.0") 
task = Task.init(project_name='SpanClassifier', task_name='EntitySpanClassifier', tags=["maxpool", args.dataset], output_uri="s3://experiment-logging/storage/")
clearlogger = task.get_logger()

task.connect(args)
task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
task.execute_remotely(queue_name="compute", exit_process=True)

dataset = ds.get(dataset_name="processed-DWIE", dataset_project="datasets/DWIE", dataset_tags=["1st-mention"], only_published=True)
dataset_folder = dataset.get_local_copy()
dwie = json.load(open(os.path.join(dataset_folder, "data", "new_dwie.json")))["dataset"]
 
dataset = ds.get(dataset_name="muc4-processed", dataset_project="datasets/muc4", dataset_tags=["processed", "GRIT"], only_published=True)
dataset_folder = dataset.get_local_copy()
print(list(os.walk(os.path.join(dataset_folder, "data/muc4-grit/processed"))))

class bucket_ops:
    StorageManager.set_cache_file_limit(5, cache_context=None)

    def list(remote_path:str):
        return StorageManager.list(remote_path, return_full_path=False)

    def upload_folder(local_path:str, remote_path:str):
        StorageManager.upload_folder(local_path, remote_path, match_wildcard=None)
        print("Uploaded {}".format(local_path))

    def download_folder(local_path:str, remote_path:str):
        StorageManager.download_folder(remote_path, local_path, match_wildcard=None, overwrite=True)
        print("Downloaded {}".format(remote_path))
    
    def get_file(remote_path:str):        
        object = StorageManager.get_local_copy(remote_path)
        return object

    def upload_file(local_path:str, remote_path:str):
        StorageManager.upload_file(local_path, remote_path, wait_for_upload=True, retries=3)


''' Writes to a jsonl file'''
def to_jsonl(filename:str, file_obj):
    resultfile = open(filename, 'wb')
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj)

def read_json(jsonfile):
    with open(jsonfile, 'rb') as file:
        file_object = [json.loads(sample) for sample in file]
    return file_object

train_data = read_json(os.path.join(dataset_folder, "data/muc4-grit/processed/train.json"))
dev_data = read_json(os.path.join(dataset_folder, "data/muc4-grit/processed/dev.json"))
test_data = read_json(os.path.join(dataset_folder, "data/muc4-grit/processed/test.json"))

muc4 = {
    "train": train_data,
    "val": dev_data,
    "test": test_data
}
 
import os, random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerForMaskedLM, LongformerModel, LongformerTokenizer, LongformerConfig, AutoTokenizer
from transformers.models.longformer.modeling_longformer import LongformerLMHead, _compute_global_attention_mask
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

from typing import Callable, Any, Dict, List, Tuple, Optional, Sequence, Tuple, TypeVar, Union, NamedTuple

from loss.focal_loss import FocalLoss
from loss.dice_loss import DiceLoss

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return type(self), (self.message,)

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message

def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)

def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.
    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```
    # Parameters
    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.
    # Returns
    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices

def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.
    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/main/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.
    The key reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.
    # Returns
    selected_targets : `torch.Tensor`
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets

def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()

def enumerate_spans(
    sentence: List,
    offset: int = 0,
    max_span_width: int = None,
    min_span_width: int = 1,
    filter_function: Callable[[List], bool] = None,
) -> List[Tuple[int, int]]:
    """
    Given a sentence, return all token spans within the sentence. Spans are `inclusive`.
    Additionally, you can provide a maximum and minimum span width, which will be used
    to exclude spans outside of this range.
    Finally, you can provide a function mapping `List[T] -> bool`, which will
    be applied to every span to decide whether that span should be included. This
    allows filtering by length, regex matches, pos tags or any Spacy `Token`
    attributes, for example.
    # Parameters
    sentence : `List[T]`, required.
        The sentence to generate spans for. The type is generic, as this function
        can be used with strings, or Spacy `Tokens` or other sequences.
    offset : `int`, optional (default = `0`)
        A numeric offset to add to all span start and end indices. This is helpful
        if the sentence is part of a larger structure, such as a document, which
        the indices need to respect.
    max_span_width : `int`, optional (default = `None`)
        The maximum length of spans which should be included. Defaults to len(sentence).
    min_span_width : `int`, optional (default = `1`)
        The minimum length of spans which should be included. Defaults to 1.
    filter_function : `Callable[[List[T]], bool]`, optional (default = `None`)
        A function mapping sequences of the passed type T to a boolean value.
        If `True`, the span is included in the returned spans from the
        sentence, otherwise it is excluded..
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end))
    return spans

def get_entity_token_id(context_encodings, role_ans:list, label_idx=None):
    overflow_count = 0
    entity_spans=[]            
    entity_labels=[]

    for role, ans_char_start, ans_char_end, mention in role_ans:
        sequence_ids = context_encodings.sequence_ids()

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        qns_offset = sequence_ids.index(0)+1
        pad_start_idx = sequence_ids[sequence_ids.index(0):].index(None)
        offsets_wo_pad = context_encodings["offset_mapping"][0][qns_offset:pad_start_idx]

        # if char indices more than end idx in last word span
        if ans_char_end>offsets_wo_pad[-1][1] or ans_char_start>offsets_wo_pad[-1][1]:
            overflow_count+=1
            ans_char_start = 0
            ans_char_end = 0

        if ans_char_start==0 and ans_char_end==0:
            token_span=[0,0,0]
        else:
            token_span=[]
            for idx, span in enumerate(offsets_wo_pad):
                if ans_char_start>=span[0] and ans_char_start<=span[1] and len(token_span)==0:
                    token_span.append(idx) 

                if ans_char_end>=span[0] and ans_char_end<=span[1] and len(token_span)==1:
                    token_span.append(idx)
                    break                        
                        
        # If token span is incomplete
        if len(token_span)<2:
            token_span=[0,0,0]
            # ipdb.set_trace()

        token_span = [token_span[0], token_span[1], token_span[1]-token_span[0]]
        entity_spans.append(token_span)

    return entity_spans

class DWIE_Data(Dataset):
    def __init__(self, dataset, tokenizer, args):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processed_dataset = {
            "docid": [],
            "context": [],
            "input_ids": [],
            "attention_mask": [],
            "gold_mentions": [],
            "labels":[],
            "span_len_mask":[],
            "spans": []
        }
        self.max_num_spans = args.max_num_spans
        self.max_span_len = args.max_span_len

        for idx, doc in enumerate(self.dataset):
            context = ' '.join(doc["sentences"])
            tokens = self.tokenizer.tokenize(context)
            context_len = len(tokens)
            spans = enumerate_spans(tokens, min_span_width=1, max_span_width=self.max_span_len)[:self.max_num_spans]
            spans = [(span[0], span[1], span[1]-span[0]) for span in spans]
            spans[len(spans):self.max_num_spans] = [(0,0,0)]*(self.max_num_spans-len(spans))

            context_encodings = self.tokenizer(context, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")

            entity_spans = [(entity["start"],entity["end"], entity["end"]-entity["start"])  for entity in doc["entities"] if entity["start"]<self.tokenizer.model_max_length]
            entity_spans = [span for span in entity_spans if span!=[0,0,0]]
            labels = [1.0 if (list(span) in entity_spans) else 0.0 for span in spans]

            self.processed_dataset["input_ids"].append(context_encodings["input_ids"]),
            self.processed_dataset["attention_mask"].append(context_encodings["attention_mask"]),
            self.processed_dataset["spans"].append(torch.tensor(spans))
            self.processed_dataset["labels"].append(torch.tensor(labels))
            # self.processed_dataset["span_len_mask"].append(torch.stack(span_len_mask))
 
    def __len__(self):
        return len(self.processed_dataset["input_ids"])
 
    def __getitem__(self, idx):
        item={}
        item['input_ids'] = self.processed_dataset["input_ids"][idx]
        item['attention_mask'] = self.processed_dataset["attention_mask"][idx]
        # item['span_len_mask'] = self.processed_dataset["span_len_mask"][idx]
        item['spans'] = self.processed_dataset["spans"][idx]
        item['labels'] = self.processed_dataset["labels"][idx]
        return item

    def collate_fn(self, batch):
        input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze(1) 
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch]).squeeze(1)
        spans = torch.stack([ex['spans'] for ex in batch])
        labels = torch.stack([ex['labels'] for ex in batch])
        # span_len_mask = torch.stack([ex["span_len_mask"] for ex in batch])
         
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spans": spans,
            "labels": labels,
        }

class NERDataset(Dataset):
    def __init__(self, dataset, tokenizer, args):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_num_spans = args.max_num_spans
        self.args = args

        self.processed_dataset = {
            "docid": [],
            "context": [],
            "input_ids": [],
            "attention_mask": [],
            "gold_mentions": [],
            "labels":[],
            "span_len_mask":[],
            "spans": []
        }
        self.max_span_len = args.max_span_len
        self.max_num_spans = args.max_num_spans
        
        labels=[]
    
        for doc in dataset:
            docid = doc["docid"]
            context = doc["doctext"] #self.tokenizer.decode(self.tokenizer.encode(doc["doctext"]))
            
            tokens = self.tokenizer.tokenize(context)[:self.tokenizer.model_max_length]
            context_len = len(tokens)

            spans = enumerate_spans(tokens, min_span_width=1, max_span_width=self.max_span_len)[:self.max_num_spans]
            spans = [(span[0], span[1], span[1]-span[0]) for span in spans]
            spans[len(spans):self.max_num_spans] = [(0,0,0)]*(self.max_num_spans-len(spans))
                
            ### Only take the 1st label of each role
            # role_ans = [[key, doc["extracts"][key][0][0][1] if len(doc["extracts"][key])>0 else 0, doc["extracts"][key][0][0][1]+len(doc["extracts"][key][0][0][0]) if len(doc["extracts"][key])>0 else 0, doc["extracts"][key][0][0][0] if len(doc["extracts"][key])>0 else ""] for key in doc["extracts"].keys()]

            context_encodings = self.tokenizer(context, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_offsets_mapping=True, return_tensors="pt")
    
            ### expand on all labels in each role
            role_ans = [[key, mention[1] if len(mention)>0 else 0, mention[1]+len(mention[0]) if len(mention)>0 else 0, mention[0] if len(mention)>0 else ""] for key in doc["extracts"].keys() for cluster in doc["extracts"][key] for mention in cluster]    

            entity_spans = get_entity_token_id(context_encodings, role_ans)
            entity_spans = [span for span in entity_spans if span!=[0,0,0]]
            labels = [1.0 if (list(span) in entity_spans) else 0.0 for span in spans]

            # entity_len_mask=[]
            # noise_len_mask=[]
            # for span in spans:
            #     span_mask = torch.zeros(self.max_span_len)
            #     span_mask[:(span[1]-span[0])] = torch.arange(span[0], span[1])
                    
            #     if list(span) in entity_spans:
            #         # labels.append(1.0)
            #         entity_len_mask.append(span_mask)
            #     else:
            #         # labels.append(0.0)
            #         noise_len_mask.append(span_mask)
            
            # try:
            #     num_noise_samples = 20 - (10 if len(entity_len_mask)>10 else len(entity_len_mask))
            #     sample_noise_len_mask = random.sample(noise_len_mask, num_noise_samples)
            # except:
            #     ipdb.set_trace()

            # span_len_mask = entity_len_mask + sample_noise_len_mask
            
            # labels = [1.0]*len(entity_len_mask) + [0.0]*len(sample_noise_len_mask)

            self.processed_dataset["input_ids"].append(context_encodings["input_ids"]),
            self.processed_dataset["attention_mask"].append(context_encodings["attention_mask"]),
            self.processed_dataset["spans"].append(torch.tensor(spans))
            self.processed_dataset["labels"].append(torch.tensor(labels))
            # self.processed_dataset["span_len_mask"].append(torch.stack(span_len_mask))

 
    def __len__(self):
        return len(self.processed_dataset["input_ids"])
 
    def __getitem__(self, idx):
        item={}
        item['input_ids'] = self.processed_dataset["input_ids"][idx]
        item['attention_mask'] = self.processed_dataset["attention_mask"][idx]
        # item['span_len_mask'] = self.processed_dataset["span_len_mask"][idx]
        item['spans'] = self.processed_dataset["spans"][idx]
        item['labels'] = self.processed_dataset["labels"][idx]
        return item
 
    def collate_fn(self, batch):
        input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze(1) 
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch]).squeeze(1)
        spans = torch.stack([ex['spans'] for ex in batch])
        labels = torch.stack([ex['labels'] for ex in batch])
        # span_len_mask = torch.stack([ex["span_len_mask"] for ex in batch])
         
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spans": spans,
            "labels": labels,
              # "span_len_mask": span_len_mask
        }
 
class NERLongformer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args=args
        self.config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        self.config.gradient_checkpointing = True

        self.tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096', use_fast=True)
        self.tokenizer.model_max_length = self.args.max_length

        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096', config=self.config)
        #self.longformerMLM = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)

        self.lm_head = LongformerLMHead(self.config) 
        # self.lm_head = LongformerLMHead(self.config) 
 
        # train_data = DWIE_Data(dwie["train"], self.tokenizer, self.class2id, self.args)
        # y_train = torch.stack([doc["bio_labels"] for doc in train_data.consolidated_dataset]).view(-1).cpu().numpy()
        # #self.class_weights[0] = self.class_weights[0]/2
        # self.class_weights = torch.cuda.FloatTensor([0.20, 1, 1.2, 1, 1.2])
        # print("weights: {}".format(self.class_weights))
        self.dropout = nn.Dropout(0.1)
 
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size*2+self.args.width_embed_len, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
            # nn.Sigmoid(),
         )
        # self.size_embeddings = nn.Embedding(self.args.max_span_len, self.config.hidden_size)
        self.width_embedding = nn.Embedding(self.args.max_span_len+1, self.args.width_embed_len)

    def test_dataloader(self):
        if self.args.dataset=="muc4":
            test = muc4["test"]
            test_data = NERDataset(test, self.tokenizer, self.args)
        else:
            test = dwie["test"]
            test_data = DWIE_Data(test, self.tokenizer, self.args)
        test_dataloader = DataLoader(test_data, batch_size=self.args.eval_batch_size, collate_fn = test_data.collate_fn)

        return test_dataloader

    def val_dataloader(self):
        if self.args.dataset=="muc4":
            val = muc4["val"]
            val_data = NERDataset(val, self.tokenizer, self.args)
        else:
            val = dwie["test"]
            val_data = DWIE_Data(val, self.tokenizer, self.args)
        val_dataloader = DataLoader(val_data, batch_size=self.args.eval_batch_size, collate_fn = val_data.collate_fn)
        return val_dataloader
 
    def train_dataloader(self):
        if self.args.dataset=="muc4":
            train = muc4["train"]
            train_data = NERDataset(train, self.tokenizer, self.args)
        else:
            train = dwie["train"]
            train_data = DWIE_Data(train, self.tokenizer, self.args)
        train_dataloader = DataLoader(train_data, batch_size=self.args.train_batch_size, collate_fn = train_data.collate_fn)
        return train_dataloader

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Gradient Accumulation caveat 1:
        # For gradient accumulation to work, all model parameters should contribute
        # to the computation of the loss. Remember that the self-attention layers in the LED model
        # have two sets of qkv layers, one for local attention and another for global attention.
        # If we don't use any global attention, the global qkv layers won't be used and
        # PyTorch will throw an error. This is just a PyTorch implementation limitation
        # not a conceptual one (PyTorch 1.8.1).
        # The following line puts global attention on the <s> token to make sure all model
        # parameters which is necessery for gradient accumulation to work.
        global_attention_mask[:, :1] = 1

        # # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1

        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        return global_attention_mask

    def _get_span_embeddings(self, input_ids, spans, token_type_ids=None, attention_mask=None):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)       
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(sequence_output, spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = torch.cat((spans_start_embedding, spans_end_embedding, spans_width_embedding), dim=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding
 
    def forward(self, **batch):
      # in lightning, forward defines the prediction/inference actions             
      #span_len_mask = batch.pop("span_len_mask", None)
        spans = batch.pop("spans", None)
        labels = batch.pop("labels", None)
        total_loss=0
 
        span_embeds = self._get_span_embeddings(batch["input_ids"], spans, token_type_ids=None, attention_mask=batch["attention_mask"])

        if labels!=None:
            pass

        ### Span CLF Objective
        ############################################################################################################
        # span_index = span_len_mask.view(span_len_mask.size(0), -1).unsqueeze(-1).expand(span_len_mask.size(0), span_len_mask.size(1)*span_len_mask.size(2), sequence_output.size(-1)) # (bs, num_samples*max_span_len, 768)
        # span_len_binary_mask = span_len_mask.gt(0).long().unsqueeze(-1)
        # span_len_binary_index = span_len_binary_mask.sum(-2).squeeze(-1)
        # span_len_embeddings = self.size_embeddings(span_len_binary_index)

        # span_len_binary_mask_embeddings = span_len_binary_mask.expand(span_len_binary_mask.size(0), span_len_binary_mask.size(1), span_len_binary_mask.size(2), sequence_output.size(-1))
        # span_embeddings = torch.gather(sequence_output, 1, span_index.long()) # (bs, num_samples*max_span_len, 768)
        # span_embedding_groups = torch.stack(torch.split(span_embeddings, 15, 1)).transpose(0,1) # (bs, num_samples, max_span_len, 768)
        # extracted_spans_after_binary_mask = span_len_binary_mask_embeddings*span_embedding_groups #makes all padding positions into zero vectors (bs, num_samples, max_span_len, 768) 
        # maxpooled_embeddings = torch.max(extracted_spans_after_binary_mask, dim=-2).values # (bs, num_samples, 768)
 
        # #combined_embeds = torch.cat([cls_output.unsqueeze(1).repeat(1, 40, 1), maxpooled_embeddings], dim=-1) #(bs, num_samples, 2*768)
        # combined_embeds = torch.cat([span_len_embeddings, maxpooled_embeddings], dim=-1) #(bs, num_samples, 2*768)
 
        logits = self.classifier(span_embeds).squeeze()        
 
        masked_lm_loss = None        
        span_clf_loss = None
        if labels!=None:
            #torch.count_nonzero(labels.view(-1))
            #loss_fct = WeightedFocalLoss(alpha=self.args.alpha, gamma=self.args.gamma)
            loss_fct = DiceLoss(with_logits=True, smooth=self.args.dice_smooth, ohem_ratio=self.args.dice_ohem,
                                alpha=self.args.dice_alpha, square_denominator=None,
                                reduction="mean", index_label_position=False)
            span_clf_loss = loss_fct(logits.view(-1), labels.view(-1))
            total_loss+=span_clf_loss
 
        return (total_loss, logits)
 
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss, _ = self(**batch)
        # logits = torch.argmax(self.softmax(logits), dim=-1)
        return {"loss": loss}
 
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        #input_ids, attention_mask, labels = batch
        loss, logits = self(**batch)
        # clearlogger.report_text(logits)
        preds = (logits>0.5).long()
        return {"val_loss": loss, "preds": preds, "labels": batch["labels"]}
 
    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_preds = torch.cat([x["preds"] for x in outputs]).view(-1, self.args.max_num_spans)#.cpu().detach().tolist()
        val_labels = torch.cat([x["labels"] for x in outputs]).view(-1, self.args.max_num_spans)#.cpu().detach().tolist()
    
        pred_indices = torch.nonzero(val_preds).squeeze()
        target_indices = torch.nonzero(val_labels).squeeze()
        matches = np.intersect1d(pred_indices.cpu(), target_indices.cpu())

        precision = len(matches)/(len(pred_indices.cpu())+0.000000000001)
        recall = len(matches)/(len(target_indices.cpu())+0.00000000001)
        f1 = 2*(precision*recall) / (precision+recall+0.00000000001)

        logs = {
            "val_loss": val_loss_mean,
        }

        self.log("val_loss", logs["val_loss"])
        self.log("val_recall", recall)
        self.log("val_precision", precision)
        self.log("val_f1", f1)
 
    def test_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        #input_ids, attention_mask, labels = batch
        loss, logits = self(**batch)
        # clearlogger.report_text(logits)
        #clearlogger.report_scalar(title='mlm_loss', series = 'val', value=masked_lm_loss, iteration=batch_idx) 
        preds = (logits>0.5).long()
        return {"test_loss": loss, "preds": preds, "labels": batch["labels"], "spans": batch["spans"]}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_spans = torch.cat([x["spans"] for x in outputs]).view(-1, self.args.max_num_spans, 3).cpu().detach().tolist()
        test_preds = torch.cat([x["preds"] for x in outputs]).view(-1, self.args.max_num_spans)#.cpu().detach().tolist()
        test_labels = torch.cat([x["labels"] for x in outputs]).view(-1, self.args.max_num_spans)#.cpu().detach().tolist()

        pred_indices = torch.nonzero(test_preds).squeeze()
        target_indices = torch.nonzero(test_labels).squeeze()
        matches = np.intersect1d(pred_indices.cpu(), target_indices.cpu())

        # list of (sample_idx, position)
        pred_indices_list = [[pred_idx[0],pred_idx[1]] for pred_idx in pred_indices.cpu().detach().tolist()]
        positive_docs = set([element[0] for element in pred_indices_list])
        
        docspans = []
        test_text = [self.tokenizer.tokenize(doc["doctext"]) for doc in muc4["test"]]
        
        for idx, (sample, tokens) in enumerate(zip(test_spans, test_text)):
            if idx in positive_docs:
                template_spans = [self.tokenizer.convert_tokens_to_string(tokens[sample[val][0]+1:sample[val][1]+2]).strip() for id, val in pred_indices_list if id==idx]
                docspans.append(template_spans)
            else:
                docspans.append([])              
        
        to_jsonl("./predictions.jsonl", docspans)
        task.upload_artifact(name="predictions", artifact_object="./predictions.jsonl")

        precision = len(matches)/(len(pred_indices.cpu())+0.000000000001)
        recall = len(matches)/(len(target_indices.cpu())+0.000000000001)
        f1 = 2*(precision*recall) / ((precision+recall)+0.000000000001)

        logs = {
            "test_loss": test_loss_mean,
        }

        self.log("test_loss", logs["test_loss"])
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)


     # Freeze weights?
    def configure_optimizers(self):
        # Freeze alternate layers of longformer
        for idx, (name, parameters) in enumerate(self.longformer.named_parameters()):

            if idx%2==0:
                parameters.requires_grad=False
            else:
                parameters.requires_grad=True

            # if idx<6:
            #     parameters.requires_grad=False
            # else:
            #     parameters.requires_grad=True

        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = "./",
    filename="best_entity_lm", 
    monitor="val_f1", 
    mode="max", 
    save_top_k=1, 
    save_weights_only=True,
    period=3,
)

early_stop_callback = EarlyStopping(monitor="val_loss", patience=8, verbose=False, mode="min")

model = NERLongformer(args)

# trained_model_path = bucket_ops.get_file(
#             remote_path="s3://experiment-logging/storage/SpanClassifier/EntitySpanClassifier.206d902b9b8f4c05ae5ca1bfd93f3fbf/models/best_entity_lm-v2.ckpt"
#             )
# model = NERLongformer.load_from_checkpoint(trained_model_path, args = args)

trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, callbacks=[checkpoint_callback, early_stop_callback])

if args.train:
    trainer.fit(model)

trainer.test(model)