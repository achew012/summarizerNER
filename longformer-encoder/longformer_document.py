from clearml import Task, StorageManager
import argparse
import json, os, jsonlines
from datasets import load_dataset
import ipdb

config = {
"seed": 1234, 
"lr": 1e-04, 
"warmup": 1000, 
"num_workers": 4, 
"limit_val_batches": 0.005, 
"limit_test_batches": 0.005, 
"limit_train_batches": 0.002, 
"max_output_len": 64, 
"data_dir": "/data",
"output_dir": "./saved_models/test", 
"val_every": 0.33, 
"max_input_len": 1536, 
"batch_size": 1, 
"eval_batch_size": 4, 
"grad_accum": 1, 
"fp16": False, 
"grad_ckpt": True, 
"attention_window": 256,
"num_epochs": 10,
"model_name": "allenai/longformer-base-4096",
"num_labels": 118
}
args = argparse.Namespace(**config)

# task = Task.init(project_name='LangGen', task_name='Document Encoding', output_uri="s3://experiment-logging/storage/")
# clearlogger = task.get_logger()

# task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
# task.connect(args)
# task.execute_remotely(queue_name="128RAMv100", exit_process=True)

def read_json(jsonfile):
    with open(jsonfile, 'rb') as file:
        file_object = [json.loads(sample) for sample in file]
    return file_object

''' Writes to a jsonl file'''
def to_jsonl(filename:str, file_obj):
    resultfile = open(filename, 'wb')
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj)

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

# #Download Pretrained Models
# bucket_ops.download_folder(
#     local_path="/models/led-base-16384", 
#     remote_path="s3://experiment-logging/pretrained/led-base-16384", 
#     )


def read_json(jsonfile):
    with open(jsonfile, 'rb') as file:
        file_object = [json.loads(sample) for sample in file]
    return file_object

import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from rouge_score import rouge_scorer

#########################################################################################################################################
import itertools

class DocumentDataset(Dataset):
    # doc_list
    # extracted_list as template
    def __init__(self, docs, tokenizer, args):
        self.text = docs["text"]
        self.tokenizer = tokenizer
        self.context = tokenizer(self.text, padding="max_length", truncation=True, max_length=args.max_input_len, return_tensors="pt")
        self.idx2topics = {idx: value for idx, value in enumerate(set(itertools.chain.from_iterable(docs["topics"])))}
        self.topics2idx = {value: key for key, value in self.idx2topics.items()}
        self.label_indices = [[self.topics2idx[topic] for topic in doc] for doc in docs["topics"]]
        self.labels = torch.zeros(len(docs["topics"]), len(self.topics2idx.keys()), dtype=torch.uint8)
        for doc_idx, doc_labels in enumerate(self.label_indices):  
            for label_idx in doc_labels:
                self.labels[doc_idx, label_idx] = 1

        #convert_back = [self.idx2topics[idx] for idx, value in enumerate(self.labels[1]) if value==1]
        # or (self.labels == 1).nonzero(as_tuple=False)
    
    def __len__(self):
        """Returns length of the dataset"""
        return len(self.text)

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        item = {key: val[idx] for key, val in self.context.items()}
        item["labels"] = self.labels[idx]
        return item

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """

        input_ids = torch.stack([ex['input_ids'] for ex in batch]) 
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch]) 
        labels = torch.stack([ex['labels'] for ex in batch]) 
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

####################################################################################################################
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

# tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
# docs = load_dataset('reuters21578', 'ModHayes') #["text"] #["topics"]
# dataset = DocumentDataset(docs["test"], tokenizer, args)

# ipdb.set_trace()
# import sys; sys.exit()

####################################################################################################################

class LongformerEncoder(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""
    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.save_hyperparameters()
        self.args = params

        # Load and update config then load a pretrained LEDForConditionalGeneration
        self.config = AutoConfig.from_pretrained(self.args.model_name, gradient_checkpointing = True)
        self.model = AutoModel.from_pretrained(self.args.model_name, config=self.config)       

        #self.config.num_labels = self.args.num_labels
        #self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_name, config=self.config)
        
        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)

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
        global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        return global_attention_mask

    def forward(self, **batch): 
        
        #ipdb.set_trace()
        outputs = self.model(**batch, output_hidden_states=True)
        return outputs 

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""

        labels = batch.pop("labels", None)

        if labels!=None:
            #outputs = self({**batch, "labels": labels.type(torch.cuda.FloatTensor)})
            outputs = self({**batch})
            loss = outputs.loss
        else:
            loss = None

        return {'loss': loss}

    def _get_dataloader(self, split_name, is_train):
        """Get training and validation dataloaders"""
        docs = load_dataset('reuters21578', 'ModHayes')[split_name]
        dataset = DocumentDataset(docs, self.tokenizer, self.args)

        if split_name in ["val", "test"]:
            return DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers)
        else:
            return DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    def train_dataloader(self):
        return self._get_dataloader('train', is_train=True)

    def test_dataloader(self):
        return self._get_dataloader('test', is_train=False)

    def _evaluation_step(self, split, batch, batch_nb):
        # input_ids, attention_mask  = batch["input_ids"], batch["attention_mask"]
        labels = batch.pop("labels", None)
        outputs = self(**batch)
        
        # Get mean embeddings of last 4 layers of model (dim=0) and across the sequence (dim=seqlength)               
        combined_layers = torch.mean(torch.stack(outputs.hidden_states[-4:]), dim=0) # (bs, seqlen, embdim)
        doc_embeddings = torch.mean(combined_layers, dim=1) #(bs, 768)        
        return doc_embeddings

    def test_epoch_end(self, outputs):
        doc_embeddings = []
        for batch in outputs:
            for sample in batch["embeddings"]:
                doc_embeddings.append(sample)

        torch.save(doc_embeddings, 'doc_embeddings.pt')

        return {"embeddings": doc_embeddings}

    def test_step(self, batch, batch_nb):
        doc_embeddings = self._evaluation_step('test', batch, batch_nb)
        return {"embeddings": doc_embeddings}        

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parser):
        # **************** Parameters that we will NOT change during this tutorial **************** #
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--limit_val_batches", default=0.005, type=float, help='Percent of validation data used')
        parser.add_argument("--limit_test_batches", default=0.005, type=float, help='Percent of test data used')
        parser.add_argument("--limit_train_batches", default=0.002, type=float, help='Percent of training data used')
        parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces in the summary")
        parser.add_argument("--output_dir", type=str, default='./saved_models/test', help="Location of output dir")
        parser.add_argument("--val_every", default=0.33, type=float, help='Validation every')

        # **************** Parameters that we will change during this tutorial **************** #
        parser.add_argument("--max_input_len", type=int, default=8192, help="maximum num of wordpieces in the input")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--fp16", action='store_true', help="Use fp16 ")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_window", type=int, default=1024, help="Attention window")
        return parser


# trained_model_path = bucket_ops.get_file(
#     remote_path="s3://experiment-logging/storage/ner-pretraining/NER-LM.0b6dc1f3db3f41e1ad9c3db53bbd1b31/models/best_entity_lm.ckpt "
#     )

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = "./",
    filename="best_ner_model", 
    monitor="val_accuracy", 
    mode="max", 
    save_top_k=1, 
    save_weights_only=True,
    period=5
)

model = LongformerEncoder(args)
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, callbacks=[checkpoint_callback])
#trainer.fit(model)
results = trainer.test(model)

