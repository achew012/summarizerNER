from datasets import load_dataset
from clearml import Task, StorageManager, Dataset
import argparse
import json, os, ipdb
import jsonlines

config = json.load(open('config.json'))
args = argparse.Namespace(**config)

task = Task.init(project_name='LongformerNER', task_name='simpleTokenClassification', tags=["muc4"], output_uri="s3://experiment-logging/storage/")
clearlogger = task.get_logger()

task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
task.connect(args)
task.execute_remotely(queue_name="128RAMv100", exit_process=True)

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


# #Read args from config file instead, use vars() to convert namespace to dict
dataset = Dataset.get(dataset_name="muc4-processed", dataset_project="datasets/muc4", dataset_tags=["processed", "GRIT"], only_published=True)
dataset_folder = dataset.get_local_copy()
print(list(os.walk(os.path.join(dataset_folder, "data/muc4-grit/processed"))))

# dataset = load_dataset("wnut_17")

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


import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

role_map = {
    'PerpOrg': 'perpetrator organizations', 
    'PerpInd': 'perpetrator individuals',
    'Victim': 'victims',
    'Target': 'targets',
    'Weapon': 'weapons'
}

#########################################################################################################################################
class NERDataset(Dataset):
    # doc_list
    # extracted_list as template
    def __init__(self, dataset, tokenizer, labels2idx, args):
        self.tokenizer = tokenizer
        self.processed_dataset = {
            "docid": [],
            "context": [],
            "input_ids": [],
            "attention_mask": [],
            "gold_mentions": [],
            "labels": [],
        }
        self.labels2idx = labels2idx

        for doc in dataset:
            docid = doc["docid"]
            context = doc["doctext"] #self.tokenizer.decode(self.tokenizer.encode(doc["doctext"]))
            ### Only take the 1st label of each role
            qns_ans = [[key, doc["extracts"][key][0][0][1] if len(doc["extracts"][key])>0 else 0, doc["extracts"][key][0][0][1]+len(doc["extracts"][key][0][0][0]) if len(doc["extracts"][key])>0 else 0, doc["extracts"][key][0][0][0] if len(doc["extracts"][key])>0 else ""] for key in doc["extracts"].keys()]    

            ### expand on all labels in each role
            #qns_ans = [["who are the {} entities?".format(role_map[key].lower()), mention[1] if len(mention)>0 else 0, mention[1]+len(mention[0]) if len(mention)>0 else 0, mention[0] if len(mention)>0 else ""] for key in doc["extracts"].keys() for cluster in doc["extracts"][key] for mention in cluster]    

            #labels = [-100]*(self.tokenizer.model_max_length)
            labels = [self.labels2idx["O"]]*(self.tokenizer.model_max_length)
            
            length_of_sequence = len(self.tokenizer.tokenize(context)) if len(self.tokenizer.tokenize(context))<=len(labels) else len(labels)
            
            labels[:length_of_sequence] = [self.labels2idx["O"]]*length_of_sequence            

            context_encodings = self.tokenizer(context, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_offsets_mapping=True, return_tensors="pt")

            for qns, ans_char_start, ans_char_end, mention in qns_ans:
                sequence_ids = context_encodings.sequence_ids()

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                pad_start_idx = sequence_ids[sequence_ids.index(0):].index(None)
                offsets_wo_pad = context_encodings["offset_mapping"][0][sequence_ids.index(0):pad_start_idx]

                if ans_char_end>offsets_wo_pad[-1][1] or ans_char_start>offsets_wo_pad[-1][1]:
                    ans_char_start = 0
                    ans_char_end = 0

                if ans_char_start==0 and ans_char_end==0:
                    token_span=[0,0]
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
                    ipdb.set_trace()

                # self.processed_dataset["docid"].append(docid)
                # self.processed_dataset["context"].append(context)
                # self.processed_dataset["input_ids"].append(context_encodings["input_ids"].squeeze(0))
                # self.processed_dataset["attention_mask"].append(context_encodings["attention_mask"].squeeze(0))
                # self.processed_dataset["qns"].append(docid)
                # self.processed_dataset["gold_mentions"].append(mention)
                # self.processed_dataset["start"].append(token_span[0])
                # self.processed_dataset["end"].append(token_span[1]+1)

                if token_span!=[0,0]:
                    labels[token_span[0]:token_span[0]+1] = [self.labels2idx["B-"+qns]]
                    if len(labels[token_span[0]+1:token_span[1]+1])>0 and (token_span[1]-token_span[0])>0:
                        labels[token_span[0]+1:token_span[1]+1] = [self.labels2idx["I-"+qns]]*(token_span[1]-token_span[0])

            self.processed_dataset["docid"].append(docid)
            self.processed_dataset["context"].append(context)
            self.processed_dataset["input_ids"].append(context_encodings["input_ids"].squeeze(0))
            self.processed_dataset["attention_mask"].append(context_encodings["attention_mask"].squeeze(0))
            self.processed_dataset["gold_mentions"].append(mention)
            self.processed_dataset["labels"].append(torch.tensor(labels))

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.processed_dataset["docid"])

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        item = {key: val[idx] for key, val in self.processed_dataset.items()}
        return item

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """
        #pad_token_id = 1
        #input_ids, output_ids = list(zip(*batch))
        #input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        #output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)

        input_ids = torch.stack([ex['input_ids'] for ex in batch]) 
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch]) 
        labels = torch.stack([ex['labels'] for ex in batch]) 
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

####################################################################################################################
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class NERLongformer(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""
    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.save_hyperparameters()
        self.args = params
        self.dataset = muc4

        self.labels2idx = {classname: idx+1 for idx, classname in enumerate([prefix+key for key in role_map.keys() for prefix in ["B-", "I-"]])}
        self.labels2idx["O"] = 0

        # Load and update config then load a pretrained LEDForConditionalGeneration
        self.config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
        self.config.gradient_checkpointing = True
        self.config.num_labels = len(self.labels2idx.keys())
        self.model = AutoModelForTokenClassification.from_pretrained("allenai/longformer-base-4096", config=self.config)
        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True)
        self.tokenizer.model_max_length = 1024
        self.softmax = nn.Softmax(dim=-1)

        dataset_split = self.dataset["train"]
        dataset = NERDataset(dataset=dataset_split, tokenizer=self.tokenizer, labels2idx=self.labels2idx, args=self.args)
        y_train = torch.stack(dataset.processed_dataset["labels"]).view(-1).cpu().numpy()
        self.loss_weights=torch.cuda.FloatTensor(compute_class_weight("balanced", np.unique(y_train), y_train))

        # pretrained_lm_path = bucket_ops.get_file(
        #     remote_path="s3://experiment-logging/storage/ner-pretraining/MLM-Loss.118725620595422d84ed3379b630a0c9/models/best_entity_lm.ckpt"
        # )
        # lm_args={
        #     "lr": 3e-4,
        #     "train_batch_size":1,
        #     "eval_batch_size":1,
        # }
        # self.embeds = NERLongformer.load_from_checkpoint(pretrained_lm_path, args = lm_args).longformer



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

    def forward(self, **batch): 
        # input_ids, attention_mask = batch["input_ids"], batch["attention_mask"], 

        outputs = self.model(**batch) #, global_attention_mask=self._set_global_attention_mask(batch["input_ids"]))
        logits = outputs.logits
        
        loss=None
        
        if "labels" in batch.keys():
            loss_fct = nn.CrossEntropyLoss(self.loss_weights, ignore_index=-100)
            # Only keep active parts of the loss
            if batch["attention_mask"] is not None:
                active_loss = batch["attention_mask"].view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = batch["labels"].view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), batch["labels"].view(-1))

        return (loss, logits)

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        outputs = self.forward(**batch)
        return {'loss': outputs[0]}

    def _get_dataloader(self, split_name, is_train):
        """Get training and validation dataloaders"""
        dataset_split = self.dataset[split_name]
        dataset = NERDataset(dataset=dataset_split, tokenizer=self.tokenizer, labels2idx=self.labels2idx, args=self.args)

        if split_name in ["val", "test"]:
            return DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=NERDataset.collate_fn)
        else:
            return DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=NERDataset.collate_fn)

    def train_dataloader(self):
        return self._get_dataloader('train', is_train=True)

    def val_dataloader(self):
        return self._get_dataloader('val', is_train=False)

    def test_dataloader(self):
        return self._get_dataloader('test', is_train=False)

    def _evaluation_step(self, split, batch, batch_nb):
        """Validaton or Testing - predict output, compare it with gold, compute rouge1, 2, L, and log result"""
        outputs = self.forward(**batch)
        return {'loss': outputs[0], "logits": outputs[1]}

    def validation_step(self, batch, batch_nb):
        outputs = self._evaluation_step('val', batch, batch_nb)
        preds = torch.argmax(outputs["logits"], dim=-1)
        labels = batch["labels"]
        return {'val_loss': outputs["loss"], "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_logits = torch.stack([x["preds"] for x in outputs], dim=0).view(-1, self.tokenizer.model_max_length)
        val_labels = torch.stack([x["labels"] for x in outputs]).view(-1, self.tokenizer.model_max_length)
        #print(classification_report(val_logits.view(-1).cpu().detach(), val_labels.view(-1).cpu().detach()))
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_nb):
        outputs = self._evaluation_step('val', batch, batch_nb)
        preds = torch.argmax(self.softmax(outputs["logits"]), dim=-1)
        labels = batch["labels"]
        return {'test_loss': outputs["loss"], "preds": preds, "labels": labels}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_preds = torch.stack([x["preds"] for x in outputs], dim=0).view(-1, self.tokenizer.model_max_length)
        test_labels = torch.stack([x["labels"] for x in outputs]).view(-1, self.tokenizer.model_max_length)
        print(classification_report(test_preds.view(-1).cpu().detach(), test_labels.view(-1).cpu().detach()))

        self.idx2labels = {key: value for value, key in self.labels2idx.items()}
        predictions = [[(idx, self.idx2labels[tag]) for idx, tag in enumerate(doc)] for doc in test_preds.cpu().detach().tolist()]        
        
        focused = [[(idx, self.idx2labels[tag]) for idx, tag in enumerate(doc) if tag!="O"] for doc in test_preds.cpu().detach().tolist()]        

        to_jsonl("./predictions.jsonl", focused)
        task.upload_artifact(name='predictions', artifact_object="./predictions.jsonl")


    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
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

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = "./",
    filename="best_ner_model", 
    monitor="val_loss", 
    mode="min", 
    save_top_k=1, 
    save_weights_only=True,
    period=5
)

#trained_model_path = "best_entity_lm.ckpt"
model = NERLongformer(args)
#model = NERLED.load_from_checkpoint(trained_model_path, params = args)
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, callbacks=[checkpoint_callback])
trainer.fit(model)
results = trainer.test(model)
print(results)

