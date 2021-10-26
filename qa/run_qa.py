from clearml import Task, StorageManager, Dataset
import argparse
import json, os
import jsonlines
import ipdb
from collections import OrderedDict

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
"max_input_len": 470, 
"batch_size": 4, 
"eval_batch_size": 4, 
"grad_accum": 1, 
"fp16": False, 
"grad_ckpt": True, 
"attention_window": 256,
"num_epochs": 15,
"use_entity_embeddings": True,
"embedding_path":"s3://experiment-logging/storage/ner-pretraining/EntitySpanPretraining.16fec40c8a0e4198b97b231f8cdfcb7f/models/best_entity_lm.ckpt",
"model_name": "allenai/longformer-base-4096",
#"model_name": "mrm8488/longformer-base-4096-finetuned-squadv2",  
"debug": False
}
#json.load(open('config.json'))

args = argparse.Namespace(**config)
task = Task.init(project_name='LangGen', task_name='MRC-NER-PRETRAINEDSPANS-mlm+span_loss', output_uri="s3://experiment-logging/storage/")
clearlogger = task.get_logger()

task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
task.connect(args)
task.execute_remotely(queue_name="128RAMv100", exit_process=True)

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

role_list = ["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]

#############################################################################################

import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from eval import eval_ceaf

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
    def __init__(self, dataset, tokenizer, args):
        self.tokenizer = tokenizer

        self.processed_dataset = {
            "docid": [],
            "context": [],
            "input_ids": [],
            "attention_mask": [],
            "qns": [],
            "gold_mentions": [],
            "start": [],
            "end": []
        }

        overflow_count = 0

        for doc in dataset:
            docid = doc["docid"]
            context = doc["doctext"] #self.tokenizer.decode(self.tokenizer.encode(doc["doctext"]))

            ### Only take the 1st label of each role
            qns_ans = [["who are the {} entities?".format(role_map[key].lower()), doc["extracts"][key][0][0][1] if len(doc["extracts"][key])>0 else 0, doc["extracts"][key][0][0][1]+len(doc["extracts"][key][0][0][0]) if len(doc["extracts"][key])>0 else 0, doc["extracts"][key][0][0][0] if len(doc["extracts"][key])>0 else ""] for key in doc["extracts"].keys()]    
            ### expand on all labels in each role
            #qns_ans = [["who are the {} entities?".format(role_map[key].lower()), mention[1] if len(mention)>0 else 0, mention[1]+len(mention[0]) if len(mention)>0 else 0, mention[0] if len(mention)>0 else ""] for key in doc["extracts"].keys() for cluster in doc["extracts"][key] for mention in cluster]    

            for qns, ans_char_start, ans_char_end, mention in qns_ans:
                context_encodings = self.tokenizer(qns, context, padding="max_length", truncation=True, max_length=args.max_input_len, return_offsets_mapping=True, return_tensors="pt")
                sequence_ids = context_encodings.sequence_ids()

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                qns_offset = sequence_ids.index(1)-1
                pad_start_idx = sequence_ids[sequence_ids.index(1):].index(None)
                offsets_wo_pad = context_encodings["offset_mapping"][0][qns_offset:pad_start_idx]

                # if char indices more than end idx in last word span
                if ans_char_end>offsets_wo_pad[-1][1] or ans_char_start>offsets_wo_pad[-1][1]:
                    overflow_count+=1
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

                # print("span: ", tokenizer.decode(context_encodings["input_ids"][0][token_span[0]+qns_offset:token_span[1]+1+qns_offset]))
                # print("mention: ", mention)

                self.processed_dataset["docid"].append(docid)
                self.processed_dataset["context"].append(context)
                self.processed_dataset["input_ids"].append(context_encodings["input_ids"].squeeze(0))
                self.processed_dataset["attention_mask"].append(context_encodings["attention_mask"].squeeze(0))
                self.processed_dataset["qns"].append(docid)
                self.processed_dataset["gold_mentions"].append(mention)
                self.processed_dataset["start"].append(token_span[0]+qns_offset)
                self.processed_dataset["end"].append(token_span[1]+qns_offset+1)

            # ipdb.set_trace()

        print("OVERFLOW COUNT: ", overflow_count)

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.processed_dataset["docid"])

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        #item = {key: val[idx] for key, val in self.processed_dataset["encodings"].items()}
        item={}
        item['input_ids'] = self.processed_dataset["input_ids"][idx]
        item['attention_mask'] = self.processed_dataset["attention_mask"][idx]
        item['docid'] = self.processed_dataset["docid"][idx]
        item['gold_mentions'] = self.processed_dataset["gold_mentions"][idx]
        item['start'] = torch.tensor(self.processed_dataset["start"])[idx]
        item['end'] = torch.tensor(self.processed_dataset["end"])[idx]
        return item

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """

        docids = [ex['docid'] for ex in batch]
        gold_mentions = [ex['gold_mentions'] for ex in batch]
        input_ids = torch.stack([ex['input_ids'] for ex in batch]) 
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch]) 
        start = torch.stack([ex['start'] for ex in batch]) 
        end = torch.stack([ex['end'] for ex in batch]) 
        
        return {
            'docid': docids,
            'gold_mentions': gold_mentions,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start,
            'end_positions': end,
        }

##################################################################################

import re, string, collections

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def read_golds_from_test_file(data_dir, tokenizer):
    golds = OrderedDict()
    doctexts_tokens = OrderedDict()
    file_path = os.path.join(data_dir, "test.json")
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            docid=line["docid"]
            #docid = int(line["docid"].split("-")[0][-1])*10000 + int(line["docid"].split("-")[-1]) # transform TST1-MUC3-0001 to int(0001)
            doctext, extracts_raw = line["doctext"], line["extracts"]

            extracts = OrderedDict()
            for role, entitys_raw in extracts_raw.items():
                extracts[role] = []
                for entity_raw in entitys_raw:
                    entity = []
                    for mention_offset_pair in entity_raw:
                        entity.append(mention_offset_pair[0])
                    if entity:
                        extracts[role].append(entity)
            doctexts_tokens[docid] = tokenizer.tokenize(doctext)
            golds[docid] = extracts
    return doctexts_tokens, golds

####################################################################################################################
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from pretrain import NERLongformer

class NERLongformerQA(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""
    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.save_hyperparameters()
        self.args = params
        self.dataset = muc4

        # Load and update config then load a pretrained LEDForConditionalGeneration
        self.config = AutoConfig.from_pretrained(self.args.model_name)
        self.config.gradient_checkpointing = True
        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.args.model_name, config=self.config)

        # Use pretrained embeddings
        if self.args.use_entity_embeddings:

            pretrained_lm_path = bucket_ops.get_file(
                remote_path=self.args.embedding_path
            )

            lm_args={
                "lr": 5e-4,
                "num_epochs":5,
                "train_batch_size":12,
                "eval_batch_size":1,
                "max_length": 2048, # be mindful underlength will cause device cuda side error
                "max_span_len": 15,
                "max_spans": 25,
                "mlm_task": False,
                "bio_task": True,
            }
            lm_args = argparse.Namespace(**lm_args)

            # WARNING different longformer models have different calls
            #self.model.longformer = NERLongformer.load_from_checkpoint(pretrained_lm_path, args = lm_args).longformerMLM
            self.model.longformer = NERLongformer.load_from_checkpoint(pretrained_lm_path, args = lm_args).longformer

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
        #global_attention_mask[:, :1] = 1

        # # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1

        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        # Sets the questions for global attention
        batch_size = input_ids.size()[0]
        question_separators = (input_ids == 2).nonzero(as_tuple=True)
        sep_indices_batch = [torch.masked_select(question_separators[1], torch.eq(question_separators[0], batch_num))[0] for batch_num in range(batch_size)]

        for batch_num in range(batch_size):
            global_attention_mask[batch_num, :sep_indices_batch[batch_num]]=1

        return global_attention_mask

    def forward(self, **batch): 
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"], 

        if "start_positions" in batch.keys():
            start, end = batch["start_positions"], batch["end_positions"]

            outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,  # mask padding tokens
                            global_attention_mask=self._set_global_attention_mask(input_ids),
                            start_positions=start,
                            end_positions=end
                            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask  # mask padding tokens
            )

        return outputs

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        docids = batch.pop("docid", None)
        gold_mentions = batch.pop("gold_mentions", None)

        outputs = self.forward(**batch)
        return {'loss': outputs.loss}

    def _get_dataloader(self, split_name, is_train):
        """Get training and validation dataloaders"""
        if self.args.debug:
            dataset_split = self.dataset[split_name]
        else:
            dataset_split = self.dataset[split_name]

        dataset = NERDataset(dataset=dataset_split, tokenizer=self.tokenizer, args=self.args)
        if split_name in ["val", "test"]:
            return DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=NERDataset.collate_fn)
        else:
            return DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, collate_fn=NERDataset.collate_fn)

    def train_dataloader(self):
        print("train")
        return self._get_dataloader('train', is_train=True)

    def val_dataloader(self):
        print("val")
        return self._get_dataloader('val', is_train=False)

    def test_dataloader(self):
        print("test")
        return self._get_dataloader('test', is_train=False)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def _evaluation_step(self, split, batch, batch_nb):
        """Validaton or Testing - predict output, compare it with gold, compute rouge1, 2, L, and log result"""        
        # input_ids, attention_mask, start, end  = batch["input_ids"], batch["attention_mask"], batch["start_positions"], batch["end_positions"]

        docids = batch.pop("docid", None)
        gold_mentions = batch.pop("gold_mentions", None)
        
        outputs = self(**batch)  # mask padding tokens        
        candidates_start_batch = torch.topk(outputs.start_logits, 20)
        candidates_end_batch = torch.topk(outputs.end_logits, 20)

        # Get qns mask
        batch_size = batch["input_ids"].size()[0]
        question_separators = (batch["input_ids"] == 2).nonzero(as_tuple=True)
        sep_indices_batch = [torch.masked_select(question_separators[1], torch.eq(question_separators[0], batch_num))[0] for batch_num in range(batch_size)]
        question_indices_batch = [[i+1 for i,token in enumerate(tokens[1:sep_idx+1])] for tokens, sep_idx in zip(batch["input_ids"], sep_indices_batch)]
        batch_outputs = []

        #For each sample in batch
        for start_candidates, start_candidates_logits, end_candidates, end_candidates_logits, tokens, question_indices, start_gold, end_gold, attention_mask, docid, gold_mention in zip(candidates_start_batch.indices, candidates_start_batch.values, candidates_end_batch.indices, candidates_end_batch.values, batch["input_ids"], question_indices_batch, batch["start_positions"], batch["end_positions"], batch["attention_mask"], docids, gold_mentions):
            valid_candidates = []
            # For each candidate in sample
            for start_index, start_score in zip(start_candidates, start_candidates_logits):
                for end_index, end_score in zip(end_candidates, end_candidates_logits):
                    # throw out invalid predictions
                    if start_index in question_indices:
                        continue
                    elif end_index in question_indices:
                        continue
                    elif end_index < start_index:
                        continue
                    elif (end_index-start_index)>30:
                        continue
                    elif (start_index)>(torch.count_nonzero(attention_mask)):
                        continue

                    if start_index==0:
                        if len(valid_candidates)<1:
                            valid_candidates.append((start_index.item(), end_index.item(), "", 0))
                    else:
                        valid_candidates.append((start_index.item(), end_index.item(),  self.tokenizer.decode(tokens[start_index:end_index]), (start_score+end_score).item()))

            batch_outputs.append(
                {
                    "docid": docid,
                    "qns": self.tokenizer.decode(tokens[1:len(question_indices)]),
                    "gold_mention": gold_mention,
                    "context": self.tokenizer.decode(torch.masked_select(tokens, torch.gt(attention_mask, 0))[1+len(question_indices):]), 
                    "start_gold": start_gold.item(),
                    "end_gold": end_gold.item(),
                    "gold": self.tokenizer.decode(tokens[start_gold:end_gold]),
                    "candidates": valid_candidates[:5]
                }
            )

        logs = {
            "loss": outputs.loss
        }        
        
        return {"loss": logs["loss"], "preds": batch_outputs}

    def validation_step(self, batch, batch_nb):
        out = self._evaluation_step('val', batch, batch_nb)
        return {"results": out["preds"], "loss": out["loss"]}

    def validation_epoch_end(self, outputs):
        total_loss = []
        for batch in outputs:
            total_loss.append(batch["loss"])
        self.log("val_loss", sum(total_loss)/len(total_loss))

    def test_step(self, batch, batch_nb):
        out = self._evaluation_step('test', batch, batch_nb)
        return {"results": out["preds"]}

    #################################################################################
    def test_epoch_end(self, outputs):
        #gold_list = NERDataset(dataset=self.dataset["test"], tokenizer=self.tokenizer, args=self.args).processed_dataset["gold_mentions"]       
        pred_list = []

        logs={}
        doctexts_tokens, golds = read_golds_from_test_file(os.path.join(dataset_folder, "data/muc4-grit/processed/"), self.tokenizer)
        
        if self.args.debug:
            golds = {key:golds[key] for idx, key in enumerate(golds.keys()) if idx<10}

        predictions = {}
        for batch in outputs:
            for sample in batch["results"]:                
                pred_list.append(sample["candidates"][:1][0][2] if sample["candidates"][:1][0][2]!="</s>" else "")
                if sample["docid"] not in predictions.keys():
                    predictions[sample["docid"]]={
                        "docid": sample["docid"],
                        "context": sample["context"],
                        "qns": [sample["qns"]],
                        "gold_mention": [sample["gold_mention"]],
                        "gold": [sample["gold"]],
                        "candidates": [sample["candidates"][:1]]
                   }
                else:
                    predictions[sample["docid"]]["qns"].append(sample["qns"])
                    predictions[sample["docid"]]["gold_mention"].append(sample["gold_mention"])
                    predictions[sample["docid"]]["gold"].append(sample["gold"])
                    predictions[sample["docid"]]["candidates"].append(sample["candidates"][:1])

        preds = OrderedDict()
        for key, doc in predictions.items():
            if key not in preds:
                preds[key] = OrderedDict()
                for idx, role in enumerate(role_list):
                    preds[key][role] = []
                    if idx+1 > len(doc["candidates"]): 
                        continue
                    elif doc["candidates"][idx]:
                        if doc["candidates"][idx][0][2]=="</s>":
                            continue
                        else:    
                            preds[key][role] = [[doc["candidates"][idx][0][2].replace("</s>", "")]]                

        preds_list = [{**doc, "docid": key} for key, doc in preds.items()]
        results = eval_ceaf(preds, golds)
        print("================= CEAF score =================")
        print("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
        print("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
        print("==============================================")
        logs["test_micro_avg_f1_phi_strict"] = results["strict"]["micro_avg"]["f1"]
        logs["test_micro_avg_precision_phi_strict"] = results["strict"]["micro_avg"]["p"]
        logs["test_micro_avg_recall_phi_strict"] = results["strict"]["micro_avg"]["r"]
        clearlogger.report_scalar(title='f1', series = 'test', value=logs["test_micro_avg_f1_phi_strict"], iteration=1) 
        clearlogger.report_scalar(title='precision', series = 'test', value=logs["test_micro_avg_precision_phi_strict"], iteration=1) 
        clearlogger.report_scalar(title='recall', series = 'test', value=logs["test_micro_avg_recall_phi_strict"], iteration=1) 

        # f1_list = [compute_f1(gold, pred) for pred, gold in zip(pred_list, gold_list)]
        # mean_F1 = sum(f1_list)/len(f1_list)
        # EM_list = [compute_exact(gold, pred) for pred, gold in zip(pred_list, gold_list)]
        # mean_EM = sum(EM_list)/len(EM_list)

        #clearlogger.report_scalar(title='f1', series = 'test', value=mean_F1, iteration=1) 
        #clearlogger.report_scalar(title='EM', series = 'test', value=mean_EM, iteration=1) 

        to_jsonl("./predictions.jsonl", preds_list)
        task.upload_artifact(name='predictions', artifact_object="./predictions.jsonl")
        return {"results": results}

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        # if self.args.use_entity_embeddings: 
        #     for (_, parameters) in self.longformer.named_parameters():
        #         parameters.requires_grad=False 

        # Freeze the model
        # for idx, (name, parameters) in enumerate(self.model.named_parameters()):
        #     if idx<6:
        #         parameters.requires_grad=False
        #     else:
        #         parameters.requires_grad=True

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
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

#base
# trained_model_path = bucket_ops.get_file(
#             remote_path="s3://experiment-logging/storage/LangGen/promptNER-QA-Base.7ffa23a05839464980272aa317353fc3/models/best_ner_model.ckpt"
#             )

#squad-finetune
# trained_model_path = bucket_ops.get_file(
#             remote_path="s3://experiment-logging/storage/LangGen/promptNER-QA-Squad.2bfce82a26464e37a945c4696a8036f6/models/best_ner_model.ckpt"
#             )

# trained_model_path = bucket_ops.get_file(
#             remote_path="s3://experiment-logging/storage/LangGen/MRC-NER-PRETRAINSSPANS.962f18a7e03b4c99a18972f53b667d42/models/best_ner_model.ckpt"
#             )

#model = NERLongformerQA.load_from_checkpoint(trained_model_path, params = args)
model = NERLongformerQA(args)
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, callbacks=[checkpoint_callback])
trainer.fit(model)
results = trainer.test(model)
