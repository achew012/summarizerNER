from clearml import Task, StorageManager, Dataset
import argparse
import json, os, ipdb

config = {
"seed": 1234, 
"lr": 3e-04, 
"warmup": 1000, 
"num_workers": 4, 
"max_output_len": 96, 
"data_dir": "/data",
"output_dir": "./saved_models/test", 
"val_every": 0.33, 
"max_input_len": 1024, 
"batch_size": 1, 
"eval_batch_size": 1, 
"accumulate_grad_batches": 1, 
"fp16": False, 
"grad_ckpt": True, 
"attention_window": 256,
"num_epochs": 5,
"max_steps": -1,
"weight_decay": 0.0,
"adam_epsilon": 1e-8,
"gradient_clip_val": 1.0,
"warmup_steps": 0,
"model_name": 'allenai/led-base-16384',
#"model_name": 'facebook/bart-large'
}
args = argparse.Namespace(**config)

Task.add_requirements('transformers', package_version='4.2.0')
task = Task.init(project_name='LangGen', task_name='promptNER-fixedwords-led', output_uri="s3://experiment-logging/storage/")
clearlogger = task.get_logger()

# task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
# task.connect(args)
# task.execute_remotely(queue_name="128RAMv100", exit_process=True)

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
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, LEDModel, AutoConfig, get_linear_schedule_with_warmup 
from led_model import LEDConstrainedGen
from bart_gen import BartConstrainedGen

role_map = {
    'PerpOrg': '<PerpOrg>', 
    'PerpInd': '<PerpInd>',
    'Victim': '<Victim>',
    'Target': '<Target>',
    'Weapon': '<Weapon>'
}

#########################################################################################################################################
def convert_templates_to_prompts(templates, tokenizer):
    input_template = ["<PerpOrg><PerpInd><Victim><Target><Weapon>" for doc in templates]
    
    filled_templates = ["<PerpInd>{}<PerpOrg>{}<Victim>{}<Target>{}<Weapon>{}".format(doc["<PerpInd>"], doc["<PerpOrg>"], doc["<Victim>"], doc["<Target>"], doc["<Weapon>"]) for doc in templates]
    return input_template, filled_templates

class NERDataset(Dataset):
    # doc_list
    # extracted_list as template
    def __init__(self, dataset, tokenizer, args):
        self.tokenizer = tokenizer
        self.docs = [doc["doctext"] for doc in dataset]
        # take only 1st mention of each role
        first_mention_extracts = [{role_map[key]: doc["extracts"][key][0][0][0] if len(doc["extracts"][key])>0 else "" for key in doc["extracts"].keys()} for doc in dataset]
        # convert extracts to prompt template
        self.input_template, self.filled_templates = convert_templates_to_prompts(first_mention_extracts, tokenizer)
        #self.encodings = self.tokenizer(self.input_template, self.docs, padding="max_length", truncation=True, max_length=args.max_input_len, return_tensors="pt")        
        self.encodings = self.tokenizer(self.docs, padding="max_length", truncation=True, max_length=args.max_input_len, return_tensors="pt")
        self.decoder_encodings = self.tokenizer(self.filled_templates, padding="max_length", truncation=True, max_length=args.max_output_len, return_tensors="pt")

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.docs)

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        item = {key: val[idx] for key, val in self.encodings.items()}
        # item['labels'] = self.labels["input_ids"][idx]
        item['decoder_input_ids'] = self.decoder_encodings["input_ids"][idx]
        item['decoder_mask'] = ~self.decoder_encodings["attention_mask"][idx]
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
        decoder_input_ids = torch.stack([ex['decoder_input_ids'] for ex in batch]) 
        decoder_mask = torch.stack([ex['decoder_mask'] for ex in batch]) 
        
        return {
            "input_token_ids": input_ids,
            "input_attn_mask": attention_mask,
            "tgt_token_ids": decoder_input_ids,
            "tgt_attn_mask": decoder_mask 
        }

####################################################################################################################
class NERLED(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""
    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.save_hyperparameters()
        self.args = params
        self.dataset = muc4

        # Load and update config then load a pretrained LEDForConditionalGeneration
        self.config = AutoConfig.from_pretrained(self.args.model_name)
        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)
        #self.tokenizer.add_tokens(['<ent>'])
        self.tokenizer.add_tokens([value for key, value in role_map.items()])

        self.vocab_size = len(self.tokenizer) 
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        if self.args.model_name == "allenai/led-base-16384":
            self.model = LEDConstrainedGen(self.config, self.tokenizer)
        elif self.args.model_name == "facebook/bart-large":
            self.model = BartConstrainedGen(self.config, self.tokenizer)

        self.model.resize_token_embeddings()

        
    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        inputs = {
                    "input_ids": batch["input_token_ids"],
                    "attention_mask": batch["input_attn_mask"],
                    "decoder_input_ids": batch['tgt_token_ids'],
                    "decoder_attention_mask": batch["tgt_attn_mask"],   
                    "task": 0 
                }
        outputs = self.model(**inputs)

        loss = outputs[0]
        #loss = torch.mean(loss)

        log = {
            'train/loss': loss, 
        } 

        return {
            'loss': loss, 
            'log': log
        }

    def _get_dataloader(self, split_name, is_train):
        """Get training and validation dataloaders"""
        dataset_split = self.dataset[split_name]
        dataset = NERDataset(dataset=dataset_split, tokenizer=self.tokenizer, args=self.args)
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

    def validation_step(self,batch, batch_idx):
        inputs = {
                    "input_ids": batch["input_token_ids"],
                    "attention_mask": batch["input_attn_mask"],
                    "decoder_input_ids": batch['tgt_token_ids'],
                    "decoder_attention_mask": batch["tgt_attn_mask"],  
                    "task" :0,   
                }
        outputs = self.model(**inputs)
        loss = outputs[0]
        loss = torch.mean(loss)
        
        return loss  

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        log = {
            'val_loss': avg_loss, 
        } 

        self.log("val_loss", log["val_loss"])

    def test_step(self, batch, batch_idx):
        # if self.args.sample_gen:
        #     sample_output = self.model.generate(batch['input_token_ids'], do_sample=True, 
        #                         top_k=20, top_p=0.95, max_length=30, num_return_sequences=1,num_beams=1,
        #                     )
        # else:

        sample_output = self.model.generate(batch['input_token_ids'], do_sample=False, 
                            max_length=self.args.max_output_len, num_beams=1,
                        )
        
        sample_output = sample_output.reshape(batch['input_token_ids'].size(0), 1, -1)
        # doc_key = batch['doc_key'] # list 
        tgt_token_ids = batch['tgt_token_ids']

        return {"preds": sample_output, "gold": tgt_token_ids} 

    def test_epoch_end(self, outputs):
        # evaluate F1 
        with open('predictions.jsonl','w') as writer:
            for x in outputs:
                for idx in range(len(x["preds"])):
                    pred = {
                        'predicted': self.tokenizer.decode(x["preds"][idx].squeeze(0), skip_special_tokens=True),
                        'gold': self.tokenizer.decode(x["gold"][idx].squeeze(0), skip_special_tokens=True) 
                    }
                    writer.write(json.dumps(pred)+'\n')


        task.upload_artifact(name='predictions', artifact_object="./predictions.jsonl")

        return {} 

    # def configure_optimizers(self):
    #     """Configure the optimizer and the learning rate scheduler"""
    #     # Freeze the model
    #     # for idx, (name, parameters) in enumerate(self.model.named_parameters()):
    #     #     if idx<6:
    #     #         parameters.requires_grad=False
    #     #     else:
    #     #         parameters.requires_grad=True

    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
    #     return [optimizer]    

    def configure_optimizers(self):
        self.train_len = len(self.train_dataloader())
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_epochs = self.args.max_steps // self.train_len // self.args.accumulate_grad_batches + 1
        else:
            t_total = self.train_len // self.args.accumulate_grad_batches * self.args.num_epochs
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
        # scheduler is called only once per epoch by default 
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'linear-schedule',
        }
        return [optimizer, ], [scheduler_dict,]

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = "./",
    filename="best_ner_model", 
    monitor="val_loss", 
    mode="min", 
    save_top_k=1, 
    save_weights_only=True,
    period=5
)

# trained_model_path = bucket_ops.get_file(
#     remote_path="s3://experiment-logging/storage/LangGen/promptNER-fixedwords-led.9884106e43884dcda03b8ab5e0e5b792/models/best_ner_model-v6.ckpt"
#     )

#trained_model_path = "best_entity_lm.ckpt"

model = NERLED(args)
#model = NERLED.load_from_checkpoint(trained_model_path, params = args)
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, callbacks=[checkpoint_callback])
trainer.fit(model)
results = trainer.test(model)

