from clearml import Task, StorageManager, Dataset
import argparse
import json, os

config = json.load(open('configQA.json'))
args = argparse.Namespace(**config)

Task.add_requirements('transformers', package_version='4.2.0')
task = Task.init(project_name='LangGen', task_name='promptNER-fixedwords', output_uri="s3://experiment-logging/storage/")
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
from constrained_gen import LEDConstrainedGen

role_map = {
    'PerpOrg': 'perpetrator organizations', 
    'PerpInd': 'perpetrator individuals',
    'Victim': 'victims',
    'Target': 'targets',
    'Weapon': 'weapons'
}

#########################################################################################################################################
def convert_templates_to_prompts(templates, tokenizer):
    input_template = ["The <arg> from <arg> used a <arg> to attack <arg> injuring <arg>"  for doc in templates]
    filled_templates = ["{} from {} used {} to attack {} harming {}".format(doc["perpetrator individuals"], doc["perpetrator organizations"], doc["weapons"], doc["targets"], doc["victims"]) for doc in templates]
    return input_template, filled_templates

class NERDataset(Dataset):
    # doc_list
    # extracted_list as template
    def __init__(self, dataset, tokenizer, args):
        self.tokenizer = tokenizer
        self.docs = [doc["doctext"] for doc in dataset]
        # take only 1st mention of each role
        first_mention_extracts = [{role_map[key].lower(): doc["extracts"][key][0][0][0] if len(doc["extracts"][key])>0 else "<ent>" for key in doc["extracts"].keys()} for doc in dataset]
        # convert extracts to prompt template
        self.input_template, self.filled_templates = convert_templates_to_prompts(first_mention_extracts, tokenizer)
        self.encodings = self.tokenizer(self.input_template, self.docs, padding="max_length", truncation=True, max_length=args.max_input_len, return_tensors="pt")        
        self.decoder_encodings = self.tokenizer(self.input_template, padding="max_length", truncation=True, max_length=args.max_output_len, return_tensors="pt")
        # import ipdb; ipdb.set_trace()

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
            "tgt_input_ids": decoder_input_ids,
            "tgt_attn_mask": decoder_mask 
        }

####################################################################################################################
from transformers import LEDTokenizer, LEDModel, AutoConfig 

class NERLED(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""
    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.save_hyperparameters()
        self.args = params
        self.dataset = muc4

        # Load and update config then load a pretrained LEDForConditionalGeneration
        self.config = AutoConfig.from_pretrained('allenai/led-base-16384')
        self.config.gradient_checkpointing = True

        # Load tokenizer and metric
        self.tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384', use_fast=True)
        self.tokenizer.add_tokens(['<ent>'])
        self.vocab_size = len(self.tokenizer) 
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        self.model = LEDConstrainedGen(self.config, self.tokenizer)
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
        loss = torch.mean(loss)

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
            'val/loss': avg_loss, 
        } 
        return {
            'loss': avg_loss, 
            'log': log 
        }

    def test_step(self, batch, batch_idx):
        if self.hparams.sample_gen:
            sample_output = self.model.generate(batch['input_token_ids'], do_sample=True, 
                                top_k=20, top_p=0.95, max_length=30, num_return_sequences=1,num_beams=1,
                            )
        else:
            sample_output = self.model.generate(batch['input_token_ids'], do_sample=False, 
                                max_length=30, num_return_sequences=1,num_beams=1,
                            )
        
        sample_output = sample_output.reshape(batch['input_token_ids'].size(0), 1, -1)
        doc_key = batch['doc_key'] # list 
        tgt_token_ids = batch['tgt_token_ids']

        return (doc_key, sample_output, tgt_token_ids) 

    def test_epoch_end(self, outputs):
        # evaluate F1 
        with open('checkpoints/{}/predictions.jsonl'.format(self.hparams.ckpt_name),'w') as writer:
            for tup in outputs:
                for idx in range(len(tup[0])):
                    
                    pred = {
                        'doc_key': tup[0][idx],
                        'predicted': self.tokenizer.decode(tup[1][idx].squeeze(0), skip_special_tokens=True),
                        'gold': self.tokenizer.decode(tup[2][idx].squeeze(0), skip_special_tokens=True) 
                    }
                    writer.write(json.dumps(pred)+'\n')

        return {} 

    # def configure_optimizers(self):
    #     """Configure the optimizer and the learning rate scheduler"""
    #     # Freeze the model
    #     # for idx, (name, parameters) in enumerate(self.model.named_parameters()):
    #     #     if idx<6:
    #     #         parameters.requires_grad=False
    #     #     else:
    #     #         parameters.requires_grad=True

    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
    #     return [optimizer]    

    def configure_optimizers(self):
        self.train_len = len(self.train_dataloader())
        if self.hparams.max_steps > 0:
            t_total = self.hparams.max_steps
            self.hparams.num_train_epochs = self.hparams.max_steps // self.train_len // self.hparams.accumulate_grad_batches + 1
        else:
            t_total = self.train_len // self.hparams.accumulate_grad_batches * self.hparams.num_train_epochs

        logger.info('{} training steps in total.. '.format(t_total)) 
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # scheduler is called only once per epoch by default 
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'linear-schedule',
        }

        return [optimizer, ], [scheduler_dict,]

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = "./",
    filename="best_ner_model", 
    monitor="val_accuracy", 
    mode="max", 
    save_top_k=1, 
    save_weights_only=True,
    period=5
)

# trained_model_path = bucket_ops.get_file(
#     remote_path="s3://experiment-logging/storage/ner-pretraining/NER-LM.0b6dc1f3db3f41e1ad9c3db53bbd1b31/models/best_entity_lm.ckpt "
#     )

#trained_model_path = "best_entity_lm.ckpt"

model = NERLED(args)
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, callbacks=[checkpoint_callback])
trainer.fit(model)
results = trainer.test(model)
print(results)

