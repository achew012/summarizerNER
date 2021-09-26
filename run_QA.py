from clearml import Task, StorageManager, Dataset
import argparse
import json, os

config = json.load(open('config.json'))
args = argparse.Namespace(**config)

Task.add_requirements('transformers', package_version='4.2.0')
task = Task.init(project_name='LangGen', task_name='promptNER-QA', output_uri="s3://experiment-logging/storage/")
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
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from rouge_score import rouge_scorer


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
        # first_mention_position = [{
        #     "qns": [role_map[key].lower() for key in doc["extracts"].keys()], 
        #     "context": [doc["doctext"] for key in doc["extracts"].keys()], 
        #     "start": [doc["extracts"][key][0][0][1] if len(doc["extracts"][key])>0 else None for key in doc["extracts"].keys()], 
        #     "end": [doc["extracts"][key][0][0][1]+len(doc["extracts"][key][0][0][0]) if len(doc["extracts"][key])>0 else None for key in doc["extracts"].keys()]
        #     } for doc in dataset]

        self.first_mention_position = {
            "qns": [role_map[key].lower()  for doc in dataset for key in doc["extracts"].keys()], 
            "context": [doc["doctext"] for doc in dataset for key in doc["extracts"].keys()], 
            "start": [doc["extracts"][key][0][0][1] if len(doc["extracts"][key])>0 else 0 for doc in dataset for key in doc["extracts"].keys()], 
            "end": [doc["extracts"][key][0][0][1]+len(doc["extracts"][key][0][0][0]) if len(doc["extracts"][key])>0 else 0 for doc in dataset for key in doc["extracts"].keys()]
            }

        self.context = tokenizer(self.first_mention_position["qns"], self.first_mention_position["context"], padding=True, truncation=True, max_length=768, return_tensors="pt")

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.first_mention_position["context"])

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        item = {key: val[idx] for key, val in self.context.items()}
        item['start'] = torch.tensor(self.first_mention_position["start"])[idx]
        item['end'] = torch.tensor(self.first_mention_position["end"])[idx]
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
        start = torch.stack([ex['start'] for ex in batch]) 
        end = torch.stack([ex['end'] for ex in batch]) 
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start,
            'end_positions': end,
        }

####################################################################################################################
from transformers import LEDTokenizer, LEDForConditionalGeneration, LEDForQuestionAnswering

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
        self.model = LEDForQuestionAnswering.from_pretrained("allenai/led-base-16384", config=self.config)

        # Load tokenizer and metric
        self.tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384', use_fast=True)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

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
        outputs = self.model(**batch)
        # outputs = self.model(input_ids=input_ids,
        #             attention_mask=attention_mask,  # mask padding tokens
        #             start_positions=start,
        #             end_positions=end,
        #             use_cache=False)

        # start_logits = outputs.start_logits
        # end_logits = outputs.end_logits          
        return outputs

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        outputs = self.forward(**batch)
        return {'loss': outputs.loss}

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

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def _evaluation_step(self, split, batch, batch_nb):
        """Validaton or Testing - predict output, compare it with gold, compute rouge1, 2, L, and log result"""

        # input_ids, attention_mask, start, end  = batch["input_ids"], batch["attention_mask"], batch["start"], batch["end"]
        outputs = self(**batch)  # mask padding tokens
        
        candidates = torch.topk(outputs.start_logits, 3)

        for sample in candidates.indices:
            print(self.tokenizer.decode(sample))

        # import ipdb; ipdb.set_trace()

        logs = {
            "val_loss": outputs.loss
        }
            
        return {"val_loss": logs["val_loss"]}

    # def test_epoch_end(self, outputs):
    #     return {"preds": outputs}

    def validation_step(self, batch, batch_nb):
        self._evaluation_step('val', batch, batch_nb)

    def test_step(self, batch, batch_nb):
        self._evaluation_step('test', batch, batch_nb)

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        # Freeze the model
        # for idx, (name, parameters) in enumerate(self.model.named_parameters()):
        #     if idx<6:
        #         parameters.requires_grad=False
        #     else:
        #         parameters.requires_grad=True

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
#model = NERLED.load_from_checkpoint(trained_model_path, params = args)
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, callbacks=[checkpoint_callback])
trainer.fit(model)
results = trainer.test(model)
print(results)

