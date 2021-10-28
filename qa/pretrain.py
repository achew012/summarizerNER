from clearml import Task, StorageManager, Dataset as ds
import argparse, json, os, random, math, ipdb

import os, random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerForMaskedLM, LongformerModel, LongformerTokenizer, LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerLMHead, _compute_global_attention_mask
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

from typing import Callable, List, Set, Tuple, TypeVar, Optional

class NERLongformer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        self.config.gradient_checkpointing = True

        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.tokenizer.model_max_length = self.args.max_length
        # self.longformer = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)

        self.longformer = LongformerModel(self.config)
        if self.args.mlm_task:
            self.lm_head = LongformerLMHead(self.config) 

        # train_data = DWIE_Data(dwie["train"], self.tokenizer, self.class2id, self.args)
        # y_train = torch.stack([doc["bio_labels"] for doc in train_data.consolidated_dataset]).view(-1).cpu().numpy()
        # y_train = y_train[y_train != -100]
        # self.class_weights=torch.cuda.FloatTensor(compute_class_weight("balanced", np.unique(y_train), y_train))
        # #self.class_weights[0] = self.class_weights[0]/2
        # self.class_weights = torch.cuda.FloatTensor([0.20, 1, 1.2, 1, 1.2])
        # print("weights: {}".format(self.class_weights))
        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size*2, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
            nn.Sigmoid(),
        )
        self.size_embeddings = nn.Embedding(self.args.max_span_len, self.config.hidden_size)
        
    # def val_dataloader(self):
    #     val = dwie["test"]
    #     val_data = DWIE_Data(val, self.tokenizer, self.args)
    #     val_dataloader = DataLoader(val_data, batch_size=self.args.eval_batch_size, collate_fn = val_data.collate_fn)
    #     return val_dataloader

    # def train_dataloader(self):
    #     train = dwie["train"]
    #     train_data = DWIE_Data(train, self.tokenizer, self.args)
    #     train_dataloader = DataLoader(train_data, batch_size=self.args.train_batch_size, collate_fn = train_data.collate_fn)
    #     return train_dataloader

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
        # in lightning, forward defines the prediction/inference actions
        masked_input_ids = batch.pop("masked_input_ids", None)
        entity_span = batch.pop("entity_span", None)
        entity_mask = batch.pop("entity_mask", None)

        span_len_mask = batch.pop("span_len_mask", None)
        labels = batch.pop("labels", None)

        outputs = self.longformer(
            **batch, 
            global_attention_mask=self._set_global_attention_mask(batch["input_ids"]), output_hidden_states=True
            )

        # cls_output = outputs[0][:, 1]
        sequence_output = outputs[0][:, 1:] 

        ### Span CLF Objective
        ############################################################################################################
        span_index = span_len_mask.view(span_len_mask.size(0), -1).unsqueeze(-1).expand(span_len_mask.size(0), span_len_mask.size(1)*span_len_mask.size(2), sequence_output.size(-1)) # (bs, num_samples*max_span_len, 768)
        span_len_binary_mask = span_len_mask.gt(0).long().unsqueeze(-1)
        span_len_binary_index = span_len_binary_mask.sum(-2).squeeze(-1)
        span_len_embeddings = self.size_embeddings(span_len_binary_index)

        span_len_binary_mask_embeddings = span_len_binary_mask.expand(span_len_binary_mask.size(0), span_len_binary_mask.size(1), span_len_binary_mask.size(2), sequence_output.size(-1))
        span_embeddings = torch.gather(sequence_output, 1, span_index.long()) # (bs, num_samples*max_span_len, 768)
        span_embedding_groups = torch.stack(torch.split(span_embeddings, 15, 1)).transpose(0,1) # (bs, num_samples, max_span_len, 768)
        extracted_spans_after_binary_mask = span_len_binary_mask_embeddings*span_embedding_groups #makes all padding positions into zero vectors (bs, num_samples, max_span_len, 768) 
        maxpooled_embeddings = torch.max(extracted_spans_after_binary_mask, dim=-2).values # (bs, num_samples, 768)

        # if len(cls_output.repeat(sequence_output.size()[0], span_len_mask.size()[1], 1).size())!=len(maxpooled_embeddings.size()):
        #     ipdb.set_trace()

        #combined_embeds = torch.cat([cls_output.unsqueeze(1).repeat(1, 40, 1), maxpooled_embeddings], dim=-1) #(bs, num_samples, 2*768)
        combined_embeds = torch.cat([span_len_embeddings, maxpooled_embeddings], dim=-1) #(bs, num_samples, 2*768)
        logits = self.classifier(combined_embeds).squeeze()        

        total_loss=0
        
        span_clf_loss = None
        if labels!=None:
            loss_fct = nn.BCELoss()
            span_clf_loss = loss_fct(logits.view(-1), labels.view(-1))
            total_loss+=span_clf_loss

        ### MLM Objective
        ##############################################################################################################
        if self.args.mlm_task:
            prediction_scores = self.lm_head(outputs[0])
 
            mlm_loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = mlm_loss_fct(prediction_scores.view(-1, self.config.vocab_size), batch["input_ids"].view(-1))
            total_loss+=masked_lm_loss
        ##############################################################################################################

        return (total_loss, logits, span_clf_loss, masked_lm_loss)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss, _, span_clf_loss, masked_lm_loss = self(**batch)
        # logits = torch.argmax(self.softmax(logits), dim=-1)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {
            "train_loss": train_loss_mean,
        }

        self.log("train_loss", logs["train_loss"])

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        #input_ids, attention_mask, labels = batch
        loss, logits, span_clf_loss, masked_lm_loss = self(**batch)
        clearlogger.report_text(logits)
        clearlogger.report_scalar(title='mlm_loss', series = 'val', value=masked_lm_loss, iteration=batch_idx) 
        preds = (logits>0.5).long()
        return {"val_loss": loss, "preds": preds, "labels": batch["labels"]}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_preds = torch.stack([x["preds"] for x in outputs]).view(-1).cpu().detach().tolist()
        val_labels = torch.stack([x["labels"] for x in outputs]).view(-1).cpu().detach().tolist()

        logs = {
            "val_loss": val_loss_mean,
        }
        precision, recall, f1, support = precision_recall_fscore_support(val_labels, val_preds, average='macro')

        self.log("val_loss", logs["val_loss"])
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

    # Freeze weights?
    def configure_optimizers(self):
        # Freeze alternate layers of longformer
        for idx, (name, parameters) in enumerate(self.longformer.named_parameters()):
            # if idx%2==0:
            #     parameters.requires_grad=False
            # else:
            #     parameters.requires_grad=True
            if idx<6:
                parameters.requires_grad=False
            else:
                parameters.requires_grad=True

        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer


# class NERLongformer(pl.LightningModule):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
#         self.config.gradient_checkpointing = True

#         self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
#         self.tokenizer.model_max_length = self.args.max_length

#         self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096', config=self.config)
#         self.longformerMLM = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', config=self.config)

#         self.wte = self.longformerMLM.get_input_embeddings()
#         self.sigmoid = nn.Sigmoid()
#         self.classifier = nn.Sequential(
#             nn.Linear(self.config.hidden_size*2, self.config.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.config.hidden_size, 1),
#             nn.Sigmoid(),
#         )
#         self.size_embeddings = nn.Embedding(self.args.max_span_len, self.config.hidden_size)
        
#     def val_dataloader(self):
#         val = dwie["test"]
#         val_data = DWIE_Data(val, self.tokenizer, self.args)
#         val_dataloader = DataLoader(val_data, batch_size=self.args.eval_batch_size, collate_fn = val_data.collate_fn)
#         return val_dataloader

#     def train_dataloader(self):
#         train = dwie["train"]
#         train_data = DWIE_Data(train, self.tokenizer, self.args)
#         train_dataloader = DataLoader(train_data, batch_size=self.args.train_batch_size, collate_fn = train_data.collate_fn)
#         return train_dataloader

#     def _set_global_attention_mask(self, input_ids):
#         """Configure the global attention pattern based on the task"""

#         # Local attention everywhere - no global attention
#         global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

#         # Gradient Accumulation caveat 1:
#         # For gradient accumulation to work, all model parameters should contribute
#         # to the computation of the loss. Remember that the self-attention layers in the LED model
#         # have two sets of qkv layers, one for local attention and another for global attention.
#         # If we don't use any global attention, the global qkv layers won't be used and
#         # PyTorch will throw an error. This is just a PyTorch implementation limitation
#         # not a conceptual one (PyTorch 1.8.1).
#         # The following line puts global attention on the <s> token to make sure all model
#         # parameters which is necessery for gradient accumulation to work.
#         global_attention_mask[:, :1] = 1

#         # # Global attention on the first 100 tokens
#         # global_attention_mask[:, :100] = 1

#         # # Global attention on periods
#         # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

#         return global_attention_mask


#     def forward(self, **batch):
#         # in lightning, forward defines the prediction/inference actions
#         masked_input_ids = batch.pop("masked_input_ids", None)
#         entity_span = batch.pop("entity_span", None)
#         entity_mask = batch.pop("entity_mask", None)
#         span_len_mask = batch.pop("span_len_mask", None)
#         labels = batch.pop("labels", None)
#         total_loss=0

#         outputs = self.longformer(
#             **batch, 
#             global_attention_mask=self._set_global_attention_mask(batch["input_ids"]), output_hidden_states=True
#             )

#         # cls_output = outputs[0][:, 1]
#         sequence_output = outputs[0][:, 1:] 

#         ### Span CLF Objective
#         ############################################################################################################
#         span_index = span_len_mask.view(span_len_mask.size(0), -1).unsqueeze(-1).expand(span_len_mask.size(0), span_len_mask.size(1)*span_len_mask.size(2), sequence_output.size(-1)) # (bs, num_samples*max_span_len, 768)
#         span_len_binary_mask = span_len_mask.gt(0).long().unsqueeze(-1)
#         span_len_binary_index = span_len_binary_mask.sum(-2).squeeze(-1)
#         span_len_embeddings = self.size_embeddings(span_len_binary_index)

#         span_len_binary_mask_embeddings = span_len_binary_mask.expand(span_len_binary_mask.size(0), span_len_binary_mask.size(1), span_len_binary_mask.size(2), sequence_output.size(-1))
#         span_embeddings = torch.gather(sequence_output, 1, span_index.long()) # (bs, num_samples*max_span_len, 768)
#         span_embedding_groups = torch.stack(torch.split(span_embeddings, 15, 1)).transpose(0,1) # (bs, num_samples, max_span_len, 768)
#         extracted_spans_after_binary_mask = span_len_binary_mask_embeddings*span_embedding_groups #makes all padding positions into zero vectors (bs, num_samples, max_span_len, 768) 
#         maxpooled_embeddings = torch.max(extracted_spans_after_binary_mask, dim=-2).values # (bs, num_samples, 768)

#         #combined_embeds = torch.cat([cls_output.unsqueeze(1).repeat(1, 40, 1), maxpooled_embeddings], dim=-1) #(bs, num_samples, 2*768)
#         combined_span_embeds = torch.cat([span_len_embeddings, maxpooled_embeddings], dim=-1) #(bs, num_samples, 2*768)
#         logits = self.classifier(combined_span_embeds).squeeze()        

#         ### MLM Objective
#         ##############################################################################################################       
#         token_entity_embeds = torch.sum(torch.stack([self.wte(masked_input_ids), outputs[0]], dim=0), dim=0)

#         mlm_loss = None        
#         outputs = self.longformerMLM(
#             inputs_embeds = token_entity_embeds,
#             attention_mask = batch["attention_mask"],
#             global_attention_mask=self._set_global_attention_mask(batch["input_ids"]), 
#             labels = batch["input_ids"]
#         )
#         mlm_loss=outputs.loss

#         ##############################################################################################################

#         span_clf_loss = None
#         if labels!=None:
#             loss_fct = nn.BCELoss()
#             span_clf_loss = loss_fct(logits.view(-1), labels.view(-1))
#             total_loss+=span_clf_loss

#         if mlm_loss!=None:
#             total_loss+=mlm_loss

#         return (total_loss, logits, span_clf_loss, mlm_loss)

#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop. It is independent of forward
#         loss, _, span_clf_loss, masked_lm_loss = self(**batch)
#         # logits = torch.argmax(self.softmax(logits), dim=-1)
#         return {"loss": loss, "mlm_loss": masked_lm_loss}

#     def training_epoch_end(self, outputs):
#         train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
#         mlm_loss_mean = torch.stack([x["mlm_loss"] for x in outputs]).mean()

#         logs = {
#             "train_loss": train_loss_mean,
#         }

#         self.log("train_loss", logs["train_loss"])
#         self.log("mlm_loss", mlm_loss_mean)


#     def validation_step(self, batch, batch_idx):
#         # training_step defines the train loop. It is independent of forward
#         #input_ids, attention_mask, labels = batch
#         loss, logits, span_clf_loss, masked_lm_loss = self(**batch)
#         clearlogger.report_text(logits)
#         #clearlogger.report_scalar(title='mlm_loss', series = 'val', value=masked_lm_loss, iteration=batch_idx) 
#         preds = (logits>0.5).long()
#         return {"val_loss": loss, "preds": preds, "labels": batch["labels"]}

#     def validation_epoch_end(self, outputs):
#         val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
#         val_preds = torch.stack([x["preds"] for x in outputs]).view(-1).cpu().detach().tolist()
#         val_labels = torch.stack([x["labels"] for x in outputs]).view(-1).cpu().detach().tolist()

#         logs = {
#             "val_loss": val_loss_mean,
#         }
#         precision, recall, f1, support = precision_recall_fscore_support(val_labels, val_preds, average='macro')

#         self.log("val_loss", logs["val_loss"])
#         self.log("val_precision", precision)
#         self.log("val_recall", recall)
#         self.log("val_f1", f1)

#     # Freeze weights?
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
#         return optimizer
