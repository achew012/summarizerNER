from clearml import Task, StorageManager, Dataset as ds
import argparse, json, os, random, math, ipdb

# config={
#     "lr": 3e-4,
#     "num_epochs":50,
#     "train_batch_size":1,
#     "eval_batch_size":1,
#     "max_length": 2048
# }
# args = argparse.Namespace(**config)

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerForMaskedLM, LongformerModel, LongformerTokenizer, LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerLMHead
import pytorch_lightning as pl
import random

class NERLongformer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        self.config.gradient_checkpointing = True
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.tokenizer.model_max_length = 2048
        
        self.longformer = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)
        
        #self.longformer = LongformerModel(self.config)
        #self.lm_head = LongformerLMHead(self.config) 

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 3)
        self.class_weights = torch.tensor([1, 1, 0.01]).cuda()

        # SBO representation
        self.sbo = nn.Sequential(
          nn.Linear(3*self.config.hidden_size, self.config.hidden_size), # 3 =  start + end + position embeddings,  output is arbitrary
          nn.GELU(),
          nn.LayerNorm(self.config.hidden_size),
          nn.Linear(self.config.hidden_size, self.config.vocab_size),
          nn.GELU(),
          nn.LayerNorm(self.config.vocab_size),
        )

    def val_dataloader(self):
        val = dwie["test"]
        val_data = DWIE_Data(val, self.tokenizer)
        val_dataloader = DataLoader(val_data, batch_size=self.args.eval_batch_size, collate_fn = val_data.collate_fn)
        return val_dataloader

    def train_dataloader(self):
        train = dwie["train"]
        train_data = DWIE_Data(train, self.tokenizer)
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


    def forward(self, **batch):
        # in lightning, forward defines the prediction/inference actions

        masked_input_ids, input_ids, attention_mask, entity_span, entity_mask = batch["masked_input_ids"], batch["input_ids"], batch["attention_mask"], batch["entity_span"], batch["entity_mask"]
        mlm_labels = input_ids
        sbo_labels = torch.masked_select(input_ids, entity_mask.gt(0))

        ## Common Outputs
        outputs = self.longformer(
            input_ids = masked_input_ids,
            attention_mask = attention_mask,
            global_attention_mask=self._set_global_attention_mask(masked_input_ids),
            labels = mlm_labels,
            output_hidden_states=True
        )
        
        #sequence_output = outputs[0]
        sequence_output = outputs.hidden_states[-1]
        
        ## loss, logits for MLM
        #prediction_scores = self.lm_head(sequence_output)
        prediction_scores = outputs.logits

        ## loss, logits for Span Boundary Objective
        # get boundary pair indices to look up embedding
        span_index = entity_span.view(entity_span.size(0), -1).unsqueeze(-1).expand(entity_span.size(0), entity_span.size(1)*entity_span.size(2), sequence_output.size(-1))
        # Lookup from output of longformer
        span_embeddings = torch.gather(sequence_output, 1, span_index) #
        # group into entity boundary pairs
        span_embedding_pairs = torch.stack(torch.split(span_embeddings, 2, 1)).squeeze() # Number of Entity Boundary Pair Embeddings - (bs * N * 2 * 768)
        # get entity positional indices from mask
        entity_boundaries = torch.masked_select(entity_mask, entity_mask.gt(0)).long()
        # replace positional indices with lookup from span boundary embedding pairs
        span_embedding_sequence = torch.index_select(span_embedding_pairs, 0, entity_boundaries).squeeze() # Seq
        # Get positional indices of masked entity
        pos_idx = torch.stack([((sample > 0).nonzero(as_tuple=False)).squeeze(-1) for sample in entity_mask])
        # Look up positional embeddings
        #masked_token_position_embeddings = self.longformer.embeddings.position_embeddings(pos_idx).view(-1, 1, sequence_output.size(-1))        
        masked_token_position_embeddings = self.longformer.longformer.embeddings.position_embeddings(pos_idx).view(-1, 1, sequence_output.size(-1))        

        if len(span_embedding_sequence.size())<3:
            span_embedding_sequence = span_embedding_sequence.unsqueeze(0)

        if len(masked_token_position_embeddings.size())<3:
            masked_token_position_embeddings = masked_token_position_embeddings.unsqueeze(0)

        # Combine x_start, x_end and token_embedding_i to single dim
        combined_representation = torch.cat((span_embedding_sequence, masked_token_position_embeddings), 1).view(-1, 3*sequence_output.size(-1))
        # pass through spanbert sbo representation
        sbo_scores = self.sbo(combined_representation)

        # ipdb.set_trace()
        span_loss = None
        #masked_lm_loss = None
        masked_lm_loss = outputs.loss
        
        if mlm_labels is not None and sbo_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            #masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            span_loss = loss_fct(sbo_scores.view(-1, self.config.vocab_size), sbo_labels.view(-1))

        total_loss = masked_lm_loss + span_loss

        return (total_loss, sequence_output)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss, logits = self(**batch)
        # logits = torch.argmax(self.softmax(logits), dim=-1)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {
            "train_loss": train_loss_mean,
            "train_perplexity": torch.exp(train_loss_mean)
        }

        self.log("train_loss", logs["train_loss"])
        self.log("train_perplexity", logs["train_perplexity"])

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        #input_ids, attention_mask, labels = batch
        loss, logits = self(**batch)
        # print(hidden_states)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        logs = {
            "val_loss": val_loss_mean,
            "val_perplexity": torch.exp(val_loss_mean)
        }
        self.log("val_loss", logs["val_loss"])
        self.log("val_perplexity", logs["val_perplexity"])

    # Freeze weights?
    def configure_optimizers(self):
        # Freeze alternate layers of longformer
        for idx, (name, parameters) in enumerate(self.longformer.named_parameters()):
            if idx%2==0:
                parameters.requires_grad=False
            else:
                parameters.requires_grad=True

        optimizer = torch.optim.Adam(self.longformer.parameters(), lr=self.args.lr)
        return optimizer
