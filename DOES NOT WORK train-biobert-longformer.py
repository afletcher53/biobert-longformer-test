import logging
import os
import math
import copy
import torch
from dataclasses import dataclass, field
from transformers import BertForMaskedLM, BertTokenizerFast, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
import wandb
from transformers.integrations import WandbCallback
from datasets import load_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BertLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)

class BertLongForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.bert.encoder.layer):
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)

def create_long_model(save_model_to, attention_window, max_pos):
    model = BertForMaskedLM.from_pretrained('dmis-lab/biobert-v1.1')
    tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-v1.1', model_max_length=max_pos)
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
    config.max_position_embeddings = max_pos
    
    # allocate a larger position embedding matrix
    new_pos_embed = model.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_pos:
        if k + step > max_pos:
            step = max_pos - k
        new_pos_embed[k:(k + step)] = model.bert.embeddings.position_embeddings.weight[:step]
        k += step

    model.bert.embeddings.position_embeddings.weight.data = new_pos_embed
    model.bert.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.bert.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer

def copy_proj_layers(model):
    for i, layer in enumerate(model.bert.encoder.layer):
        layer.attention.self.query_global = copy.deepcopy(layer.attention.self.query)
        layer.attention.self.key_global = copy.deepcopy(layer.attention.self.key)
        layer.attention.self.value_global = copy.deepcopy(layer.attention.self.value)
    return model

class CustomWandbCallback(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            if 'eval_loss' in logs:
                logs['eval_bpc'] = logs['eval_loss'] / math.log(2)
            wandb.log(logs)
def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path, train_datapath, val_datapath):
    data_files = {
        "train": train_datapath,
        "validation": val_datapath
    }
    datasets = load_dataset("text", data_files=data_files)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length)

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    if eval_only:
        train_dataset = tokenized_datasets["validation"]
    else:
        train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    callbacks = [CustomWandbCallback]
    
    trainer = Trainer(
        model=model, 
        args=args, 
        data_collator=data_collator,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        callbacks=callbacks
    )

    def safe_evaluate():
        try:
            return trainer.evaluate()
        except ValueError as e:
            logger.warning(f"Evaluation failed with error: {e}")
            return {"eval_loss": float('inf')}

    eval_results = safe_evaluate()
    eval_loss = eval_results.get('eval_loss', float('inf'))
    eval_bpc = eval_loss / math.log(2)
    logger.info(f'Initial eval bpc: {eval_bpc}')
    wandb.log({"initial_eval_bpc": eval_bpc})
    
    if not eval_only:
        # Check if the model_path exists and contains a checkpoint
        if os.path.exists(model_path) and any(fname.startswith("checkpoint-") for fname in os.listdir(model_path)):
            logger.info(f"Resuming training from {model_path}")
            train_result = trainer.train(resume_from_checkpoint=model_path)
        else:
            logger.info("Starting training from scratch")
            train_result = trainer.train()
        
        trainer.save_model()
        
        # Log and print the training results
        logger.info(f"Training results: {train_result}")
        for key, value in train_result.metrics.items():
            wandb.log({f"train_{key}": value})

        eval_results = safe_evaluate()
        eval_loss = eval_results.get('eval_loss', float('inf'))
        eval_bpc = eval_loss / math.log(2)
        logger.info(f'Eval bpc after pretraining: {eval_bpc}')
        wandb.log({"final_eval_bpc": eval_bpc})

    return eval_results
@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})

parser = HfArgumentParser((TrainingArguments, ModelArgs,))

def main():
    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
        '--output_dir', 'tmp',
        '--warmup_steps', '500',
        '--learning_rate', '0.00003',
        '--weight_decay', '0.01',
        '--adam_epsilon', '1e-6',
        '--max_steps', '3000',
        '--logging_steps', '500',
        '--save_steps', '500',
        '--max_grad_norm', '5.0',
        '--per_device_eval_batch_size', '8',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '32',
        '--do_train', 'True',
        '--do_eval', 'True',
    ])
    train_datapath = './pubmed_train.txt'
    val_datapath = './pubmed_val.txt'

    wandb.init(project="biobert-long", config=vars(training_args))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # biobert_base = BertForMaskedLM.from_pretrained('dmis-lab/biobert-v1.1')
    # biobert_base_tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-v1.1')
    # logger.info('Evaluating biobert-base (seqlen: 512) for reference ...')
    # pretrain_and_evaluate(training_args, biobert_base, biobert_base_tokenizer, eval_only=True, model_path=None, train_datapath=train_datapath, val_datapath=val_datapath)

    logger.info(f'Creating long model with attention window: {model_args.attention_window} and max pos: {model_args.max_pos}')
    save_model_to = f'biobert-long-window{model_args.attention_window}-pos{model_args.max_pos}'
    long_model, long_tokenizer = create_long_model(save_model_to, model_args.attention_window, model_args.max_pos)
  
    
    logger.info(f'Training long model...')
    pretrain_and_evaluate(training_args, long_model, long_tokenizer, eval_only=False, model_path=save_model_to, train_datapath=train_datapath, val_datapath=val_datapath)

    model_path = f'{training_args.output_dir}/biobert-base-{model_args.max_pos}'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(f'Converting biobert-base into biobert-base-{model_args.max_pos}')
    model, tokenizer = create_long_model(
        save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)
    
    logger.info(f'Loading the model from {model_path}')

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertLongForMaskedLM.from_pretrained(model_path)

    logger.info(f'Pretraining biobert-base-{model_args.max_pos} ... ')

    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir, train_datapath=train_datapath, val_datapath=val_datapath)

    logger.info(f'Copying local projection layers into global projection layers ... ')
    model = copy_proj_layers(model)
    logger.info(f'Saving model to {model_path}')
    model.save_pretrained(model_path)

    wandb.finish()

if __name__ == "__main__":
    main()