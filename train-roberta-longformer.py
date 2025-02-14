import logging
import os
import math
import copy
import torch
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
import wandb
from logging.handlers import RotatingFileHandler

import torch


# Set up file handler
file_handler = RotatingFileHandler('training.log', maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)

# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the logger and add the handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# logger.info(f"CUDA available: {torch.cuda.is_available()}")
# logger.info(f"Current device: {torch.cuda.current_device()}")
# logger.info(f"Device name: {torch.cuda.get_device_name()}")

os.environ["WANDB_API_KEY"] = "ceff7aa0c8155b19c88199c687e74f8e22b4024c"

from transformers.integrations import WandbCallback
from transformers import TrainingArguments, HfArgumentParser, Trainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def print_gpu_memory():
    if torch.cuda.is_available():
        logger.info(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")


class RobertaLongSelfAttention(LongformerSelfAttention):
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


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)

def create_long_model(save_model_to, attention_window, max_pos):
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', model_max_length=max_pos)
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
    model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
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
    logger.info(model.config)
    tokenizer.model_max_length = 4098
    return model, tokenizer


def copy_proj_layers(model):
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = copy.deepcopy(layer.attention.self.query)
        layer.attention.self.key_global = copy.deepcopy(layer.attention.self.key)
        layer.attention.self.value_global = copy.deepcopy(layer.attention.self.value)
    return model

class CustomWandbCallback(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            # Convert 'eval_loss' to 'eval_bpc'
            if 'eval_loss' in logs:
                logs['eval_bpc'] = logs['eval_loss'] / math.log(2)
            wandb.log(logs)

def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
                              block_size=tokenizer.model_max_length)
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=args.train_datapath,
                                    block_size=tokenizer.model_max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    # Add CustomWandbCallback to the callbacks
    callbacks = [CustomWandbCallback]
    
    trainer = Trainer(
        model=model, 
        args=args, 
        data_collator=data_collator,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        callbacks=callbacks
    )
    print_gpu_memory()
    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    eval_bpc = eval_loss / math.log(2)
    logger.info(f'Initial eval bpc: {eval_bpc}')
    wandb.log({"initial_eval_bpc": eval_bpc})
    
    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        eval_bpc = eval_loss / math.log(2)
        logger.info(f'Eval bpc after pretraining: {eval_bpc}')
        wandb.log({"final_eval_bpc": eval_bpc})

@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})

parser = HfArgumentParser((TrainingArguments, ModelArgs,))




def main():
    print_gpu_memory()
    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
        '--output_dir', 'tmp',
        '--warmup_steps', '500',
        '--learning_rate', '0.00003',
        '--weight_decay', '0.01',
        '--adam_epsilon', '1e-6',
        '--max_steps', '3000',
        '--logging_steps', '500',
        '--save_steps', '250',  # Save more frequently
        '--max_grad_norm', '5.0',
        '--per_device_eval_batch_size', '4',  # Reduced from 8
        '--per_device_train_batch_size', '1',  # Reduced from 2
        '--gradient_accumulation_steps', '64',  # Increased from 32
        '--do_train',
        '--do_eval',
    ])
    training_args.val_datapath = './pubmed_val.txt'
    training_args.train_datapath = './pubmed_train.txt'
    print_gpu_memory()
    # Initialize wandb
    wandb.init(project="roberta-long", config=vars(training_args))

    # Choose GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    roberta_base = RobertaForMaskedLM.from_pretrained('roberta-base')
    roberta_base_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    # print_gpu_memory()
    # logger.info('Evaluating roberta-base (seqlen: 512) for reference ...')
    # pretrain_and_evaluate(training_args, roberta_base, roberta_base_tokenizer, eval_only=True, model_path=None)

    # Create and train long-range model
    logger.info(f'Creating long model with attention window: {model_args.attention_window} and max pos: {model_args.max_pos}')
    save_model_to = f'roberta-long-window{model_args.attention_window}-pos{model_args.max_pos}'
    long_model, long_tokenizer = create_long_model(save_model_to, model_args.attention_window, model_args.max_pos)
    
    logger.info(f'Training long model...')
    pretrain_and_evaluate(training_args, long_model, long_tokenizer, eval_only=False, model_path=save_model_to)

    model_path = f'{training_args.output_dir}/roberta-base-{model_args.max_pos}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(f'Converting roberta-base into roberta-base-{model_args.max_pos}')
    model, tokenizer = create_long_model(
        save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)
    
    logger.info(f'Loading the model from {model_path}')

    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = RobertaLongForMaskedLM.from_pretrained(model_path)

    logger.info(f'Pretraining roberta-base-{model_args.max_pos} ... ')

    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)

    logger.info(f'Copying local projection layers into global projection layers ... ')
    model = copy_proj_layers(model)
    logger.info(f'Saving model to {model_path}')
    model.save_pretrained(model_path)

    wandb.finish()

if __name__ == "__main__":
    main()