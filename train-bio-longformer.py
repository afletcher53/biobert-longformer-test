import logging
import os
import math
import copy
import torch
from dataclasses import dataclass, field
from transformers import BertForMaskedLM, BertTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention

import gzip
import shutil
import urllib.request
import xml.etree.ElementTree as ET

def download_pubmed(year, month):
    base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed24n{:04d}.xml.gz".format(year)
    filename = "pubmed22n{:04d}.xml.gz".format(year)
    urllib.request.urlretrieve(base_url, filename)
    return filename

def extract_abstracts(xml_file, output_file):
    with gzip.open(xml_file, 'rb') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        tree = ET.parse(f_in)
        root = tree.getroot()
        for article in root.findall('.//PubmedArticle'):
            abstract = article.find('.//AbstractText')
            if abstract is not None and abstract.text:
                f_out.write(abstract.text + '\n\n')

def prepare_pubmed_data(num_files=10):
    train_file = 'pubmed_train.txt'
    val_file = 'pubmed_val.txt'
    
    with open(train_file, 'w') as train, open(val_file, 'w') as val:
        for i in range(1, num_files + 1):
            filename = download_pubmed(i, 1)
            output = 'pubmed_extracted_{}.txt'.format(i)
            extract_abstracts(filename, output)
            
            # Split data: 90% train, 10% val
            with open(output, 'r') as f:
                lines = f.readlines()
                split = int(len(lines) * 0.9)
                train.writelines(lines[:split])
                val.writelines(lines[split:])
            
            os.remove(filename)
            os.remove(output)
    
    return train_file, val_file

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
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)

def create_long_model(save_model_to, attention_window, max_pos):
    model = BertForMaskedLM.from_pretrained('dmis-lab/biobert-v1.1')
    tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-v1.1', max_len=max_pos)
    config = model.config

    # extend position embeddings
    tokenizer.max_length = max_pos
    tokenizer.init_kwargs['max_length'] = max_pos
    current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: BERT has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    
    # allocate a larger position embedding matrix
    new_pos_embed = model.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_pos:
        end_pos = min(k + step, max_pos)
        new_pos_embed[k:end_pos] = model.bert.embeddings.position_embeddings.weight[:end_pos-k]
        k = end_pos
    
    model.bert.embeddings.position_embeddings.weight.data = new_pos_embed

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

    logger.info('saving model to {}'.format(save_model_to))
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer

def copy_proj_layers(model):
    for i, layer in enumerate(model.bert.encoder.layer):
        layer.attention.self.query_global = copy.deepcopy(layer.attention.self.query)
        layer.attention.self.key_global = copy.deepcopy(layer.attention.self.key)
        layer.attention.self.value_global = copy.deepcopy(layer.attention.self.value)
    return model

def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
                              block_size=tokenizer.max_len)
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info('Loading and tokenizing training data is usually slow: {}'.format(args.train_datapath))
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=args.train_datapath,
                                    block_size=tokenizer.max_len)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True,)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info('Initial eval bpc: {}'.format(eval_loss/math.log(2)))

    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info('Eval bpc after pretraining: {}'.format(eval_loss/math.log(2)))

class ModelArgs:
    def __init__(self, attention_window=512, max_pos=4096):
        self.attention_window = attention_window
        self.max_pos = max_pos
        self.__post_init__()

    def __post_init__(self):
        self.max_pos = (self.max_pos // self.attention_window) * self.attention_window
        if self.max_pos < self.attention_window:
            self.max_pos = self.attention_window

# Replace the existing parser section with this:
parser = HfArgumentParser((TrainingArguments,))
training_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    '--output_dir', 'tmp',
    '--warmup_steps', '500',
    '--learning_rate', '0.00003',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    '--max_steps', '3000',
    '--logging_steps', '500',
    '--save_steps', '500',
    '--max_grad_norm', '5.0',
    '--per_gpu_eval_batch_size', '8',
    '--per_gpu_train_batch_size', '2',  # 32GB gpu with fp32
    '--gradient_accumulation_steps', '32',
    '--evaluate_during_training',
    '--do_train',
    '--do_eval',
])[0]  # We only need the first item from the returned tuple

# training_args.val_datapath = './pubmed_val.txt'
# training_args.train_datapath = './pubmed_train.txt'
# Choose GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Main execution
if __name__ == "__main__":
    logger.info("Preparing PubMed data...")
    train_file, val_file = prepare_pubmed_data(num_files=1)  # You can adjust the number of files
    
    # make copies of train/val files for future use
    # shutil.copy(train_file, 'pubmed_train_copy.txt')
    # shutil.copy(val_file, 'pubmed_val_copy.txt')
    
    training_args.train_datapath = train_file
    training_args.val_datapath = val_file

    model_args = ModelArgs(attention_window=512, max_pos=4096)  # This will automatically adjust to 4096
    model_path = '{}/biobert-long-{}'.format(training_args.output_dir, model_args.max_pos)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info('Converting BioBERT into BioBERT-long-{}'.format(model_args.max_pos))
    model, tokenizer = create_long_model(
        save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)

    logger.info('Loading the model from {}'.format(model_path))
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertLongForMaskedLM.from_pretrained(model_path)

    logger.info('Pretraining BioBERT-long-{} ... '.format(model_args.max_pos))
    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)

    logger.info('Copying local projection layers into global projection layers ... ')
    model = copy_proj_layers(model)
    logger.info('Saving model to {}'.format(model_path))
    model.save_pretrained(model_path)

    logger.info('Loading the final model from {}'.format(model_path))
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertLongForMaskedLM.from_pretrained(model_path)

    logger.info("BioBERT-long model is ready for use!")

    # Clean up
    # os.remove(train_file)
    # os.remove(val_file)