import sentencepiece as spm
from sentencepiece import SentencePieceTrainer

SentencePieceTrainer.train(
    input="./pubmed_train.txt",
    model_prefix="big_bird_tokenizer", 
    vocab_size=50358,
)

from transformers import BigBirdTokenizer

tokenizer = BigBirdTokenizer(
    "big_bird_tokenizer.model",
    "big_bird_tokenizer.vocab"
)
tokenizer.save_pretrained('big_bird_tokenizer')