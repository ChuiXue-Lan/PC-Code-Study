from transformers import BertTokenizer

BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./bert-base-uncased")