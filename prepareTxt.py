import os
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'Data/PatentAbstractHighTempSeal.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://github.com/Vahid-Kh/NLP/blob/da12adf98d303843e554adc71a4bd9ec1814bd93/test.txt'
    with open(input_file_path, 'w', encoding="utf-8") as f:

        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()

print(data)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
print(train_data[:1000])
# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")


# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))



