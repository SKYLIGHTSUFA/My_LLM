import torch
from torch import nn
import os
from byte_pair_encoder import BPE

import glob

all_text = []
for file_path in glob.glob('/Users/ilnaz/Projecta/My_LLM/data/texts/*.txt'):
    print(f"Loading file: {file_path}")
    file = open(file_path, 'r', encoding='utf8')
    all_text.append(file.read())
    
all_text = '\n\n\n'.join(all_text)
print(f"Length of text: {len(all_text)} characters")