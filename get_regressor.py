
import pandas as pd
import numpy as np
import requests
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import sys
import csv
from tqdm import tqdm

sys.path.insert(0, "/home/nllnwang/SP/SyntacticSurprisal/CCGMultitask/")
from model import MultiTaskModel

sys.path.insert(0, "/home/nllnwang/SP/SyntacticSurprisal/sapbenchmark/Surprisals/")
from util import align

# Suppress warnings
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

# Helper Functions
def indexify(word, w2idx):
    return w2idx.get(word, w2idx.get("<oov>"))

def tokenize(sent):
    sent = " ,".join(sent.split(","))
    sent = " .".join(sent.split("."))
    sent = " 's".join(sent.split("'s"))
    sent = " n't".join(sent.split("n't"))
    return sent.split()

# Model Loading
model = None  # Global variable to store the loaded model

def get_model(model_path, w2idx, c2idx, model_type, cuda):
    global model
    if model is None:
        model = MultiTaskModel(len(w2idx), 650, 650, [len(w2idx), len(c2idx)], 2)
        model.load_state_dict(torch.load(f"{model_path}.pt", map_location=torch.device("cuda" if cuda else "cpu")))
        model = model.cuda() if cuda else model.cpu()
    return model

def compute_surprisal_quicker(sentence, model_path, model_type="lm_ambig", cuda=False, uncased=False):
    torch.manual_seed(1)
    np.random.seed(1)

    # Load w2idx and c2idx only once
    with open(f"{model_path}.w2idx", "rb") as w2idx_f:
        w2idx = pickle.load(w2idx_f)
    with open(f"{model_path}.c2idx", "rb") as c2idx_f:
        c2idx = pickle.load(c2idx_f)

    model = get_model(model_path, w2idx, c2idx, model_type, cuda)
    model.eval()

    tokens = ["<eos>"] + tokenize(sentence)
    input_ids = torch.LongTensor([indexify(w.lower() if uncased else w, w2idx) for w in tokens])

    if cuda:
        input_ids = input_ids.cuda()

    h, c = model.init_hidden(1)

    with torch.no_grad():
        for token, next_token in zip(input_ids[:-1], input_ids[1:]):
            lm_n, ccg_n, (h, c) = model(token.view(-1, 1), (h, c))
            next_word_prob = lm_n[-1].view(-1)

            if next_token == input_ids[-1]:  # When reaching the last token
                vocab_size = len(w2idx)
                tagset_size = len(c2idx)
                all_vocab = torch.arange(vocab_size).view(1, -1).to(input_ids.device)

                h_ = h.repeat(1, all_vocab.shape[1], 1)
                c_ = c.repeat(1, all_vocab.shape[1], 1)

                # Compute all tags predictions for the last token
                _, out, _ = model(all_vocab, (h_, c_))
                out_o = out.transpose(0, 1)

                _, out_gold, _ = model(next_token.view(1, 1), (h, c))
                p_predtag = torch.logsumexp(out_o + next_word_prob.repeat(tagset_size, 1), dim=1)

                if model_type.endswith("ambig"):
                    tag_surprisal = -torch.logsumexp(p_predtag + out_gold.view(-1), dim=0).item()
                elif model_type.endswith("klambig"):
                    tag_surprisal = F.kl_div(p_predtag, out_gold.view(-1), log_target=True).item()
                else:
                    raise ValueError("Invalid model_type provided.")

                return tag_surprisal

def compute_gaussian_prior(sentence, mean=0.0, std=1.0):
    tokens = sentence.split()
    last_word = tokens[-1]
    gaussian_prior = torch.normal(mean=float(mean), std=float(std), size=())  # Add size argument

    return gaussian_prior.item()



WORD_FREQUENCY_URL = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt"
FILE_NAME = "word_frequency_list.txt"

def download_word_frequency_list(url, file_name):
    """Downloads a word frequency list if not already present."""
    if not os.path.exists(file_name):
        print(f"Downloading word frequency list from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, "wb") as file:
                file.write(response.content)
            print(f"File saved as {file_name}")
        else:
            print("Failed to download file.")
    else:
        print(f"File {file_name} already exists. Skipping download.")

def load_word_frequency(file_name):
    """Loads the word frequency list into a Pandas DataFrame."""
    df_freq = pd.read_csv(file_name, sep=" ", header=None, names=["word", "frequency"])
    return df_freq

download_word_frequency_list(WORD_FREQUENCY_URL, FILE_NAME)
df_freq = load_word_frequency(FILE_NAME)

df_freq["word"] = df_freq["word"].str.lower()


df = pd.read_csv('word_properties.csv')
df["Word"] = df["Word"].str.lower()

df = df.merge(df_freq, how="left", left_on="Word", right_on="word")

df["frequency"] = df["frequency"].fillna(1)  # Assign 1 to avoid log(0) issues
df["log_frequency"] = np.log10(df["frequency"] + 1)

df.drop(columns=["word"], inplace=True)
print(df)

