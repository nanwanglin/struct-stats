
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertModel.from_pretrained('bert-large-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  

def get_last_word_embedding(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # should i modify the input and output here
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        all_hidden_states, _ = model(input_tensor)

    last_word_embedding = all_hidden_states[-1][0, -2, :].cpu().numpy() # last layer, last word

    return last_word_embedding


def calculate_cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

def compare_sentences(real_sen, alter_sen):
    real_embedding = get_last_word_embedding(real_sen[0])
    
    similarities = []
    for sentence in alter_sen:
        alter_embedding = get_last_word_embedding(sentence)
        similarity = calculate_cosine_similarity(real_embedding, alter_embedding)
        s = 1-similarity # from similarity to distance (dissimilarity)
        similarities.append(s)

    return np.array(similarities)


if __name__ == "__main__":
    real_sen = ["the movie is very great"]
    alter_sen = ["the movie is very much", "the movie is very well", "the movie is very good"]

    similarities = compare_sentences(real_sen, alter_sen)
    
# word = "This"
# layer_idx = 0  
# embedding = word_embeddings.get((word, layer_idx))
# embedding = word_embeddings.get((word))
# embedding.shape


