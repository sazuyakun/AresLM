import tiktoken
import torch
import torch.nn as nn

def get_embeddings(tokenizer_type="gpt2", output_dim=256, max_len=1024, max_length=4, dataloader=None):
    tokenizer = tiktoken.get_encoding(tokenizer_type)

    vocab_size = tokenizer.n_vocab
    context_length = max_len

    max_length = max_length

    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = nn.Embedding(context_length, output_dim)

    for batch in dataloader:
        x, y = batch
        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))
        input_embeddings = token_embeddings + pos_embeddings
        break

    return input_embeddings
