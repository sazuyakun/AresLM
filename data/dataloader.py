import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must atleast be equal to max_length + 1"

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# Example usage:
# if __name__ == "__main__":
#     with open("the-verdict.txt", "r", encoding="utf-8") as f:
#         raw_text = f.read()

#     vocab_size = 50257
#     output_dim = 256
#     context_length = 1024


#     token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#     pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

#     batch_size = 8
#     max_length = 4
#     dataloader = create_dataloader_v1(
#         raw_text,
#         batch_size=batch_size,
#         max_length=max_length,
#         stride=max_length
#     )
#     for batch in dataloader:
#         x, y = batch

#         token_embeddings = token_embedding_layer(x)
#         pos_embeddings = pos_embedding_layer(torch.arange(max_length))

#         input_embeddings = token_embeddings + pos_embeddings

#         break

#     print(input_embeddings.shape)

