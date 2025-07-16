from data.dataloader import create_dataloader_v1
from data.embedding import get_embeddings
from attention.multi_head_attention import MultiHeadAttention

# TEXT_PATH = "./data/the-verdict.txt"
TEXT_PATH = "./small-text-sample.txt"

STAGE_NAME = "Data Loading"
if __name__ == "__main__":
    print(f"<======== Stage: {STAGE_NAME} ========>")
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    max_len = 1024

    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)

    input_embeddings = get_embeddings(
        tokenizer_type="gpt2",
        output_dim=output_dim,
        max_len=max_len,
        max_length=max_length,
        dataloader=dataloader
    )

    print(input_embeddings.shape)


STAGE_NAME = "Multi-head attention"
if __name__ == "__main__":
    print(f"\n<======== Stage: {STAGE_NAME} ========>")
    context_length = max_length
    d_in = output_dim

    num_heads=2
    d_out = d_in // num_heads

    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)

    batch = input_embeddings
    context_vecs = mha(batch)

    print("context_vecs.shape:", context_vecs.shape)

