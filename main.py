import math
import onnx
import onnxruntime as ort
import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self, embedding_size: int = 64, n_heads: int = 2, n_items: int = 5):
        super(MyModel, self).__init__()
        self.item_embedding = nn.Embedding(n_items, embedding_size, padding_idx=0)

        self.trm_encoder = MultiHeadAttention(
            n_heads=n_heads,
            hidden_size=embedding_size,
        )

    def forward(self, item_seq):
        attention_mask = item_seq.ne(0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        expand_attention_mask = extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
        extended_attention_mask = torch.tril(expand_attention_mask)
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)

        item_emb = self.item_embedding(item_seq)
        alpha = self.trm_encoder(
            item_emb, extended_attention_mask
        ).to(torch.double)
        return alpha


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, hidden_size):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        context_layer = torch.matmul(attention_scores.to(torch.double), value_layer.to(torch.double)).to(torch.float32)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


if __name__ == '__main__':
    model = MyModel()
    model.eval()

    x_train = torch.ones([1, 2], dtype=torch.long)
    print(model.forward(x_train))

    bigmodel_onnx_filename = 'mymodel.onnx'
    torch.onnx.export(
        model,
        x_train,
        bigmodel_onnx_filename,
        input_names=['x'],
        output_names=['output'],
    )

    onnx.load(bigmodel_onnx_filename)

    # When loading neural graph, Onnx will crash with a Trilu Node NOT_IMPLEMENTED error
    ort_sess = ort.InferenceSession(bigmodel_onnx_filename, providers=['CPUExecutionProvider'])
    key = {'x': x_train.numpy()}
    print(ort_sess.run(None, key))
