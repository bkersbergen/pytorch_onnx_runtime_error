import onnx
import onnxruntime as ort
import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, item_seq):
        attention_mask = item_seq < 100
        tril_mask = torch.tril(attention_mask)
        query_layer = torch.rand((1, 2, 2, 32))
        key_layer = torch.rand((1, 2, 32, 2))
        attention_scores = torch.matmul(query_layer, key_layer)
        return attention_scores + tril_mask


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
