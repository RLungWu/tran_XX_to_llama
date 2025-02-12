# 查看模型结构
from transformers import AutoModel


model = AutoModel.from_pretrained("./downloads/Llama-3.2-1B")
print(model)

# 查看参数列表
for name, param in model.named_parameters():
    print(name, param.shape)
