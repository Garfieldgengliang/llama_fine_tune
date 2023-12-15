
# 运行: python load-and-run.py

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

# 指定模型名
model_name_or_path = "gpt2"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 根据模型名加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True
)

# 根据模型名加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, trust_remote_code=True
).to(device)

# tokenize: 文本转 token ids
inputs = tokenizer('I am', return_tensors='pt').to(device)

print("===input===")
print(inputs)

# 生产输出(token ids)
pred = model.generate(**inputs, max_new_tokens=32)

# token id转文本
output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

print("===output===")
print(output)
