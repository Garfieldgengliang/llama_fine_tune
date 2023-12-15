import torch

# 在Bert的基础上加了一个线性分类器
class MyClassifier(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.bert_encoder = backbone # 模型底座（预训练的模型）
        self.linear = torch.nn.Linear(768, 2) # 线性分类器的头
        self.dropout = torch.nn.Dropout(p=0.1) # dropout

    def compute_loss(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss()
        return loss_fct(logits, labels)

    def forward(self, input_ids, attention_mask,labels=None):
        # 底座模型推理
        output = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 取第一个 token [CLS] 对应的隐层 
        output = output.last_hidden_state[:, 0, :] 
        # 应用 dropout
        output = self.dropout(output)
        # 传入线性分类器的头
        output = self.linear(output)
        if labels is not None:
            # 计算 loss
            loss = self.compute_loss(output, labels)
            return loss, output
        return output
