import numpy as np
import torch
def logits_to_max_prob(logits, axis=-1):
    # 找到每个批次中概率最高的类别的索引
    max_prob_index = np.argmax(logits, axis=axis)

    exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)
    max_probabilities = np.max(probabilities,axis=-1)
    return max_probabilities, max_prob_index

# 示例：创建一个包含 logits 的数组（带有批次维度）
logits_array = np.array([[1,2,3],
                         [2,2,2]])
ttt = torch.from_numpy(logits_array).to(torch.float32)
print(torch.softmax(ttt,dim=1))

# 调用函数获取每个批次中最高概率的值和对应的类别索引，指定 dtype 为 float32
max_probabilities, max_prob_index = logits_to_max_prob(logits_array, axis=-1)

print("Logits:")
print(logits_array)
print("\n每个批次中最高概率的值:")
print(max_probabilities)
print("\n每个批次中最高概率的类别索引:")
print(max_prob_index)
