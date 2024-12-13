
import torch

# 加载文件
data = torch.load('/opt/data/private/CMI-TFC/data/myTensor_1.pt')

# 检查是否是列表或字典（常见的存储方式）
if isinstance(data, list):
    print(f"数据是一个列表，包含 {len(data)} 个元素。")
    for i, item in enumerate(data):
        if isinstance(item, torch.Tensor):
            print(f"元素 {i}: 形状为 {item.shape}")
        else:
            print(f"元素 {i}: 类型为 {type(item)}，不是张量")
elif isinstance(data, dict):
    print(f"数据是一个字典，包含 {len(data)} 个键。")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: 形状为 {value.shape}")
        else:
            print(f"{key}: (不是张量类型的数据)")
else:
    print(f"数据的类型是 {type(data)}，不明确是否包含标签")
