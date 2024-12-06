import torch

# 加载文件
data = torch.load('/opt/data/private/CMI-TFC/data/myTensor_1.pt')

# 检查是否是列表
if isinstance(data, list):
    print(f"数据是一个列表，包含 {len(data)} 个元素。")
    for i, item in enumerate(data):
        if isinstance(item, torch.Tensor):
            print(f"元素 {i}: 形状为 {item.shape}")
        else:
            print(f"元素 {i}: 类型为 {type(item)}，不是张量")
else:
    # 如果不是列表，可能是张量或字典
    if isinstance(data, torch.Tensor):
        print(f"张量的形状为: {data.shape}")
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: (不是张量类型的数据)")

