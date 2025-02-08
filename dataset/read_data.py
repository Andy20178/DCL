import numpy as np
import pdb
import torch
# 加载npz文件
# data = np.load('/data/lcs/PACS_data/pacs_npz/pacs_object0001.npz')
data = torch.load('/data/zs/T_npy/object0000/audio/object0000aud.pt')
# # npz文件可能包含多个数组，可以使用files属性查看所有的数组名称
# print(data.files)
pdb.set_trace()
# 假设你想获取其中一个名为'array1'的数组
array1 = data['array1']

# 使用数据...
print(array1)
