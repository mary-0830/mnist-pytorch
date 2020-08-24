
import torch

pre_model = "mnist_cnn.pt"
dict = torch.load(pre_model)
for key in list(dict.keys()):
    print("key:", key)
    if key.startswith('cnn.conv2.weight'):
        del dict[key]
    elif  key.startswith('cnn.conv2.bias'):
        del dict[key]
torch.save(dict, 'mnist_deleted.pt')
# # #验证修改是否成功
changed_dict = torch.load('mnist_deleted.pt')
for key in dict.keys():
    print(key)
