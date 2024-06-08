import torch
with open('/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00001-of-00002.bin', 'rb') as file:
        binary_data = file.read()
num=0
print(len(binary_data))
for key,value in binary_data.items():
    print(key)
    print(value)
    break
kill
# 加载policy.pt文件
policy = torch.load('/home/wxt/huatong/huggingface/hub/llama2_7b_sft_halos_2_3/LATEST/policy.pt')
cont=1
print(len(policy))
kill
for key,value in policy.items():
    cont+=1

    print("="*20)
    print(key)
    print(value)
    print("="*20)
    if cont==10:
        break
