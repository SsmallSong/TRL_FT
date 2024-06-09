import torch
with open('/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00001-of-00002.bin', 'rb') as file:
        binary_data_1 = file.read()
with open('/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00002-of-00002.bin', 'rb') as file:
        binary_data_2 = file.read()
num=0
print(len(binary_data_1)+len(binary_len_2))
print(binary_data)

print("="*20)
print("="*20)
print("="*20)

# for key,value in binary_data.items():
#     print(key)
#     print(value)
#     break
# kill
# 加载policy.pt文件
policy =torch.load('/home/wxt/huatong/huggingface/hub/llama2_7b_sft_halos_2_3/LATEST/policy.pt')
cont=1
print(len(policy))
state_dict=policy['state']
print(len(state_dict))
print((state_dict))
# kill
# for key,value in policy.items():
#     cont+=1

#     print("="*20)
#     print(key)
#     print(value)
#     print("="*20)
#     if cont==10:
#         break
