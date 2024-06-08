import torch

# 加载policy.pt文件
policy = torch.load('/home/wxt/huatong/huggingface/hub/llama2_7b_sft_halos_2_3/LATEST/policy.pt')
cont=1
for key,value in policy.items():
    cont+=1

    print("="*20)
    print(key)
    print(value)
    print("="*20)
    if cont==10:
        break
