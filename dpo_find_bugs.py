# from datasets import load_dataset

# dataset_name="trl-internal-testing/hh-rlhf-trl-style"

# dataset_name = "snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset"

# ds = load_dataset(dataset_name)

# # def process(row):
# #     row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
# #     row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
# #     return row

# # ds = ds.map(
# #     process,
# #     num_proc=multiprocessing.cpu_count(),
# #     load_from_cache_file=False,
# # )

# # train_dataset = ds["train"]
# # eval_dataset = ds["test"]

# train_dataset = ds["train_iteration_3"]
# eval_dataset = ds["test_iteration_3"]
# print(type(train_dataset))
# print(train_dataset[0].keys())

