import torch
# Load model directly
encodeds_sorn_chosen=torch.tensor([[    1,  8290,   368,  2270, 26812,   272,  4735,   302,   574,  6940,
         28804,     1,   733, 16289, 28793,  8290,   368,  2270, 26812,   272,
          4735,   302,   574,  6940, 28804,   733, 28748, 16289, 28793, 28737,
          1580, 15884,   653,   354,   707, 16630, 28725,   562,   315,   837,
           776,   264,  6074,  2007,  3822,   396, 18278, 10895,  9464, 28725,
           315,   949, 28742, 28707,   506,   264,  9994,  2659, 28725,  3564,
           697, 28725,   442,   264,  5277,  9025,   298,  2996,   586,  6940,
         28723,   315,  1759,  1871,   304,  8270, 14915,  2818,   356, 16292,
           304,  1178,   315, 28742,   333,   750,  2078, 28723,  1984,   345,
           267,  2045, 28739,   349,   272,  7153,  4466,   304,  5287,   302,
           586,  5225,   390,  6740,   486,   586, 24518,   304,  5443, 28723,
           315,   949, 28742, 28707,   506,   272, 21368,   298,  6217,   442,
          2996,   378, 28725,   390,   315,   949, 28742, 28707,   506,  3817,
           495,  9021,   442, 16508, 28723,   851,  5935,   302,  6940, 27203,
         15588,   298,  3687, 16905, 10637,   302,   716,  2737, 16455,   304,
          1008, 28733, 15950,  1467, 28723,  1263, 10589, 28725,   456,   541,
           347,   396,  7677,  8499,   288,  8829,   745,   297, 18831, 28725,
         20421,   356,  1424,  2366,   846,  1063, 28725, 20968, 28725,   304,
           272, 11656,   302,  4788, 28723,   661, 28742, 28713,   396,  5853,
         23083,   302,   272,  2930,  4644,   304,   813,  8280,   298,  2380,
           272,  4735,   302,   813,  9025, 28723,  1263,   528, 28725,   378,
         28742, 28713,   776,   264,  5935,  2373,   272, 18745,   302,  2930,
          2312,  8306, 28725,  1847,   806,   302,  3327,  2659, 28723,     2,
             2]])
encodeds_hh_chosen=torch.tensor([[    1,  1602,   511,   315, 21170,  2493, 28804,     1,   733, 16289,
         28793,  1602,   511,   315, 21170,  2493, 28804,   733, 28748, 16289,
         28793,  5985,   368,  7201,   684,   272,  2309,   442,   684,   767,
           368, 28809, 28715,   511,   297,  5245, 28804,     2,   733, 16289,
         28793,  1824,   315,   682,   511,   297,  5245,   733, 28748, 16289,
         28793, 28737, 28809, 28719,   459,  1215,  8304,   395,   272,  2948,
         28713,   302,   272,  2309,   442, 10879,   302,  6965, 13349, 28723,
          1537,   513,   368,  2613,   586,  2787,   368, 28809, 28715,   927,
           298,  1912,   528,   767,   272,  4382,   460,   304,   767,   368,
         28809, 28715,   737,   586,  7403,   298,   347,   684,   910,   368,
          1659,   938,   706, 28723,     2,     2]])

encodeds_sorn_labels=torch.tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,     1,   733,
         16289, 28793, 12628,   264,  5964,   302,  3085, 28725,  8270,   272,
          9378,  1444,   706, 28723,   415,   907,  1707,   349,  1987,   272,
           464,   514,  1358, 28742,   304,   272,  1676,  1707,   349,  1987,
           272,   464,  3045,   270,   383, 28742,   415,  9378,  1580,   347,
           624,   302,   272,  2296, 28747,  1001, 28733,  5942,   766,  1082,
           325, 17213,  4296,   557, 11503, 15416, 28725,  3051,  8095, 28725,
          6836, 28725,  1951, 28725,   442,  5509, 28723,   330, 17237,  9378,
         12825,   369,   272,  5935,   304,  1016,   270,   383,  6847,   298,
           272,  1348,  3546,  6164,   875, 28723,   330, 11503, 15416,  9378,
         12825,   369,   272,  1016,   270,   383,   349,   264,  8011,   302,
           690,   272,  5935,   349,   264,  2948,  3335, 28723,   330,  3051,
          8095,  9378, 12150,   369,  1016,   270,   383,   349,   264,   744,
         28748,  7451, 28748,  8974, 28748,  9978,   302,   272,  5935, 28723,
          1094,  6836,  9378,   349,   739,   272,  1016,   270,   383,   349,
           396,   616,   664,   495,  4072,   288,   396,  6836,   302,   272,
          5935, 28723,  1094,  1951,  9378,  8288,   739,   272,  1016,   270,
           383,   349,   264, 12143, 18723,   298,   396,  2992, 28748, 15996,
         28748, 28716,   763,  3250, 28748,  2883,   369,   349,  7885,   486,
           442,   395,   272,  5935, 28723,  1047,   272,  1016,   270,   383,
           304,  5935,   460,   521,  9646, 28725,   272,  9378,  1444,   706,
           349,   464,  9121,  4135,  4950,  1575,   574, 11194,   778,  6618,
         28725, 11503, 28725,   290,  2036, 28725,   998,   373, 28725,  1951,
         28725,   304,  5509, 28723,    13,    13, 28792, 28824,  9582,  1325,
          1358, 28747, 12072, 28725,  5855,   270,   383, 28747,  3944, 28723,
            13, 28792, 28741,  9582,   998,   373,    13,    13,    13, 28792,
         28824,  9582,  1325,  1358, 28747,   337,  4755, 28725,  5855,   270,
           383, 28747,  8114, 28723,    13, 28792, 28741,  9582, 11503,    13,
            13,    13, 28792, 28824,  9582,  1325,  1358, 28747, 14693,  8565,
         28725,  5855,   270,   383, 28747, 10409, 28723,    13, 28792, 28741,
          9582,   733, 28748, 16289, 28793,  1061,   373, 28747,   560,   456,
          1222, 28725,   345,   452,  3953, 28739,   349,   396,  6836,   302,
           272,  5935,   345,   824,  2516,  8565,   611,   330, 14693,  8565,
           541,   347,  1269,   302, 10409, 28725, 18063,   624,   302,   871,
         15559,   442, 20800, 28723,  8469, 28725,   272,  9378,   349,   345,
          9122, 28739,   325,  1061,   373,   609,    13,    13, 28792, 28824,
          9582,  1325,  1358, 28747,  7607, 28725,  5855,   270,   383, 28747,
         21177, 28723,    13, 28792, 28741,  9582, 11503,    13,    13,    13,
         28792, 28824,  9582,  1325,  1358, 28747,  4607,  4414, 28725,  5855,
           270,   383, 28747,  4456, 28723,    13, 28792, 28741,  9582,  3051,
          2557,  1033,    13,    13,    13, 28792, 28824,  9582,  1325,  1358,
         28747,  4297, 22326, 28725,  5855,   270,   383, 28747,  4397, 28723,
            13, 28792, 28741,  9582,  1951,    13,    13,    13, 28792, 28824,
          9582,  1325,  1358, 28747, 19767, 28725,  5855,   270,   383, 28747,
         11057, 28723,    13, 28792, 28741,  9582,  3051,  8095,    13,    13,
            13, 28792, 28824,  9582,  1325,  1358, 28747,   261,  4886, 28725,
          5855,   270,   383, 28747,  4216,  4384, 28723,    13, 28792, 28741,
          9582,   998,   373,    13,    13,    13, 28792, 28824,  9582,  1325]])
encodeds_hh_labels=torch.tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,     1,   733, 16289,
         28793,  1602,   511,   315, 21170,  2493, 28804,   733, 28748, 16289,
         28793,  5985,   368,  7201,   684,   272,  2309,   442,   684,   767,
           368, 28809, 28715,   511,   297,  5245, 28804,     2,   733, 16289,
         28793,  1824,   315,   682,   511,   297,  5245,   733, 28748, 16289,
         28793, 28737, 28809, 28719,   459,  1215,  8304,   395,   272,  2948,
         28713,   302,   272,  2309,   442, 10879,   302,  6965, 13349, 28723,
          1537,   513,   368,  2613,   586,  2787,   368, 28809, 28715,   927,
           298,  1912,   528,   767,   272,  4382,   460,   304,   767,   368,
         28809, 28715,   737,   586,  7403,   298,   347,   684,   910,   368,
          1659,   938,   706, 28723,     2,     2]])
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import torch
device = "cuda" # the device to load the model onto
# export CUDA_VISIBLE_DEVICES=1
model_id='/home/wxt/huatong/huggingface/hub/mistral_7b_instruct_dpo'

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
print("Tokenizer Loading Finished!")
model = AutoModelForCausalLM.from_pretrained(model_id)
print("Model Loading Finished!")

model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.to(device)
model_inputs_1 = encodeds_sorn_chosen.to(device)
model_inputs_2 = encodeds_hh_chosen.to(device)
model_inputs_3 = encodeds_sorn_labels.to(device)
model_inputs_4 = encodeds_hh_labels.to(device)

decoded_1 = tokenizer.batch_decode(model_inputs_1)
decoded_2 = tokenizer.batch_decode(model_inputs_2)
#decoded_3 = tokenizer.batch_decode(model_inputs_3)
#decoded_4 = tokenizer.batch_decode(model_inputs_4)
#decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print("=============================================")
print(decoded_1[0])
print("=============================================")
print(decoded_2[0])
print("=============================================")
#print(decoded_3[0])
print("=============================================")
#print(decoded_4[0])
