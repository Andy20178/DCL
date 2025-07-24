from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download, AutoTokenizer
import torch
import json
from tqdm import tqdm
import os
import transformers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--tokenizer_dir", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--data_type", type=str, default="data")
args = parser.parse_args()
model_dir = args.model_dir
tokenizer_dir = args.tokenizer_dir
#下载模型
# model_dir = snapshot_download(model_dir, cache_dir="./", revision="master")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(tokenizer_dir)
#读取jsonl文件
split = [args.split]
data_types = [args.data_type]
message_data_list = []
result_list = []
for s in split:
    for data_type in data_types:
        with open(f"PACS_data/inference_multi_video_json/{s}_{data_type}.jsonl", "r") as f:
            for line in f:
                message_data = json.loads(line)
                message_data_list.append(message_data)
        for messages in tqdm(message_data_list):
            messages_data, label_data = [messages['messages']], [messages['label']]
            text = processor.apply_chat_template(
                messages_data, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )
            # import pdb; pdb.set_trace()
            # 4. 处理视频输入
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages_data, return_video_kwargs=True)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to("cuda")

            # 5. 推理
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=32)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            print(output_text)

            # file_name = os.path.basename(file_path)           # 得到 object0102.mp4
            # name_without_ext = os.path.splitext(file_name)[0] # 得到 object0102
            # print(name_without_ext)
            video_1_name = os.path.basename(messages_data[0]['content'][0]['video'])
            video_2_name = os.path.basename(messages_data[0]['content'][1]['video'])
            video_1_name = os.path.splitext(video_1_name)[0]
            video_2_name = os.path.splitext(video_2_name)[0]
            
            result_list.append(
                {
                    "video_pair": f"{video_1_name}_{video_2_name}",
                    "question": messages_data[0]['content'][2]['text'],
                    "label": label_data,
                    "output": [output_text[0]]
                }
            )
            
        with open(f"PACS_data/inference_multi_video_json/{s}_{data_type}_result_3B_only_PACS_m.jsonl", "w") as f:
            for result in result_list:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            