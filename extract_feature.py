import os
import json
import torch
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLModel
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import argparse
# 初始化模型和处理器
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, default="3B")
args = parser.parse_args()
model_size = [args.model_size]
# model_dir = "Qwen/Qwen2.5-VL-32B-Instruct"

for size in model_size:
    # model_dir = snapshot_download(f"Qwen/Qwen2.5-VL-{size}-Instruct", cache_dir="./", revision="master")
    model_dir = f"Qwen/Qwen2.5-VL-{size}-Instruct"
#下载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True
    )
    model_visual = model.visual
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_dir)

    # 遍历目录
    base_dir = "PACS_data/frame252"
    output_dir = f"PACS_data/frame252_features_{size}"  # 指定保存特征的路径

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    def process_image(image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        image_pixel_values = inputs['pixel_values']
        image_grid_thw = inputs['image_grid_thw']
        with torch.no_grad():  # 禁用梯度计算
            image_embeds = model.visual(image_pixel_values, grid_thw=image_grid_thw)
        
        image_embeds = torch.mean(image_embeds, dim=0) 
        image_embeds = image_embeds.unsqueeze(0)
        # import pdb; pdb.set_trace()
        # 释放不必要的显存
        del inputs, image_pixel_values, image_grid_thw
        torch.cuda.empty_cache()

        return image_embeds


    # 遍历所有子目录和文件
    for object_dir in tqdm(os.listdir(base_dir), desc="Processing objects"):
        object_path = os.path.join(base_dir, object_dir)
        if os.path.isdir(object_path):
            print(f"Processing object: {object_dir}")
            object_output_path = os.path.join(output_dir, f"{object_dir}.pt")  # 每个object保存一个pt文件
            
            # 创建一整个object对应的feature
            object_visual_feature = []

            # 限制处理的图片数量为100帧
            frame_count = 0
            for frame_file in tqdm(sorted(os.listdir(object_path)), desc=f"Processing frames for {object_dir}"):
                if frame_file.endswith(".png") and frame_count < 100:
                    frame_path = os.path.join(object_path, frame_file)
                    frame_id = frame_file.split(".")[0]  # 假设文件名是数字
                    print(f"Processing frame: {frame_file}")
                    
                    # 处理图像并获取特征
                    image_embeds = process_image(frame_path)
                    
                    # 将特征添加到object_visual_feature列表中
                    object_visual_feature.append(image_embeds)
                    
                    # 更新计数器
                    frame_count += 1
            
            # 将所有帧的特征堆叠成一个张量并保存
            object_visual_feature = torch.cat(object_visual_feature, dim=0)  # 假设特征可以堆叠
            torch.save(object_visual_feature, object_output_path)
            # import pdb; pdb.set_trace()
            print(f"Saved features for object {object_dir} to {object_output_path}")
            del object_visual_feature
            torch.cuda.empty_cache()