import json
# import cv2
import numpy as np
import os
from tqdm import tqdm
split = ['train', 'val', 'test']
#   data_type =['data']
data_type = ['data_mat']
for s in split:
    samples = []
    output_jsonl = f"PACS_data/inference_multi_video_json/{s}_{data_type[0]}.jsonl"
    #创建对应的文件夹
    os.makedirs(f"PACS_data/inference_multi_video_json", exist_ok=True)
    for small_data_type in data_type:
        # 路径配置
        input_json = f"PACS_data/json/{s}_{small_data_type}.json"
        video_dir = "PACS_data/videos/"
        # 读取原始数据
        with open(input_json, "r") as f:
            data = json.load(f)
        for pair_key, questions in tqdm(data.items()):
            # pair_key 形如 object0004_object0005
            obj1, obj2 = pair_key.split("_")
            video1_path = f"{video_dir}{obj1}.mp4"
            video2_path = f"{video_dir}{obj2}.mp4"
            for qid, qinfo in questions.items():
                # label: 1表示video1，2表示video2
                label = qinfo["label"] + 1
                question = qinfo["text"]
                sample = {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video1_path,
                            "max_pixels": 360 * 420,
                            "fps": 1.0,
                        },
                        {
                            "type": "video",
                            "video": video2_path,
                            "max_pixels": 360 * 420,
                            "fps": 1.0,
                        },
                        {
                            "type": "text",
                            "text": f"I need you to act as a robot with physical common sense. "
                                    "Analyze the objects in the two videos above, and choose the one that would be a worse makeshift parachute for the question: "
                                    f"{question} "
                                    "You should only answer 'video 1' or 'video 2', and nothing else."
                        }
                    ]
                }
                samples.append([sample, label])

    # 写入jsonl
    #将label和sample写成一个jsonl文件
    with open(output_jsonl, "w") as f:
        for sample, label in samples:
            f.write(json.dumps({"messages": sample, "label": f"video {label}"}, ensure_ascii=False) + "\n")

    print(f"处理完成，共生成{len(samples)}条样本，保存在{output_jsonl}")