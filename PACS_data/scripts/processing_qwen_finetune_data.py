import json
import cv2
import numpy as np
import os
from tqdm import tqdm

#当前的版本：
#将两个video拼接然后问话，并且PACS和PACS-Material的数据分开保存，分开训练
split = ['val', 'test']
data_type =['data', 'data_mat']
for s in split:
    for small_data_type in data_type:
        samples = []
        output_jsonl = f"PACS_data/finetune_json/{s}_{small_data_type}_data.jsonl"
        # 路径配置
        input_json = f"PACS_data/json/{s}_{small_data_type}.json"
        video_dir = "PACS_data/videos/"
        output_video_dir = "PACS_data/concatenated_videos/"
        # 读取原始数据
        with open(input_json, "r") as f:
            data = json.load(f)
        for pair_key, questions in tqdm(data.items()):
            # pair_key 形如 object0004_object0005
            obj1, obj2 = pair_key.split("_")
            video1_path = f"{video_dir}{obj1}.mp4"
            video2_path = f"{video_dir}{obj2}.mp4"
            if os.path.exists(f"{output_video_dir}{obj1}_{obj2}_concatenated_video.mp4"):   
                # print(f'{obj1}_{obj2} video already exists')
                pass
            else:
                # 读取视频1
                video1 = cv2.VideoCapture(video1_path)
                fps = video1.get(cv2.CAP_PROP_FPS)
                width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video1_frames = []
                while True:
                    ret, frame = video1.read()
                    if not ret:
                        break
                    video1_frames.append(frame)
                video1.release()

                # 读取视频2
                video2 = cv2.VideoCapture(video2_path)
                video2_frames = []
                while True:
                    ret, frame = video2.read()
                    if not ret:
                        break
                    video2_frames.append(frame)
                video2.release()

                # 生成2秒黑屏
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                black_frames = [black_frame.copy() for _ in range(int(2 * fps))]

                # 合并所有帧
                all_frames = video1_frames + black_frames + video2_frames

                # 保存为mp4
                out_path = f"{output_video_dir}{obj1}_{obj2}_concatenated_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                for idx, frame in enumerate(all_frames):
                    if frame is None:
                        print(f"Frame {idx} is None! 跳过该帧。")
                        continue
                    if frame.shape != (height, width, 3):
                        frame = cv2.resize(frame, (width, height))
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    out.write(frame)
                out.release()
                print(f'{obj1}_{obj2} video concatenated')
            for qid, qinfo in questions.items():
                # label: 1表示video1，2表示video2
                label = qinfo["label"] + 1
                question = qinfo["text"]
                sample = {
                    "video": f"{output_video_dir}{obj1}_{obj2}_concatenated_video.mp4",
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<video>\n You are an expert with knowledge of physical principles. You need to pay attention to certain physical properties of objects in the videos I provide and answer questions based on these properties. In the videos I provide, the content before the blackout is referred to as 'video1', and the content after the blackout is referred to as 'video2'. For each question, you need to choose either 'video1' or 'video2' as the most suitable answer. Please note that you are only allowed to answer with 'video1' or 'video2'. The question is: {question}"
                        },
                        {
                            "from": "gpt",
                            "value": f"video {label}"
                        }
                    ]
                }
                samples.append(sample)

    # 写入jsonl
        with open(output_jsonl, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"处理完成，共生成{len(samples)}条样本，保存在{output_jsonl}")