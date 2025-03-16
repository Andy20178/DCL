import os
from torch.utils.data import Dataset
import json
import cv2
from PIL import Image, ImageFile
import clip
import random
import librosa
import numpy as np
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PACSImageAudioDataset(Dataset):
    def __init__(self, data_dir, split, img_transform=None, q_transform=None, second_transform=None, test_mode=False, extra_imgs=False, audio_transform=None, loss_fn = None):
        self.data_dir = data_dir
        self.split = split
        self.img_transform = img_transform
        self.second_transform = second_transform
        self.q_transform = q_transform
        self.audio_transform = audio_transform
        self.test_mode = test_mode
        self.extra_imgs = extra_imgs
        self.data = self.load_data()
        self.length = len(self.data)
        self.loss_fn = loss_fn
        self.obj_list = []
    def __len__(self):
        return self.length
    def get_obj_list(self):
        return self.obj_list
    def _load_image(self, sample_img):
        img = Image.open(os.path.join(self.data_dir,"square_crop", sample_img + ".png"))
        if self.second_transform and random.random() < 0.1 and not self.test_mode:
            img = self.second_transform(img)
        elif self.img_transform:
            img = self.img_transform(img)
        return img

    def _load_audio(self, sample_audio):
        wav, _ = librosa.load(os.path.join("/data/lcs/PACS_data", "audio44100", sample_audio + ".wav"), sr=44100, mono=True, dtype=np.float32)

        if wav.ndim == 1:
            wav = wav[:, np.newaxis]

        wav = wav.T

        if self.audio_transform:
            wav = self.audio_transform(wav)
        return wav
    def _load_video(self, sample_video):
        random_indices = np.random.randint(1, 100, 8)
        random_indices.sort()
        video_frames = torch.zeros((8, 3, 224, 224))
        for i in range(8):
            video_frame = Image.open(os.path.join(self.data_dir,"frames252", sample_video, f"{random_indices[i]:06d}.png"))
            if self.second_transform and random.random() < 0.1 and not self.test_mode:
                video_frame = self.second_transform(video_frame)
            elif self.img_transform:
                video_frame = self.img_transform(video_frame)
            video_frames[i] = video_frame
        return video_frames
    def _load_label(self, sample_label):
        
        label_dict = json.load(open("/data/lcs/PACS_data/json/label_dict.json"))
        label = label_dict[sample_label]
        return label
    def __getitem__(self, idx):
        sample = self.data[idx]
        #
        img1 = self._load_image(sample["obj1"])
        img2 = self._load_image(sample["obj2"])
        
        audio1 = self._load_audio(sample["obj1"])
        audio2 = self._load_audio(sample["obj2"])
        
        video1 = self._load_video(sample["obj1"])
        video2 = self._load_video(sample["obj2"])
        
        label1 = self._load_label(sample["obj1"])
        label2 = self._load_label(sample["obj2"])
        if self.loss_fn == "CE":
            target = {"obj1": sample["obj1"], "obj2": sample["obj2"], "question": sample["question"],"tokens":sample["tokens"],
                  "label": sample["label"], "img1": img1, "img2": img2, "video1": video1, "video2": video2,
                  "audio1": audio1, "audio2": audio2}
            return target
        elif self.loss_fn == "Triplet":
            if sample["label"] == 0:
                target = {"obj1":sample["obj1"], "obj2":sample["obj2"], "question":sample["question"],"question_id":sample["question_id"], "tokens":sample["tokens"], "img1":img1, "img2":img2, "audio1":audio1, "audio2":audio2, "video1":video1, "video2":video2, "label":0, "label1_m":label1, "label2_m":label2}
            else:
                target = {"obj1":sample["obj2"], "obj2":sample["obj1"], "question":sample["question"],"question_id":sample["question_id"], "tokens":sample["tokens"], "img1":img2, "img2":img1, "audio1":audio2, "audio2":audio1, "video1":video2, "video2":video1, "label":0, "label1_m":label2, "label2_m":label1}
            return target

    def load_data(self):
        data_json = json.load(open(os.path.join(self.data_dir, 'json', self.split + '.json')))
        data = []
        for pair in data_json:
            v1, v2 = pair.split("_")

            if not (os.path.exists(os.path.join(self.data_dir,"square_crop", v1 + ".png"))):
                print(f"did not find image: {v1}")
                continue
            if not (os.path.exists(os.path.join(self.data_dir,"square_crop", v2 + ".png"))):
                print(f"did not find image: {v2}")
                continue
            if not (os.path.exists(os.path.join("/data/lcs/PACS_data/audio44100/", v1 + ".wav"))):
                print(f"did not find audio: {v1}")
                continue
            if not (os.path.exists(os.path.join("/data/lcs/PACS_data/audio44100/", v2 + ".wav"))):
                print(f"did not find audio: {v2}")
                continue
            if not (os.path.exists(os.path.join(self.data_dir,"frames252", v1))):
                print(f"did not find video: {v1}")
                continue
            if not (os.path.exists(os.path.join(self.data_dir,"frames252", v2))):
                print(f"did not find video: {v2}")
                continue
            for q in data_json[pair]:
                sample = {"obj1":v1, "obj2":v2, "question":data_json[pair][q]["text"], "label":data_json[pair][q]["label"], "question_id":q, "tokens":clip.tokenize(data_json[pair][q]["text"])[0]}

                data.append(sample)
        return data
    