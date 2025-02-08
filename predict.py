import torch
import numpy as np
import torch.nn as nn
from utils_extra import load_model
from torchvision import transforms
from PIL import Image
import json
import clip
from collections import defaultdict
from model import AudioCLIPFinetune
import utils.transforms as audio_transforms
import librosa
import os
import argparse
from dataset.dataset import PACSImageAudioDataset

parser = argparse.ArgumentParser(description='predict on audioclip')
parser.add_argument(
        '--model_path',
        default="/home/lcs/PACS-lcs/original/experiments/AudioCLIP/logs/9/model_audioclip_9_39.pt",
        type=str,
        help='Model path'
    )
parser.add_argument(
        '--save_dir',
        dest='save_dir',
        default="results/",
        type=str,
        help='Directory containing PACS data'
    )

parser.add_argument(
        '--split',
        default="test_data",
        type=str,
        help='which split to predict'
    )

parser.add_argument(
        '--data_dir',
        default="/data/lcs/PACS_data",
        type=str,
        help='which split to predict'
    )
parser.add_argument(
    '--use_VAE',
    default=True,
    type=bool,
)
parser.add_argument(
    '--use_TransVAE',
    default=True,
    type=bool,
)
parser.add_argument(
    '--use_knowledge_relation',
    default=False,
    type=bool,
)
parser.add_argument(
    '--loss_fn',
    default="Triplet",
    type=str,
)

parser.add_argument(
    '--use_static_knowledge',
    default=False,
    type=bool,
)
parser.add_argument(
    '--use_dynamic_knowledge',
    default=False,
    type=bool,
)
parser.add_argument(
    '--use_original_video_knowledge',
    default=False,
    type=bool,
)
parser.add_argument(
    '--use_audio_knowledge',
    default=False,
    type=bool,
)
parser.add_argument(
    '--use_image_knowledge',
    default=False,
    type=bool,
)
parser.add_argument(
    '--run_num',
    default="9",
    type=str,
)
parser.add_argument(
    '--batch_size',
    default=64,
    type=int,
)
args = parser.parse_args()

PRELOAD = "assets/AudioCLIP-Partial-Training.pt"
MODEL_PATH = args.model_path

PRE_LOAD = "assets/ViT-B-16.pt"
DATA_DIR = args.data_dir
SPLIT = args.split

audio_model = AudioCLIPFinetune(pretrained=f'{PRELOAD}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model(PRE_LOAD, device,args=args)
model.audio_model = audio_model.audio
# model.text_projection = nn.Parameter(torch.empty(512, 512))
# model.audio_image_fuse = nn.Linear(1024+512,512)

checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint)
# model.text_projection.data = model.text_projection.data.half()
# for param in model.audio_image_fuse.parameters():
#     param.data = param.data.half()
model = model.to(device)
if args.loss_fn == "CE":
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
elif args.loss_fn == "Triplet":
    loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(), margin=1)
#裁剪图像
img_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

audio_transform = transforms.Compose([
        audio_transforms.ToTensor1D(),
        audio_transforms.RandomCrop(out_len=44100*5, train=False), 
        audio_transforms.RandomPadding(out_len=44100*5, train=False),
        ])

similarities = defaultdict(dict)
#加载数据集
q_transform = None
test_dataset = PACSImageAudioDataset(args.data_dir, "test_data", img_transform=img_transform, q_transform=q_transform, audio_transform=audio_transform, extra_imgs=True, loss_fn=args.loss_fn)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)
data_size_test = len(test_dataset)
# test_data = json.load(open(f"{DATA_DIR}/json/{SPLIT}.json", 'r'))
print("data_size_test",data_size_test)
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(test_dataloader):
        img1 = data["img1"].to(device)
        img2 = data["img2"].to(device)
        audio1 = data["audio1"].to(device)
        audio2 = data["audio2"].to(device)
        video1 = data["video1"].to(device)
        video2 = data["video2"].to(device)
        token = data["tokens"].to(device)
        label = data["label"].to(device)
        if args.use_VAE:
            if args.loss_fn == "CE":
                output, VAE_Loss_obj_1, VAE_loss_obj_2 = model(img1, audio1, img2, audio2, video1, video2, token, data_size_test,args)
                pred = torch.softmax(output, dim=1)
                
                loss = loss_fn(pred, label) + VAE_Loss_obj_1 + VAE_loss_obj_2
                preds = (pred.argmax(dim=1) )
                correct = (preds == label).sum().item()
            elif args.loss_fn == "Triplet":
                if args.use_TransVAE:
                    objf1, objf2, textf1, VAE_Loss_obj_1, VAE_loss_obj_2, loss_dict = model(img1, audio1, img2, audio2, video1, video2, token, data_size_test,args)
                    loss = loss_fn(textf1, objf2, objf1) + VAE_Loss_obj_1 + VAE_loss_obj_2
                else:
                    objf1, objf2, textf1, VAE_Loss_obj_1, VAE_loss_obj_2 = model(img1, audio1, img2, audio2, video1, video2, token, data_size_test,args)
                    loss = loss_fn(textf1, objf2, objf1) + VAE_Loss_obj_1 + VAE_loss_obj_2
        cs1 = nn.CosineSimilarity()(textf1, objf1)
        cs2 = nn.CosineSimilarity()(textf1, objf2)
        correct = (cs1 > cs2).sum().item()
        total += correct
        print(correct, total)
    total_acc = total / 11044
    ##train是11044，test是1164
    print(correct, total_acc)
print(correct, total_acc)
        
# json.dump(dict(similarities), open(os.path.join(args.save_dir, f"preds_{SPLIT}.json"), 'w'))