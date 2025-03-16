import logging
import torch
from typing import Any, Union, List 
from model.build_model import build_model
from torchvision import transforms
import utils.transforms as audio_transforms
from PIL import Image
from dataset.dataset import PACSImageAudioDataset
def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def load_model(MODEL_PATH, device="cuda", args=None):
    
    model = torch.jit.load(MODEL_PATH, map_location="cpu")#
    model = build_model(model.state_dict(),args = args).to(device)#

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model

def load_dataset(args):
    '''
    input: args
    output: dataset_collecter, dataloader_collecter, dataset_size_collecter
    dataset_collecter: dict, key: train, val, test
    dataloader_collecter: dict, key: train, val, test
    dataset_size_collecter: dict, key: train, val, test
    '''
    train_img_transform = transforms.Compose([
        transforms.RandomResizedCrop((224,224), (0.85, 1.0), ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    train_second_transform = transforms.Compose([
            transforms.Resize((224,224), interpolation=Image.BICUBIC),
            transforms.RandomCrop((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


    train_audio_transform = transforms.Compose([
            audio_transforms.ToTensor1D(),
            # audio_transforms.RandomScale(),
            audio_transforms.RandomCrop(out_len=44100*5), 
            audio_transforms.RandomPadding(out_len=44100*5),
            audio_transforms.RandomNoise(p=0.8),
            audio_transforms.RandomFlip(p=0.5),
    ])
    train_q_transform = None
    train_dataset = PACSImageAudioDataset(args.data_dir, "train_data", img_transform=train_img_transform, q_transform=train_q_transform, second_transform=train_second_transform, audio_transform=train_audio_transform, extra_imgs=True, loss_fn=args.loss_fn)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    train_data_size = len(train_dataset)
    val_img_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    val_audio_transform = transforms.Compose([
            audio_transforms.ToTensor1D(),
            audio_transforms.RandomCrop(out_len=44100*5, train=False), 
            audio_transforms.RandomPadding(out_len=44100*5, train=False),
            ])

    val_q_transform = None
    val_dataset = PACSImageAudioDataset(args.data_dir, "val_data", img_transform=val_img_transform, q_transform=val_q_transform, test_mode=True, audio_transform=val_audio_transform, loss_fn=args.loss_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_data_size = len(val_dataset)
    img_transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    audio_transform_test = transforms.Compose([
            audio_transforms.ToTensor1D(),
            audio_transforms.RandomCrop(out_len=44100*5, train=False), 
            audio_transforms.RandomPadding(out_len=44100*5, train=False),
            ])
    q_transform_test = None
    test_dataset = PACSImageAudioDataset(args.data_dir, "test_data", img_transform=img_transform_test, q_transform=q_transform_test, audio_transform=audio_transform_test, extra_imgs=True, loss_fn=args.loss_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)
    test_data_size = len(test_dataset)
    dataset_size_collecter = dict()
    dataset_collecter = dict()
    dataloader_collecter = dict()
    dataset_size_collecter["train"] = train_data_size
    dataset_size_collecter["val"] = val_data_size
    dataset_size_collecter["test"] = test_data_size
    dataset_collecter["train"] = train_dataset
    dataset_collecter["val"] = val_dataset
    dataset_collecter["test"] = test_dataset
    dataloader_collecter["train"] = train_dataloader
    dataloader_collecter["val"] = val_dataloader
    dataloader_collecter["test"] = test_dataloader
    return dataset_collecter, dataloader_collecter, dataset_size_collecter
def compute_acc_loss(output, target, args, loss_dict, feature_dict):
    '''
    input: output, target, args
    output: loss, acc
    '''
    if args.loss_fn == "CrossEntropy":
        loss_fn = torch.nn.CrossEntropyLoss()
        pred = torch.softmax(output, dim=1)
        preds = (pred.argmax(dim=1) ).cpu()
        correct = (preds == target).sum().item()
        loss_classification = loss_fn(output, target)
    elif args.loss_fn == "Triplet":
        loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.CosineSimilarity(), margin=1)
        cs1 = output['cs1'].cpu()
        cs2 = output['cs2'].cpu()
        correct = (cs1 > cs2).sum().item()
        loss_classification = loss_fn(feature_dict['text_feature'], feature_dict['obj2_feature'], feature_dict['obj1_feature'])
    loss = loss_classification + loss_dict['loss_1_VAE'] + loss_dict['loss_2_VAE']
    if args.miss_modal != "None" and args.use_modal_share:
        loss += loss_dict['obj1_spec_video_loss'] + loss_dict['obj2_spec_video_loss'] + loss_dict['obj1_spec_audio_loss'] + loss_dict['obj2_spec_audio_loss'] + loss_dict['obj1_share_loss'] + loss_dict['obj2_share_loss']
    return loss, correct