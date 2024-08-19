import torch 
import json
from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset


class VQADataset(Dataset):
    def __init__(self, dataset, processor=None):
        self.data = json.load(open(os.path.join(dataset, "VQA_RAD_Dataset_Public.json"), 'r'))
        self.processor = processor
        self.image_dir = os.path.join(dataset, "images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.image_dir, sample['image_name'])
        image = Image.open(img_path).convert("RGB")
        question = sample["question"]
        prompt = "USER: <analysis_image>\n%s \nASSISTANT:" % question
        # if self.transform:
        #     analysis_image = self.transform(analysis_image)
        # if self.processor:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # question = annotation['question']
        # answer = annotation['answer']
        answer = sample["answer"]
        return inputs, question, answer
