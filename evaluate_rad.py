import torch
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm
from VQADataset import VQADataset
import os
import json
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LLava:
    def __init__(self, dataset):
        self.dataset =dataset
        self.load_data()

        self.image_dir = os.path.join(self.dataset, "images")

        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.evaluate(self.model)

    def load_data(self, dataset="dataset/"):
        self.data = json.load(open(os.path.join(dataset, "VQA_RAD_Dataset_Public.json"), 'r'))

    def save_csv(self, answer, question, prediction):
        with open("result/predict_1.csv", "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(["question", "answer", "prediction"])

            for line in zip(question, answer, prediction):
                writer.writerow(line)

    # Define the evaluation function
    def evaluate(self, model):
        answers, questions, predictions = [], [], []

        for idx, sample in tqdm(enumerate(self.data[0: 500])):
            # sample = self.data[idx]
            img_path = os.path.join(self.image_dir, sample['image_name'])
            image = Image.open(img_path).convert("RGB")
            question = sample["question"]
            prompt = "USER: <analysis_image>\n%s \nASSISTANT:" % question
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            if sample["answer_type"] == "CLOSED":
                # usually closed question has only one token answer
                max_new_tokens = 1
            else:
                max_new_tokens = 20
            # max_new_tokens = 20
            generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            prediction = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answer = sample["answer"]

            answers.append(answer)
            questions.append(question)
            predictions.append(prediction[prediction.find("\nASSISTANT:")+12:])

        self.save_csv(answers, questions, predictions)


if __name__ == '__main__':
    LLava("dataset/")
