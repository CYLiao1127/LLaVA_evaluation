# LLaVA evaluation

## Folder content
    .
    ├── calculate_bert_score.py      # Test the result of bert score
    ├── llava_VQA.py                 # Code from huggingface: https://huggingface.co/docs/transformers/model_doc/llava
    ├── dataset_analysis.py          # Plot and analysis dataset
    ├── evaluate_rad.py              # Evaluate the VQA_RAD_dataset
    ├── VQADataset.py                # Dataloader for parallel Computing (not use now)
    ├── dataset                      # Download the VQA dataset and unzip in this folder
    │   ├── image
    │   │   └── ...                  # images
    │   └── VQA_RAD_Dataset_Public.json                
    ├── analysis_image               # Save the dataset analysis result
    │   └── ...                      # analysis results
    ├── result                       # Save the evaluate result
    │   └── ...                      # evaluate result
    │   requirements.txt              # Package use in this project
    └── README.md

## Requirements
* If PyTorch fails to install successfully, it might be due to issues with **CUDA versioning and PyTorch compatibility**.
* Some packages may not install through *pip install*. Please search for a solution on Google.
```
conda create -n venv python=3.10
conda activate venv
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```


## Evaluate VQA_Rad dataset

* download the dataset for the [VQA_Rad](https://osf.io/89kps/)
* unzip the data into dataset folder
* run the evaluation code **evaluate_rad.py**
```script
python evaluate_rad.py
```

## Result of bert score
```
Prediction: 巴黎
Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000

Prediction: 法國首都是巴黎
Precision: 0.6286, Recall: 0.8455, F1 Score: 0.7211

Prediction: 巴黎是法國首都
Precision: 0.6230, Recall: 0.8311, F1 Score: 0.7122

Prediction: 法國首都是倫敦
Precision: 0.5490, Recall: 0.6511, F1 Score: 0.5957
```
