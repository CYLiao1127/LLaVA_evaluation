from bert_score import score

# The answer and predictions
answer = ["巴黎", "巴黎", "巴黎", "巴黎"]
predictions = [
    "巴黎",
    "法國首都是巴黎",
    "巴黎是法國首都",
    "法國首都是倫敦"
]

# Calculate BERTScore
P, R, F1 = score(predictions, answer, lang='zh', verbose=True)

# Print the results
for i, pred in enumerate(predictions):
    print(f"Prediction: {pred}")
    print(f"Precision: {P[i].item():.4f}, Recall: {R[i].item():.4f}, F1 Score: {F1[i].item():.4f}\n")


''' 
Prediction: 巴黎
Precision: 1.0000, Recall: 1.0000, F1 Score: 1.0000

Prediction: 法國首都是巴黎
Precision: 0.6286, Recall: 0.8455, F1 Score: 0.7211

Prediction: 巴黎是法國首都
Precision: 0.6230, Recall: 0.8311, F1 Score: 0.7122

Prediction: 法國首都是倫敦
Precision: 0.5490, Recall: 0.6511, F1 Score: 0.5957
'''