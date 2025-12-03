import pandas as pd
import torch
import json
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dataset import FraudDataset
from model import FraudClassifier

df = pd.read_csv("processed_dataset.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = FraudDataset(train_df)
test_dataset = FraudDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FraudClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 訓練
for epoch in range(8):
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        risk = batch["risk"].to(device)
        keywords = batch["keywords"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, risk, keywords)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# 測試與評估數據記錄
model.eval()
all_preds, all_labels = [], []
evaluation_results = []  # 新增：用於儲存詳細評估結果

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        risk = batch["risk"].to(device)
        keywords = batch["keywords"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, risk, keywords)
        probabilities = torch.softmax(outputs, dim=1)  # 新增：獲取概率分佈
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 新增：詳細記錄每一筆預測結果
        batch_size = input_ids.size(0)
        for i in range(batch_size):

            original_text = f"batch_{batch_idx}_sample_{i}"  # 臨時標識
            
            record = {
                "message_id": f"eval_{batch_idx}_{i}",
                "input": {
                    "text_identifier": original_text,
                    "source_risk": risk[i].item(),
                    "keyword_hits": keywords[i].item()
                },
                "ground_truth": labels[i].item(),
                "model_output": {
                    "prediction": preds[i].item(),
                    "confidence": probabilities[i][1].item(),  # 詐騙類別的信心分數
                    "probabilities": {
                        "normal_prob": probabilities[i][0].item(),
                        "fraud_prob": probabilities[i][1].item()
                    }
                },
                "evaluation": {
                    "is_correct": (preds[i].item() == labels[i].item()),
                    "error_type": None  # 下面會判斷
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # 判斷錯誤類型
            if not record["evaluation"]["is_correct"]:
                if record["model_output"]["prediction"] == 1 and record["ground_truth"] == 0:
                    record["evaluation"]["error_type"] = "False Positive"
                elif record["model_output"]["prediction"] == 0 and record["ground_truth"] == 1:
                    record["evaluation"]["error_type"] = "False Negative"
            
            evaluation_results.append(record)

print(classification_report(all_labels, all_preds, target_names=["normal", "fraud"]))

# 新增：計算整體評估指標
total_samples = len(evaluation_results)
correct_predictions = sum(1 for record in evaluation_results if record["evaluation"]["is_correct"])
accuracy = correct_predictions / total_samples

false_positives = sum(1 for record in evaluation_results if record["evaluation"]["error_type"] == "False Positive")
false_negatives = sum(1 for record in evaluation_results if record["evaluation"]["error_type"] == "False Negative")

# 新增：保存詳細評估結果到 JSON 檔案
evaluation_summary = {
    "model_info": {
        "model_name": "FraudClassifier",
        "evaluation_date": datetime.now().isoformat(),
        "device_used": str(device)
    },
    "performance_metrics": {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    },
    "detailed_results": evaluation_results
}

# 保存 JSON 檔案
with open("model_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)

print(f"✅ 模型訓練完成")
print(f"✅ 評估結果已保存至 'model_evaluation_results.json'")
print(f"📊 整體準確率: {accuracy:.4f}")
print(f"📊 誤報數 (False Positives): {false_positives}")
print(f"📊 漏報數 (False Negatives): {false_negatives}")

torch.save(model.state_dict(), "fraud_model.pt")
print("✅ 模型權重已儲存為 'fraud_model.pt'")