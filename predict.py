import torch
from transformers import BertTokenizer
from model import FraudClassifier

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = FraudClassifier()
model.load_state_dict(torch.load("fraud_model.pt", map_location="cpu"))
model.eval()

# 風險來源與關鍵詞庫
source_risk_map = {
    "IG": 0.9, "Thread": 0.9, "WhatsApp": 0.8, "Facebook": 0.8,
    "Moovup": 0.2, "JobsDB": 0.2, "Indeed": 0.2,"Other":0.5
}
money_keywords = ["日薪", "高薪", "即日結算", "每單佣金", "紅包", "Alipay", "WeChat", "FPS", "支付寶", "回贈", "bonus"]
comm_keywords = ["加 WhatsApp", "加群", "入群", "Telegram", "任務群", "官方群", "老師", "客服", "回覆 OK", "回 有興趣"]
task_keywords = ["刷單", "模擬購買", "點 Like", "轉發", "留言", "觀看", "訂閱", "衝銷量", "寫好評", "打字員", "資料輸入"]
bait_keywords = ["無需經驗", "包教會", "彈性時間", "多勞多得", "屋企手機搞掂", "平台墊付", "無需出錢", "即刻領任務","學生","兼職","畢業生","主婦","在家工作","零經驗","高薪酬","日結","現金","現金出糧","無需投資","保證收入","快速賺錢","輕鬆賺錢","網上工作","彈性工時","自由工作","無需面試","即日上工","全職","兼職工作","短期工作","長期工作","高收入","穩定收入","額外收入","額外賺錢","無需技能","簡單工作","易做","易賺錢","無需資格","無需證書","無需學歷","無需經驗要求","無需培訓","無需指導","無需監督","無需管理","無需報告","無需考核","無需評估","無需面試官","無需人事部","無需招聘流程","無需工作證明","無需工作經驗證明","無需推薦信","無需背景調查","無需信用檢查","無需身份驗證"]
risk_keywords = money_keywords + comm_keywords + task_keywords + bait_keywords

import torch
from transformers import BertTokenizer
from model import FraudClassifier

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = FraudClassifier()
model.load_state_dict(torch.load("fraud_model.pt", map_location="cpu"))
model.eval()

source_risk_map = {
    "IG": 0.9, "Thread": 0.9, "WhatsApp": 0.8, "Facebook": 0.8,
    "Moovup": 0.2, "JobsDB": 0.2, "Indeed": 0.2, "Other": 0.5,"PolyU":0.0,"Labour Department":0.0
}

money_keywords = ["日薪", "高薪", "即日結算", "每單佣金", "紅包", "Alipay", "WeChat", "FPS", "支付寶", "回贈", "bonus"]
comm_keywords = ["加 WhatsApp", "加群", "入群", "Telegram", "任務群", "官方群", "老師", "客服", "回覆 OK", "回 有興趣"]
task_keywords = ["刷單", "模擬購買", "點 Like", "轉發", "留言", "觀看", "訂閱", "衝銷量", "寫好評", "打字員", "資料輸入"]
bait_keywords = [
    "無需經驗", "包教會", "彈性時間", "多勞多得", "屋企手機搞掂", "平台墊付", "無需出錢", "即刻領任務","學生","兼職","畢業生","主婦",
    "在家工作","零經驗","高薪酬","日結","現金","現金出糧","無需投資","保證收入","快速賺錢","輕鬆賺錢","網上工作","彈性工時",
    "自由工作","無需面試","即日上工","全職","兼職工作","短期工作","長期工作","高收入","穩定收入","額外收入","額外賺錢",
    "無需技能","簡單工作","易做","易賺錢","無需資格","無需證書","無需學歷","無需經驗要求","無需培訓","無需指導","無需監督",
    "無需管理","無需報告","無需考核","無需評估","無需面試官","無需人事部","無需招聘流程","無需工作證明","無需工作經驗證明",
    "無需推薦信","無需背景調查","無需信用檢查","無需身份驗證"
]
risk_keywords = money_keywords + comm_keywords + task_keywords + bait_keywords


def predict_message(text, source, lang="zh"):
    """Predict and explain in Chinese or English based on lang."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    risk = torch.tensor([[source_risk_map.get(source, 0.5)]], dtype=torch.float)
    keyword_hits = torch.tensor([[sum(kw in text for kw in risk_keywords)]], dtype=torch.float)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"], risk, keyword_hits)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    # === 根據語言輸出 ===
    if lang == "zh":
        label = "❌ 詐騙招聘" if pred == 1 else "✅ 真實招聘"
        reasons = []
        if label == "❌ 詐騙招聘" and source_risk_map.get(source, 0.5) >= 0.8:
            reasons.append(f"來自高風險平台 {source}")

        def match_keywords(text, keywords, label):
            return [f"{label}：{kw}" for kw in keywords if kw in text]

        reasons += match_keywords(text, money_keywords, "金錢誘因")
        reasons += match_keywords(text, comm_keywords, "通訊操作")
        reasons += match_keywords(text, task_keywords, "任務類型")
        reasons += match_keywords(text, bait_keywords, "詐騙話術")

        if label == "❌ 詐騙招聘" and reasons:
            explanation = "\n📌 原因：" + "\n- " + "\n- ".join(reasons)
        elif label == "✅ 真實招聘":
            explanation = "\n📌 原因：暫無明顯詐騙特徵。"
        else:
            explanation = "\n📌 原因：暫無明顯詐騙特徵。"

        return {
            "label": f"{label}（可信度：{confidence:.2f}）",
            "explanation": explanation
        }

    else:
        label = "❌ Fraudulent Job Offer" if pred == 1 else "✅ Genuine Job Posting"
        reasons = []
        if label == "❌ Fraudulent Job Offer" and source_risk_map.get(source, 0.5) >= 0.8:
            reasons.append(f"Posted on high-risk platform: {source}")

        def match_keywords(text, keywords, category):
            return [f"{category}: {kw}" for kw in keywords if kw in text]

        reasons += match_keywords(text, money_keywords, "Financial incentive")
        reasons += match_keywords(text, comm_keywords, "Communication pattern")
        reasons += match_keywords(text, task_keywords, "Task type")
        reasons += match_keywords(text, bait_keywords, "Scam language")

        if label == "❌ Fraudulent Job Offer" and reasons:
            explanation = "\n📌 Reasons detected:\n- " + "\n- ".join(reasons)
        elif label == "✅ Genuine Job Posting":
            explanation = "\n📌 Reasons: No significant scam indicators found."   
        else:
            explanation = "\n📌 Reason: No significant scam indicators found."

        return {
            "label": f"{label} (Confidence: {confidence:.2f})",
            "explanation": explanation
        }

# 測試
# print(predict_message("Hi 我係 Mandy～而家急聘網上打字員📝 日薪 $1800！加 WhatsApp 了解", source="IG"))
# print(predict_message("""偉大嘅Threads
# 我哋搵緊part-time幫手返Pop-up Store Helper 📅 11月起返工
# 19-40歲(歡迎畢業生/大學生/轉工者兼職)
# 歡迎搵埋朋友一齊返！
# $90 一個鐘 8個鐘包飯鐘
# 要求有禮貌・有責任心
# 有意dm wts 6429 4986""", source="IG"))
# test_message = """Hello, this is Sarah from HR Dept. We found your CV on Indeed. Part-time online reviewer needed, $500-1500/day, review apps & write comments. Reply "REVIEW" to join."""
# print(f"Test Message Prediction: {test_message}")
# print(predict_message(test_message,source="IG",lang="en"))