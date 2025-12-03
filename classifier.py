import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 模擬訓練資料（你可以換成真實資料）
data = [
    ("刷單任務，墊付返還", "刷單詐騙"),
    ("帳號異常，請重新綁定", "假客服詐騙"),
    ("穩賺不賠，加入投資群", "投資詐騙"),
    ("高薪打字員，加 WhatsApp", "假招聘詐騙"),
    ("這是一份真實招聘", "非詐騙")
]

df = pd.DataFrame(data, columns=["text", "label"])

# TF-IDF 向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 訓練分類器
clf = LogisticRegression()
clf.fit(X, y)

# 儲存模型
with open("scam_classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, clf), f)