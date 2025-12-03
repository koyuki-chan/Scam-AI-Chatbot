import pandas as pd

# 載入資料
df = pd.read_csv("AI_dataset.csv")
df["Label"] = df["Label"].map({"fraud": 1, "normal": 0})

# 平台風險分數
source_risk_map = {
    "IG": 0.9, "Thread": 0.9, "WhatsApp": 0.8, "Facebook": 0.8,
    "Moovup": 0.2, "JobsDB": 0.2, "Indeed": 0.2, "PolyU":0.0,"Labour Department":0.0,"Other":0.5,"Unknown":0.5
}
df["source_risk"] = df["Source"].map(lambda x: source_risk_map.get(x, 0.5))

# 高風險關鍵詞分類
money_keywords = ["日薪", "高薪", "即日結算", "每單佣金", "紅包", "Alipay", "WeChat", "FPS", "支付寶", "回贈", "bonus"]
comm_keywords = ["加 WhatsApp", "加群", "入群", "Telegram", "任務群", "官方群", "老師", "客服", "回覆 OK", "回 有興趣"]
task_keywords = ["刷單", "模擬購買", "點 Like", "轉發", "留言", "觀看", "訂閱", "衝銷量", "寫好評", "打字員", "資料輸入"]
bait_keywords = ["無需經驗", "包教會", "彈性時間", "多勞多得", "屋企手機搞掂", "平台墊付", "無需出錢", "即刻領任務","學生","兼職","畢業生","主婦","在家工作","零經驗","高薪酬","日結","現金","現金出糧","無需投資","保證收入","快速賺錢","輕鬆賺錢","網上工作","彈性工時","自由工作","無需面試","即日上工","全職","兼職工作","短期工作","長期工作","高收入","穩定收入","額外收入","額外賺錢","無需技能","簡單工作","易做","易賺錢","無需資格","無需證書","無需學歷","無需經驗要求","無需培訓","無需指導","無需監督","無需管理","無需報告","無需考核","無需評估","無需面試官","無需人事部","無需招聘流程","無需工作證明","無需工作經驗證明","無需推薦信","無需背景調查","無需信用檢查","無需身份驗證"]

# 合併所有關鍵詞
risk_keywords = money_keywords + comm_keywords + task_keywords + bait_keywords

# 計算命中關鍵詞數量
def count_keywords(text):
    return sum(kw in text for kw in risk_keywords)

df["keyword_hits"] = df["Message"].apply(count_keywords)

# 分析 source 中 normal 和 fraud 的分佈
print("=== Source 分佈分析 ===")
source_distribution = df.groupby('Source')['Label'].agg(['count', 'sum', 'mean']).reset_index()
source_distribution.columns = ['Source', '總數', '詐騙數量', '詐騙比例']
source_distribution['正常數量'] = source_distribution['總數'] - source_distribution['詐騙數量']
source_distribution['詐騙比例'] = (source_distribution['詐騙比例'] * 100).round(2)

# 按總數排序
source_distribution = source_distribution.sort_values('總數', ascending=False)

print("\n各 Source 的詐騙分佈:")
# 設定顯示格式讓對齊更好
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', None)
print(source_distribution.to_string(index=False, formatters={
    'Source': '{{:<{}}}'.format(source_distribution['Source'].str.len().max()).format,
    '總數': '{:<6}'.format,
    '詐騙數量': '{:<8}'.format,
    '正常數量': '{:<8}'.format,
    '詐騙比例': '{:<8}'.format
}))

# 總計統計
total_fraud = df['Label'].sum()
total_normal = len(df) - total_fraud
total_records = len(df)

print(f"\n=== 總體統計 ===")
print(f"總記錄數: {total_records}")
print(f"詐騙數量 (fraud): {total_fraud} ({total_fraud/total_records*100:.2f}%)")
print(f"正常數量 (normal): {total_normal} ({total_normal/total_records*100:.2f}%)")

# 詐騙比例最高的前幾個 source
print(f"\n=== 詐騙比例最高的 Source (至少5筆資料) ===")
high_risk_sources = source_distribution[source_distribution['總數'] >= 5].nlargest(5, '詐騙比例')
print(high_risk_sources[['Source', '總數', '詐騙數量', '詐騙比例']].to_string(index=False))

df.to_csv("processed_dataset.csv", index=False)
print("\n✅ 資料處理完成，已儲存為 processed_dataset.csv")
# 在原本的程式碼後面加上這部分：

# 分析 source 中 normal 和 fraud 的分佈
print("\n" + "="*50)
print("Source 分佈分析")
print("="*50)

source_stats = df.groupby('Source').agg(
    total=('Label', 'count'),
    fraud=('Label', 'sum')
).reset_index()
source_stats['normal'] = source_stats['total'] - source_stats['fraud']
source_stats['fraud_ratio'] = (source_stats['fraud'] / source_stats['total'] * 100).round(2)

# 按總數排序
source_stats = source_stats.sort_values('total', ascending=False)

print(f"\n{'Source':<20} {'Total':<8} {'Fraud':<8} {'Normal':<8} {'Fraud %':<10}")
print("-" * 60)
for _, row in source_stats.iterrows():
    print(f"{row['Source']:<20} {row['total']:<8} {row['fraud']:<8} {row['normal']:<8} {row['fraud_ratio']:<10}")

# 總計
total_records = len(df)
total_fraud = df['Label'].sum()
total_normal = total_records - total_fraud

print(f"\n總記錄數: {total_records}")
print(f"詐騙數量: {total_fraud} ({total_fraud/total_records*100:.2f}%)")
print(f"正常數量: {total_normal} ({total_normal/total_records*100:.2f}%)")

df.to_csv("processed_dataset.csv", index=False)
print("\n✅ 資料處理完成，已儲存為 processed_dataset.csv")