# anti_fraud_chatbot_ui_dual_switch.py
import re
import json
import requests
import pandas as pd
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from predict import predict_message  # 你的 FraudClassifier 模組
import random
import time

# ========== 模型與 Ollama ==========
sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# ========== 資料集處理 ==========
def load_dataset_examples(csv_path="dataset.csv", lang="zh"):
    """從 CSV 載入詐騙和正常訊息的例子"""
    try:
        df = pd.read_csv(csv_path)
        examples = {
            "scam": [],
            "legit": []
        }
        
        for _, row in df.iterrows():
            message = str(row['Message']).strip()
            label = str(row['Label']).strip().lower()
            source = str(row.get('Source', 'Unknown'))
            
            if not message:
                continue
                
            # 根據 fraud 和 normal 標籤分類
            if label == "fraud":
                examples["scam"].append({
                    "message": message,
                    "source": source,
                    "label": label
                })
            elif label == "normal":
                examples["legit"].append({
                    "message": message,
                    "source": source,
                    "label": label
                })
        
        return examples
    except Exception as e:
        print(f"載入資料集錯誤: {e}")
        return {"scam": [], "legit": []}

def get_examples_output(example_type="both", count=3, lang="zh"):
    """根據類型隨機獲取例子並直接輸出"""
    examples = load_dataset_examples(lang=lang)
    
    # 隨機抽樣
    if example_type == "scam":
        selected_examples = random.sample(examples["scam"], min(count, len(examples["scam"]))) if examples["scam"] else []
    elif example_type == "legit":
        selected_examples = random.sample(examples["legit"], min(count, len(examples["legit"]))) if examples["legit"] else []
    else:  # both
        scam_samples = random.sample(examples["scam"], min(count, len(examples["scam"]))) if examples["scam"] else []
        legit_samples = random.sample(examples["legit"], min(count, len(examples["legit"]))) if examples["legit"] else []
        selected_examples = None  # 標記為 both 類型
    
    if lang == "zh":
        if example_type == "scam":
            if not selected_examples:
                return "⚠️ 目前沒有找到詐騙訊息的例子"
            
            result = ["## 🛑 詐騙訊息例子 (fraud)\n"]
            result.append("以下是一些隨機抽取的詐騙訊息例子：\n")
            
            for i, example in enumerate(selected_examples, 1):
                result.append(f"**例子 {i}:**")
                result.append(f"- 📱 來源: {example['source']}")
                result.append(f"- 🏷️ 標籤: {example['label']}")
                result.append(f"- 💬 訊息: {example['message']}")
                result.append("")
            
            result.append("---")
            result.append("### 🔍 詐騙訊息常見特徵:")
            result.append("- 💰 高薪誘惑、輕鬆賺錢")
            result.append("- 📞 要求加 LINE/WhatsApp 等通訊軟體")
            result.append("- 💳 要求先付押金或保證金")
            result.append("- ⚠️ 保證收益、零風險")
            result.append("- 🔗 可疑的連結或聯絡方式")
            result.append(f"\n*本次隨機顯示 {len(selected_examples)} 個詐騙例子*")
            
        elif example_type == "legit":
            if not selected_examples:
                return "⚠️ 目前沒有找到正常訊息的例子"
            
            result = ["## ✅ 正常訊息例子 (normal)\n"]
            result.append("以下是一些隨機抽取的正常招聘訊息例子：\n")
            
            for i, example in enumerate(selected_examples, 1):
                result.append(f"**例子 {i}:**")
                result.append(f"- 📱 來源: {example['source']}")
                result.append(f"- 🏷️ 標籤: {example['label']}")
                result.append(f"- 💬 訊息: {example['message']}")
                result.append("")
            
            result.append("---")
            result.append("### ✅ 正常訊息特徵:")
            result.append("- 🏢 明確的公司名稱和職位")
            result.append("- 📍 具體的工作地點和內容")
            result.append("- 💼 合理的薪資範圍")
            result.append("- 📧 正式的公司聯絡方式")
            result.append("- 📄 完整的申請流程")
            result.append(f"\n*本次隨機顯示 {len(selected_examples)} 個正常例子*")
            
        else:  # both
            if not scam_samples and not legit_samples:
                return "⚠️ 目前沒有找到任何例子"
            
            result = ["## 📋 詐騙 (fraud) vs 正常 (normal) 訊息例子對比\n"]
            result.append("以下是一些隨機抽取的例子對比：\n")
            
            if scam_samples:
                result.append("### 🛑 詐騙訊息例子 (fraud):")
                for i, example in enumerate(scam_samples, 1):
                    result.append(f"{i}. **{example['source']}** - {example['label']}")
                    result.append(f"   💬 {example['message']}")
                    result.append("")
            
            if legit_samples:
                result.append("### ✅ 正常訊息例子 (normal):")
                for i, example in enumerate(legit_samples, 1):
                    result.append(f"{i}. **{example['source']}** - {example['label']}")
                    result.append(f"   💬 {example['message']}")
                    result.append("")
            
            result.append("---")
            result.append("### 🔍 如何區分詐騙與正常訊息:")
            result.append("**🚩 詐騙紅旗 (fraud):**")
            result.append("- 過度高薪且工作輕鬆")
            result.append("- 要求先付款或提供銀行資料")
            result.append("- 聯絡方式只有通訊軟體")
            result.append("- 語焉不詳、急迫性強")
            result.append("")
            result.append("**✅ 正常特徵 (normal):**")
            result.append("- 明確的公司和職位資訊")
            result.append("- 合理的薪資和工作要求")
            result.append("- 正式的面試流程")
            result.append("- 公開透明的聯絡方式")
            result.append(f"\n*本次隨機顯示 {len(scam_samples)} 個詐騙例子和 {len(legit_samples)} 個正常例子*")
    
    else:  # English
        if example_type == "scam":
            if not selected_examples:
                return "⚠️ No fraud examples found"
            
            result = ["## 🛑 Fraud Message Examples\n"]
            result.append("Here are some randomly selected fraud message examples:\n")
            
            for i, example in enumerate(selected_examples, 1):
                result.append(f"**Example {i}:**")
                result.append(f"- 📱 Source: {example['source']}")
                result.append(f"- 🏷️ Label: {example['label']}")
                result.append(f"- 💬 Message: {example['message']}")
                result.append("")
            
            result.append("---")
            result.append("### 🔍 Common Fraud Characteristics:")
            result.append("- 💰 High salary with minimal work")
            result.append("- 📞 Request to contact via messaging apps")
            result.append("- 💳 Ask for deposit or advance payment")
            result.append("- ⚠️ Guaranteed returns, zero risk claims")
            result.append("- 🔗 Suspicious links or contact methods")
            result.append(f"\n*Randomly showing {len(selected_examples)} fraud examples*")
            
        elif example_type == "legit":
            if not selected_examples:
                return "⚠️ No normal examples found"
            
            result = ["## ✅ Normal Message Examples\n"]
            result.append("Here are some randomly selected normal job message examples:\n")
            
            for i, example in enumerate(selected_examples, 1):
                result.append(f"**Example {i}:**")
                result.append(f"- 📱 Source: {example['source']}")
                result.append(f"- 🏷️ Label: {example['label']}")
                result.append(f"- 💬 Message: {example['message']}")
                result.append("")
            
            result.append("---")
            result.append("### ✅ Normal Message Features:")
            result.append("- 🏢 Clear company name and position")
            result.append("- 📍 Specific work location and details")
            result.append("- 💼 Reasonable salary range")
            result.append("- 📧 Official company contact information")
            result.append("- 📄 Proper application process")
            result.append(f"\n*Randomly showing {len(selected_examples)} normal examples*")
            
        else:  # both
            if not scam_samples and not legit_samples:
                return "⚠️ No examples found"
            
            result = ["## 📋 Fraud vs Normal Message Examples\n"]
            result.append("Here are some randomly selected examples for comparison:\n")
            
            if scam_samples:
                result.append("### 🛑 Fraud Examples:")
                for i, example in enumerate(scam_samples, 1):
                    result.append(f"{i}. **{example['source']}** - {example['label']}")
                    result.append(f"   💬 {example['message']}")
                    result.append("")
            
            if legit_samples:
                result.append("### ✅ Normal Examples:")
                for i, example in enumerate(legit_samples, 1):
                    result.append(f"{i}. **{example['source']}** - {example['label']}")
                    result.append(f"   💬 {example['message']}")
                    result.append("")
            
            result.append("---")
            result.append("### 🔍 How to Distinguish Fraud from Normal Messages:")
            result.append("**🚩 Red Flags for Fraud:**")
            result.append("- Unrealistically high pay for easy work")
            result.append("- Request for payment or bank details")
            result.append("- Contact only through messaging apps")
            result.append("- Vague details and urgency")
            result.append("")
            result.append("**✅ Normal Features:**")
            result.append("- Clear company and position information")
            result.append("- Reasonable salary and job requirements")
            result.append("- Formal interview process")
            result.append("- Public and transparent contact methods")
            result.append(f"\n*Randomly showing {len(scam_samples)} fraud and {len(legit_samples)} normal examples*")
    
    return "\n".join(result)

# ========== Ollama 功能 ==========
def generate_with_ollama(prompt, model=OLLAMA_MODEL, timeout=30):
    """使用 Ollama 生成回應"""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response", "").strip() or "⚠️ Ollama 沒回覆。"
        else:
            return f"⚠️ Ollama 錯誤: {resp.status_code}"
    except Exception as e:
        return f"⚠️ 無法連線到 Ollama：{e}"

# ========== 建立多意圖 prompt ==========
def build_prompt(label, explanation, lang, intent="judgment", user_message=""):
    """根據意圖和語言建立提示詞"""
    
    if intent == "judgment":
        
        if lang == "zh" and "❌ 詐騙招聘" in label:
            return (
                f"請分析以下招聘訊息是否為詐騙：\n\n"
                f"【原始訊息】\n{user_message}\n\n"
                f"【分類器分析結果】\n標籤: {label}\n說明: {explanation}\n\n"
                f"請根據以下重點進行分析：\n"
                f"1. 為什麼這個訊息被標記為「{label}」？\n"
                f"2. 具體指出原始訊息中的可疑之處\n"
                f"3. 分析詐騙手法的具體運作方式\n"
                f"4. 提供實用的防詐建議\n"
                f"請用繁體中文回答，語氣專業且具體。"
            )
        elif lang == "zh" and "✅ 真實招聘" in label:
            return (
                f"請分析以下招聘訊息是否為詐騙：\n\n"
                f"【原始訊息】\n{user_message}\n\n"
                f"【分類器分析結果】\n標籤: {label}\n說明: {explanation}\n\n"
                f"請根據以下重點進行分析：\n"
                f"1. 為什麼這個訊息被標記為「{label}」？\n"
                f"2. 具體指出原始訊息中的可信之處\n"
                f"3. 分析這類正常招聘的運作方式\n"
                f"4. 提供實用的求職建議\n"
                f"請用繁體中文回答，語氣專業且具體。"
            )
        elif lang == "en" and "❌ Fraudulent Job Offer" in label:
            return (
                f"Please analyze whether the following job message is a scam:\n\n"
                f"【Original Message】\n{user_message}\n\n"
                f"【Classifier Analysis】\nLabel: {label}\nExplanation: {explanation}\n\n"
                f"Please analyze based on the following points:\n"
                f"1. Why this message was classified as '{label}'\n"
                f"2. Specifically point out suspicious elements in the original message\n"
                f"3. Analyze how the scam operation works\n"
                f"4. Provide practical anti-scam advice\n"
                f"Please answer in English with professional and specific tone."
            )
        else:  # English + legit
            return (
                f"Please analyze whether the following job message is a scam:\n\n"
                f"【Original Message】\n{user_message}\n\n"
                f"【Classifier Analysis】\nLabel: {label}\nExplanation: {explanation}\n\n"
                f"Please analyze based on the following points:\n"
                f"1. Why this message was classified as '{label}'\n"
                f"2. Specifically point out credible elements in the original message\n"
                f"3. Analyze how this type of legitimate job operation works\n"
                f"4. Provide practical job-seeking advice\n"
                f"Please answer in English with professional and specific tone."
            )
    
    elif intent == "followup":
        if label and explanation:  # 有歷史分析結果
            if lang == "zh":
                return (
                    f"這是之前分析的詐騙訊息：\n\n"
                    f"【原始訊息】\n{user_message}\n\n"
                    f"【分析結果】\n標籤: {label}\n說明: {explanation}\n\n"
                    f"請詳細說明這種信息的詐騙手法：\n"
                    f"- 詐騙的具體步驟和流程\n"
                    f"- 常見的誘餌和話術\n" 
                    f"- 最終的詐騙目的和手段\n"
                    f"- 如何識破和防範\n\n"
                    f"請使用繁體中文，語氣具教學性。"
                )
            else:
                return (
                    f"This is the previously analyzed scam message:\n\n"
                    f"【Original Message】\n{user_message}\n\n"
                    f"【Analysis Result】\nLabel: {label}\nExplanation: {explanation}\n\n"
                    f"Please explain the typical scam mechanism step by step:\n"
                    f"- How the scammer operates\n"
                    f"- Common bait and tactics\n"
                    f"- Final scam objectives\n"
                    f"- How to recognize and prevent it"
                )
        else:  # 沒有歷史分析結果
            if lang == "zh":
                return (
                    f"使用者詢問：{user_message}\n\n"
                    f"請說明常見的詐騙手法，包括：\n"
                    f"- 詐騙的基本運作流程\n"
                    f"- 常見的詐騙類型\n"
                    f"- 識別詐騙的特徵\n"
                    f"- 防範建議\n\n"
                    f"請使用繁體中文，提供實用的資訊。"
                )
            else:
                return (
                    f"User asked: {user_message}\n\n"
                    f"Please explain common scam techniques, including:\n"
                    f"- Basic scam operation process\n"
                    f"- Common scam types\n"
                    f"- How to identify scams\n"
                    f"- Prevention advice"
                )
    
    elif intent == "knowledge":
        if lang == "zh":
            return (
                f"使用者詢問：{user_message}\n\n"
                f"請列出近期常見的詐騙手法，包括：\n"
                f"- 3-4 種最新或常見的詐騙類型\n"
                f"- 每種手法的特徵和話術\n"
                f"- 防範方法和注意事項\n"
                f"請使用繁體中文，提供最新且實用的資訊。"
            )
        else:
            return (
                f"User asked: {user_message}\n\n"
                f"Please list recent common scam techniques, including:\n"
                f"- 3-4 latest or common scam types\n"
                f"- Characteristics and tactics of each\n"
                f"- Prevention methods and precautions"
            )
    
    else:
        if lang == "zh":
            return f"請用繁體中文回答：{user_message}"
        else:
            return f"Please answer in English: {user_message}"

# ========== Predict + Ollama 整合 ==========
def analyze_and_summarize(message, source, intent, history=None, lang="zh", use_ollama=True):
    """分析訊息並生成摘要"""
    
    # 初始化變數
    label = ""
    explanation = ""
    analysis_text = ""
    
    # 步驟 1: 獲取分析結果（只有 judgment 意圖才使用分類器）
    if intent == "judgment":
        analysis_text, label, explanation = get_classifier_analysis(message, source, lang, history)
        # for debug
        # print("Classifier Analysis Text:", analysis_text)
        # print("Classifier Label:", label)
        # print("Classifier Explanation:", explanation)
    elif intent == "followup" and history and history.get("label"):
        # 對於 followup 意圖，使用歷史記錄
        label = history.get("label", "")
        explanation = history.get("explanation", "")
        analysis_text = history.get("analysis_text", "")

    # 步驟 2: 如果不使用 Ollama，直接返回
    if not use_ollama:
        return handle_non_ollama_response(intent, analysis_text, lang)

    # 步驟 3: 生成 Ollama 回應
    reply = generate_ollama_response(intent, message, label, explanation, analysis_text, history, lang)
    
    return analysis_text, reply


def handle_non_ollama_response(intent, analysis_text, lang):
    """處理不使用 Ollama 的回應"""
    if intent == "judgment":
        return analysis_text, analysis_text  # 返回分析結果作為回應
    elif intent == "followup":
        if lang == "zh":
            reply = (
                "⚠️ 抱歉，目前不支援進一步的詐騙手法說明功能。"
                "請啟用 Ollama 功能以獲取詳細解釋。"
            )
        else:
            reply = (
                "⚠️ Sorry, further scam mechanism explanation is not supported at the moment. "
                "Please enable Ollama for detailed explanations."
            )
        return analysis_text, reply
    elif intent == "knowledge":
        if lang == "zh":
            reply = (
                "⚠️ 抱歉，目前不支援近期詐騙手法介紹功能。"
                "請啟用 Ollama 功能以獲取最新資訊。"
            )
        else:
            reply = (
                "⚠️ Sorry, recent scam techniques introduction is not supported at the moment. "
                "Please enable Ollama for the latest information."
            )
        return analysis_text, reply
    else:
        return analysis_text, analysis_text  # 其他意圖直接返回分析結果

def get_classifier_analysis(message, source, lang, history):
    """獲取分類器分析結果"""
    try:
        lang_code = "zh" if lang == "zh" else "en"

        analysis = predict_message(message, source,lang_code)
        
        # 確保 analysis 是字典格式
        if isinstance(analysis, dict):
            label = analysis.get("label", "")
            explanation = analysis.get("explanation", "")
            analysis_text = (
                f"📋 {'Label' if lang == 'en' else '標籤'}: {label}\n"
                f"🔍 {'Explanation' if lang == 'en' else '說明'}: {explanation}"
            )
            
            # Debug 資訊
            # print("===== DEBUG =====")
            # print("Raw Analysis:", analysis)
            # print("Label:", label)
            # print("Explanation:", explanation)
            # print("================")
            
            # 更新歷史記錄
            if history is not None:
                history.clear()
                history.update({
                    "label": label,
                    "explanation": explanation,
                    "analysis_text": analysis_text,
                    "original_message": message
                })
            
            return analysis_text, label, explanation
        
        # 如果返回的不是字典，嘗試轉換或處理
        elif isinstance(analysis, str):
            # 如果還是字串格式，作為標籤處理
            label = analysis
            explanation = ""
            analysis_text = f"📋 {'Label' if lang == 'en' else '標籤'}: {label}"
            
            if history is not None:
                history.clear()
                history.update({
                    "label": label,
                    "explanation": explanation,
                    "analysis_text": analysis_text,
                    "original_message": message
                })
            
            return analysis_text, label, explanation
        
        else:
            # 其他未知格式
            error_msg = f"⚠️ 未知的分析結果格式: {type(analysis)}"
            return error_msg, "", ""
            
    except Exception as e:
        error_msg = f"⚠️ 分析錯誤: {e}" if lang == "zh" else f"⚠️ Analysis error: {e}"
        return error_msg, "", ""



def generate_ollama_response(intent, message, label, explanation, analysis_text, history, lang):
    """生成 Ollama 回應"""
    if intent == "examples":
        return get_examples_output("both", count=3, lang=lang)
    
    elif intent == "judgment":
        if analysis_text and "⚠️" not in analysis_text:
            prompt = build_prompt(label, explanation, lang, intent, user_message=message)
            reply = generate_with_ollama(prompt)
        else:
            reply = analysis_text
        return reply
    
    else:
        prompt = build_prompt(label, explanation, lang, intent, user_message=message)
        reply = generate_with_ollama(prompt)
        
        if intent == "followup" and analysis_text:
            prefix = "📋 參考分析: {label}\n\n" if lang == "zh" else "📋 Reference: {label}\n\n"
            reply = prefix.format(label=history.get('label', '')) + reply
        
        return reply

# ========== Intent 分類 ==========
def detect_intent(text, lang="zh"):
    """偵測使用者意圖"""
    if not text or not text.strip():
        return "casual"
        
    txt = text.lower().strip()
    
    # 中文意圖偵測
    if lang == "zh":
        if re.search(r"(詐騙|騙人|假的|真的嗎|是不是詐|判斷|分析|檢查|安全嗎|幫我看)", txt):
            return "judgment"
        if re.search(r"(怎麼騙|為什麼|原因|手法|流程|方式|how.*work|運作|操作)", txt):
            return "followup"
        if re.search(r"(最近|種類|類型|案例|例子|what is|types of|recent|常見|有什麼)", txt):
            return "knowledge"
        if re.search(r"(例子|範例|示例|example|sample|實例|案例)", txt):
            return "examples"
        if re.search(r"(你好|hi|hello|哈囉|嗨|hey|您好)", txt):
            return "greeting"
    else:
        # 英文意圖偵測
        if re.search(r"(scam|fraud|fake|real|check|verify|judgment|analysis|is this|safe)", txt):
            return "judgment"
        if re.search(r"(how.*work|why|method|technique|process|operation)", txt):
            return "followup"
        if re.search(r"(recent|types?|examples?|cases?|common|what.*are)", txt):
            return "knowledge"
        if re.search(r"(example|sample|instance|case)", txt):
            return "examples"
        if re.search(r"(hi|hello|hey|greetings)", txt):
            return "greeting"
    
    return "casual"

# ========== 快速回應處理 ==========
TEST_CASES = {
    "zh": {
        "job_scam": ("高薪兼職💰 日薪$1800，加 WhatsApp 領任務", "判斷是否為詐騙"),
        "task_scam": ("兼職刷單，每單返$1000，需先付押金", "判斷是否為詐騙"),
        "investment_scam": ("投資平台，保證日收益5%，穩賺不賠", "判斷是否為詐騙"),
        "how_it_works": ("這種詐騙是怎麼運作的？", "詐騙手法 / 為何"),
        "recent_scams": ("最近有什麼常見的詐騙手法？", "最近的詐騙手法"),
        "examples": ("給我一些詐騙和正常訊息的例子", "例子 / 範例")
    },
    "en": {
        "job_scam": ("High-paying job $200/day, contact WhatsApp for tasks", "Check if it's a scam"),
        "task_scam": ("Earn $1000 per task, pay deposit first", "Check if it's a scam"),
        "investment_scam": ("Investment platform 5% daily return guaranteed", "Check if it's a scam"),
        "how_it_works": ("How does this scam work?", "How the scam works"),
        "recent_scams": ("What are some recent scam types?", "Recent scam types"),
        "examples": ("Give me some examples of scam and legitimate messages", "Examples / Samples")
    }
}

def quick_respond(test_type, chat_history, source, last_history, use_ollama, lang):
    """快速測試按鈕回覆（帶即時提示）"""
    lang_code = "zh" if lang == "zh" else "en"
    
    # 判斷語言提示文字
    thinking_msg = "🤔 正在分析訊息，請稍候..." if lang == "zh" else "🤔 Analyzing the message, please wait..."
    
    # 如果測試類型不存在
    if test_type not in TEST_CASES[lang_code]:
        warn_msg = "⚠️ 找不到測試案例。" if lang == "zh" else "⚠️ Test case not found."
        chat_history.append(("N/A", warn_msg))
        yield chat_history, last_history, ""
        return

    # 取出測試訊息與模式
    message, mode = TEST_CASES[lang_code][test_type]

    # 先顯示思考中
    chat_history.append((message, thinking_msg))
    yield chat_history, last_history, message  # 即時更新畫面

    # 執行主要邏輯
    reply, new_last_history = chatbot_response(
        message, chat_history, source, mode, last_history, use_ollama, lang
    )

    # 更新為真實回覆
    chat_history[-1] = (message, reply)
    yield chat_history, new_last_history, ""

# ========== 回覆邏輯 ==========
def chatbot_response(message, chat_history, source, selected_option, last_history, use_ollama, lang):
    """處理聊天機器人回應"""
    if not message or not message.strip():
        empty_msg = "請輸入訊息" if lang == "zh" else "Please enter a message"
        return empty_msg, last_history

    # 轉換語言代碼
    lang_code = "zh" if lang == "zh" else "en"
    
    # 判斷意圖
    if selected_option in ["自動判斷 (推薦)", "Auto detect"]:
        intent = detect_intent(message, lang_code)
    else:
        mapping = {
            "判斷是否為詐騙": "judgment",
            "詐騙手法 / 為何": "followup", 
            "最近的詐騙手法": "knowledge",
            "例子 / 範例": "examples",
            "打招呼 / 你好": "greeting",
            "其他 / 我想聊": "casual",
            "Check if it's a scam": "judgment",
            "How the scam works": "followup",
            "Recent scam types": "knowledge",
            "Examples / Samples": "examples",
            "Greeting / Hello": "greeting",
            "Other / Chat": "casual",
        }
        intent = mapping.get(selected_option, "casual")
    # for debug
    # print(f"Detected intent: {intent}")
    # 處理不同意圖
    if intent in ["judgment", "followup", "knowledge", "examples"]:
        analysis_text, reply = analyze_and_summarize(
            message, source, intent=intent, history=last_history, lang=lang_code, use_ollama=use_ollama
        )
        
        # 建構回應
        if intent == "judgment":
            if use_ollama and reply and "⚠️" not in reply:
                if lang_code == "zh":
                    combined_reply = f"{analysis_text}\n\n---\n💬 Ollama 回覆:\n{reply}"
                else:
                    combined_reply = f"{analysis_text}\n\n---\n💬 Ollama Summary:\n{reply}"
            else:
                combined_reply = analysis_text
        else:
            # 對於其他意圖，直接顯示回應
            combined_reply = reply
                
        return combined_reply, last_history

    elif intent == "greeting":
        if lang_code == "zh":
            reply = "👋 你好！我是反詐騙助理，請貼上訊息讓我幫你判斷是否為詐騙，或選擇其他功能。"
        else:
            reply = "👋 Hello! I'm an anti-scam assistant. Paste a message and I'll check if it's a scam, or choose other functions."
        return reply, last_history

    else:
        if lang_code == "zh":
            reply = "👀 想讓我幫你看看訊息是不是詐騙嗎？請貼上內容或選擇判斷功能。"
        else:
            reply = "👀 Want me to check if something is a scam? Just paste the message or select the judgment function."
        return reply, last_history

# ========== UI ==========
def build_ui(lang="zh"):
    """建立使用者介面"""
    is_zh = lang == "zh"
    
    title = "反詐騙智能助理 - FFless" if is_zh else "🧠 Anti-Scam Assistant - FFless"
    placeholder = "貼上招聘內容或輸入問題，如：這是不是詐騙？" if is_zh else "Paste a message or ask: Is this a scam?"
    send_label = "送出" if is_zh else "Send"
    clear_label = "清除對話" if is_zh else "Clear Chat"
    quick_title = "⚡ 快速測試" if is_zh else "⚡ Quick Actions"
    use_ollama_label = "使用 Ollama 生成自然回答" if is_zh else "Use Ollama for natural answers"

    with gr.Column(visible=(lang=="zh")) as ui:
        gr.Markdown(f"## {title}")

        chatbot = gr.Chatbot(
            label="💬 Chat", 
            height=450,
            show_copy_button=True
        )

        msg = gr.Textbox(
            label="輸入訊息" if is_zh else "Message", 
            placeholder=placeholder, 
            lines=2,
            max_lines=4
        )

        # 👇 將選項移到輸入框下方
        with gr.Row():
            source = gr.Dropdown(
                label="訊息來源" if is_zh else "Source",
                choices=["IG", "Thread", "WhatsApp", "Facebook", "Moovup", "JobsDB", "Indeed", "PolyU","Labour Department","Other"],
                value="IG",
                scale=1
            )
            option = gr.Dropdown(
                label="模式" if is_zh else "Mode",
                choices=(
                    ["自動判斷 (推薦)", "判斷是否為詐騙", "詐騙手法 / 為何", "最近的詐騙手法", "例子 / 範例", "打招呼 / 你好", "其他 / 我想聊"]
                    if is_zh else 
                    ["Auto detect", "Check if it's a scam", "How the scam works", "Recent scam types", "Examples / Samples", "Greeting / Hello", "Other / Chat"]
                ),
                value="自動判斷 (推薦)" if is_zh else "Auto detect",
                scale=1
            )

        use_ollama = gr.Checkbox(label=use_ollama_label, value=True)

        with gr.Row():
            send = gr.Button(send_label, variant="primary", scale=2)
            clear = gr.Button(clear_label, variant="secondary", scale=1)

        gr.Markdown(f"### {quick_title}")

        # 快速測試按鈕
        with gr.Row():
            test1 = gr.Button("🔍 高薪詐騙" if is_zh else "🔍 High Salary Scam", variant="secondary", size="sm", scale=1)
            test2 = gr.Button("💼 刷單任務詐騙" if is_zh else "💼 Task Scam", variant="secondary", size="sm", scale=1)
            test3 = gr.Button("📈 投資陷阱" if is_zh else "💸 Investment Scam", variant="secondary", size="sm", scale=1)
        with gr.Row():
            test4 = gr.Button("❓ 手法解析" if is_zh else "🔍 How Scam Works?", variant="secondary", size="sm", scale=1)
            test5 = gr.Button("📚 近期詐騙趨勢" if is_zh else "📊 Recent Trends", variant="secondary", size="sm", scale=1)
            test6 = gr.Button("📋 例子對比" if is_zh else "📋 Examples", variant="secondary", size="sm", scale=1)
        with gr.Row():
            test7 = gr.Button("🧹 清除" if is_zh else "🧹 Clear", variant="stop", size="sm", scale=1)

        last_history = gr.State({})

        def respond(message, chat_history, src, selected, last_history_state, use_ollama_flag):
            # 即時顯示「正在思考...」
            thinking_msg = "🤔 正在分析訊息，請稍候..." if is_zh else "🤔 Analyzing the message, please wait..."
            chat_history.append((message, thinking_msg))
            yield "", chat_history, last_history_state
            
            # 模擬延遲後（真正回覆）
            reply, new_last_history = chatbot_response(
                message, chat_history, src, selected, last_history_state, use_ollama_flag, lang
            )
            
            # 更新為真實回覆
            chat_history[-1] = (message, reply)
            yield "", chat_history, new_last_history

        send.click(
            respond,
            inputs=[msg, chatbot, source, option, last_history, use_ollama],
            outputs=[msg, chatbot, last_history]
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot, source, option, last_history, use_ollama],
            outputs=[msg, chatbot, last_history]
        )
        clear.click(lambda: ([], {}, ""), outputs=[chatbot, last_history, msg])

        test_types = ["job_scam", "task_scam", "investment_scam", "how_it_works", "recent_scams", "examples"]
        test_buttons = [test1, test2, test3, test4, test5, test6]
        for test_btn, test_type in zip(test_buttons, test_types):
            test_btn.click(
                quick_respond,
                inputs=[gr.State(test_type), chatbot, source, last_history, use_ollama, gr.State(lang)],
                outputs=[chatbot, last_history, msg]
            )
        test7.click(lambda: ([], {}, ""), outputs=[chatbot, last_history, msg])

    return ui

# ========== 主介面 ==========
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        font=[gr.themes.GoogleFont("Noto Sans TC"), "Arial", "sans-serif"]
    ),
    title="反詐騙智能助理",
    css="""
    /* 手機響應式設計 */
    @media (max-width: 768px) {
        .gradio-container {
            max-width: 100% !important;
            padding: 10px !important;
        }
        .gradio-row {
            flex-wrap: wrap !important;
        }
        .gradio-button {
            min-width: 60px !important;
            font-size: 12px !important;
        }
        .gradio-dropdown, .gradio-textbox {
            font-size: 14px !important;
        }
    }
    .gradio-chatbot {
        height: calc(100vh - 380px) !important;
        max-height: 600px !important;
        overflow-y: auto !important;
    }
    """
) as demo:
    
    # 響應式標題
    gr.Markdown("""
    <div style="text-align: center;">
        <h3 style="margin-bottom: 5px;">🌍 反詐騙智能助理 - FFless</h1>
        <p style="margin-top: 0; color: #666; font-size: 14px;">Anti-Fraud Chatbot</p>
    </div>
    """)
    
    # 語言選擇 - 手機優化
    with gr.Row():
        lang_radio = gr.Radio(
            ["繁體中文", "English"],
            value="繁體中文", 
            label="選擇語言 / Choose Language",
            interactive=True,
            elem_id="language_radio"
        )
    
    ui_zh = build_ui("zh")
    ui_en = build_ui("en")

    def toggle_language(lang):
        """切換語言"""
        if lang == "繁體中文":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    lang_radio.change(toggle_language, inputs=lang_radio, outputs=[ui_zh, ui_en])

if __name__ == "__main__":
    # 啟動時顯示手機訪問訊息
    print("=" * 50)
    print("🚀 反詐騙智能助理已啟動！")
    print("📱 請在手機瀏覽器訪問以下網址：")
    print("   http://你的IP位址:7860")
    print("🌐 或分享此網址給其他人：")
    print("   (啟動後會顯示公開網址)")
    print("=" * 50)
    
    demo.launch(
        share=True, 
        server_name="0.0.0.0", 
        server_port=7860,
        inbrowser=False  # 不在電腦自動打開瀏覽器
    )