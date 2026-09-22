# 🛡️ AI Anti-Fraud Chatbot

An AI-powered chatbot for detecting and explaining potential fraud messages.

The project combines a BERT-based text classification model, rule-based risk analysis, and an Ollama-powered LLM to provide users with fraud detection results and human-readable explanations.

---

## Overview

Online scams often contain suspicious wording, requests, or patterns that may be difficult for users to recognize.

This project provides a conversational interface that analyzes user input and produces a fraud risk assessment.

The system combines:

- BERT-based text classification
- Rule-based fraud keyword detection
- Risk score analysis
- Ollama LLM
- Conversational interaction
- Chinese and English explanations

The goal is to demonstrate how traditional machine learning, rule-based analysis, and LLMs can be combined into a practical AI application.

---

## System Architecture


```mermaid
flowchart TD

User[User enters hiring job message] --> WebUI[Web UI]

WebUI --> Predict[Scam Detection AI]

subgraph Detection Engine
    Predict --> BERT[BERT reads text]
    Predict --> SourceRisk["Platform Risk Score<br>(IG=0.9, WhatsApp=0.8"]
    Predict --> Keywords["Scam Keyword Detector<br>(e.g. money + task + bait keywords)"]
end

BERT --> Result[Prediction Result]
SourceRisk --> Result
Keywords --> Result
Result --> |return result| LLM["LLM (Ollama) use the result to generate and analysis explanation (EN / CN)"]


LLM --> |return response| WebUI

```

```mermaid
graph TD
    A[**📥 User Input**] --> B{**Intent Detection**}
    
    B -->|Scam Judgment| C[**🔍 Classifier Analysis**]
    B -->|Other Intents| D[**🧠 Direct Processing**]
    
    C --> E{**Ollama Analysis**}
    E -->|Enabled| F[**🤖 AI Explanation**]
    E -->|Disabled| G[**📋 Basic Results**]
    
    D --> H[**📚 Database Query<br>or AI Response**]
    
    F --> I[**💬 Final Response**]
    G --> I
    H --> I
    
    I --> J[**📱 Display Results**]
    J --> K[**💾 Update History**]
    K --> A

    style A fill:#e1f5fe
    style I fill:#c8e6c9
    style C fill:#fff9c4
    style F fill:#fce4ec
```
```mermaid
graph TB
    subgraph **Frontend**
        A[**📱 Web UI**<br/>Gradio Interface]
        A --> B[**🌐 HTTP Requests**]
    end
    
    subgraph **Backend**
        B --> C[**Python Server**<br/>Gradio App]
        C --> D[**Request Router**]
        D --> F[**💬 Chat Manager**]
        F --> G[**📊 Data Processor**]
    end
    
    subgraph **ML Core**
        G --> H[**🧠 Fraud Classifier**<br/>PyTorch Model]
        G --> I[**🤖 Ollama Client**<br/>LLM Interface]
        G --> J[**📁 Dataset Manager**<br/>CSV Handler]
    end
    
    subgraph **External Services**
        H --> K[**🔢 Custom Model**<br/>fraud_model.pt]
        I --> L[**🔄 Ollama API**]
        J --> M[**💾 Local Dataset**<br/>dataset.csv]
    end
    
    subgraph **Data Flow**
        N[**📥 User Input**] --> A
        A -->|**Real-time UI Update**| O[**📤 Response Output**]
        H -->|**Prediction Results**| G
        I -->|**AI Explanations**| G
        J -->|**Example Data**| G
        G -->|**Formatted Response**| F
        F -->|**Final Output**| C
        C -->|**HTTP Response**| A
    end
```
