# Anti-Fraud Chatbot – System Architecture (English)

```mermaid
graph TD
    %% ==== User ====
    A[User\n(Browser / Mobile)] -->|HTTP / WebSocket| B[Gradio Web UI]

    %% ==== Presentation Layer ====
    subgraph UI[Presentation Layer – Gradio UI]
        B --> C1[Chat Interface\n(Chatbot)]
        B --> C2[Message Input + Source Dropdown]
        B --> C3[Language Switch\n(Chinese / English)]
        B --> C4[Mode Dropdown\n(Auto / Judgment / How / Examples…)]
        B --> C5[Quick-Test Buttons\n(High-Salary / Task / Investment…)]
        B --> C6[Ollama Toggle]
    end

    %% ==== Application Logic Layer ====
    subgraph Logic[Application Logic – Python Backend]
        D1[Intent Detection\ndetect_intent()] --> D2{Intent?}
        D2 -->|judgment| D3[Fraud Classifier\npredict_message()]
        D2 -->|followup / knowledge| D4[Prompt Builder\nbuild_prompt()]
        D2 -->|examples| D5[Load Examples\nload_dataset_examples()]
        D2 -->|greeting / casual| D6[Static Response]

        D3 --> D7[Result\n(label, explanation)]
        D7 --> D8[History State\nlast_history]
    end

    %% ==== Services & Models Layer ====
    subgraph Services[Services & Models]
        E1[Fraud Classifier\npredict.py] --> D3
        E2[Sentence Transformer\nparaphrase-multilingual-MiniLM] -.->|not in core flow| D1
        E3[Ollama LLM\nllama3.1:8b\nhttp://localhost:11434] --> F[POST /api/generate]
        F --> D4
        F --> D7
    end

    %% ==== Data Layer ====
    subgraph Data[Data Layer]
        G1[dataset.csv\n(Message, Label, Source)] --> D5
        G2[Runtime State\nlast_history dict] --> D8
    end

    %% ==== Response Flow ====
    D3 & D4 & D5 & D6 --> H[Response Builder\n(Classifier + Ollama Summary)]
    H --> I[Update Chat\nGradio Chatbot]
    I --> B

    %% ==== Quick-Test Flow ====
    C5 -->|click| J[quick_respond()]
    J --> D1

    %% ==== Error Path ====
    E3 -->|fail| K[Warning\nFallback Response]
    K --> H

    %% ==== Styling ====
    classDef ui fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000;
    classDef logic fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000;
    classDef service fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef data fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000;

    class B,C1,C2,C3,C4,C5,C6 ui
    class D1,D2,D3,D4,D5,D6,D7,D8,J logic
    class E1,E2,E3,F service
    class G1,G2 data
    class K error