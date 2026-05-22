# 📅 AI Appointment Booking Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/LangGraph-7C3AED?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

**An intelligent AI-powered appointment booking system with user authentication, RAG knowledge base, and calendar view.**

[Features](#-features) • [Tech Stack](#-tech-stack) • [Installation](#-installation) • [Usage](#-usage) • [Screenshots](#-screenshots)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔐 **User Authentication** | Secure login/register with bcrypt password hashing |
| 💬 **Conversational AI** | Natural language appointment booking |
| 🧠 **Intent Detection** | Auto-detects book/cancel/reschedule/view intents |
| 🔄 **LangGraph Workflows** | Multi-step agent decision making |
| 📚 **RAG Knowledge Base** | Upload documents for smarter AI responses |
| 🗄️ **SQLite Database** | Persistent per-user appointment storage |
| 📅 **Calendar View** | Visual calendar showing scheduled appointments |
| 🔍 **Search** | Filter and find appointments instantly |
| 📄 **PDF Export** | Download appointments as professional PDF |
| 📱 **Mobile Friendly** | Responsive design for all devices |
| 🎨 **Beautiful UI** | Purple dark theme with bubble chat |
| 🔐 **Secure** | API keys protected with .env file |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Web UI Framework |
| **LangChain** | AI Framework |
| **LangGraph** | Agent Workflow Orchestration |
| **LLaMA 3.3 70B** | Large Language Model via Groq |
| **Groq** | Fast LLM Inference |
| **FAISS** | Vector Store for RAG |
| **HuggingFace Embeddings** | Document Embeddings |
| **SQLite** | Relational Database |
| **bcrypt** | Password Hashing |
| **ReportLab** | PDF Generation |
| **streamlit-calendar** | Calendar Component |

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Groq API Key — free at [console.groq.com](https://console.groq.com)

### Step 1 — Clone the repository
```bash
git clone https://github.com/Sabeen-Ali/ai-appointment-agent.git
cd ai-appointment-agent
```

### Step 2 — Install dependencies
```bash
pip install streamlit langchain langchain-groq langchain-core langgraph
pip install langchain-community langchain-text-splitters
pip install faiss-cpu sentence-transformers pypdf
pip install python-dotenv reportlab bcrypt streamlit-calendar
```

### Step 3 — Set up environment variables
Create a `.env` file in the root directory:
GROQ_API_KEY=your-groq-api-key-here
### Step 4 — Run the application
```bash
streamlit run app.py
```

### Step 5 — Open browser
Navigate to `http://localhost:8501`

---

## 💡 Usage

### 1. Register & Login
- Create an account with username, email and password
- Each user has their own private appointments

### 2. Book an Appointment
Type naturally — the AI understands:
- *"Book a doctor appointment on Monday at 3pm"*
- *"Schedule a dentist visit for Friday morning"*
- *"I need a consultation on May 27"*

### 3. AI Agent Flow
User Input
↓
Intent Detection (book/cancel/reschedule/view/rag_query)
↓
RAG Context Retrieval (if needed)
↓
LLM Response Generation
↓
Save to Database (if confirmed)
### 4. Knowledge Base
Upload PDF or TXT files about your clinic:
- Doctor availability
- Services offered
- Pricing information
- Opening hours

### 5. View Appointments
- 📅 **Calendar View** — see appointments on correct dates
- 📋 **List View** — search, filter, and manage appointments
- 📄 **Export PDF** — download all appointments

---

## 🎯 Intent Detection

| Intent | Trigger Words | Example |
|--------|--------------|---------|
| 📅 Book | book, schedule, reserve | "Book a dentist appointment" |
| ❌ Cancel | cancel, delete, remove | "Cancel my appointment" |
| 🔄 Reschedule | reschedule, change, move | "Move my appointment to Friday" |
| 👁️ View | show, list, view | "Show my appointments" |
| 📚 RAG Query | doctor, hours, price, service | "What are the clinic hours?" |

---

## 📁 Project Structure
ai-appointment-agent/
├── app.py                  # Main application
├── appointments.db         # SQLite database (auto-created)
├── vector_store/           # FAISS vector store (auto-created)
├── .env                    # API keys (not in repo)
├── .gitignore              # Git ignore rules
└── README.md               # Documentation
---

## 🔐 Security

- Passwords hashed with **bcrypt** — never stored in plain text
- API keys stored in `.env` file — excluded from version control
- Each user can only see their own appointments
- No sensitive data pushed to GitHub

---

## 🌟 Roadmap

- [ ] Email confirmation notifications
- [ ] Google Calendar integration
- [ ] Voice input support
- [ ] Analytics dashboard
- [ ] Multi-language support (Urdu, Arabic)
- [ ] Cloud deployment

---

## 👩‍💻 Author

**Sabeen Ali**

[![GitHub](https://img.shields.io/badge/GitHub-Sabeen--Ali-181717?style=flat&logo=github)](https://github.com/Sabeen-Ali)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
Built with ❤️ using Python and AI
<br>
⭐ Star this repo if you found it helpful!
</div>
