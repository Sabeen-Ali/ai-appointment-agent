# 📅 AI Appointment Booking Agent

An AI-powered appointment management system that allows users to **book, cancel, reschedule, and view appointments using natural-language conversation**.

The application combines **LangChain, LangGraph, RAG, LLaMA 3.3, Groq, FAISS, SQLite, and Streamlit** to create an intelligent appointment assistant with authentication, a knowledge base, calendar management, search, and PDF export.

---

## 🚀 Demo

> **Live Demo:** Coming soon

> **Demo Video:** Coming soon

---

## ✨ Key Features

### 🤖 Conversational Appointment Management

Interact with the application using natural language instead of traditional forms.

Examples:

```text
"Book a dentist appointment on Monday at 3 PM"

"Cancel my appointment"

"Move my appointment to Friday"

"Show my appointments"
```

The AI identifies the user's intention and processes the appropriate workflow.

---

### 🧠 Intelligent Intent Detection

The system identifies different types of user requests:

| Intent             | Example                              |
| ------------------ | ------------------------------------ |
| 📅 Book            | "Book a dentist appointment"         |
| ❌ Cancel           | "Cancel my appointment"              |
| 🔄 Reschedule      | "Move my appointment to Friday"      |
| 👁️ View           | "Show my appointments"               |
| 📚 Knowledge Query | "What are the clinic opening hours?" |

---

### 🔄 LangGraph Agent Workflow

The application uses **LangGraph** to organize the appointment assistant into a multi-step workflow.

```text
                    User
                      │
                      ▼
             ┌─────────────────┐
             │  User Message   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Intent Detection│
             └────────┬────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
      Booking       Cancel       Reschedule
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  RAG Retrieval  │
             │   if required   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ LLM Processing  │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ User Confirmation│
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ SQLite Database │
             └─────────────────┘
```

---

## 📚 RAG Knowledge Base

The application includes a **Retrieval-Augmented Generation (RAG)** knowledge base.

Users can upload documents containing information such as:

* Doctor availability
* Services offered
* Pricing
* Clinic information
* Opening hours
* Other appointment-related information

The system uses:

**Documents → Text Processing → Embeddings → FAISS → Retrieval → LLM Response**

This allows the assistant to answer knowledge-based questions using the uploaded information.

---

## 🔐 User Authentication

The application provides user registration and login functionality.

Security features include:

* User registration
* Login authentication
* Password hashing using **bcrypt**
* Per-user appointment data
* API key protection through environment variables

Passwords are not stored as plain text.

---

## 🗄️ Appointment Database

Appointment information is persistently stored using **SQLite**.

Each authenticated user has their own appointment records.

The system supports appointment management including:

* Creating appointments
* Viewing appointments
* Cancelling appointments
* Rescheduling appointments
* Searching appointments

---

## 📅 Calendar & List Views

Appointments can be viewed through:

### Calendar View

Provides a visual representation of scheduled appointments.

### List View

Allows users to:

* Search appointments
* Filter appointments
* Review appointment details
* Manage existing appointments

---

## 📄 PDF Export

Users can export appointment information as a professionally formatted PDF.

The PDF generation functionality is implemented using **ReportLab**.

---

## 🛠️ Technology Stack

| Technology                  | Purpose                        |
| --------------------------- | ------------------------------ |
| **Python**                  | Core programming language      |
| **Streamlit**               | Web application interface      |
| **LangChain**               | LLM application framework      |
| **LangGraph**               | Agent workflow orchestration   |
| **LLaMA 3.3 70B**           | Large Language Model           |
| **Groq**                    | Fast LLM inference             |
| **FAISS**                   | Vector database for RAG        |
| **Hugging Face Embeddings** | Document embeddings            |
| **SQLite**                  | Persistent appointment storage |
| **bcrypt**                  | Password hashing               |
| **ReportLab**               | PDF generation                 |
| **streamlit-calendar**      | Calendar interface             |

---

## 📂 Project Structure

```text
ai-appointment-agent/
│
├── app.py
├── appointments.db
├── vector_store/
├── .env
├── .gitignore
└── README.md
```

### Main Components

**`app.py`**
Main Streamlit application containing the user interface and application logic.

**`appointments.db`**
SQLite database used for persistent appointment storage.

**`vector_store/`**
FAISS vector store generated for the RAG knowledge base.

**`.env`**
Stores sensitive configuration such as API keys.

**`.gitignore`**
Prevents sensitive and generated files from being committed.

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Sabeen-Ali/ai-appointment-agent.git
```

Navigate to the project directory:

```bash
cd ai-appointment-agent
```

---

## 2. Install Dependencies

Make sure you have **Python 3.10 or later** installed.

Install the required packages:

```bash
pip install streamlit langchain langchain-groq langchain-core langgraph
```

```bash
pip install langchain-community langchain-text-splitters
```

```bash
pip install faiss-cpu sentence-transformers pypdf
```

```bash
pip install python-dotenv reportlab bcrypt streamlit-calendar
```

---

## 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-api-key
```

> ⚠️ Never commit your actual API key to GitHub.

---

## 4. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 💡 How to Use

## Step 1 — Register

Create an account using:

* Username
* Email
* Password

---

## Step 2 — Login

Sign in using your registered credentials.

---

## Step 3 — Interact with the AI Assistant

Use natural language to manage appointments.

### Book

```text
Book a doctor appointment on Monday at 3 PM.
```

### Cancel

```text
Cancel my appointment.
```

### Reschedule

```text
Move my appointment to Friday.
```

### View

```text
Show my appointments.
```

### Ask a Knowledge Question

```text
What are the clinic opening hours?
```

---

# 🧠 AI Processing Flow

The application processes user requests through an intelligent workflow:

```text
User Input
    ↓
Intent Detection
    ↓
Identify Request Type
    ↓
┌──────────────┬──────────────┬──────────────┐
│    Booking   │    Cancel    │  Reschedule  │
└──────────────┴──────────────┴──────────────┘
    ↓
RAG Retrieval (when required)
    ↓
LLM Response Generation
    ↓
User Confirmation
    ↓
Database Update
```

---

# 🔒 Security

The project includes several security considerations:

* Passwords are hashed using **bcrypt**
* API keys are stored using environment variables
* `.env` is excluded from version control
* Appointment data is associated with individual users
* Sensitive credentials should not be committed to the repository

> **Production Note:** Additional security hardening would be required before deploying this application for real-world production use.

---

# 📸 Screenshots

Add screenshots of the following application flows here:

### 🔐 Login / Registration

```text
[Add screenshot here]
```

### 💬 AI Appointment Conversation

```text
[Add screenshot here]
```

### 📅 Calendar View

```text
[Add screenshot here]
```

### 📋 Appointment List

```text
[Add screenshot here]
```

### 📄 PDF Export

```text
[Add screenshot here]
```

---

# 🎯 Use Cases

This project can serve as a foundation for appointment-based businesses such as:

* 🏥 Clinics
* 🦷 Dental practices
* 💇 Salons
* 🧑‍⚕️ Healthcare consultants
* 💼 Professional consultants
* 🏢 Service-based businesses

The same architecture can be adapted for different appointment and customer-support workflows.

---

# 🛣️ Future Improvements

Possible future enhancements include:

* 📧 Email appointment confirmations
* 📅 Google Calendar integration
* 🎙️ Voice input
* 📊 Analytics dashboard
* 🌐 Multi-language support
* ☁️ Cloud deployment
* 🔔 Appointment reminders
* 👥 Admin dashboard
* 🔑 Role-based access control

---

# 📌 What I Learned

This project provided practical experience with:

* Building AI-powered applications with Python
* Designing agent workflows using LangGraph
* Integrating LLMs into applications
* Implementing Retrieval-Augmented Generation
* Working with vector databases
* Managing persistent application data with SQLite
* Implementing authentication and password hashing
* Building interactive interfaces with Streamlit
* Connecting multiple AI and software components into a complete application

---

# 👩‍💻 Author

**Sabeen Ali**

BS Information Technology Student
AI Agent & Automation Developer

### Areas of Interest

* 🤖 Agentic AI
* 🧠 Large Language Models
* 🔎 Retrieval-Augmented Generation
* 🔄 AI Workflow Automation
* 🎙️ Voice AI
* 🐍 Python
* 🌐 AI-powered Applications

---

# 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

### Built with ❤️ using Python, LangChain, LangGraph, Groq, RAG and Streamlit.
