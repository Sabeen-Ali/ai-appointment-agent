import streamlit as st
import json
import os
import sqlite3
import io
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import operator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# ── SQLite Database ──────────────────────────────────────────
DB_FILE = "appointments.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request TEXT,
            response TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def load_appointments(search_query=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if search_query:
        c.execute("SELECT * FROM appointments WHERE request LIKE ? OR response LIKE ?",
                 (f"%{search_query}%", f"%{search_query}%"))
    else:
        c.execute("SELECT * FROM appointments ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "request": r[1], "response": r[2], "timestamp": r[3]} for r in rows]

def save_appointment(user_message, ai_response):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO appointments (request, response, timestamp) VALUES (?, ?, ?)",
             (user_message, ai_response, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def delete_appointment(apt_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM appointments WHERE id = ?", (apt_id,))
    conn.commit()
    conn.close()

def get_total_count():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM appointments")
    count = c.fetchone()[0]
    conn.close()
    return count

init_db()

# ── RAG System ───────────────────────────────────────────────
VECTOR_STORE_PATH = "vector_store"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def load_documents(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        try:
            if file_extension == "pdf":
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path)
            docs.extend(loader.load())
        finally:
            os.remove(temp_path)
    return docs

def create_vector_store(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        embeddings = get_embeddings()
        return FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None

def search_documents(query, k=3):
    vector_store = load_vector_store()
    if vector_store:
        docs = vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    return ""

# ── PDF Export ───────────────────────────────────────────────
def generate_pdf(appointments):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("AI Appointment Booking Agent", styles["Title"]))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 20))
    data = [["#", "Date Booked", "Request"]]
    for apt in appointments:
        data.append([str(apt["id"]), apt["timestamp"], apt["request"][:60]])
    table = Table(data, colWidths=[40, 150, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7C3AED")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f0ff")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ── LangGraph ────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_input: str
    intent: str
    appointment_confirmed: bool
    response: str
    rag_context: str

def detect_intent(state: AgentState) -> AgentState:
    user_input = state["user_input"].lower()
    if any(word in user_input for word in ["book", "schedule", "appointment", "reserve"]):
        intent = "book"
    elif any(word in user_input for word in ["cancel", "delete", "remove"]):
        intent = "cancel"
    elif any(word in user_input for word in ["show", "list", "view", "my appointments"]):
        intent = "view"
    elif any(word in user_input for word in ["reschedule", "change", "move"]):
        intent = "reschedule"
    elif any(word in user_input for word in ["doctor", "available", "service", "hours", "cost", "price", "specialist"]):
        intent = "rag_query"
    else:
        intent = "general"
    return {**state, "intent": intent}

def retrieve_context(state: AgentState) -> AgentState:
    if state["intent"] == "rag_query":
        context = search_documents(state["user_input"])
        return {**state, "rag_context": context}
    return {**state, "rag_context": ""}

def generate_response(state: AgentState) -> AgentState:
    rag_context = state.get("rag_context", "")
    if rag_context:
        system_prompt = f"""You are a helpful AI appointment booking assistant.
        Use the following information from our documents to answer the user's question:
        
        {rag_context}
        
        If the information is not in the documents, say so politely.
        When an appointment is confirmed, start with 'APPOINTMENT CONFIRMED:'."""
    else:
        system_prompt = """You are a helpful AI appointment booking assistant.
        Help users schedule, reschedule, or cancel appointments.
        When confirmed, start with 'APPOINTMENT CONFIRMED:'."""

    messages = [SystemMessage(content=system_prompt)]
    for msg in state["messages"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    response = llm.invoke(messages)
    reply = response.content
    confirmed = "APPOINTMENT CONFIRMED" in reply.upper()
    return {**state, "response": reply, "appointment_confirmed": confirmed}

def save_if_confirmed(state: AgentState) -> AgentState:
    if state["appointment_confirmed"]:
        save_appointment(state["user_input"], state["response"])
    return state

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    graph.add_node("save_if_confirmed", save_if_confirmed)
    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", "save_if_confirmed")
    graph.add_edge("save_if_confirmed", END)
    return graph.compile()

agent = build_graph()

# ── Streamlit UI ─────────────────────────────────────────────
st.set_page_config(page_title="AI Appointment Agent", page_icon="📅", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #1a1025; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b69 0%, #1a1025 100%);
        border-right: 1px solid #7C3AED;
    }
    [data-testid="stSidebar"] * { color: #e9d5ff !important; }
    .stApp, .stMarkdown, p, h1, h2, h3 { color: #e9d5ff; }
    .intent-badge {
        background: rgba(124,58,237,0.3);
        color: #c4b5fd;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        border: 1px solid #7C3AED;
    }
    .rag-badge {
        background: rgba(16,185,129,0.3);
        color: #6ee7b7;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        border: 1px solid #10b981;
    }
    .stat-box {
        background: linear-gradient(135deg, #2d1b69, #3b1f7a);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(124,58,237,0.3);
        border: 1px solid #7C3AED;
    }
    .appointment-card {
        background: linear-gradient(135deg, #2d1b69, #3b1f7a);
        border-left: 4px solid #7C3AED;
        border-radius: 12px;
        padding: 14px;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(124,58,237,0.3);
        color: #e9d5ff;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED, #9333ea) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #7C3AED, #9333ea) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
    }
    .stTextInput > div > div > input {
        background-color: #2d1b69 !important;
        color: #e9d5ff !important;
        border: 1px solid #7C3AED !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInput"] {
        background-color: #2d1b69 !important;
        border: 2px solid #7C3AED !important;
        border-radius: 25px !important;
    }
    .user-bubble {
        display: flex;
        justify-content: flex-end;
        margin: 8px 0;
    }
    .user-bubble-inner {
        background: linear-gradient(135deg, #7C3AED, #9333ea);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        box-shadow: 0 4px 15px rgba(124,58,237,0.4);
    }
    .ai-bubble {
        display: flex;
        justify-content: flex-start;
        margin: 8px 0;
    }
    .ai-bubble-inner {
        background: linear-gradient(135deg, #2d1b69, #3b1f7a);
        color: #e9d5ff;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        border: 1px solid #7C3AED;
        box-shadow: 0 4px 15px rgba(124,58,237,0.2);
    }
    @media (max-width: 768px) {
        .main .block-container { padding: 1rem; }
        h1 { font-size: 1.5rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/calendar--v1.png", width=80)
    st.title("📅 Appointment Agent")
    st.caption("Powered by LangChain + LangGraph + RAG")
    st.markdown("---")
    page = st.radio("Navigate", ["💬 Book Appointment", "📚 Knowledge Base", "📋 My Appointments"])
    st.markdown("---")
    total = get_total_count()
    st.markdown("### 📊 Stats")
    st.markdown(f"""
    <div class='stat-box'>
        <h2 style='color:#a855f7'>{total}</h2>
        <p style='color:#e9d5ff'>Total Appointments</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    rag_status = "✅ Active" if os.path.exists(VECTOR_STORE_PATH) else "❌ No Documents"
    st.markdown(f"### 📚 RAG Status\n**{rag_status}**")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Book Appointment Page ────────────────────────────────────
if page == "💬 Book Appointment":
    st.title("💬 Book an Appointment")
    st.markdown("*Powered by **LangChain + LangGraph + RAG** AI Agent*")
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='user-bubble'>
                <div class='user-bubble-inner'>
                    <b>👤 You</b><br>{msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            intent_html = f"<br><span class='intent-badge'>🎯 Intent: {msg['intent']}</span>" if msg.get("intent") else ""
            rag_html = "<br><span class='rag-badge'>📚 RAG Enhanced</span>" if msg.get("rag_used") else ""
            st.markdown(f"""
            <div class='ai-bubble'>
                <div class='ai-bubble-inner'>
                    <b style='color:#a855f7'>🤖 AI Agent</b><br>{msg["content"]}{intent_html}{rag_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

    user_input = st.chat_input("Type your appointment request...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f"""
        <div class='user-bubble'>
            <div class='user-bubble-inner'>
                <b>👤 You</b><br>{user_input}
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🤖 Agent thinking..."):
            result = agent.invoke({
                "messages": st.session_state.messages,
                "user_input": user_input,
                "intent": "",
                "appointment_confirmed": False,
                "response": "",
                "rag_context": ""
            })
            reply = result["response"]
            intent = result["intent"]
            rag_used = bool(result.get("rag_context"))

        rag_html = "<br><span class='rag-badge'>📚 RAG Enhanced</span>" if rag_used else ""
        st.markdown(f"""
        <div class='ai-bubble'>
            <div class='ai-bubble-inner'>
                <b style='color:#a855f7'>🤖 AI Agent</b><br>{reply}
                <br><span class='intent-badge'>🎯 Intent: {intent}</span>{rag_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "intent": intent,
            "rag_used": rag_used
        })

        if result["appointment_confirmed"]:
            st.success("✅ Appointment saved to database!")
            st.balloons()

# ── Knowledge Base Page ──────────────────────────────────────
elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base")
    st.markdown("*Upload documents to make AI smarter about your services*")
    st.markdown("---")

    st.subheader("📤 Upload Documents")
    st.markdown("Upload PDF or TXT files about your clinic, doctors, services, or schedules")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🔄 Process Documents"):
            with st.spinner("Processing documents..."):
                docs = load_documents(uploaded_files)
                create_vector_store(docs)
            st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
            st.info(f"📊 Total chunks created from documents ready for search!")

    st.markdown("---")
    st.subheader("📝 Or Add Text Directly")
    manual_text = st.text_area(
        "Type information about your clinic/services:",
        placeholder="Example: Dr. Ahmed is available Monday to Friday from 9am to 5pm. He specializes in general medicine...",
        height=150
    )

    if st.button("💾 Save to Knowledge Base"):
        if manual_text:
            temp_path = "temp_manual.txt"
            with open(temp_path, "w") as f:
                f.write(manual_text)
            loader = TextLoader(temp_path)
            docs = loader.load()
            os.remove(temp_path)
            create_vector_store(docs)
            st.success("✅ Information saved to knowledge base!")
        else:
            st.warning("Please enter some text first!")

    st.markdown("---")
    if os.path.exists(VECTOR_STORE_PATH):
        st.success("✅ Knowledge base is active — AI will use it to answer questions!")
        if st.button("🗑️ Clear Knowledge Base"):
            import shutil
            shutil.rmtree(VECTOR_STORE_PATH)
            st.success("Knowledge base cleared!")
            st.rerun()
    else:
        st.warning("⚠️ No knowledge base yet — upload documents to get started!")

# ── My Appointments Page ─────────────────────────────────────
elif page == "📋 My Appointments":
    st.title("📋 My Appointments")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Search appointments...",
                              placeholder="e.g. doctor, Monday, dentist")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Search"):
            st.rerun()

    appointments = load_appointments(search_query=search)

    if not appointments:
        st.info("No appointments found!")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<p style='color:#e9d5ff'>Found <b>{len(appointments)}</b> appointment(s)</p>",
                       unsafe_allow_html=True)
        with col2:
            pdf_buffer = generate_pdf(appointments)
            st.download_button(
                label="📄 Export PDF",
                data=pdf_buffer,
                file_name=f"appointments_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

        st.markdown("---")
        st.subheader("📊 Overview")
        table_data = [{"#": a["id"], "Date Booked": a["timestamp"],
                      "Request": a["request"]} for a in appointments]
        st.table(table_data)

        st.markdown("---")
        st.subheader("📋 Details")
        for apt in appointments:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div class='appointment-card'>
                    <b style='color:#a855f7'>#{apt['id']} — {apt['timestamp']}</b><br>
                    <b style='color:#c4b5fd'>Request:</b>
                    <span style='color:#e9d5ff'>{apt['request']}</span><br>
                    <b style='color:#c4b5fd'>AI Response:</b>
                    <span style='color:#e9d5ff'>{apt['response'][:200]}...</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button(f"🗑️ Delete", key=f"del_{apt['id']}"):
                    delete_appointment(apt['id'])
                    st.success("Deleted!")
                    st.rerun()