import os
import re
import json
import uuid  # NEW
import mysql.connector
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import HumanMessage, SystemMessage
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# In-memory chat memory (arrays)
# -------------------------
HISTORY_TURNS = 10  # keep last N turns per session
SESSION_HISTORY: Dict[str, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=HISTORY_TURNS))
# shape per item: (role, content), where role in {"user","assistant"}

def add_history(session_id: str, role: str, content: str):
    SESSION_HISTORY[session_id].append((role, content))

def get_history(session_id: str) -> List[Tuple[str, str]]:
    return list(SESSION_HISTORY[session_id])  # oldest -> newest order preserved by deque

# -------------------------
# LLM + DB setup
# -------------------------
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    max_new_tokens=150,
    temperature=0.2,
)
model = ChatHuggingFace(llm=llm)

DB_CONFIG = {
    "host": "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    "port": 4000,
    "user": "4FqQsswjPt3ZaUg.bot",
    "password": "password",  # <-- keep in env in real code
    "database": "test",
    "ssl_ca": r"C:\tidb\isrgrootx1.pem",
}

# -------------------------
# API models
# -------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None   # NEW: optional
    debug: Optional[bool] = False

@app.get("/")
def root():
    return {"message": "Inventory chatbot is running"}

# -------------------------
# Chat endpoint
# -------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    user_query = req.query

    # NEW: auto-generate session_id if not provided
    session_id = req.session_id or str(uuid.uuid4())

    # record the user turn in memory
    add_history(session_id, "user", user_query)

    # 1) Build SQL prompt with short-term memory (arrays)
    history = get_history(session_id)
    sql_messages: List = []
    # include last turns as context (helps follow-ups)
    for role, content in history:
        if role == "user":
            sql_messages.append(HumanMessage(content=content))
        else:
            sql_messages.append(SystemMessage(content=f"[{role}] {content}"))

    sql_messages.append(SystemMessage(content=(
        "You are a SQL assistant.\n"
        "Only generate a single MySQL SELECT query on the products table using ONLY these columns:\n"
        "products(product_id INT, name VARCHAR, description TEXT, sku VARCHAR, unit_price DECIMAL, "
        "reorder_level INT, category_id INT).\n"
        "Never invent columns. Do not add explanations. Return ONLY the SQL."
    )))
    sql_messages.append(HumanMessage(content=f"User question: {user_query}\nWrite a SQL query for MySQL."))

    # 2) Get SQL
    sql_candidate = model(sql_messages).content.strip()
    print("Raw LLM SQL candidate:", sql_candidate)

    match = re.search(r"(select .*?from\s+products\b.*?)(;|$)", sql_candidate, re.IGNORECASE | re.DOTALL)
    if not match:
        add_history(session_id, "assistant", "Sorry, I couldn’t form a valid SQL for that. Try rephrasing?")
        raise HTTPException(status_code=400, detail=f"Could not extract a valid SQL query: {sql_candidate}")
    sql_candidate = match.group(1).strip()
    print("✅ Clean SQL to execute:", sql_candidate)

    # 3) Execute SQL
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql_candidate)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
    except Exception as e:
        add_history(session_id, "assistant", f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

    result = [dict(zip(columns, row)) for row in rows]

    # 4) Build final English answer using memory + results
    answer_messages: List = []
    for role, content in history:
        if role == "user":
            answer_messages.append(HumanMessage(content=content))
        else:
            answer_messages.append(SystemMessage(content=f"[{role}] {content}"))

    answer_messages += [
        SystemMessage(content=(
            "You are a helpful assistant.\n"
            "Reply in clear, friendly English (one short paragraph only).\n"
            "Summarize the result for a non-technical user. If no rows, give helpful next steps.\n"
            "Do NOT include SQL unless asked.After giving the relevent name, do not generate anymore text like do not write if you are looking for further details etc." \
            "Also do not give response to any query which is not related to inventories. for eg-you should not give answer of 3+4=7 etc."
        )),
        HumanMessage(content=(
            f"User asked: {user_query}\n"
            f"SQL executed: {sql_candidate}\n"
            f"Results JSON: {json.dumps(result, ensure_ascii=False)}\n"
            f"Write the conversational answer now."
        ))
    ]
    explanation = model(answer_messages).content.strip()

    # 5) Save assistant reply in memory
    add_history(session_id, "assistant", explanation)

    # 6) Return minimal English by default; always include session_id so the client can reuse it
    resp = {"session_id": session_id, "message": explanation}
    if req.debug:
        resp.update({
            "sql": sql_candidate,
            "raw_result": result,
            "history_sample": get_history(session_id),
        })
    return resp
