import os
import re
import mysql.connector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import HumanMessage, SystemMessage

# Load env (Hugging Face API key)
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to ["http://localhost:5173", "https://your-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# LLM Setup (GPT-OSS 20B)
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    max_new_tokens=150,
    temperature=0.2,
)
model = ChatHuggingFace(llm=llm)

# TiDB Connection
DB_CONFIG = {
    "host": "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    "port": 4000,
    "user": "4FqQsswjPt3ZaUg.bot",
    "password": "password",  # change to your bot password
    "database": "test",
    "ssl_ca": r"C:\tidb\isrgrootx1.pem",
}

# Request
class ChatRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Inventory chatbot is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    user_query = req.query

    # 1️⃣ Ask LLM to generate SQL
    system_prompt = SystemMessage(
        content=(
            "You are a SQL assistant. "
            "Only generate SELECT queries on the 'products' table. "
            "The table schema is:\n"
            "products(product_id INT, name VARCHAR, description TEXT, sku VARCHAR, unit_price DECIMAL, reorder_level INT, category_id INT). "
            "Never guess column names. Always use only these columns.Also user can ask anything related to products table like give me the product by its product_id so you should give him query of SELECT name from products where product_id='7'-eg." \
            "He can ask anything related to the products schema. For eg give me the name of product whose id is this or give me the quantity of the particular product.Use the schema and generate appropriate SQL query"
        )
    )

    human_prompt = HumanMessage(content=f"User question: {user_query}\nWrite a SQL query for MySQL.")
    sql_candidate = model([system_prompt, human_prompt]).content.strip()
    print("Raw LLM SQL candidate:", sql_candidate)

    # 2️⃣ Extract clean SELECT query from LLM output
    match = re.search(r"(select .*?from products.*?)(;|$)", sql_candidate, re.IGNORECASE | re.DOTALL)
    if match:
        sql_candidate = match.group(1).strip()
    else:
        raise HTTPException(status_code=400, detail=f"Could not extract a valid SQL query: {sql_candidate}")

    print("✅ Clean SQL to execute:", sql_candidate)

    # 3️⃣ Execute SQL safely
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql_candidate)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

    # 4️⃣ Convert result to dicts
    result = [dict(zip(columns, row)) for row in rows]

    # 5️⃣ Explanation in plain English
    explain_prompt = HumanMessage(
        content=f"User asked: {user_query}\nSQL: {sql_candidate}\nResults: {result}\n"
                f"Explain this in like chat gpt should explain"
    )
    explanation = model([SystemMessage(content="You are a helpful assistant."), explain_prompt])

    # 6️⃣Format the answer neatly
    if not result:
        answer_text = f"I searched the database but found no matching products for your query: \"{user_query}\"."
    else:
        if len(columns) == 1:
            values = [str(r[columns[0]]) for r in result]
            answer_text = "\n".join([f"- {v}" for v in values])
        else:
            lines = []
            for row in result:
                row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
                lines.append(f"- {row_str}")
            answer_text =  "\n".join(lines)
    return {
        "sql": sql_candidate,
        "raw_result": result,
        "reply": explanation.content,  # Layman explanation
        "answer": answer_text         # Clean conversational text
    }
