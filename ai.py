import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "RAG-App"
    }
)
MODEL = "openai/gpt-oss-120b:free"


def answer_question(vector_store, question):

    results = vector_store.search(question, k=5)

    context = ""

    for r in results:
        context += f"\n[SOURCE: {r['source']} | CHUNK: {r['chunk_id']}]\n{r['text']}\n"

    prompt = f"""
Odpověz pouze na základě kontextu.

KONTEXT:
{context}

OTÁZKA:
{question}

Pravidla:
- používej jen informace z kontextu
- pokud odpověď není v kontextu, řekni že nevíš
- uváděj zdroj (source + chunk id)
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
