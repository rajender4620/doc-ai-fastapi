from fastapi import FastAPI, File, UploadFile
from openai import OpenAI
import pdfplumber
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from docx import Document
import spacy

app = FastAPI()
nlp = spacy.load("en_core_web_sm")
# Create embeddings object (small + fast model)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Chroma DB; stored in "chroma_db" folder
vector_db = Chroma(
    collection_name="docs", embedding_function=embeddings, persist_directory="chroma_db"
)

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)


@app.get("/health")
async def read_health():
    return {"message": "ok"}


@app.post("/upload")
async def getUserPdfOrDoc(file: UploadFile = File(...)):
    content = ""
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        # Read PDF
        with pdfplumber.open(file.file) as pdf:
            content = "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif filename.endswith(".docx"):
        # Read DOCX
        doc = Document(file.file)
        content = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        return {"error": "Unsupported file type"}

    # Step: redact PII
    doc = nlp(content)
    redacted_text = content
    for ent in doc.ents:
        if ent.label_ in [
            "PERSON",
            "GPE",
            "LOC",
            "EMAIL",
            "PHONE_NUMBER",
            "ORG",
        ]:  # GPE=geo-political entity (places)
            redacted_text = redacted_text.replace(ent.text, "[REDACTED]")

            # Call OpenRouter for summary

    summary_response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {
                "role": "user",
                "content": f"Summarize this document briefly:\n\n{redacted_text}",
            }
        ],
    )
    summary = summary_response.choices[0].message.content

    # Call OpenRouter for risks
    risks_response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {
                "role": "user",
                "content": f"List potential compliance risks in this document as bullet points:\n\n{redacted_text}",
            }
        ],
    )
    risks_text = risks_response.choices[0].message.content
    risks = [line.strip("-• \n") for line in risks_text.splitlines() if line.strip()]

    # Store in vector DB
    metadata = {
        "filename": file.filename,
        "risks": "\n".join(risks),  # ✅ convert list to string
    }
    vector_db.add_texts([summary], metadatas=[metadata])
    vector_db.persist()

    # ✅ Print what’s inside the DB
    existing = vector_db.get()
    print("=== Stored documents in vector DB ===")
    for doc, meta in zip(existing["documents"], existing["metadatas"]):
        print(f"Document: {doc}")
    print(f"Metadata: {meta}")

    return {"redacted_text": redacted_text, "summary": summary, "risks": risks}


@app.get("/search")
async def search_docs(q: str):
    # Search vector DB with the question text
    results = vector_db.similarity_search(query=q, k=3)  # top 3 matches

    response = []
    for doc in results:
        response.append({"summary": doc.page_content, "metadata": doc.metadata})
    return {"results": response}
