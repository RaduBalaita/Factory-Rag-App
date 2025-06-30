import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set the API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAFIqgCUu2s8m9r9telIbTKkRvtVKhkoBo"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Query(BaseModel):
    machine: str
    query: str

# Configure the paths to your docs and FAISS index
DOCS_PATH = "C:/Users/Tempest/Desktop/RAG/docs"
FAISS_INDEX_PATH = "C:/Users/Tempest/Desktop/RAG/backend/faiss_index"

# Mappings from machine names to their documentation
MACHINE_CONFIG = {
    "Yaskawa Alarm 380500": {
        "doc": f"{DOCS_PATH}/PDF/1-1cvjgdf_ys840di_errorcode_of_alarm_380500-deu.pdf",
        "index": f"{FAISS_INDEX_PATH}/yaskawa_alarm_380500.faiss"
    },
    "General Error Codes": {
        "doc": f"{DOCS_PATH}/PDF/error-codes.pdf",
        "index": f"{FAISS_INDEX_PATH}/general_error_codes.faiss"
    },
    "Fagor CNC 8055": {
        "doc": f"{DOCS_PATH}/PDF/Fagor-CNC-8055-Error-Solution-English.pdf",
        "index": f"{FAISS_INDEX_PATH}/fagor_cnc_8055.faiss"
    },
    "FC-GTR V2.1": {
        "doc": f"{DOCS_PATH}/PDF/FC-GTR-V2.1-OP-EN.pdf",
        "index": f"{FAISS_INDEX_PATH}/fc_gtr_v2_1.faiss"
    },
    "NUM CNC": {
        "doc": f"{DOCS_PATH}/PDF/NUM CNC Error List.pdf",
        "index": f"{FAISS_INDEX_PATH}/num_cnc.faiss"
    }
}

# Initialize the LLM
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create FAISS indexes
if not os.path.exists(FAISS_INDEX_PATH):
    os.makedirs(FAISS_INDEX_PATH)

for machine, config in MACHINE_CONFIG.items():
    if os.path.exists(config["index"]):
        print(f"Loading existing FAISS index for {machine}")
        config["db"] = FAISS.load_local(config["index"], embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Creating new FAISS index for {machine}")
        loader = PyMuPDFLoader(config["doc"])
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(config["index"])
        config["db"] = db

@app.post("/query")
async def ask_query(query: Query):
    config = MACHINE_CONFIG.get(query.machine)
    if not config:
        return {"error": "Machine not found"}

    # 1. Use the pre-loaded FAISS index
    db = config["db"]
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 2. Use MultiQueryRetriever
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

    # 3. Define the prompt template
    template = """
    You are a helpful factory maintenance assistant. You will be given context from a machine's technical manual and an error code.
    Structure your response in three distinct sections:
    1. **Problem Description:** Briefly explain what the error code means based on the context.
    2. **Probable Causes:** List the most likely reasons for this error from the context.
    3. **Solution Steps:** Provide a clear, step-by-step guide to fix the issue using only information from the context.

    Context: {context}

    Query: {query}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])

    # 4. Create the processing chain
    chain = (
        {"context": mq_retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Invoke the chain with the user's query
    response = await chain.ainvoke(f"error code {query.query}")

    # 6. Debugging: Print the retrieved context
    retrieved_docs = await mq_retriever.ainvoke(f"error code {query.query}")
    print("Retrieved Context:", "\n".join([doc.page_content for doc in retrieved_docs]))

    return {"response": response}