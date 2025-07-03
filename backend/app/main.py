import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set the API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD7PrAIzcfpThTw00OnMIW_WDv394FTLMQ"

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
    language: str = 'en'
    system_prompt: str = 'You are a helpful assistant.'

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
        # Custom splitting logic based on error codes
        full_text = " ".join([doc.page_content for doc in documents])
        
        # Regex to find error codes and their descriptions
        pattern = r"(\d{3,})\s*['\]([^\]]+)['\]"
        matches = list(re.finditer(pattern, full_text))
        
        texts = []
        if matches:
            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i+1].start() if i+1 < len(matches) else len(full_text)
                
                chunk_content = full_text[start:end]
                
                # Clean the chunk content
                chunk_content = re.sub(r"·M· Model Ref\\. \\d+", "", chunk_content)
                chunk_content = re.sub(r"Error solution", "", chunk_content)
                chunk_content = re.sub(r"CNC \\d+ ·M·", "", chunk_content)
                chunk_content = chunk_content.replace("", "")
                chunk_content = re.sub(r'\\s+', ' ', chunk_content).strip()
                
                if chunk_content:
                    texts.append(Document(page_content=chunk_content))

        if texts:
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(config["index"])
            config["db"] = db
        else:
            print(f"WARNING: No text chunks were created for {machine}. The index will be empty.")
            config["db"] = None

@app.post("/query")
async def ask_query(query: Query):
    config = MACHINE_CONFIG.get(query.machine)
    if not config:
        return {"error": "Machine not found"}

    # 1. Use the pre-loaded FAISS index
    db = config.get("db")
    if not db:
        return {"response": f"The document index for '{query.machine}' is empty. This likely means the PDF could not be processed correctly. Please check the document format."}
    
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 2. Use MultiQueryRetriever
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

    # 3. Define the prompt template
    template = f"""
    {query.system_prompt}
    You will be given context from a machine's technical manual and an error code.
    Structure your response in three distinct sections:
    1. **Problem Description:** Briefly explain what the error code means based on the context.
    2. **Probable Causes:** List the most likely reasons for this error from the context.
    3. **Solution Steps:** Provide a clear, step-by-step guide to fix the issue using only information from the context.

    **IMPORTANT:** Only use information from the context that is explicitly and exactly about the provided error code. Ignore any information related to other error codes, even if they are numerically similar. If the context does not contain specific information for the queried error code, state that the information is not available.

    Translate the final response to {query.language}.

    Context: {{context}}

    Query: {{query}}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])

    # 4. Create the processing chain
    chain = (
        {{"context": mq_retriever, "query": RunnablePassthrough()}}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Invoke the chain with the user's query
    response = await chain.ainvoke(f"What is the meaning of error code {query.query}?")

    # 6. Debugging: Print the retrieved context
    retrieved_docs = await mq_retriever.ainvoke(f"What is the meaning of error code {query.query}?")
    print("Retrieved Context:", "\n".join([doc.page_content for doc in retrieved_docs]))

    return {"response": response}