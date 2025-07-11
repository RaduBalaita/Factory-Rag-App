import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import re
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from fastapi.responses import StreamingResponse
from typing import AsyncIterator, Optional, Dict, Any
from langchain_community.llms import LlamaCpp

# Set the API key
load_dotenv()  # Load environment variables from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

DEFAULT_PROMPT_TEMPLATE = """You will be given context from a machine's technical manual and an error code.
Structure your response in three distinct sections:
1. **Problem Description:** Briefly explain what the error code means based on the context.
2. **Probable Causes:** List the most likely reasons for this error from the context.
3. **Solution Steps:** Provide a clear, step-by-step guide to fix the issue using only information from the context.

**IMPORTANT:** Only use information from the provided context. If the context is empty or does not contain specific information for the queried error code, state that "No specific information was found for this error code in the manual."

Context: {context}

Query: {query}

Answer:
"""

# Global variable to store the prompt template
prompt_template_str = DEFAULT_PROMPT_TEMPLATE

class ModelConfig(BaseModel):
    type: str = 'api'
    provider: Optional[str] = 'google'
    api_key: Optional[str] = None
    path: Optional[str] = None

class Query(BaseModel):
    machine: str
    query: str
    language: str = 'en'
    system_prompt: str = 'You are a helpful assistant.'
    model_config: ModelConfig = ModelConfig()

@app.get("/prompt_template")
async def get_prompt_template():
    return {"prompt_template": prompt_template_str}

@app.post("/prompt_template")
async def set_prompt_template(request: dict):
    global prompt_template_str
    prompt_template_str = request.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    return {"prompt_template": prompt_template_str}

# Configure the paths to your docs and FAISS index
DOCS_PATH = "C:/Users/Tempest/Desktop/RAG/docs"
FAISS_INDEX_PATH = "C:/Users/Tempest/Desktop/RAG/backend/faiss_index"

# Mappings from machine names to their documentation
MACHINE_CONFIG = {
    "Yaskawa Alarm 380500": {
        "doc": f"{DOCS_PATH}/PDF/1-1cvjgdf_ys840di_errorcode_of_alarm_380500-deu.pdf",
        "index": f"{FAISS_INDEX_PATH}/yaskawa_alarm_380500.faiss"
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

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm(model_config: ModelConfig):
    if model_config.type == 'local' and model_config.path:
        if os.path.exists(model_config.path):
            return LlamaCpp(
                model_path=model_config.path,
                n_gpu_layers=-1, # Offload all layers to GPU
                n_batch=512,
                n_ctx=2048,
                f16_kv=True,  # Must be True on new models
                verbose=True,
            )
        else:
            raise ValueError(f"Local model path does not exist: {model_config.path}")
    
    # Default to Google Gemini
    # Here you could add logic for other providers like OpenAI, Claude
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")
        
    return GoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

if not os.path.exists(FAISS_INDEX_PATH):
    os.makedirs(FAISS_INDEX_PATH)

for machine, config in MACHINE_CONFIG.items():
    if os.path.exists(config["index"]):
        print(f"Loading existing Chroma index for {machine}")
        config["db"] = Chroma(persist_directory=config["index"], embedding_function=embeddings)
    else:
        print(f"Creating new Chroma index for {machine}")
        loader = PyMuPDFLoader(config["doc"])
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])

        pattern = r"\n(\d{3,})\s*[‘’']"
        split_text = re.split(pattern, full_text)
        texts_with_metadata = []
        if len(split_text) > 1:
            for i in range(1, len(split_text), 2):
                error_code = split_text[i].strip()
                content = split_text[i+1].strip() if (i + 1) < len(split_text) else ""
                full_chunk_content = f"{error_code} ‘{content}"
                full_chunk_content = re.sub(r"·M· Model Ref\\. \\d+", "", full_chunk_content)
                full_chunk_content = re.sub(r"Error solution", "", full_chunk_content)
                full_chunk_content = re.sub(r"CNC \\d+ ·M·", "", full_chunk_content)
                full_chunk_content = full_chunk_content.replace("", "")
                full_chunk_content = re.sub(r'\\s+', ' ', full_chunk_content).strip()
                if full_chunk_content and error_code:
                    metadata = {"error_code": error_code, "machine": machine}
                    doc = Document(page_content=full_chunk_content, metadata=metadata)
                    texts_with_metadata.append(doc)
        if texts_with_metadata:
            db = Chroma.from_documents(texts_with_metadata, embeddings, persist_directory=config["index"])
            db.persist()
            config["db"] = db
        else:
            print(f"WARNING: No text chunks were created for {machine}. The index will be empty.")
            config["db"] = None

async def stream_response(chain, user_query: str) -> AsyncIterator[str]:
    async for chunk in chain.astream(user_query):
        yield chunk

@app.post("/query")
async def ask_query(query: Query):
    try:
        llm = get_llm(query.model_config)
    except ValueError as e:
        return {"error": str(e)}

    config = MACHINE_CONFIG.get(query.machine)
    if not config:
        return {"error": "Machine not found"}


    db = config.get("db")
    if not db:
        return {"response": f"The document index for '{query.machine}' is empty. This likely means the PDF could not be processed correctly. Please check the document format."}

    # 1. Define metadata fields for the SelfQueryRetriever
    metadata_field_info = [
        AttributeInfo(
            name="error_code",
            description="The numeric error code for a machine fault, like '1066' or '0079'.",
            type="string",
        ),
        AttributeInfo(
            name="machine",
            description=f"The machine model the document is for, which is '{query.machine}'",
            type="string",
        ),
    ]
    document_content_description = "A description of a machine fault, its cause, and its solution."

    # 2. Set up the SelfQueryRetriever
    retriever = SelfQueryRetriever.from_llm(
        llm,
        db,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={"k": 1}
    )

    # 3. Define the prompt template using the global template
    # The user-editable part is now the whole template.
    # The 'system_prompt' from the query can still be used if you want to prepend it.
    language_instruction = ""
    if query.language == "ro":
        language_instruction = "\nIMPORTANT: Respond ONLY in Romanian language."
    elif query.language == "en":
        language_instruction = "\nIMPORTANT: Respond ONLY in English language."
    
    final_template_str = f"{{system_prompt}}\n\n{prompt_template_str}{language_instruction}"
    
    prompt = PromptTemplate(
        template=final_template_str,
        input_variables=["system_prompt", "context", "query"]
    )

    # 4. Create the processing chain
    chain = (
        {
            "context": retriever,
            "query": RunnablePassthrough(),
            "system_prompt": lambda x: query.system_prompt, # Pass the system prompt from the query
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Return a streaming response
    return StreamingResponse(stream_response(chain, query.query), media_type="text/event-stream")