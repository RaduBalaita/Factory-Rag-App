import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import shutil
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
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

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

Answer:"""

# Paths
DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "machines_config.json")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    query: str
    machine: str

class ModelConfig(BaseModel):
    type: str
    provider: Optional[str] = None
    path: Optional[str] = None

class PromptTemplateRequest(BaseModel):
    template: str

@app.get("/models")
@app.get("/api/models")
async def get_models():
    """Get list of available local models"""
    models_path = os.path.join(os.path.dirname(__file__), "..", "..", ".models")
    try:
        if os.path.exists(models_path):
            files = [f for f in os.listdir(models_path) if f.endswith('.gguf')]
            return {"models": files}
        else:
            return {"models": [], "error": f"Models directory not found: {models_path}"}
    except Exception as e:
        return {"models": [], "error": str(e)}

@app.get("/api/machines")
async def get_machines():
    """Get list of available machines"""
    global MACHINE_CONFIG
    return {"machines": list(MACHINE_CONFIG.keys())}

@app.post("/api/machines/upload")
async def upload_machine_document(name: str, file: UploadFile = File(...)):
    """Upload a new machine document"""
    global MACHINE_CONFIG
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create safe filename
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        filename = f"{safe_name}.pdf"
        doc_path = os.path.join(DOCS_PATH, "PDF", filename)
        index_path = os.path.join(FAISS_INDEX_PATH, f"{safe_name}.faiss")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        
        # Save uploaded file
        with open(doc_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add to machine config
        MACHINE_CONFIG[name] = {
            "doc": doc_path,
            "index": index_path
        }
        
        # Create index for the new document
        await create_index_for_machine(name, MACHINE_CONFIG[name])
        
        # Save updated configuration
        save_machine_config(MACHINE_CONFIG)
        
        return {"message": f"Document '{name}' uploaded successfully", "filename": filename}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/machines/{machine_name}")
async def delete_machine_document(machine_name: str):
    """Delete a machine document and its index"""
    global MACHINE_CONFIG
    try:
        # URL decode the machine name (FastAPI should handle this, but let's be explicit)
        import urllib.parse
        decoded_name = urllib.parse.unquote(machine_name)
        print(f"Attempting to delete machine: '{decoded_name}'")
        print(f"Available machines: {list(MACHINE_CONFIG.keys())}")
        
        # Check both encoded and decoded names
        target_name = None
        if decoded_name in MACHINE_CONFIG:
            target_name = decoded_name
        elif machine_name in MACHINE_CONFIG:
            target_name = machine_name
        
        if not target_name:
            raise HTTPException(status_code=404, detail=f"Machine '{decoded_name}' not found. Available machines: {list(MACHINE_CONFIG.keys())}")
        
        config = MACHINE_CONFIG[target_name]
        print(f"Machine config: {config}")
        
        # Close the database connection first if it exists
        if "db" in config and config["db"] is not None:
            try:
                # Try to close the Chroma connection
                if hasattr(config["db"], '_client'):
                    config["db"]._client.reset()
                config["db"] = None
                print("Closed database connection")
            except Exception as e:
                print(f"Error closing database: {e}")
        
        # Delete document file
        if "doc" in config and os.path.exists(config["doc"]):
            os.remove(config["doc"])
            print(f"Deleted document file: {config['doc']}")
        
        # Delete index directory
        if "index" in config and os.path.exists(config["index"]):
            import time
            # Wait a moment for file handles to be released
            time.sleep(1)
            try:
                shutil.rmtree(config["index"])
                print(f"Deleted index directory: {config['index']}")
            except Exception as e:
                print(f"Error deleting index directory: {e}")
                # Try to delete individual files if directory deletion fails
                try:
                    for root, dirs, files in os.walk(config["index"]):
                        for file in files:
                            try:
                                os.remove(os.path.join(root, file))
                            except Exception:
                                pass
                    shutil.rmtree(config["index"])
                    print(f"Successfully deleted index directory on second attempt")
                except Exception as e2:
                    print(f"Could not delete index directory: {e2}")
        
        # Remove from config
        del MACHINE_CONFIG[target_name]
        
        # Save updated configuration
        save_machine_config(MACHINE_CONFIG)
        
        return {"message": f"Machine '{target_name}' deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting machine '{machine_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting machine: {str(e)}")

# Configure the paths to your docs and FAISS index

def load_machine_config():
    """Load machine configuration from JSON file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Validate that all configured documents exist
                valid_config = {}
                for machine, machine_config in config.items():
                    if os.path.exists(machine_config["doc"]):
                        valid_config[machine] = machine_config
                    else:
                        print(f"WARNING: Document not found for {machine}: {machine_config['doc']}")
                return valid_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    else:
        # Create default configuration if file doesn't exist
        # Only add machines for files that actually exist
        default_config = {}
        
        # Check for available PDF files and create default configs
        pdf_dir = os.path.join(DOCS_PATH, "PDF")
        if os.path.exists(pdf_dir):
            for filename in os.listdir(pdf_dir):
                if filename.endswith('.pdf'):
                    # Create better machine names from filenames
                    base_name = filename.replace('.pdf', '')
                    
                    # Handle specific known files
                    if 'fagor' in base_name.lower() and '8055' in base_name:
                        machine_name = "Fagor CNC 8055"
                    elif 'yaskawa' in base_name.lower() or '380500' in base_name:
                        machine_name = "Yaskawa Alarm 380500"
                    elif 'fc-gtr' in base_name.lower() or 'fc_gtr' in base_name.lower():
                        machine_name = "FC-GTR V2.1"
                    elif 'num' in base_name.lower() and 'cnc' in base_name.lower():
                        machine_name = "NUM CNC"
                    else:
                        # Generic cleanup for other files
                        machine_name = base_name.replace('-', ' ').replace('_', ' ')
                        machine_name = ' '.join(word.capitalize() for word in machine_name.split())
                    
                    doc_path = os.path.join(pdf_dir, filename)
                    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name.lower())
                    index_path = os.path.join(FAISS_INDEX_PATH, f"{safe_name}.faiss")
                    
                    default_config[machine_name] = {
                        "doc": doc_path,
                        "index": index_path
                    }
                    
        # Save default config
        save_machine_config(default_config)
        return default_config

def save_machine_config(config):
    """Save machine configuration to JSON file"""
    try:
        # Create a copy of config without the 'db' objects for JSON serialization
        json_config = {}
        for machine, machine_config in config.items():
            json_config[machine] = {
                "doc": machine_config["doc"],
                "index": machine_config["index"]
                # Don't save the 'db' object as it's not JSON serializable
            }
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(json_config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")

# Load machine configuration
MACHINE_CONFIG = load_machine_config()

async def create_index_for_machine(machine_name: str, config: dict):
    """Create vector index for a machine document"""
    try:
        print(f"Creating new Chroma index for {machine_name}")
        print(f"Document path: {config['doc']}")
        
        # Check if document file exists
        if not os.path.exists(config["doc"]):
            print(f"ERROR: Document file does not exist: {config['doc']}")
            config["db"] = None
            return
            
        loader = PyMuPDFLoader(config["doc"])
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from document")
        
        if not documents:
            print(f"ERROR: No documents loaded from {config['doc']}")
            config["db"] = None
            return
            
        full_text = "\n".join([doc.page_content for doc in documents])
        print(f"Full text length: {len(full_text)} characters")
        print(f"First 200 characters: {full_text[:200]}")

        # Use a more comprehensive approach to extract error codes
        # Pattern to find error codes like "1083 'Range exceeded'" 
        error_pattern = r"(\d{3,})\s+'([^']+)'"
        error_matches = list(re.finditer(error_pattern, full_text))
        
        print(f"Found {len(error_matches)} error code matches with basic pattern")
        
        texts_with_metadata = []
        
        if error_matches:
            print(f"Processing {len(error_matches)} error codes")
            for match in error_matches:
                error_code = match.group(1).strip()
                error_title = match.group(2).strip()
                
                # Find the full content for this error code
                start_pos = match.start()
                
                # Look for the next error code to know where this one ends
                next_error = None
                for next_match in error_matches:
                    if next_match.start() > start_pos:
                        next_error = next_match
                        break
                
                if next_error:
                    end_pos = next_error.start()
                    full_content = full_text[start_pos:end_pos].strip()
                else:
                    # This is the last error code, take rest of the text (but limit it)
                    full_content = full_text[start_pos:start_pos+2000].strip()
                
                # Clean up the content
                full_content = re.sub(r"·M· Model Ref\\. \\d+", "", full_content)
                full_content = re.sub(r"Error solution", "", full_content)
                full_content = re.sub(r"CNC \\d+ ·M·", "", full_content)
                full_content = full_content.replace("", "")
                full_content = re.sub(r'\s+', ' ', full_content).strip()
                
                if full_content and error_code:
                    metadata = {"error_code": error_code, "machine": machine_name}
                    doc = Document(page_content=full_content, metadata=metadata)
                    texts_with_metadata.append(doc)
                    
                    # Debug: print first few error codes to verify
                    if len(texts_with_metadata) <= 5:
                        print(f"Indexed error {error_code}: '{error_title}' - {full_content[:100]}...")
        else:
            # If no error code pattern, try different patterns or chunk by pages
            print("No error code pattern found, trying alternative approaches...")
            
            # Try to find error codes with different patterns
            error_patterns = [
                r"(\d{3,})\s*[:\-\s]",  # Error codes followed by : or -
                r"Error\s+(\d{3,})",     # "Error" followed by number
                r"Alarm\s+(\d{3,})",     # "Alarm" followed by number
            ]
            
            found_errors = False
            for pattern in error_patterns:
                matches = re.finditer(pattern, full_text, re.IGNORECASE)
                error_chunks = []
                
                for match in matches:
                    error_code = match.group(1)
                    start_pos = match.start()
                    # Get surrounding context (500 chars before and after)
                    context_start = max(0, start_pos - 500)
                    context_end = min(len(full_text), start_pos + 1500)
                    context = full_text[context_start:context_end].strip()
                    
                    if len(context) > 50:  # Only add if we have substantial content
                        metadata = {"error_code": error_code, "machine": machine_name}
                        doc = Document(page_content=context, metadata=metadata)
                        error_chunks.append(doc)
                
                if error_chunks:
                    print(f"Found {len(error_chunks)} errors using pattern: {pattern}")
                    texts_with_metadata.extend(error_chunks)
                    found_errors = True
                    break
            
            # If still no luck, chunk by pages
            if not found_errors:
                print("No error patterns found, chunking by pages")
                for i, doc in enumerate(documents):
                    if doc.page_content.strip() and len(doc.page_content.strip()) > 100:
                        metadata = {"page": i+1, "machine": machine_name}
                        texts_with_metadata.append(Document(page_content=doc.page_content.strip(), metadata=metadata))
                    
        print(f"Created {len(texts_with_metadata)} text chunks")
        
        if texts_with_metadata:
            db = Chroma.from_documents(texts_with_metadata, embeddings, persist_directory=config["index"])
            # Note: Chroma 0.4.x automatically persists, no need to call persist()
            config["db"] = db
            print(f"Successfully created index for {machine_name} with {len(texts_with_metadata)} chunks")
        else:
            print(f"WARNING: No text chunks were created for {machine_name}. The index will be empty.")
            config["db"] = None
    except Exception as e:
        print(f"Error creating index for {machine_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        config["db"] = None

def get_llm(model_config):
    if model_config['type'] == 'local' and model_config.get('path'):
        if not os.path.exists(model_config['path']):
            raise ValueError(f"Local model path does not exist: {model_config['path']}")
        return LlamaCpp(
            model_path=model_config['path'],
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            verbose=True,
        )
    elif model_config['type'] == 'api':
        provider = model_config.get('provider', 'google').lower()
        if provider == 'google':
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Google API key not provided")
            return GoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.7,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
        elif provider == 'openai':
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=api_key,
                temperature=0.7
            )
        elif provider == 'anthropic':
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not provided")
            return ChatAnthropic(
                model="claude-3-sonnet-20240229",
                anthropic_api_key=api_key,
                temperature=0.7
            )
        else:
            raise ValueError(f"Unsupported API provider: {provider}")
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")

if not os.path.exists(FAISS_INDEX_PATH):
    os.makedirs(FAISS_INDEX_PATH)

for machine, config in MACHINE_CONFIG.items():
    if os.path.exists(config["index"]):
        print(f"Loading existing Chroma index for {machine}")
        config["db"] = Chroma(persist_directory=config["index"], embedding_function=embeddings)
    else:
        print(f"Creating new Chroma index for {machine}")
        try:
            if not os.path.exists(config["doc"]):
                print(f"WARNING: Document not found for {machine}: {config['doc']}")
                config["db"] = None
                continue
                
            loader = PyMuPDFLoader(config["doc"])
            documents = loader.load()
            full_text = "\n".join([doc.page_content for doc in documents])

            # Try multiple patterns to extract error codes  
            error_code_patterns = [
                r"\n(\d{3,})\s*[''']",  # Original pattern: error codes followed by quotes
                r"(\d{3,})\s+'[^']*'",  # Pattern: error codes followed by quoted text
                r"(\d{3,})\s+'[^']*'\s*\n",  # Pattern: error codes with quotes and newline
            ]
            
            texts_with_metadata = []
            best_pattern = None
            best_split = []
            
            # Try each pattern and use the one that finds the most error codes
            for pattern in error_code_patterns:
                split_text = re.split(pattern, full_text)
                if len(split_text) > len(best_split):
                    best_pattern = pattern
                    best_split = split_text
            
            print(f"Best pattern '{best_pattern}' split text into {len(best_split)} parts")
            
            # Try to extract error codes with their descriptions using the best pattern
            if len(best_split) > 1:
                print(f"Found error code pattern, processing {(len(best_split)-1)//2} error codes")
                for i in range(1, len(best_split), 2):
                    error_code = best_split[i].strip()
                    content = best_split[i+1].strip() if (i + 1) < len(best_split) else ""
                    
                    # Look for the specific error in the full text around this error code
                    error_search = rf"(\d{{3,}})\s+'{re.escape(error_code)}[^']*'.*?(?=\d{{3,}}\s+'|$)"
                    error_match = re.search(error_search, full_text, re.DOTALL)
                    
                    if error_match:
                        full_chunk_content = error_match.group(0).strip()
                    else:
                        # Fallback to the split method
                        full_chunk_content = f"{error_code} '{content}"
                    
                    # Clean up the content
                    full_chunk_content = re.sub(r"·M· Model Ref\\. \\d+", "", full_chunk_content)
                    full_chunk_content = re.sub(r"Error solution", "", full_chunk_content)
                    full_chunk_content = re.sub(r"CNC \\d+ ·M·", "", full_chunk_content)
                    full_chunk_content = full_chunk_content.replace("", "")
                    full_chunk_content = re.sub(r'\s+', ' ', full_chunk_content).strip()
                    
                    if full_chunk_content and error_code:
                        metadata = {"error_code": error_code, "machine": machine}
                        doc = Document(page_content=full_chunk_content, metadata=metadata)
                        texts_with_metadata.append(doc)
            else:
                # If no error code pattern, try different patterns or chunk by pages
                print("No error code pattern found, trying alternative approaches...")
                
                # Try to find error codes with different patterns
                error_patterns = [
                    r"(\d{3,})\s*[:\-\s]",  # Error codes followed by : or -
                    r"Error\s+(\d{3,})",     # "Error" followed by number
                    r"Alarm\s+(\d{3,})",     # "Alarm" followed by number
                ]
                
                found_errors = False
                for pattern in error_patterns:
                    matches = re.finditer(pattern, full_text, re.IGNORECASE)
                    error_chunks = []
                    
                    for match in matches:
                        error_code = match.group(1)
                        start_pos = match.start()
                        # Get surrounding context (500 chars before and after)
                        context_start = max(0, start_pos - 500)
                        context_end = min(len(full_text), start_pos + 1500)
                        context = full_text[context_start:context_end].strip()
                        
                        if len(context) > 50:  # Only add if we have substantial content
                            metadata = {"error_code": error_code, "machine": machine}
                            doc = Document(page_content=context, metadata=metadata)
                            error_chunks.append(doc)
                    
                    if error_chunks:
                        print(f"Found {len(error_chunks)} errors using pattern: {pattern}")
                        texts_with_metadata.extend(error_chunks)
                        found_errors = True
                        break
                
                # If still no luck, chunk by pages
                if not found_errors:
                    print("No error patterns found, chunking by pages")
                    for i, doc in enumerate(documents):
                        if doc.page_content.strip() and len(doc.page_content.strip()) > 100:
                            metadata = {"page": i+1, "machine": machine}
                            texts_with_metadata.append(Document(page_content=doc.page_content.strip(), metadata=metadata))
            if texts_with_metadata:
                db = Chroma.from_documents(texts_with_metadata, embeddings, persist_directory=config["index"])
                config["db"] = db
                print(f"Successfully created index for {machine} with {len(texts_with_metadata)} chunks")
            else:
                print(f"WARNING: No text chunks were created for {machine}. The index will be empty.")
                config["db"] = None
        except Exception as e:
            print(f"Error creating index for {machine}: {str(e)}")
            config["db"] = None

async def stream_response_with_fallback(retriever, user_query: str, prompt_template: str) -> AsyncIterator[str]:
    """Stream response with automatic fallback to different LLM providers on failure"""
    
    # Define fallback order: try current model first, then fallback providers
    current_provider = os.getenv("MODEL_PROVIDER", "google")
    current_type = os.getenv("MODEL_TYPE", "api")
    current_path = os.getenv("MODEL_PATH")
    
    # Define fallback chain
    if current_type == "local":
        fallback_providers = ["google", "openai", "anthropic"]  # If local fails, try API providers
    else:
        # If current API provider fails, try other API providers, then local
        all_api_providers = ["google", "openai", "anthropic"]
        fallback_providers = [p for p in all_api_providers if p != current_provider]
        
        # Add local as last resort if available
        local_models_dir = os.path.join(os.getcwd(), ".models")
        if os.path.exists(local_models_dir) and os.listdir(local_models_dir):
            fallback_providers.append("local")
    
    # Try current model first
    try:
        current_model_config = {
            "type": current_type,
            "provider": current_provider,
            "path": current_path
        }
        llm = get_llm(current_model_config)
        if llm:
            chain = create_chain(retriever, prompt_template, llm)
            yield f"🤖 Using {current_provider} model...\n\n"
            
            async for chunk in chain.astream(user_query):
                yield chunk
            return  # Success, no need to try fallbacks
    except ValueError as e:
        error_msg = str(e)
        if "finish_reason" in error_msg and "1" in error_msg:
            yield f"⚠️ {current_provider} blocked content due to safety filters. Trying fallback models...\n\n"
        elif "API key not provided" in error_msg:
            yield f"⚠️ {current_provider} API key not configured. Trying fallback models...\n\n"
        else:
            yield f"⚠️ {current_provider} error: {error_msg}. Trying fallback models...\n\n"
    except Exception as e:
        yield f"⚠️ {current_provider} failed: {str(e)}. Trying fallback models...\n\n"
    
    # Try fallback providers
    for provider in fallback_providers:
        try:
            yield f"🔄 Trying {provider}...\n"
            
            # Create model config for fallback provider
            if provider == "local":
                # Use the first available local model
                local_models_dir = os.path.join(os.getcwd(), ".models")
                if os.path.exists(local_models_dir):
                    model_files = [f for f in os.listdir(local_models_dir) if f.endswith('.gguf')]
                    if model_files:
                        fallback_config = {
                            "type": "local",
                            "provider": "local",
                            "path": os.path.join(local_models_dir, model_files[0])
                        }
                    else:
                        yield f"❌ No local models found in {local_models_dir}.\n"
                        continue
                else:
                    yield f"❌ Local models directory not found.\n"
                    continue
            else:
                fallback_config = {
                    "type": "api",
                    "provider": provider,
                    "path": None
                }
            
            llm = get_llm(fallback_config)
            if llm:
                chain = create_chain(retriever, prompt_template, llm)
                
                async for chunk in chain.astream(user_query):
                    yield chunk
                return  # Success
            else:
                yield f"❌ {provider} not available.\n"
                
        except ValueError as e:
            if "API key not provided" in str(e):
                yield f"❌ {provider} API key not configured.\n"
            else:
                yield f"❌ {provider} failed: {str(e)}\n"
        except Exception as e:
            yield f"❌ {provider} failed: {str(e)}\n"
    
    # If all providers failed
    yield "\n❌ All LLM providers failed. Please check your API keys and try again."

def create_chain(retriever, prompt_template: str, llm):
    """Create a processing chain with the given retriever, prompt template, and LLM"""
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=prompt_template
    )
    
    return (
        {
            "context": retriever,
            "query": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

async def stream_response(chain, user_query: str) -> AsyncIterator[str]:
    """Legacy function for backward compatibility"""
    try:
        async for chunk in chain.astream(user_query):
            yield chunk
    except ValueError as e:
        error_msg = str(e)
        if "finish_reason" in error_msg:
            if "1" in error_msg:  # finish_reason = 1 is STOP due to safety
                yield "⚠️ Content was blocked by safety filters. This may be due to technical terms in the manual being misinterpreted. Try:\n1. Rephrasing your query\n2. Using a different model\n3. Using a local model"
            elif "2" in error_msg:  # finish_reason = 2 is MAX_TOKENS
                yield "⚠️ Response was cut off due to length limits. Please try a more specific query."
            else:
                yield f"⚠️ An error occurred: {error_msg}"
        else:
            yield f"⚠️ An error occurred: {error_msg}"
    except Exception as e:
        yield f"⚠️ An unexpected error occurred: {str(e)}"

@app.post("/query")
async def query_documents(query: QueryRequest):
    global MACHINE_CONFIG
    
    try:
        print(f"Query received: '{query.query}' for machine: '{query.machine}'")
        
        # Get the current model configuration from environment
        model_config = {
            "type": os.getenv("MODEL_TYPE", "api"),
            "provider": os.getenv("MODEL_PROVIDER", "google"),
            "path": os.getenv("MODEL_PATH")
        }
        print(f"Model config: {model_config}")
        
        config = MACHINE_CONFIG.get(query.machine)
        if not config:
            raise HTTPException(status_code=404, detail=f"Machine '{query.machine}' not found")
        
        print(f"Machine config found: {config}")
        
        # Create index if it doesn't exist
        if not config.get("db"):
            print("Database not found, creating index...")
            await create_index_for_machine(query.machine, config)
        
        if not config.get("db"):
            raise HTTPException(status_code=500, detail=f"Failed to create or load index for {query.machine}")
        
        print("Database loaded successfully")
        
        # Get LLM for SelfQueryRetriever (we'll handle fallbacks in the response generation)
        try:
            model_config = {
                "type": os.getenv("MODEL_TYPE", "api"),
                "provider": os.getenv("MODEL_PROVIDER", "google"),
                "path": os.getenv("MODEL_PATH")
            }
            llm = get_llm(model_config)
        except Exception as e:
            print(f"Failed to create primary LLM for SelfQueryRetriever: {e}")
            # Use Google as fallback for SelfQueryRetriever if primary fails
            try:
                fallback_config = {"type": "api", "provider": "google", "path": None}
                llm = get_llm(fallback_config)
                print("Using Google LLM as fallback for SelfQueryRetriever")
            except Exception as e2:
                print(f"Fallback LLM also failed: {e2}")
                raise HTTPException(status_code=500, detail="No LLM available for query processing")
        
        # Define metadata fields for the SelfQueryRetriever
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

        # Set up the retriever - use similarity search for simple error codes, SelfQueryRetriever for complex queries
        use_self_query = False
        
        # Check if this is a simple error code query (just numbers)
        if not re.match(r'^\d+$', query.query.strip()):
            # Complex query - try SelfQueryRetriever
            try:
                retriever = SelfQueryRetriever.from_llm(
                    llm,
                    config["db"],
                    document_content_description,
                    metadata_field_info,
                    verbose=True,
                    search_kwargs={"k": 3}
                )
                use_self_query = True
                print("Using SelfQueryRetriever for complex query")
            except Exception as e:
                print(f"Error creating SelfQueryRetriever: {e}")
                retriever = config["db"].as_retriever(search_kwargs={"k": 5})
                print("Falling back to similarity search")
        else:
            # Simple error code - use similarity search directly with enhanced strategy
            print(f"Using enhanced search for error code: {query.query}")
            
            # First try: exact metadata filter for error code
            try:
                exact_docs = config["db"].similarity_search(
                    query.query, 
                    k=3,
                    filter={"error_code": query.query.strip()}
                )
                if exact_docs:
                    print(f"Found {len(exact_docs)} documents with exact error code match")
                    retriever = config["db"].as_retriever(
                        search_type="similarity",
                        search_kwargs={
                            "k": 3,
                            "filter": {"error_code": query.query.strip()}
                        }
                    )
                else:
                    print("No exact error code match, using similarity search")
                    retriever = config["db"].as_retriever(search_kwargs={"k": 5})
            except Exception as e:
                print(f"Error with filtered search: {e}, falling back to similarity search")
                retriever = config["db"].as_retriever(search_kwargs={"k": 5})
        
        # Test the retriever directly
        try:
            test_docs = retriever.get_relevant_documents(query.query)
            print(f"Direct retriever test found {len(test_docs)} documents")
            for i, doc in enumerate(test_docs):
                print(f"Test doc {i+1}: {doc.page_content[:100]}...")
                if hasattr(doc, 'metadata'):
                    print(f"  Metadata: {doc.metadata}")
            
            # If SelfQueryRetriever found nothing, try a simple similarity search
            if len(test_docs) == 0 and use_self_query:
                print("SelfQueryRetriever found no documents, trying similarity search...")
                simple_retriever = config["db"].as_retriever(search_kwargs={"k": 5})
                fallback_docs = simple_retriever.get_relevant_documents(query.query)
                print(f"Similarity search found {len(fallback_docs)} documents")
                
                # If similarity search found documents, use it instead
                if len(fallback_docs) > 0:
                    print("Using similarity search retriever instead of SelfQueryRetriever")
                    retriever = simple_retriever
                    for i, doc in enumerate(fallback_docs[:3]):
                        print(f"Fallback doc {i+1}: {doc.page_content[:100]}...")
                        if hasattr(doc, 'metadata'):
                            print(f"  Metadata: {doc.metadata}")
        except Exception as e:
            print(f"Error testing retriever: {e}")
            # Fallback to similarity search
            print("Using similarity search as fallback")
            retriever = config["db"].as_retriever(search_kwargs={"k": 5})
        
        # Get the current prompt template
        prompt_template = get_current_prompt_template()
        print(f"Using prompt template: {prompt_template[:100]}...")
        
        print("Starting to stream response with fallback support...")
        
        # Stream the response with automatic fallback
        return StreamingResponse(
            stream_response_with_fallback(retriever, query.query, prompt_template),
            media_type="text/plain"
        )
    except Exception as e:
        print(f"Error in query_documents: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Prompt template management
CURRENT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

def get_current_prompt_template():
    return CURRENT_PROMPT_TEMPLATE

def set_current_prompt_template(template: str):
    global CURRENT_PROMPT_TEMPLATE
    CURRENT_PROMPT_TEMPLATE = template

@app.get("/prompt_template")
async def get_prompt_template():
    """Get the current prompt template"""
    return {"template": get_current_prompt_template()}

@app.post("/prompt_template")
async def update_prompt_template(request: PromptTemplateRequest):
    """Update the prompt template"""
    try:
        set_current_prompt_template(request.template)
        return {"message": "Prompt template updated successfully", "template": request.template}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/error_codes/{machine_name}")
async def debug_error_codes(machine_name: str):
    """Debug endpoint to list all indexed error codes for a machine"""
    global MACHINE_CONFIG
    try:
        config = MACHINE_CONFIG.get(machine_name)
        if not config or not config.get("db"):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_name}' not found or not indexed")
        
        # Get all documents from the database
        all_docs = config["db"].get()
        
        error_codes = []
        for i, metadata in enumerate(all_docs.get("metadatas", [])):
            if metadata and "error_code" in metadata:
                error_codes.append({
                    "error_code": metadata["error_code"],
                    "content_preview": all_docs["documents"][i][:100] if i < len(all_docs["documents"]) else "No content"
                })
        
        # Sort by error code
        error_codes.sort(key=lambda x: int(x["error_code"]) if x["error_code"].isdigit() else 9999)
        
        return {
            "machine": machine_name,
            "total_error_codes": len(error_codes),
            "error_codes": error_codes[:50],  # Limit to first 50 for readability
            "has_1083": any(ec["error_code"] == "1083" for ec in error_codes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
