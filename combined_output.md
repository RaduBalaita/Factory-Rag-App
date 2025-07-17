

# backend/app/main.py

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
import gc
import psutil
from cachetools import LRUCache

# Prompt cache (in-memory, autoclears on app close)
prompt_cache = LRUCache(maxsize=1000)

# Helper function for streaming a string as an async iterator
async def string_to_async_iterator(s: str) -> AsyncIterator[str]:
    """Helper to turn a complete string into a single-item async stream."""
    yield s

def get_cache_key(prompt, model, system_prompt):
    return f"{model}:{system_prompt}:{prompt}"

# Set the API key
load_dotenv()  # Load environment variables from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Note: API key is now optional since it can be configured via UI
if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not found in environment. You can configure API keys via the UI.")
else:
    print("✅ GEMINI_API_KEY loaded from environment.")
    
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

class ModelConfig(BaseModel):
    type: str
    provider: Optional[str] = None
    path: Optional[str] = None
    api_key: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    machine: str
    model_settings: ModelConfig
    language: Optional[str] = None
    system_prompt: Optional[str] = None

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
        # Validate file type and set folder
        ext_map = {
            '.pdf': 'PDF',
            '.md': 'MD',
            '.markdown': 'MD',
            '.docx': 'DOCX',
            '.txt': 'TXT',
            '.doc': 'DOC',
            '.rtf': 'RTF',
            '.odt': 'ODT',
            '.fodt': 'LIBRE',
            '.ott': 'LIBRE',
        }
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ext_map:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        folder = ext_map[file_ext]
        # Create safe filename
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        filename = f"{safe_name}{file_ext}"
        doc_path = os.path.join(DOCS_PATH, folder, filename)
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
        
        # Clear the database object from memory to release file handles
        if "db" in config and config.get("db"):
            try:
                # More thorough cleanup for Chroma database
                db = config["db"]
                
                # Try to close any active connections
                if hasattr(db, '_client') and db._client:
                    try:
                        db._client.reset()
                    except Exception:
                        pass
                    db._client = None
                
                if hasattr(db, '_collection') and db._collection:
                    db._collection = None
                
                # Delete the entire object
                del db
                
            except Exception as e:
                print(f"Error during database cleanup: {e}")
            
            config["db"] = None
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            print("Cleared database object from memory")
        
        # Also clean up any other references in the global config
        for machine_name, machine_config in MACHINE_CONFIG.items():
            if machine_name != target_name and machine_config.get("index") == config.get("index"):
                machine_config["db"] = None
        
        # Delete document file
        if "doc" in config and os.path.exists(config["doc"]):
            os.remove(config["doc"])
            print(f"Deleted document file: {config['doc']}")
        
        # Delete index directory with Windows file handle management
        if "index" in config and os.path.exists(config["index"]):
            import time
            import stat
            import psutil
            import sys
            
            def remove_readonly(func, path, _):
                """Clear the readonly bit and reattempt the removal"""
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass
            
            def close_file_handles(directory_path):
                """Close file handles that might be locking the directory"""
                try:
                    current_process = psutil.Process()
                    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                        try:
                            if proc.info['open_files']:
                                for file_info in proc.info['open_files']:
                                    if directory_path.lower() in file_info.path.lower():
                                        print(f"Found process {proc.info['name']} (PID: {proc.info['pid']}) holding file: {file_info.path}")
                                        if proc.info['pid'] != current_process.pid:
                                            proc.terminate()
                                            proc.wait(timeout=3)
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                            pass
                except Exception as e:
                    print(f"Could not close file handles: {e}")
            
            # Wait for file handles to be released
            time.sleep(3)
            
            try:
                shutil.rmtree(config["index"], onerror=remove_readonly)
                print(f"Deleted index directory: {config['index']}")
            except Exception as e:
                print(f"Error deleting index directory: {e}")
                
                # Try to close any file handles holding the directory
                close_file_handles(config["index"])
                time.sleep(2)
                
                # Try again after closing handles
                try:
                    shutil.rmtree(config["index"], onerror=remove_readonly)
                    print(f"Successfully deleted index directory after closing handles")
                except Exception as e2:
                    print(f"Still couldn't delete after closing handles: {e2}")
                    
                    # Final attempt: delete files one by one with aggressive retries
                    deleted_successfully = True
                    try:
                        for root, dirs, files in os.walk(config["index"], topdown=False):
                            # Delete all files in this directory
                            for file in files:
                                filepath = os.path.join(root, file)
                                for attempt in range(10):  # More attempts
                                    try:
                                        # Make file writable and delete
                                        if os.path.exists(filepath):
                                            os.chmod(filepath, stat.S_IWRITE)
                                            os.remove(filepath)
                                            break
                                    except Exception as file_error:
                                        if attempt == 9:  # Last attempt
                                            print(f"Failed to delete file {filepath}: {file_error}")
                                            deleted_successfully = False
                                        time.sleep(0.5)  # Short wait between attempts
                            
                            # Delete empty directories
                            for dir in dirs:
                                dirpath = os.path.join(root, dir)
                                for attempt in range(5):
                                    try:
                                        if os.path.exists(dirpath):
                                            os.rmdir(dirpath)
                                            break
                                    except Exception:
                                        if attempt == 4:
                                            deleted_successfully = False
                                        time.sleep(0.5)
                        
                        # Finally, try to delete the root directory
                        for attempt in range(5):
                            try:
                                if os.path.exists(config["index"]):
                                    os.rmdir(config["index"])
                                    print(f"Successfully deleted index directory after manual cleanup")
                                    break
                            except Exception:
                                if attempt == 4:
                                    deleted_successfully = False
                                time.sleep(1)
                        
                        if not deleted_successfully:
                            raise Exception("Could not delete all files/directories")
                            
                    except Exception as e3:
                        print(f"Manual cleanup failed: {e3}")
                        # As last resort, just mark it for cleanup and continue
                        try:
                            import uuid
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            temp_name = f"{config['index']}_CLEANUP_NEEDED_{timestamp}"
                            if os.path.exists(config["index"]):
                                os.rename(config["index"], temp_name)
                                print(f"❌ Could not delete directory. Renamed to: {temp_name}")
                                print("⚠️  Please manually delete this directory when convenient")
                        except Exception as e4:
                            print(f"❌ Complete cleanup failure. Directory may need manual deletion: {config['index']}")
                            print(f"Error: {e4}")
        
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
        # Return empty configuration if file doesn't exist
        # Machines should only be added through the API upload endpoint
        print("No machine configuration file found. Starting with empty configuration.")
        return {}

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

from langchain.text_splitter import RecursiveCharacterTextSplitter

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

        # Use the better chunking strategy from mainold.py for error code documents
        full_text = "\n".join([doc.page_content for doc in documents])
        
        # Pattern to split by error codes (3+ digits followed by quotes)
        # Updated to handle different quote characters including curly quotes
        pattern = r"\n(\d{3,})\s*['''\"\u2018\u2019\u201C\u201D]"
        split_text = re.split(pattern, full_text)
        texts_with_metadata = []
        
        print(f"Split text into {len(split_text)} parts")
        
        if len(split_text) > 1:
            # Debug: let's see what we get
            for i in range(1, len(split_text), 2):
                if i < len(split_text):
                    error_code = split_text[i].strip()
                    content = split_text[i+1].strip() if (i + 1) < len(split_text) else ""
                    
                    if error_code == "1083":
                        print(f"DEBUG: Found 1083 at index {i}, content starts with: {content[:100]}")
                    
                    full_chunk_content = f"{error_code} '{content}"
                    
                    # Clean up the content exactly like mainold.py
                    full_chunk_content = re.sub(r"·M· Model Ref\\. \\d+", "", full_chunk_content)
                    full_chunk_content = re.sub(r"Error solution", "", full_chunk_content)
                    full_chunk_content = re.sub(r"CNC \\d+ ·M·", "", full_chunk_content)
                    full_chunk_content = full_chunk_content.replace("", "")
                    full_chunk_content = re.sub(r'\\s+', ' ', full_chunk_content).strip()
                    
                    if full_chunk_content and error_code:
                        metadata = {"error_code": error_code, "machine": machine_name}
                        doc = Document(page_content=full_chunk_content, metadata=metadata)
                        texts_with_metadata.append(doc)
        else:
            # Fallback to regular chunking if no error codes found
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            texts_with_metadata = text_splitter.split_documents(documents)
            
        print(f"Created {len(texts_with_metadata)} text chunks")
        
        if texts_with_metadata:
            db = Chroma.from_documents(texts_with_metadata, embeddings, persist_directory=config["index"])
            db.persist()  # Add persist() call like in mainold.py
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
            n_ctx=3072,  # Reduced from 4096 to prevent context issues
            f16_kv=True,
            verbose=False,
        )
    elif model_config['type'] == 'api':
        provider = model_config.get('provider', 'google').lower()
        if provider == 'google':
            # Use API key from model config or fallback to environment
            api_key = model_config.get('api_key') or os.getenv("GEMINI_API_KEY")
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
            # Use API key from model config or fallback to environment
            api_key = model_config.get('api_key') or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=api_key,
                temperature=0.7
            )
        elif provider == 'anthropic':
            # Use API key from model config or fallback to environment
            api_key = model_config.get('api_key') or os.getenv("ANTHROPIC_API_KEY")
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
        # Use the async function we already created
        import asyncio
        try:
            # Run the async function synchronously during startup
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(create_index_for_machine(machine, config))
            loop.close()
        except Exception as e:
            print(f"Error creating index for {machine}: {str(e)}")
            config["db"] = None

async def stream_response_with_fallback(retriever, user_query: str, prompt_template: str, model_config: Dict[str, Any]) -> AsyncIterator[str]:
    """Stream response with automatic fallback to different LLM providers on failure"""
    
    # Define fallback order: try current model first, then fallback providers
    current_type = model_config.get("type", "api")
    
    # Fix provider detection for local models
    if current_type == "local":
        current_provider = "local"
    else:
        current_provider = model_config.get("provider", "google")
    
    # Define fallback chain with LOCAL MODELS AS PREFERRED FALLBACK
    local_models_dir = os.path.join(os.path.dirname(__file__), "..", "..", ".models")
    local_available = os.path.exists(local_models_dir) and os.listdir(local_models_dir)
    
    if current_type == "local":
        # If current model is local, fallback to API providers
        fallback_providers = ["google", "openai", "anthropic"]
    else:
        # If current model is API, prioritize LOCAL as first fallback for privacy/cost benefits
        fallback_providers = []
        
        # Add local as FIRST fallback if available
        if local_available:
            fallback_providers.append("local")
        
        # Then add other API providers as secondary fallbacks
        all_api_providers = ["google", "openai", "anthropic"]
        fallback_providers.extend([p for p in all_api_providers if p != current_provider])
    
    # Try current model first
    try:
        llm = get_llm(model_config)
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
                local_models_dir = os.path.join(os.path.dirname(__file__), "..", "..", ".models")
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
                    "path": None,
                    "api_key": os.getenv("GEMINI_API_KEY") if provider == "google" else (
                        os.getenv("OPENAI_API_KEY") if provider == "openai" else 
                        os.getenv("ANTHROPIC_API_KEY") if provider == "anthropic" else None
                    )
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
        model_config = query.model_settings.dict()
        config = MACHINE_CONFIG.get(query.machine)
        if not config:
            raise HTTPException(status_code=404, detail=f"Machine '{query.machine}' not found")

        # --- 1. CACHE CHECK ---
        cache_key = get_cache_key(query.query, model_config.get('type', ''), query.system_prompt or '')
        cached_response = prompt_cache.get(cache_key)
        if cached_response:
            print(f"✅ Cache hit for key: {cache_key}")
            # ALWAYS return a StreamingResponse, even for a cached result
            return StreamingResponse(
                string_to_async_iterator(cached_response),
                media_type="text/plain"
            )

        print(f"❌ Cache miss for key: {cache_key}. Generating new response.")

        # --- 2. DATABASE & RETRIEVER SETUP (Unchanged) ---
        if not config.get("db"):
            await create_index_for_machine(query.machine, config)
        if not config.get("db"):
            raise HTTPException(status_code=500, detail=f"Failed to create or load index for {query.machine}")

        error_code_pattern = r'^\d{3,}$'
        is_error_code = re.match(error_code_pattern, query.query.strip())

        if is_error_code:
            retriever = config["db"].as_retriever(search_kwargs={"k": 1, "filter": {"error_code": query.query.strip()}})
            if not retriever.get_relevant_documents(query.query):
                retriever = config["db"].as_retriever(search_kwargs={"k": 2})
        else:
            retriever = config["db"].as_retriever(search_kwargs={"k": 3})

        prompt_template = get_current_prompt_template()

        # --- 3. GENERATE, CACHE, AND STREAM ---
        # Generate the full response first by consuming the stream/generator
        response_generator = stream_response_with_fallback(retriever, query.query, prompt_template, model_config)
        full_response_chunks = [chunk async for chunk in response_generator]
        full_response = "".join(full_response_chunks)

        # Now, save the complete response to the cache
        if full_response.strip():
            print(f"💾 Saving response to cache with key: {cache_key}")
            prompt_cache[cache_key] = full_response

        # Finally, stream the fully-formed response back to the client
        return StreamingResponse(
            string_to_async_iterator(full_response),
            media_type="text/plain"
        )

    except Exception as e:
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
            "has_1083": any(ec["error_code"] == "1083" for ec in error_codes),
            "has_1056": any(ec["error_code"] == "1056" for ec in error_codes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/search_error/{machine_name}/{error_code}")
async def search_specific_error_code(machine_name: str, error_code: str):
    """Debug endpoint to search for a specific error code"""
    global MACHINE_CONFIG
    try:
        config = MACHINE_CONFIG.get(machine_name)
        if not config or not config.get("db"):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_name}' not found or not indexed")
        
        # Get all documents from the database
        all_docs = config["db"].get()
        
        found_codes = []
        for i, metadata in enumerate(all_docs.get("metadatas", [])):
            if metadata and "error_code" in metadata and metadata["error_code"] == error_code:
                found_codes.append({
                    "error_code": metadata["error_code"],
                    "full_content": all_docs["documents"][i] if i < len(all_docs["documents"]) else "No content",
                    "metadata": metadata
                })
        
        return {
            "machine": machine_name,
            "search_code": error_code,
            "found": len(found_codes) > 0,
            "results": found_codes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


---


# frontend\src\App.css

.App {
  display: flex;
  height: 100vh;
  background-color: #f0f0f0;
  overflow: hidden;
}

.App.dark {
    background-color: #333;
    color: #fff;
}

.main-content {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    transition: margin-left 0.3s ease-in-out, margin-right 0.3s ease-in-out;
    z-index: 1;
}

.App.machine-sidebar-open .main-content {
    margin-left: 250px;
}

.App.settings-sidebar-open .main-content {
    margin-right: 250px;
}

.top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  background-color: #e0e0e0;
}

.App.dark .top-bar {
    background-color: #222;
}

.top-bar button {
  font-size: 24px;
  background: none;
  border: none;
  cursor: pointer;
  color: #000;
}

.App.dark .top-bar button {
    color: #fff;
}

.top-bar .machine-name {
    font-weight: bold;
}

.sidebar {
  position: fixed;
  top: 0;
  height: 100%;
  width: 250px;
  background-color: #fff;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease-in-out;
  z-index: 100;
}

.App.dark .sidebar {
    background-color: #444;
    color: #fff;
}

.machine-sidebar {
  left: 0;
  transform: translateX(-100%);
}

.settings-sidebar {
  right: 0;
  transform: translateX(100%);
}

.sidebar.open {
  transform: translateX(0);
}

.chat-window {
  flex-grow: 1;
  padding: 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.message {
  max-width: 70%;
  padding: 10px 15px;
  border-radius: 15px;
  margin-bottom: 10px;
  line-height: 1.4;
}

.message.bot {
  background-color: #e0e0e0;
  align-self: flex-start;
  border-bottom-left-radius: 2px;
  color: #000;
}

.message.user {
  background-color: #007bff;
  color: white;
  align-self: flex-end;
  border-bottom-right-radius: 2px;
}

.input-bar {
  display: flex;
  padding: 10px;
  background-color: #e0e0e0;
}

.App.dark .input-bar {
    background-color: #222;
}

.input-bar input {
  flex-grow: 1;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  font-size: inherit;
}

.App.dark .input-bar input {
    background-color: #555;
    color: #fff;
    border-color: #666;
}

.input-bar button {
  padding: 10px 20px;
  border: none;
  background-color: #007bff;
  color: white;
  border-radius: 5px;
  cursor: pointer;
}

.input-bar .clear-btn {
    margin-right: 10px;
}

.input-bar .send-btn {
    margin-left: 10px;
}

.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 200;
}

.modal {
    background-color: #fff;
    padding: 20px;
    border-radius: 5px;
    width: 500px;
}

.App.dark .modal {
    background-color: #444;
}

.modal textarea {
    width: 100%;
    margin-bottom: 10px;
}

.sidebar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 15px;
  border-bottom: 1px solid #e0e0e0;
}

.App.dark .sidebar-header {
    border-bottom-color: #555;
}

.sidebar-header h3 {
    margin: 0;
}

.close-btn {
    font-size: 24px;
    background: none;
    border: none;
    cursor: pointer;
}

.sidebar-content {
    padding: 15px;
}

.machine-item, .setting-item {
    padding: 15px;
    margin-bottom: 10px;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.App.dark .machine-item, .App.dark .setting-item {
    border-color: #555;
}

.machine-item:hover, .setting-item:hover {
    background-color: #f0f0f0;
}

.App.dark .machine-item:hover, .App.dark .setting-item:hover {
    background-color: #555;
}

.tab-buttons {
    display: flex;
    margin-bottom: 15px;
}

.tab-buttons button {
    flex: 1;
    padding: 10px;
    border: 1px solid #ccc;
    background-color: #f0f0f0;
    cursor: pointer;
}

.tab-buttons button.active {
    background-color: #fff;
    border-bottom-color: #fff;
}

.App.dark .tab-buttons button {
    background-color: #333;
    border-color: #555;
}

.App.dark .tab-buttons button.active {
    background-color: #444;
    border-bottom-color: #444;
}

.tab-content {
    padding: 15px;
    border: 1px solid #ccc;
    border-top: none;
}

.App.dark .tab-content {
    border-color: #555;
}

.api-key-input {
    display: flex;
    flex-direction: column;
    margin-bottom: 10px;
}

.api-key-input label {
    margin-bottom: 5px;
}

.api-key-input input {
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
}

.App.dark .api-key-input input {
    background-color: #555;
    color: #fff;
    border-color: #666;
}

.local-model-list {
    max-height: 150px;
    overflow-y: auto;
    border: 1px solid #ccc;
    padding: 5px;
    margin-top: 10px;
}

.local-model-item {
    padding: 8px;
    cursor: pointer;
}

.local-model-item:hover {
    background-color: #f0f0f0;
}

.local-model-item.selected {
    background-color: #007bff;
    color: white;
}

.App.dark .local-model-list {
    border-color: #555;
}

.App.dark .local-model-item:hover {
    background-color: #555;
}

.App.dark .local-model-item.selected {
    background-color: #007bff;
}

/* Manage Documents Modal */
.manage-documents-modal {
    max-width: 600px;
    max-height: 80vh;
    overflow-y: auto;
}

.upload-section {
    border-bottom: 1px solid #ddd;
    padding-bottom: 20px;
    margin-bottom: 20px;
}

.App.dark .upload-section {
    border-bottom-color: #555;
}

.form-group {
    margin-bottom: 15px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

.form-group input {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.App.dark .form-group input {
    background-color: #555;
    border-color: #666;
    color: #fff;
}

.file-name {
    display: block;
    margin-top: 5px;
    font-style: italic;
    color: #666;
}

.App.dark .file-name {
    color: #aaa;
}

.upload-btn {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
}

.upload-btn:disabled {
    background-color: #ccc;
    cursor: not-allowed;
}

.App.dark .upload-btn:disabled {
    background-color: #555;
}

.documents-list {
    max-height: 200px;
    overflow-y: auto;
}

.document-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    margin-bottom: 10px;
}

.App.dark .document-item {
    border-color: #555;
    background-color: #444;
}

.machine-name {
    flex-grow: 1;
    margin-right: 10px;
}

.delete-btn {
    background-color: #dc3545;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 4px;
    cursor: pointer;
}

.delete-btn:hover {
    background-color: #c82333;
}

.message {
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 15px;
}

.message.success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.message.error {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.App.dark .message.success {
    background-color: #2d5a3d;
    color: #a3d3a3;
    border-color: #4d7c5d;
}

.App.dark .message.error {
    background-color: #5a2d2d;
    color: #d3a3a3;
    border-color: #7c4d4d;
}


---


# frontend\src\App.js

import React, { useState, useEffect } from 'react';
import './App.css';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import MachineSidebar from './components/MachineSidebar';
import SettingsSidebar from './components/SettingsSidebar';
import SystemPromptModal from './components/SystemPromptModal';
import ChangeModelModal from './components/ChangeModelModal';
import ManageDocumentsModal from './components/ManageDocumentsModal';

const translations = {
    en: {
        thinking: 'Thinking...',
    },
    ro: {
        thinking: 'Gândire...',
    },
};

function App() {
    const [isMachineSidebarOpen, setMachineSidebarOpen] = useState(false);
    const [isSettingsSidebarOpen, setSettingsSidebarOpen] = useState(false);
    const [isSystemPromptModalOpen, setSystemPromptModalOpen] = useState(false);
    const [isChangeModelModalOpen, setChangeModelModalOpen] = useState(false);
    const [isManageDocumentsModalOpen, setManageDocumentsModalOpen] = useState(false);

    const [machine, setMachine] = useState(() => localStorage.getItem('machine') || 'No machine selected');
    const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
    const [language, setLanguage] = useState(() => localStorage.getItem('language') || 'en');
    const [fontSize, setFontSize] = useState(() => parseInt(localStorage.getItem('fontSize'), 10) || 16);
    const [promptTemplate, setPromptTemplate] = useState(`You will be given context from a machine's technical manual and an error code.
Structure your response in three distinct sections:
1. **Problem Description:** Briefly explain what the error code means based on the context.
2. **Probable Causes:** List the most likely reasons for this error from the context.
3. **Solution Steps:** Provide a clear, step-by-step guide to fix the issue using only information from the context.

**IMPORTANT:** Only use information from the provided context. If the context is empty or does not contain specific information for the queried error code, state that "No specific information was found for this error code in the manual."`);
    const [modelConfig, setModelConfig] = useState(() => {
        const savedConfig = localStorage.getItem('modelConfig');
        return savedConfig ? JSON.parse(savedConfig) : { type: 'api', provider: 'google' };
    });
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const t = translations[language];


    useEffect(() => {
        const fetchPromptTemplate = async () => {
            try {
                const res = await fetch('http://127.0.0.1:8000/prompt_template');
                const data = await res.json();
                setPromptTemplate(data.template);
            } catch (error) {
                console.error("Failed to fetch prompt template:", error);
                // Set a fallback template if the fetch fails
                setPromptTemplate('You are a helpful assistant.');
            }
        };
        fetchPromptTemplate();
    }, []);

    useEffect(() => {
        localStorage.setItem('machine', machine);
        localStorage.setItem('theme', theme);
        localStorage.setItem('language', language);
        localStorage.setItem('fontSize', fontSize);
        localStorage.setItem('promptTemplate', promptTemplate);
        localStorage.setItem('modelConfig', JSON.stringify(modelConfig));
    }, [machine, theme, language, fontSize, promptTemplate, modelConfig]);

    const handleSendMessage = async (query) => {
    if (machine === 'No machine selected') {
        setMessages(prev => [...prev, 
            { text: query, sender: 'user' },
            { text: 'Please select a machine from the sidebar first.', sender: 'bot' }
        ]);
        return;
    }

    const userMessage = { text: query, sender: 'user' };
    
    // Step 1: Add the user message and the "Thinking..." placeholder.
    const thinkingPlaceholder = { text: t.thinking, sender: 'bot' };
    // Create an index to track which message we need to update.
    const botMessageIndex = messages.length + 1; 
    setMessages(prev => [...prev, userMessage, thinkingPlaceholder]);
    
    setLoading(true);

    try {
        const res = await fetch('http://127.0.0.1:8000/query', { // Corrected IP
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                machine, query, language, 
                system_prompt: promptTemplate,
                model_settings: modelConfig 
            }),
        });
        
        if (!res.ok) {
            const errorData = await res.json();
            throw new Error(errorData.detail || 'The server returned an error.');
        }

        // Step 2: Read the ENTIRE response from the stream at once.
        // Since the backend sends it all in one go now, this is efficient.
        const fullResponseText = await res.text();

        // Step 3: Replace the "Thinking..." message with the final, complete response.
        setMessages(prev => {
            const newMessages = [...prev];
            // Find the message at our specific index and update its text.
            newMessages[botMessageIndex] = { ...newMessages[botMessageIndex], text: fullResponseText };
            return newMessages;
        });

    } catch (error) {
        // If an error occurs, replace the "Thinking..." message with the error.
        setMessages(prev => {
            const newMessages = [...prev];
            newMessages[botMessageIndex] = { ...newMessages[botMessageIndex], text: `An error occurred: ${error.message}` };
            return newMessages;
        });
    } finally {
        setLoading(false);
    }
};

    const handleClearChat = () => {
        setMessages([]);
    };

    const handleContentClick = () => {
        if (isMachineSidebarOpen) setMachineSidebarOpen(false);
        if (isSettingsSidebarOpen) setSettingsSidebarOpen(false);
        if (isSystemPromptModalOpen) setSystemPromptModalOpen(false);
        if (isChangeModelModalOpen) setChangeModelModalOpen(false);
        if (isManageDocumentsModalOpen) setManageDocumentsModalOpen(false);
    };

    const handlePromptTemplateSave = async () => {
        try {
            await fetch('http://127.0.0.1:8000/prompt_template', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ template: promptTemplate }),
            });
        } catch (error) {
            console.error('Failed to save prompt template:', error);
        }
        setSystemPromptModalOpen(false);
    };

    return (
        <div className={`App ${theme} ${isMachineSidebarOpen ? 'machine-sidebar-open' : ''} ${isSettingsSidebarOpen ? 'settings-sidebar-open' : ''}`} style={{ fontSize: `${fontSize}px` }}>
            <MachineSidebar isOpen={isMachineSidebarOpen} onClose={() => setMachineSidebarOpen(false)} setMachine={setMachine} />
            <SettingsSidebar 
                isOpen={isSettingsSidebarOpen} 
                onClose={() => setSettingsSidebarOpen(false)} 
                theme={theme} 
                setTheme={setTheme} 
                language={language} 
                setLanguage={setLanguage} 
                fontSize={fontSize}
                setFontSize={setFontSize}
                openSystemPrompt={() => setSystemPromptModalOpen(true)} 
                openChangeModel={() => setChangeModelModalOpen(true)}
                openManageDocuments={() => setManageDocumentsModalOpen(true)}
                modelConfig={modelConfig}
            />
            <div className="main-content" onClick={handleContentClick}>
                <div className="top-bar">
                    <button onClick={(e) => { e.stopPropagation(); setMachineSidebarOpen(true); }}>&#9776;</button>
                    <span className="machine-name">{machine}</span>
                    <button onClick={(e) => { e.stopPropagation(); setSettingsSidebarOpen(true); }}>&#9881;</button>
                </div>
                <ChatWindow messages={messages} />
                <InputBar onSendMessage={handleSendMessage} onClearChat={handleClearChat} language={language} />
            </div>
            <SystemPromptModal 
                isOpen={isSystemPromptModalOpen} 
                onClose={() => setSystemPromptModalOpen(false)} 
                systemPrompt={promptTemplate} 
                setSystemPrompt={setPromptTemplate} 
                onSave={handlePromptTemplateSave}
            />
            <ChangeModelModal
                isOpen={isChangeModelModalOpen}
                onClose={() => setChangeModelModalOpen(false)}
                language={language}
                modelConfig={modelConfig}
                setModelConfig={setModelConfig}
            />
            <ManageDocumentsModal
                isOpen={isManageDocumentsModalOpen}
                onClose={() => setManageDocumentsModalOpen(false)}
                language={language}
                onDocumentChange={() => {
                    // Trigger a refresh of the machine list in MachineSidebar
                    // This could be improved with a more sophisticated state management
                }}
            />
        </div>
    );
}

export default App;


---


# frontend\src\App.test.js

import { render, screen } from '@testing-library/react';
import App from './App';

test('renders learn react link', () => {
  render(<App />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});


---


# frontend\src\index.css

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}


---


# frontend\src\index.js

import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();


---


# frontend\src\logo.svg

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 841.9 595.3"><g fill="#61DAFB"><path d="M666.3 296.5c0-32.5-40.7-63.3-103.1-82.4 14.4-63.6 8-114.2-20.2-130.4-6.5-3.8-14.1-5.6-22.4-5.6v22.3c4.6 0 8.3.9 11.4 2.6 13.6 7.8 19.5 37.5 14.9 75.7-1.1 9.4-2.9 19.3-5.1 29.4-19.6-4.8-41-8.5-63.5-10.9-13.5-18.5-27.5-35.3-41.6-50 32.6-30.3 63.2-46.9 84-46.9V78c-27.5 0-63.5 19.6-99.9 53.6-36.4-33.8-72.4-53.2-99.9-53.2v22.3c20.7 0 51.4 16.5 84 46.6-14 14.7-28 31.4-41.3 49.9-22.6 2.4-44 6.1-63.6 11-2.3-10-4-19.7-5.2-29-4.7-38.2 1.1-67.9 14.6-75.8 3-1.8 6.9-2.6 11.5-2.6V78.5c-8.4 0-16 1.8-22.6 5.6-28.1 16.2-34.4 66.7-19.9 130.1-62.2 19.2-102.7 49.9-102.7 82.3 0 32.5 40.7 63.3 103.1 82.4-14.4 63.6-8 114.2 20.2 130.4 6.5 3.8 14.1 5.6 22.5 5.6 27.5 0 63.5-19.6 99.9-53.6 36.4 33.8 72.4 53.2 99.9 53.2 8.4 0 16-1.8 22.6-5.6 28.1-16.2 34.4-66.7 19.9-130.1 62-19.1 102.5-49.9 102.5-82.3zm-130.2-66.7c-3.7 12.9-8.3 26.2-13.5 39.5-4.1-8-8.4-16-13.1-24-4.6-8-9.5-15.8-14.4-23.4 14.2 2.1 27.9 4.7 41 7.9zm-45.8 106.5c-7.8 13.5-15.8 26.3-24.1 38.2-14.9 1.3-30 2-45.2 2-15.1 0-30.2-.7-45-1.9-8.3-11.9-16.4-24.6-24.2-38-7.6-13.1-14.5-26.4-20.8-39.8 6.2-13.4 13.2-26.8 20.7-39.9 7.8-13.5 15.8-26.3 24.1-38.2 14.9-1.3 30-2 45.2-2 15.1 0 30.2.7 45 1.9 8.3 11.9 16.4 24.6 24.2 38 7.6 13.1 14.5 26.4 20.8 39.8-6.3 13.4-13.2 26.8-20.7 39.9zm32.3-13c5.4 13.4 10 26.8 13.8 39.8-13.1 3.2-26.9 5.9-41.2 8 4.9-7.7 9.8-15.6 14.4-23.7 4.6-8 8.9-16.1 13-24.1zM421.2 430c-9.3-9.6-18.6-20.3-27.8-32 9 .4 18.2.7 27.5.7 9.4 0 18.7-.2 27.8-.7-9 11.7-18.3 22.4-27.5 32zm-74.4-58.9c-14.2-2.1-27.9-4.7-41-7.9 3.7-12.9 8.3-26.2 13.5-39.5 4.1 8 8.4 16 13.1 24 4.7 8 9.5 15.8 14.4 23.4zM420.7 163c9.3 9.6 18.6 20.3 27.8 32-9-.4-18.2-.7-27.5-.7-9.4 0-18.7.2-27.8.7 9-11.7 18.3-22.4 27.5-32zm-74 58.9c-4.9 7.7-9.8 15.6-14.4 23.7-4.6 8-8.9 16-13 24-5.4-13.4-10-26.8-13.8-39.8 13.1-3.1 26.9-5.8 41.2-7.9zm-90.5 125.2c-35.4-15.1-58.3-34.9-58.3-50.6 0-15.7 22.9-35.6 58.3-50.6 8.6-3.7 18-7 27.7-10.1 5.7 19.6 13.2 40 22.5 60.9-9.2 20.8-16.6 41.1-22.2 60.6-9.9-3.1-19.3-6.5-28-10.2zM310 490c-13.6-7.8-19.5-37.5-14.9-75.7 1.1-9.4 2.9-19.3 5.1-29.4 19.6 4.8 41 8.5 63.5 10.9 13.5 18.5 27.5 35.3 41.6 50-32.6 30.3-63.2 46.9-84 46.9-4.5-.1-8.3-1-11.3-2.7zm237.2-76.2c4.7 38.2-1.1 67.9-14.6 75.8-3 1.8-6.9 2.6-11.5 2.6-20.7 0-51.4-16.5-84-46.6 14-14.7 28-31.4 41.3-49.9 22.6-2.4 44-6.1 63.6-11 2.3 10.1 4.1 19.8 5.2 29.1zm38.5-66.7c-8.6 3.7-18 7-27.7 10.1-5.7-19.6-13.2-40-22.5-60.9 9.2-20.8 16.6-41.1 22.2-60.6 9.9 3.1 19.3 6.5 28.1 10.2 35.4 15.1 58.3 34.9 58.3 50.6-.1 15.7-23 35.6-58.4 50.6zM320.8 78.4z"/><circle cx="420.9" cy="296.5" r="45.7"/><path d="M520.5 78.1z"/></g></svg>

---


# frontend\src\reportWebVitals.js

const reportWebVitals = onPerfEntry => {
  if (onPerfEntry && onPerfEntry instanceof Function) {
    import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
      getCLS(onPerfEntry);
      getFID(onPerfEntry);
      getFCP(onPerfEntry);
      getLCP(onPerfEntry);
      getTTFB(onPerfEntry);
    });
  }
};

export default reportWebVitals;


---


# frontend\src\setupTests.js

// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';


---


# frontend\src\components\ChangeModelModal.js

import React, { useState, useEffect } from 'react';

const translations = {
    en: {
        changeModel: 'Change Model',
        cloudAPIs: 'Cloud APIs',
        localModels: 'Local Models',
        googleAPIKey: 'Google API Key',
        openaiAPIKey: 'OpenAI API Key',
        claudeAPIKey: 'Claude API Key',
        save: 'Save',
        browse: 'Browse',
        currentLocalModel: 'Current Local Model',
        noModelSelected: 'No model selected',
        availableModels: 'Available Models',
    },
    ro: {
        changeModel: 'Schimbă Modelul',
        cloudAPIs: 'API-uri Cloud',
        localModels: 'Modele Locale',
        googleAPIKey: 'Cheie API Google',
        openaiAPIKey: 'Cheie API OpenAI',
        claudeAPIKey: 'Cheie API Claude',
        save: 'Salvează',
        browse: 'Răsfoiește',
        currentLocalModel: 'Model Local Actual',
        noModelSelected: 'Niciun model selectat',
        availableModels: 'Modele Disponibile',
    }
};

const ChangeModelModal = ({ isOpen, onClose, language, modelConfig, setModelConfig }) => {
    const [activeTab, setActiveTab] = useState(modelConfig.type === 'local' ? 'local' : 'cloud');
    const [cloudProvider, setCloudProvider] = useState(modelConfig.provider || 'google');
    const [apiKeys, setApiKeys] = useState({
        google: modelConfig.provider === 'google' ? modelConfig.api_key || '' : '',
        openai: modelConfig.provider === 'openai' ? modelConfig.api_key || '' : '',
        claude: modelConfig.provider === 'claude' ? modelConfig.api_key || '' : ''
    });
    const [localModelPath, setLocalModelPath] = useState(modelConfig.type === 'local' ? modelConfig.path : '');
    const [availableLocalModels, setAvailableLocalModels] = useState([]);

    const t = translations[language];

    useEffect(() => {
        // Update internal state if the modal is reopened with new props
        if (isOpen) {
            setActiveTab(modelConfig.type === 'local' ? 'local' : 'cloud');
            setCloudProvider(modelConfig.provider || 'google');
            // Load the correct API key for the current provider
            const newApiKeys = { ...apiKeys };
            if (modelConfig.provider && modelConfig.api_key) {
                newApiKeys[modelConfig.provider] = modelConfig.api_key;
            }
            setApiKeys(newApiKeys);
            setLocalModelPath(modelConfig.type === 'local' ? modelConfig.path : '');
        }
    }, [isOpen, modelConfig]);

    useEffect(() => {
        const fetchLocalModels = async () => {
            try {
                const res = await fetch('http://127.0.0.1:8000/api/models');
                const data = await res.json();
                if (data.models) {
                    setAvailableLocalModels(data.models);
                }
            } catch (error) {
                console.error("Failed to fetch local models:", error);
            }
        };

        if (isOpen && activeTab === 'local') {
            fetchLocalModels();
        }
    }, [isOpen, activeTab]);

    const handleFileChange = (event) => {
        if (event.target.files && event.target.files[0]) {
            // Note: Accessing the full path of a file selected by a user is a security restriction in modern browsers.
            // We can only get the file name. The backend will need to know the base path to the models.
            setLocalModelPath(event.target.files[0].name);
        }
    };
    
    const handleSave = () => {
        if (activeTab === 'cloud') {
            setModelConfig({ 
                type: 'api', 
                provider: cloudProvider, 
                api_key: apiKeys[cloudProvider] 
            });
        } else {
            // Prepend the base path for the backend
            const fullPath = `C:/Users/Tempest/Desktop/RAG/.models/${localModelPath}`;
            setModelConfig({ type: 'local', path: fullPath });
        }
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
                <h3>{t.changeModel}</h3>
                <div className="tab-buttons">
                    <button onClick={() => setActiveTab('cloud')} className={activeTab === 'cloud' ? 'active' : ''}>{t.cloudAPIs}</button>
                    <button onClick={() => setActiveTab('local')} className={activeTab === 'local' ? 'active' : ''}>{t.localModels}</button>
                </div>

                {activeTab === 'cloud' && (
                    <div className="tab-content">
                        <div className="provider-radios">
                            <label>
                                <input type="radio" value="google" checked={cloudProvider === 'google'} onChange={() => setCloudProvider('google')} />
                                Google
                            </label>
                            <label>
                                <input type="radio" value="openai" checked={cloudProvider === 'openai'} onChange={() => setCloudProvider('openai')} />
                                OpenAI
                            </label>
                            <label>
                                <input type="radio" value="claude" checked={cloudProvider === 'claude'} onChange={() => setCloudProvider('claude')} />
                                Claude
                            </label>
                        </div>
                        <div className="api-key-input">
                            <label>{t[`${cloudProvider}APIKey`]}</label>
                            <input 
                                type="password" 
                                value={apiKeys[cloudProvider]} 
                                onChange={(e) => setApiKeys({...apiKeys, [cloudProvider]: e.target.value})} 
                            />
                        </div>
                    </div>
                )}

                {activeTab === 'local' && (
                    <div className="tab-content">
                        <p>{t.currentLocalModel}: {localModelPath.split('/').pop() || t.noModelSelected}</p>
                        <input type="file" id="local-model-input" style={{ display: 'none' }} onChange={handleFileChange} accept=".gguf" />
                        <button onClick={() => document.getElementById('local-model-input').click()}>{t.browse}</button>
                        
                        <h4>{t.availableModels}</h4>
                        <div className="local-model-list">
                            {availableLocalModels.map(model => (
                                <div 
                                    key={model} 
                                    className={`local-model-item ${localModelPath.endsWith(model) ? 'selected' : ''}`}
                                    onClick={() => setLocalModelPath(model)}
                                >
                                    {model}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <button onClick={handleSave}>{t.save}</button>
            </div>
        </div>
    );
};

export default ChangeModelModal;


---


# frontend\src\components\ChatWindow.js

import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChatWindow = ({ messages }) => {
    // Create a ref for the very last message element.
    const lastMessageRef = useRef(null);

    useEffect(() => {
        // If the last message ref is attached to an element, scroll it into view smoothly.
        if (lastMessageRef.current) {
            lastMessageRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);

    return (
        <div className="chat-window">
            {messages.map((msg, index) => {
                // Check if this is the last message in the array.
                const isLastMessage = index === messages.length - 1;
                return (
                    // Attach the ref *only* to the div of the last message.
                    <div 
                        key={index} 
                        className={`message ${msg.sender}`} 
                        ref={isLastMessage ? lastMessageRef : null}
                    >
                        {msg.sender === 'bot' ? (
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                        ) : (
                            <p>{msg.text}</p>
                        )}
                    </div>
                );
            })}
        </div>
    );
};

export  default ChatWindow;

---


# frontend\src\components\InputBar.js

import React, { useState } from 'react';

const translations = {
    en: {
        describeProblem: 'Describe your problem...',
        clear: 'Clear',
        send: 'Send',
    },
    ro: {
        describeProblem: 'Descrie problema ta...',
        clear: 'Curăță',
        send: 'Trimite',
    },
};

const InputBar = ({ onSendMessage, onClearChat, language }) => {
    const [query, setQuery] = useState('');

    const t = translations[language];

    const handleSubmit = () => {
        if (query.trim()) {
            onSendMessage(query);
            setQuery('');
        }
    };

    return (
        <div className="input-bar">
            <button className="clear-btn" onClick={onClearChat}>{t.clear}</button>
            <input 
                type="text" 
                placeholder={t.describeProblem} 
                value={query} 
                onChange={(e) => setQuery(e.target.value)} 
                onKeyPress={(e) => e.key === 'Enter' && handleSubmit()} 
            />
            <button className="send-btn" onClick={handleSubmit}>{t.send}</button>
        </div>
    );
};

export default InputBar;

---


# frontend\src\components\MachineSidebar.js

import React, { useState, useEffect } from 'react';
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { SortableContext, useSortable, arrayMove, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

const SortableItem = ({ id, children }) => {
    const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
    };

    return (
        <div ref={setNodeRef} style={style} {...attributes} {...listeners}>
            {children}
        </div>
    );
};

const MachineSidebar = ({ isOpen, onClose, setMachine }) => {
    const [machines, setMachines] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchMachines = async () => {
            setLoading(true);
            try {
                const response = await fetch('http://127.0.0.1:8000/api/machines');
                const data = await response.json();
                if (data.machines) {
                    const machineList = data.machines.map((name, index) => ({
                        id: `machine-${index}`,
                        name: name
                    }));
                    setMachines(machineList);
                } else {
                    setMachines([]);
                }
            } catch (error) {
                console.error('Failed to fetch machines:', error);
                setMachines([]);
            } finally {
                setLoading(false);
            }
        };

        if (isOpen) {
            fetchMachines();
        }
    }, [isOpen]);

    const sensors = useSensors(
        useSensor(PointerSensor, {
            activationConstraint: {
                delay: 100, // 0.1 second delay
                tolerance: 5,
            },
        })
    );

    const handleDragEnd = (event) => {
        const { active, over } = event;
        if (active.id !== over.id) {
            setMachines((items) => {
                const oldIndex = items.findIndex((item) => item.id === active.id);
                const newIndex = items.findIndex((item) => item.id === over.id);
                return arrayMove(items, oldIndex, newIndex);
            });
        }
    };

    return (
        <div className={`sidebar machine-sidebar ${isOpen ? 'open' : ''}`}>
            <div className="sidebar-header">
                <h3>Machines</h3>
                <button onClick={onClose} className="close-btn">&times;</button>
            </div>
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                <SortableContext items={machines} strategy={verticalListSortingStrategy}>
                    <div className="sidebar-content">
                        {loading ? <p>Loading...</p> : machines.map(machine => (
                            <SortableItem key={machine.id} id={machine.id}>
                                <div className="machine-item" onClick={() => setMachine(machine.name)}>
                                    {machine.name}
                                </div>
                            </SortableItem>
                        ))}
                    </div>
                </SortableContext>
            </DndContext>
        </div>
    );
};

export default MachineSidebar;

---


# frontend\src\components\ManageDocumentsModal.js

import React, { useState, useEffect } from 'react';

const translations = {
    en: {
        manageDocuments: 'Manage Documents',
        uploadDocument: 'Upload Document',
        machineName: 'Machine Name',
        selectFile: 'Select PDF File',
        upload: 'Upload',
        existingDocuments: 'Existing Documents',
        delete: 'Delete',
        close: 'Close',
        uploading: 'Uploading...',
        noDocuments: 'No documents uploaded yet',
        confirmDelete: 'Are you sure you want to delete this document?',
        uploadSuccess: 'Document uploaded successfully!',
        uploadError: 'Error uploading document',
        deleteSuccess: 'Document deleted successfully!',
        deleteError: 'Error deleting document',
    },
    ro: {
        manageDocuments: 'Gestionează Documente',
        uploadDocument: 'Încarcă Document',
        machineName: 'Numele Mașinii',
        selectFile: 'Selectează Fișier PDF',
        upload: 'Încarcă',
        existingDocuments: 'Documente Existente',
        delete: 'Șterge',
        close: 'Închide',
        uploading: 'Se încarcă...',
        noDocuments: 'Nu au fost încărcate documente încă',
        confirmDelete: 'Sigur vrei să ștergi acest document?',
        uploadSuccess: 'Document încărcat cu succes!',
        uploadError: 'Eroare la încărcarea documentului',
        deleteSuccess: 'Document șters cu succes!',
        deleteError: 'Eroare la ștergerea documentului',
    }
};

const ManageDocumentsModal = ({ isOpen, onClose, language, onDocumentChange }) => {
    const [machines, setMachines] = useState([]);
    const [machineName, setMachineName] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState('');

    const t = translations[language];

    useEffect(() => {
        if (isOpen) {
            fetchMachines();
        }
    }, [isOpen]);

    const fetchMachines = async () => {
        try {
            const response = await fetch('http://127.0.0.1:8000/api/machines');
            const data = await response.json();
            if (data.machines) {
                setMachines(data.machines);
            }
        } catch (error) {
            console.error('Failed to fetch machines:', error);
        }
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && file.type === 'application/pdf') {
            setSelectedFile(file);
        } else {
            setMessage('Please select a PDF file');
            setTimeout(() => setMessage(''), 3000);
        }
    };

    const handleUpload = async () => {
        if (!machineName.trim() || !selectedFile) {
            setMessage('Please provide a machine name and select a file');
            setTimeout(() => setMessage(''), 3000);
            return;
        }

        setUploading(true);
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`http://127.0.0.1:8000/api/machines/upload?name=${encodeURIComponent(machineName)}`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                setMessage(t.uploadSuccess);
                setMachineName('');
                setSelectedFile(null);
                fetchMachines();
                onDocumentChange();
            } else {
                const error = await response.json();
                setMessage(t.uploadError + ': ' + error.detail);
            }
        } catch (error) {
            setMessage(t.uploadError + ': ' + error.message);
        } finally {
            setUploading(false);
            setTimeout(() => setMessage(''), 3000);
        }
    };

    const handleDelete = async (machineName) => {
        if (!window.confirm(t.confirmDelete)) {
            return;
        }

        try {
            const response = await fetch(`http://127.0.0.1:8000/api/machines/${encodeURIComponent(machineName)}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                setMessage(t.deleteSuccess);
                fetchMachines();
                onDocumentChange();
            } else {
                const error = await response.json();
                setMessage(t.deleteError + ': ' + error.detail);
            }
        } catch (error) {
            setMessage(t.deleteError + ': ' + error.message);
        } finally {
            setTimeout(() => setMessage(''), 3000);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal manage-documents-modal" onClick={(e) => e.stopPropagation()}>
                <h3>{t.manageDocuments}</h3>
                
                {message && (
                    <div className={`message ${message.includes('success') ? 'success' : 'error'}`}>
                        {message}
                    </div>
                )}

                <div className="upload-section">
                    <h4>{t.uploadDocument}</h4>
                    <div className="form-group">
                        <label>{t.machineName}</label>
                        <input 
                            type="text" 
                            value={machineName} 
                            onChange={(e) => setMachineName(e.target.value)}
                            placeholder="e.g. Siemens CNC 840D"
                        />
                    </div>
                    <div className="form-group">
                        <label>{t.selectFile}</label>
                        <input 
                            type="file" 
                            accept=".pdf" 
                            onChange={handleFileChange}
                        />
                        {selectedFile && <span className="file-name">{selectedFile.name}</span>}
                    </div>
                    <button 
                        onClick={handleUpload} 
                        disabled={uploading || !machineName.trim() || !selectedFile}
                        className="upload-btn"
                    >
                        {uploading ? t.uploading : t.upload}
                    </button>
                </div>

                <div className="existing-documents">
                    <h4>{t.existingDocuments}</h4>
                    {machines.length === 0 ? (
                        <p>{t.noDocuments}</p>
                    ) : (
                        <div className="documents-list">
                            {machines.map((machine, index) => (
                                <div key={index} className="document-item">
                                    <span className="machine-name">{machine}</span>
                                    <button 
                                        onClick={() => handleDelete(machine)}
                                        className="delete-btn"
                                    >
                                        {t.delete}
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                <button onClick={onClose}>{t.close}</button>
            </div>
        </div>
    );
};

export default ManageDocumentsModal;


---


# frontend\src\components\SettingsSidebar.js

import React, { useState, useEffect } from 'react';
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { SortableContext, useSortable, arrayMove, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

const translations = {
    en: {
        settings: 'Settings',
        theme: 'Theme',
        lightMode: 'Light Mode',
        darkMode: 'Dark Mode',
        language: 'Language',
        english: 'English',
        romanian: 'Romanian',
        fontSize: 'Font Size',
        systemPrompt: 'System Prompt',
        edit: 'Edit',
        changeModel: 'Change Model',
        manageDocuments: 'Manage Documents',
    },
    ro: {
        settings: 'Setări',
        theme: 'Temă',
        lightMode: 'Mod Luminos',
        darkMode: 'Mod Întunecat',
        language: 'Limbă',
        english: 'Engleză',
        romanian: 'Română',
        fontSize: 'Dimensiune Font',
        systemPrompt: 'Prompt Sistem',
        edit: 'Editează',
        changeModel: 'Schimbă Modelul',
        manageDocuments: 'Gestionează Documente',
    },
};

const initialSettingIds = ['theme', 'language', 'font-size', 'change-model', 'manage-documents', 'system-prompt'];

const SortableItem = ({ id, children }) => {
    const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
    };

    return (
        <div ref={setNodeRef} style={style} {...attributes} {...listeners}>
            {children}
        </div>
    );
};

const SettingsSidebar = ({ isOpen, onClose, theme, setTheme, language, setLanguage, openSystemPrompt, fontSize, setFontSize, openChangeModel, modelConfig, openManageDocuments }) => {
    const [settingIds, setSettingIds] = useState(() => {
        const savedSettingIds = localStorage.getItem('settingIds');
        if (savedSettingIds) {
            return JSON.parse(savedSettingIds);
        }
        return initialSettingIds;
    });

    useEffect(() => {
        localStorage.setItem('settingIds', JSON.stringify(settingIds));
    }, [settingIds]);

    const t = translations[language];

    const getSettingContent = (id) => {
        switch (id) {
            case 'theme':
                return <div className="setting-item"><span>{t.theme}</span><button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>{theme === 'light' ? t.darkMode : t.lightMode}</button></div>;
            case 'language':
                return <div className="setting-item"><span>{t.language}</span><button onClick={() => setLanguage(language === 'en' ? 'ro' : 'en')}>{language === 'en' ? t.romanian : t.english}</button></div>;
            case 'font-size':
                return <div className="setting-item"><span>{t.fontSize}</span><div><button onClick={() => setFontSize(14)}>14</button><button onClick={() => setFontSize(16)}>16</button><button onClick={() => setFontSize(18)}>18</button><button onClick={() => setFontSize(20)}>20</button></div></div>;
            case 'system-prompt':
                return <div className="setting-item"><span style={{ color: 'red' }}>{t.systemPrompt}</span><button onClick={openSystemPrompt}>{t.edit}</button></div>;
            case 'change-model':
                return <div className="setting-item"><span>{t.changeModel} ({modelConfig.type === 'local' ? 'Local' : `Cloud - ${modelConfig.provider}`})</span><button onClick={openChangeModel}>{t.edit}</button></div>;
            case 'manage-documents':
                return <div className="setting-item"><span>{t.manageDocuments}</span><button onClick={openManageDocuments}>{t.edit}</button></div>;
            default:
                return null;
        }
    };

    const sensors = useSensors(
        useSensor(PointerSensor, {
            activationConstraint: {
                delay: 100, // 0.1 second delay
                tolerance: 5,
            },
        })
    );

    const handleDragEnd = (event) => {
        const { active, over } = event;
        if (active.id !== over.id) {
            setSettingIds((items) => {
                const oldIndex = items.findIndex((item) => item === active.id);
                const newIndex = items.findIndex((item) => item === over.id);
                return arrayMove(items, oldIndex, newIndex);
            });
        }
    };

    return (
        <div className={`sidebar settings-sidebar ${isOpen ? 'open' : ''}`}>
            <div className="sidebar-header">
                <button onClick={onClose} className="close-btn">&times;</button>
                <h3>{t.settings}</h3>
            </div>
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                <SortableContext items={settingIds} strategy={verticalListSortingStrategy}>
                    <div className="sidebar-content">
                        {settingIds.map(id => (
                            <SortableItem key={id} id={id}>
                                {getSettingContent(id)}
                            </SortableItem>
                        ))}
                    </div>
                </SortableContext>
            </DndContext>
        </div>
    );
};

export default SettingsSidebar;

---


# frontend\src\components\SystemPromptModal.js

import React from 'react';

const SystemPromptModal = ({ isOpen, onClose, systemPrompt, setSystemPrompt, onSave }) => {
    if (!isOpen) return null;

    return (
        <div className="modal-overlay">
            <div className="modal">
                <h3>Edit System Prompt</h3>
                <textarea 
                    value={systemPrompt} 
                    onChange={(e) => setSystemPrompt(e.target.value)} 
                    rows="10"
                />
                <button onClick={onSave}>Save and Close</button>
            </div>
        </div>
    );
};

export default SystemPromptModal;

---
