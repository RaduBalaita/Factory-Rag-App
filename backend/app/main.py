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
