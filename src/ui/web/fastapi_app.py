"""
FastAPI Web Server for DocuBot - Backend API
Document-based RAG Assistant API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
from pathlib import Path
import json
import logging
import sys
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query text")
    context_length: Optional[int] = Field(500, ge=100, le=2000, description="Context length in tokens")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=1.0, description="Temperature for generation")
    include_sources: Optional[bool] = Field(True, description="Include source documents in response")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    model_used: Optional[str]
    timestamp: datetime

class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    upload_date: datetime
    processed: bool
    chunk_count: Optional[int]
    word_count: Optional[int]
    tags: List[str]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: datetime
    system_info: Dict[str, Any]

class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: Optional[str]
    estimated_completion: Optional[datetime]

# Simple Core Implementation
class SimpleDocuBotCore:
    """Simple implementation for testing web interface"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.documents = []
        self.load_documents()
    
    def load_documents(self):
        """Load existing documents"""
        documents_file = self.data_dir / "documents.json"
        if documents_file.exists():
            with open(documents_file, 'r') as f:
                self.documents = json.load(f)
    
    def process_query(self, query: str, context: dict = None):
        """Process a query - simple implementation"""
        # Simulate AI response
        import random
        
        responses = [
            f"Based on your documents, here's what I found about '{query}': This is a test response from the simple core implementation.",
            f"I've analyzed your question '{query}' and found relevant information in your uploaded documents.",
            f"Regarding '{query}', the information in your uploaded documents suggests that this topic is covered in multiple files.",
            f"Query processed successfully. For '{query}', I found matching content across {len(self.documents)} documents.",
            f"After searching through your knowledge base, I can tell you that '{query}' is an important topic mentioned in your documents.",
            f"Your question '{query}' relates to content in these documents. Here's a summary of what I found.",
            f"Based on the context from your uploaded files, here's the answer to '{query}'.",
            f"I've examined your documents and here's what I discovered about '{query}'."
        ]
        
        response = random.choice(responses)
        
        # Generate some mock sources based on actual documents
        sources = []
        if self.documents:
            for i, doc in enumerate(self.documents[:min(3, len(self.documents))]):
                sources.append({
                    "filename": doc.get("filename", f"document_{i+1}.pdf"),
                    "similarity": round(random.uniform(0.7, 0.95), 2),
                    "content": f"Relevant content about '{query}' found in {doc.get('filename')}. This section discusses topics related to your question.",
                    "page": random.randint(1, 10),
                    "document_id": doc.get("id", f"doc_{i+1}")
                })
        else:
            # Mock sources if no documents
            for i in range(3):
                sources.append({
                    "filename": f"sample_document_{i+1}.pdf",
                    "similarity": round(random.uniform(0.7, 0.95), 2),
                    "content": f"Sample content related to '{query}' from document {i+1}.",
                    "page": random.randint(1, 15),
                    "document_id": f"mock_doc_{i+1}"
                })
        
        return {
            "answer": response,
            "sources": sources,
            "model_used": "simple-mock-llm",
            "processing_time": random.uniform(0.5, 2.0)
        }
    
    def process_document(self, file_path: str):
        """Process a document - simple implementation"""
        import random
        from datetime import datetime
        
        filename = Path(file_path).name
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 10000
        
        # Add to documents list
        doc_info = {
            "id": f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.documents) + 1}",
            "filename": filename,
            "file_path": file_path,
            "file_type": Path(file_path).suffix.lower() if Path(file_path).suffix else ".txt",
            "upload_date": datetime.now().isoformat(),
            "file_size": file_size,
            "processed": True,
            "chunk_count": random.randint(3, 15),
            "word_count": random.randint(500, 5000),
            "tags": ["uploaded", "processed"]
        }
        
        self.documents.append(doc_info)
        
        # Save to file
        self.save_documents()
        
        return {
            "status": "success",
            "document_id": doc_info["id"],
            "chunks_processed": doc_info["chunk_count"],
            "processing_time": random.uniform(1.0, 5.0)
        }
    
    def save_documents(self):
        """Save documents to file"""
        documents_file = self.data_dir / "documents.json"
        documents_file.parent.mkdir(exist_ok=True)
        
        with open(documents_file, 'w') as f:
            json.dump(self.documents, f, indent=2)
    
    def get_documents(self, limit: int = 100, offset: int = 0, file_type: str = None, processed: bool = None):
        """Get filtered documents"""
        filtered = self.documents
        
        if file_type:
            filtered = [doc for doc in filtered if doc.get("file_type", "").lower() == file_type.lower()]
        
        if processed is not None:
            filtered = [doc for doc in filtered if doc.get("processed", False) == processed]
        
        # Apply pagination
        start = offset
        end = offset + limit
        return filtered[start:end]
    
    def delete_document(self, document_id: str):
        """Delete a document"""
        initial_count = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.get("id") != document_id]
        
        if len(self.documents) < initial_count:
            self.save_documents()
            return True
        return False

# Create data directory if not exists
Path("data").mkdir(exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="DocuBot API",
    description="REST API for DocuBot Document-based RAG Assistant",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application state
app.state.system_ready = False
app.state.document_count = 0
app.state.processing_tasks = {}

def initialize_system():
    """Initialize the system on startup"""
    try:
        # Initialize simple core
        app.state.docubot = SimpleDocuBotCore()
        
        # Set initial document count
        app.state.document_count = len(app.state.docubot.documents)
        
        app.state.system_ready = True
        logger.info(f"Simple DocuBot core initialized with {app.state.document_count} documents")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        app.state.system_ready = False

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting DocuBot API server with SimpleCore")
    initialize_system()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down DocuBot API server")
    app.state.system_ready = False

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "DocuBot",
        "version": "1.0.0",
        "description": "Document-based Retrieval Augmented Generation Assistant",
        "api_version": "v1",
        "endpoints": {
            "health": "/api/health",
            "documents": "/api/v1/documents",
            "query": "/api/v1/query",
            "upload": "/api/v1/upload",
            "status": "/api/v1/status/{task_id}",
            "models": "/api/v1/models",
            "system": "/api/v1/system"
        },
        "documentation": {
            "swagger": "/api/docs",
            "redoc": "/api/redoc"
        },
        "mode": "simple-core"
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    system_info = {
        "system_ready": app.state.system_ready,
        "document_count": app.state.document_count,
        "processing_tasks": len(app.state.processing_tasks),
        "python_version": sys.version,
        "platform": sys.platform,
        "mode": "simple-core"
    }
    
    return HealthResponse(
        status="healthy" if app.state.system_ready else "degraded",
        service="DocuBot API (SimpleCore)",
        version="1.0.0",
        timestamp=datetime.now(),
        system_info=system_info
    )

@app.get("/api/v1/documents", response_model=List[DocumentInfo])
async def list_documents(
    limit: Optional[int] = Query(100, ge=1, le=1000),
    offset: Optional[int] = Query(0, ge=0),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    processed: Optional[bool] = Query(None, description="Filter by processing status")
):
    """
    List all processed documents with pagination and filtering
    """
    if not app.state.system_ready:
        raise HTTPException(status_code=503, detail="System is not ready")
    
    try:
        documents = app.state.docubot.get_documents(
            limit=limit,
            offset=offset,
            file_type=file_type,
            processed=processed
        )
        
        # Convert to DocumentInfo
        result = []
        for doc in documents:
            result.append(DocumentInfo(
                id=doc.get("id", ""),
                filename=doc.get("filename", ""),
                file_type=doc.get("file_type", ""),
                file_size=doc.get("file_size", 0),
                upload_date=datetime.fromisoformat(doc.get("upload_date", datetime.now().isoformat())),
                processed=doc.get("processed", False),
                chunk_count=doc.get("chunk_count"),
                word_count=doc.get("word_count"),
                tags=doc.get("tags", [])
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/api/v1/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload")
):
    """
    Upload and process a document
    
    The file will be processed asynchronously. Returns a task ID for status tracking.
    """
    if not app.state.system_ready:
        raise HTTPException(status_code=503, detail="System is not ready")
    
    # Validate file type
    allowed_types = ['.pdf', '.txt', '.docx', '.md', '.html', '.epub', '.png', '.jpg', '.jpeg']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed types: {', '.join(allowed_types)}"
        )
    
    # Generate task ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(file.filename).stem}"
    
    # Store initial task status
    app.state.processing_tasks[task_id] = {
        "status": "pending",
        "filename": file.filename,
        "progress": 0.0,
        "started_at": datetime.now(),
        "message": "Starting upload..."
    }
    
    # Start background processing
    background_tasks.add_task(process_document_task, task_id, file)
    
    return {
        "task_id": task_id,
        "filename": file.filename,
        "status": "processing",
        "message": "Document upload accepted and processing started",
        "upload_time": datetime.now().isoformat(),
        "estimated_completion": (datetime.now().timestamp() + 30)  # 30 seconds from now
    }

@app.get("/api/v1/status/{task_id}", response_model=ProcessingStatus)
async def get_processing_status(task_id: str):
    """
    Get status of a processing task
    """
    if task_id not in app.state.processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = app.state.processing_tasks[task_id]
    
    return ProcessingStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        message=task.get("message"),
        estimated_completion=task.get("estimated_completion")
    )

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(
    query_request: QueryRequest
):
    """
    Query documents using RAG pipeline
    """
    if not app.state.system_ready:
        raise HTTPException(status_code=503, detail="System is not ready")
    
    start_time = datetime.now()
    
    try:
        # Use SimpleDocuBotCore
        result = app.state.docubot.process_query(
            query=query_request.query,
            context={
                "temperature": query_request.temperature,
                "max_tokens": query_request.context_length,
                "include_sources": query_request.include_sources
            }
        )
        
        answer = result.get("answer", "No answer available")
        sources = result.get("sources", []) if query_request.include_sources else []
        model_used = result.get("model_used")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            query=query_request.query,
            answer=answer,
            sources=sources,
            processing_time=processing_time,
            model_used=model_used,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/api/v1/models")
async def list_models():
    """
    List available AI models
    """
    # Mock models for simple core
    models = [
        {
            "name": "simple-mock-llm",
            "description": "Simple mock model for testing",
            "context_length": 2048,
            "parameters": "Mock",
            "supported_tasks": ["text-generation", "question-answering"]
        },
        {
            "name": "llama2:7b",
            "description": "Llama 2 7B parameter model (Mock)",
            "context_length": 4096,
            "parameters": "7B",
            "supported_tasks": ["text-generation", "question-answering"]
        },
        {
            "name": "mistral:7b",
            "description": "Mistral 7B model (Mock)",
            "context_length": 8192,
            "parameters": "7B",
            "supported_tasks": ["text-generation", "summarization"]
        }
    ]
    
    return {
        "models": models,
        "default_model": models[0]["name"],
        "total_models": len(models),
        "note": "Mock models for testing"
    }

@app.get("/api/v1/system")
async def system_info():
    """
    Get detailed system information
    """
    try:
        # Try to get system checker if available
        from runserver import DocuBotSystemChecker
        checker = DocuBotSystemChecker(str(project_root))
        checker.run_checks()
        summary = checker.generate_summary()
        
        system_status = summary.get('overall_status', 'HEALTHY')
        successful_checks = summary.get('successful_checks', 0)
        total_checks = summary.get('total_checks', 0)
    except ImportError:
        logger.info("System checker not available, using mock status")
        system_status = "HEALTHY"
        successful_checks = 5
        total_checks = 8
    except Exception as e:
        logger.warning(f"Could not get detailed system info: {e}")
        system_status = "HEALTHY"
        successful_checks = 5
        total_checks = 8
    
    return {
        "system": {
            "ready": app.state.system_ready,
            "document_count": app.state.document_count,
            "active_tasks": len(app.state.processing_tasks),
            "status": system_status,
            "checks_passed": f"{successful_checks}/{total_checks}",
            "mode": "simple-core"
        },
        "components": {
            "database": "Simple JSON",
            "vector_store": "Mock",
            "llm_engine": "SimpleMock",
            "embedding_model": "mock-embeddings"
        },
        "resources": {
            "cpu_usage": "Mock: 15%",
            "memory_usage": "Mock: 256MB",
            "disk_usage": "Mock: 1.2GB"
        }
    }

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all associated data
    """
    if not app.state.system_ready:
        raise HTTPException(status_code=503, detail="System is not ready")
    
    try:
        success = app.state.docubot.delete_document(document_id)
        
        if success:
            # Update document count
            app.state.document_count = len(app.state.docubot.documents)
            
            return {
                "document_id": document_id,
                "deleted": True,
                "message": "Document deleted successfully",
                "timestamp": datetime.now(),
                "remaining_documents": app.state.document_count
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# Background task functions
async def process_document_task(task_id: str, file: UploadFile):
    """
    Background task for document processing
    """
    try:
        # Update task status
        app.state.processing_tasks[task_id]["status"] = "processing"
        app.state.processing_tasks[task_id]["progress"] = 0.1
        app.state.processing_tasks[task_id]["message"] = "Receiving file..."
        
        # Create upload directory
        upload_dir = project_root / "data" / "documents" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / file.filename
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        app.state.processing_tasks[task_id]["progress"] = 0.3
        app.state.processing_tasks[task_id]["message"] = "File saved, starting processing..."
        
        # Simulate processing steps
        steps = [
            ("Validating format", 0.1),
            ("Extracting text", 0.2),
            ("Cleaning content", 0.1),
            ("Chunking text", 0.2),
            ("Generating embeddings", 0.2),
            ("Updating database", 0.1)
        ]
        
        for step_name, step_progress in steps:
            await asyncio.sleep(1)  # Simulate work
            app.state.processing_tasks[task_id]["message"] = step_name
            app.state.processing_tasks[task_id]["progress"] += step_progress
        
        # Process with SimpleDocuBotCore
        result = app.state.docubot.process_document(str(file_path))
        
        # Update document count
        app.state.document_count = len(app.state.docubot.documents)
        
        # Mark as complete
        app.state.processing_tasks[task_id]["status"] = "completed"
        app.state.processing_tasks[task_id]["progress"] = 1.0
        app.state.processing_tasks[task_id]["message"] = "Document processing complete"
        app.state.processing_tasks[task_id]["completed_at"] = datetime.now()
        app.state.processing_tasks[task_id]["document_id"] = result.get("document_id")
        
        logger.info(f"Document processing complete for task {task_id}")
        
    except Exception as e:
        app.state.processing_tasks[task_id]["status"] = "failed"
        app.state.processing_tasks[task_id]["message"] = f"Processing failed: {str(e)}"
        app.state.processing_tasks[task_id]["error"] = str(e)
        logger.error(f"Document processing failed for task {task_id}: {e}")

# Serve static files if directory exists
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logger.info(f"Serving static files from {static_path}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )