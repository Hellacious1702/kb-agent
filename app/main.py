"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import admin as admin_module
from app.api import chat, documents, workspaces
from app.core.config import get_settings
from app.core.database import close_db, init_db
from app.services.retriever import hybrid_retriever

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Templates
templates = Jinja2Templates(directory="app/web/templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting KB Agent...")
    await init_db()
    await hybrid_retriever.initialize()
    logger.info("KB Agent started successfully")

    yield

    # Shutdown
    logger.info("Shutting down KB Agent...")
    await hybrid_retriever.close()
    await close_db()
    logger.info("KB Agent shut down complete")


app = FastAPI(
    title="KB Agent",
    description="Internal Knowledge Base Agent - RAG-powered document Q&A",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")

# Include routers
app.include_router(workspaces.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(admin_module.router)


# Web UI routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main chat interface."""
    return templates.TemplateResponse(request, "chat.html")


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Admin panel."""
    return templates.TemplateResponse(request, "admin.html")


@app.get("/admin/workspaces", response_class=HTMLResponse)
async def admin_workspaces(request: Request):
    """Workspaces management page."""
    return templates.TemplateResponse(request, "admin_workspaces.html")


@app.get("/admin/workspaces/{workspace_id}", response_class=HTMLResponse)
async def admin_workspace_detail(request: Request, workspace_id: str):
    """Workspace detail page."""
    return templates.TemplateResponse(request, "admin_workspace_detail.html", {"workspace_id": workspace_id})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "kb-agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
