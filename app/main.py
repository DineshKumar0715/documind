from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.core.config import settings
from app.db.database import init_db
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.utils.logging import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown procedures."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} ({settings.APP_ENV})...")
    settings.ensure_directories()
    init_db()
    logger.info(f"{settings.APP_NAME} initialization complete. Ready to serve requests.")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")

app = FastAPI(
    title=f"{settings.APP_NAME} API",
    description="Enterprise AI-powered Document Intelligence & Retrieval-Augmented Generation (RAG) Platform powered by Google Gemini.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(documents_router, prefix=settings.API_V1_PREFIX)
app.include_router(chat_router, prefix=settings.API_V1_PREFIX)

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for container probes and monitoring."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "model": settings.GEMINI_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "gemini_api_configured": bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip() not in ("", "your_gemini_api_key_here"))
    }

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard():
    """Interactive visual dashboard for testing document upload and RAG conversation."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocuMind - AI Document Intelligence & RAG Platform</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: rgba(18, 24, 38, 0.85);
            --bg-card-hover: rgba(26, 35, 54, 0.9);
            --border-color: rgba(79, 70, 229, 0.2);
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            --accent-cyan: #06b6d4;
            --accent-indigo: #6366f1;
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.12) 0px, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 1.25rem 2.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            background: rgba(10, 14, 23, 0.8);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: var(--accent-gradient);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
        }

        .logo-text {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge-gemini {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.4);
            color: #818cf8;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .badge-pulse {
            width: 6px;
            height: 6px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10b981;
        }

        .header-links a {
            color: var(--text-muted);
            text-decoration: none;
            margin-left: 20px;
            font-size: 0.9rem;
            font-weight: 500;
            transition: color 0.2s;
        }

        .header-links a:hover {
            color: #fff;
        }

        main {
            flex: 1;
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 24px;
            padding: 24px 36px;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 24px;
            box-shadow: var(--glass-shadow);
            backdrop-filter: blur(16px);
            display: flex;
            flex-direction: column;
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .upload-dropzone {
            border: 2px dashed rgba(99, 102, 241, 0.35);
            border-radius: 14px;
            padding: 30px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
            background: rgba(99, 102, 241, 0.03);
            margin-bottom: 20px;
        }

        .upload-dropzone:hover {
            border-color: var(--accent-cyan);
            background: rgba(6, 182, 212, 0.06);
            transform: translateY(-2px);
        }

        .doc-list {
            flex: 1;
            overflow-y: auto;
            max-height: 450px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .doc-item {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 12px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s;
        }

        .doc-item:hover {
            background: rgba(255, 255, 255, 0.06);
        }

        .doc-info {
            overflow: hidden;
        }

        .doc-name {
            font-size: 0.88rem;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 200px;
        }

        .doc-meta {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .btn-delete {
            background: transparent;
            border: none;
            color: #ef4444;
            cursor: pointer;
            padding: 4px;
            border-radius: 6px;
            opacity: 0.7;
            transition: opacity 0.2s;
        }

        .btn-delete:hover {
            opacity: 1;
            background: rgba(239, 68, 68, 0.15);
        }

        .chat-container {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 160px);
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .message {
            display: flex;
            flex-direction: column;
            max-width: 85%;
            animation: fadeIn 0.25s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
        }

        .message.assistant {
            align-self: flex-start;
        }

        .message-bubble {
            padding: 14px 18px;
            border-radius: 16px;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .message.user .message-bubble {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: #ffffff;
            border-bottom-right-radius: 4px;
        }

        .message.assistant .message-bubble {
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-bottom-left-radius: 4px;
            color: #f1f5f9;
        }

        .citations-container {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .citation-tag {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.35);
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.75rem;
            color: #a5b4fc;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .chat-input-area {
            padding-top: 16px;
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .chat-input {
            flex: 1;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            padding: 14px 18px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }

        .chat-input:focus {
            border-color: var(--accent-indigo);
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
        }

        .btn-send {
            background: var(--accent-gradient);
            border: none;
            border-radius: 14px;
            padding: 14px 24px;
            color: #fff;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: transform 0.15s, opacity 0.15s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-send:hover {
            transform: scale(1.02);
            opacity: 0.95;
        }

        .btn-send:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        @media (max-width: 900px) {
            main { grid-template-columns: 1fr; }
            .chat-container { height: 600px; }
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-container">
            <div class="logo-icon">🧠</div>
            <div class="logo-text">DocuMind</div>
            <div class="badge-gemini"><div class="badge-pulse"></div> Google Gemini RAG</div>
        </div>
        <div class="header-links">
            <a href="/docs" target="_blank">Interactive Swagger Docs ↗</a>
            <a href="/health" target="_blank">Health ↗</a>
        </div>
    </header>

    <main>
        <!-- Sidebar: Documents Management -->
        <div class="card">
            <div class="card-title">
                <span>📚 Ingested Documents</span>
                <span id="doc-count" style="font-size: 0.85rem; color: var(--text-muted);">0 docs</span>
            </div>
            
            <div class="upload-dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
                <div style="font-size: 28px; margin-bottom: 8px;">📤</div>
                <div style="font-weight: 600; font-size: 0.9rem;">Click or Drag Document</div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;">PDF, TXT, MD, JSON, CSV</div>
                <input type="file" id="file-input" style="display: none;" onchange="handleFileUpload(event)">
            </div>

            <div class="doc-list" id="doc-list">
                <div style="text-align: center; color: var(--text-muted); padding: 20px; font-size: 0.85rem;">
                    No documents indexed yet. Upload a file above to begin!
                </div>
            </div>
        </div>

        <!-- Main: Interactive RAG Chat -->
        <div class="card chat-container">
            <div class="card-title">
                <span>💬 Conversational Document Intelligence</span>
                <button onclick="newSession()" style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); color: #cbd5e1; padding: 5px 12px; border-radius: 8px; cursor: pointer; font-size: 0.8rem;">+ New Chat</button>
            </div>

            <div class="chat-messages" id="chat-messages">
                <div class="message assistant">
                    <div class="message-bubble">
                        👋 Welcome to <strong>DocuMind</strong>! Upload documents on the left panel, then ask any question. Answers are retrieved in real-time and cited with source documents and page numbers.
                    </div>
                </div>
            </div>

            <div class="chat-input-area">
                <input type="text" id="chat-input" class="chat-input" placeholder="Ask a question about your documents..." onkeydown="if(event.key==='Enter') sendMessage()">
                <button id="send-btn" class="btn-send" onclick="sendMessage()">Send ➔</button>
            </div>
        </div>
    </main>

    <script>
        let currentSessionId = null;

        async function loadDocuments() {
            try {
                const res = await fetch('/api/v1/documents');
                const data = await res.json();
                const listEl = document.getElementById('doc-list');
                const countEl = document.getElementById('doc-count');
                countEl.innerText = `${data.total} doc(s)`;

                if (data.documents.length === 0) {
                    listEl.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 20px; font-size: 0.85rem;">No documents indexed yet. Upload a file above to begin!</div>';
                    return;
                }

                listEl.innerHTML = data.documents.map(d => `
                    <div class="doc-item">
                        <div class="doc-info">
                            <div class="doc-name" title="${d.filename}">📄 ${d.filename}</div>
                            <div class="doc-meta">${d.chunk_count} chunks • ${(d.file_size_bytes / 1024).toFixed(1)} KB • <span style="color: #10b981;">${d.status}</span></div>
                        </div>
                        <button class="btn-delete" onclick="deleteDoc('${d.id}', '${d.filename}')" title="Delete document">🗑️</button>
                    </div>
                `).join('');
            } catch (err) {
                console.error(err);
            }
        }

        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            const dropzone = document.getElementById('dropzone');
            dropzone.innerHTML = '<div style="font-size: 24px; margin-bottom: 8px;">⏳</div><div style="font-weight: 600; font-size: 0.9rem;">Chunking & Indexing...</div>';

            try {
                const res = await fetch('/api/v1/documents/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await res.json();
                if (res.ok) {
                    appendSystemMessage(`✅ Indexed <strong>${file.name}</strong> (${result.document.chunk_count} chunks).`);
                    await loadDocuments();
                } else {
                    alert(`Upload failed: ${result.detail || 'Unknown error'}`);
                }
            } catch (err) {
                alert(`Upload error: ${err.message}`);
            } finally {
                dropzone.innerHTML = '<div style="font-size: 28px; margin-bottom: 8px;">📤</div><div style="font-weight: 600; font-size: 0.9rem;">Click or Drag Document</div><div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;">PDF, TXT, MD, JSON, CSV</div><input type="file" id="file-input" style="display: none;" onchange="handleFileUpload(event)">';
            }
        }

        async function deleteDoc(id, name) {
            if (!confirm(`Delete document "${name}" and all its vector chunks?`)) return;
            try {
                await fetch(`/api/v1/documents/${id}`, { method: 'DELETE' });
                await loadDocuments();
            } catch (err) {
                alert(`Delete error: ${err.message}`);
            }
        }

        function newSession() {
            currentSessionId = null;
            document.getElementById('chat-messages').innerHTML = `
                <div class="message assistant">
                    <div class="message-bubble">New chat session started. Ask any question about your indexed documents!</div>
                </div>
            `;
        }

        function appendSystemMessage(text) {
            const messages = document.getElementById('chat-messages');
            const el = document.createElement('div');
            el.className = 'message assistant';
            el.innerHTML = `<div class="message-bubble" style="background: rgba(99, 102, 241, 0.1); border: 1px dashed rgba(99, 102, 241, 0.4);">${text}</div>`;
            messages.appendChild(el);
            messages.scrollTop = messages.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;

            input.value = '';
            const sendBtn = document.getElementById('send-btn');
            sendBtn.disabled = true;

            const messages = document.getElementById('chat-messages');
            
            // Add user bubble
            const userMsgEl = document.createElement('div');
            userMsgEl.className = 'message user';
            userMsgEl.innerHTML = `<div class="message-bubble">${escapeHtml(message)}</div>`;
            messages.appendChild(userMsgEl);

            // Add loading assistant bubble
            const assistantMsgEl = document.createElement('div');
            assistantMsgEl.className = 'message assistant';
            assistantMsgEl.innerHTML = `<div class="message-bubble" id="current-assistant-bubble"><span style="color: var(--text-muted);">Thinking & Retrieving context...</span></div>`;
            messages.appendChild(assistantMsgEl);
            messages.scrollTop = messages.scrollHeight;

            try {
                const res = await fetch('/api/v1/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        session_id: currentSessionId
                    })
                });

                const data = await res.json();
                if (res.ok) {
                    currentSessionId = data.session_id;
                    const bubble = assistantMsgEl.querySelector('.message-bubble');
                    
                    let citationsHtml = '';
                    if (data.citations && data.citations.length > 0) {
                        citationsHtml = `<div class="citations-container">` + data.citations.map(c => 
                            `<div class="citation-tag" title="${escapeHtml(c.content)}">📎 ${c.document_name}${c.page_number ? ` (p. ${c.page_number})` : ` (#${c.chunk_index})`} [${c.similarity_score}]</div>`
                        ).join('') + `</div>`;
                    }

                    bubble.innerHTML = formatMarkdown(data.message) + citationsHtml + `<div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 8px;">⚡ ${data.latency_seconds}s • ${data.model}</div>`;
                } else {
                    assistantMsgEl.querySelector('.message-bubble').innerHTML = `<span style="color: #ef4444;">Error: ${data.detail || 'Request failed'}</span>`;
                }
            } catch (err) {
                assistantMsgEl.querySelector('.message-bubble').innerHTML = `<span style="color: #ef4444;">Network Error: ${err.message}</span>`;
            } finally {
                sendBtn.disabled = false;
                messages.scrollTop = messages.scrollHeight;
            }
        }

        function escapeHtml(str) {
            return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }

        function formatMarkdown(text) {
            let escaped = escapeHtml(text);
            escaped = escaped.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
            escaped = escaped.replace(/\\*(.*?)\\*/g, '<em>$1</em>');
            escaped = escaped.replace(/`([^`]+)`/g, '<code style="background: rgba(0,0,0,0.3); padding: 2px 5px; border-radius: 4px; font-family: JetBrains Mono;">$1</code>');
            escaped = escaped.replace(/\\n/g, '<br>');
            return escaped;
        }

        // Initialize on load
        loadDocuments();
    </script>
</body>
</html>
"""
