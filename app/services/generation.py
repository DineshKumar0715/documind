import asyncio
from typing import List, AsyncGenerator, Optional, Dict, Any
from app.core.config import settings
from app.schemas.chat import Citation
from app.models.chat import ChatMessage
from app.utils.logging import logger

SYSTEM_PROMPT_TEMPLATE = """You are DocuMind, an intelligent document analysis and RAG assistant.
Your goal is to answer the user's questions accurately, concisely, and factually based on the provided context excerpts.

Guidelines:
1. Ground your answers ONLY in the provided context excerpts whenever available.
2. When citing information, refer to the document name and page number or chunk index (e.g. [Doc: filename, Page: X]).
3. If the context does not contain sufficient information to answer the question, clearly state that the provided documents do not contain the answer, and then provide helpful general knowledge if applicable.
4. Maintain a professional, clear, and helpful tone.
5. Format your response with markdown (bullet points, bold highlights, code blocks) where helpful.

Context Excerpts:
{context_text}
"""

class GenerationService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.temperature = settings.GEMINI_TEMPERATURE
        self.max_tokens = settings.GEMINI_MAX_OUTPUT_TOKENS
        self._llm = None
        self._init_client()

    def _init_client(self):
        if self.api_key and self.api_key.strip() not in ("", "your_gemini_api_key_here"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                logger.info(f"Initialized Gemini LLM with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize ChatGoogleGenerativeAI ({e}). Operating in fallback mode.")
                self._llm = None
        else:
            logger.info("No GEMINI_API_KEY configured. Operating in fallback mock generation mode.")
            self._llm = None

    def format_context(self, citations: List[Citation]) -> str:
        if not citations:
            return "No relevant document excerpts were found."
        
        blocks = []
        for i, cit in enumerate(citations, 1):
            page_info = f", Page {cit.page_number}" if cit.page_number else f", Chunk #{cit.chunk_index}"
            header = f"[Source #{i}: {cit.document_name}{page_info} (Score: {cit.similarity_score})]"
            blocks.append(f"{header}\n{cit.content.strip()}\n")
        return "\n".join(blocks)

    def _build_messages(
        self,
        query: str,
        citations: List[Citation],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> List[Dict[str, str]]:
        context_text = self.format_context(citations)
        system_instruction = SYSTEM_PROMPT_TEMPLATE.format(context_text=context_text)

        messages = [{"role": "system", "content": system_instruction}]
        
        if chat_history:
            for msg in chat_history[-6:]:  # Include last 6 turns
                if msg.role in ("user", "assistant"):
                    messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": query})
        return messages

    async def generate_response(
        self,
        query: str,
        citations: List[Citation],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Generate complete response asynchronously."""
        if self._llm is None:
            # Fallback mock generation
            await asyncio.sleep(0.3)
            context_summary = "\n".join([f"- {c.document_name}: {c.content[:100]}..." for c in citations[:3]])
            if citations:
                return (
                    f"**[DocuMind Analysis]**\n\n"
                    f"Based on the {len(citations)} relevant excerpt(s) found in your documents:\n\n"
                    f"{context_summary}\n\n"
                    f"> *Note: Configure your `GEMINI_API_KEY` in `.env` to enable full Google Gemini 1.5/2.0 AI synthesis.*"
                )
            else:
                return (
                    "I could not find relevant context in your uploaded documents for this query. "
                    "Please upload related documents or adjust your query."
                )

        messages = self._build_messages(query, citations, chat_history)
        
        # Convert to LangChain message objects
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))

        try:
            response = await self._llm.ainvoke(lc_messages)
            return response.content
        except Exception as e:
            logger.error(f"Error calling Gemini LLM: {e}")
            return f"Error during AI generation: {str(e)}"

    async def generate_stream(
        self,
        query: str,
        citations: List[Citation],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens asynchronously."""
        if self._llm is None:
            full_text = await self.generate_response(query, citations, chat_history)
            words = full_text.split(" ")
            for word in words:
                yield word + " "
                await asyncio.sleep(0.03)
            return

        messages = self._build_messages(query, citations, chat_history)
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))

        try:
            async for chunk in self._llm.astream(lc_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming from Gemini: {e}")
            yield f"\n[Streaming Error: {str(e)}]"
