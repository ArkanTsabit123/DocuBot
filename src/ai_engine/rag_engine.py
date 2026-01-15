"""
Retrieval-Augmented Generation (RAG) Engine
"""

from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
import json
from datetime import datetime
from .llm_client import LLMClient, LLMResponse
from ..vector_store.search_engine import HybridSearchEngine


@dataclass
class RAGResponse:
    """Structured RAG response"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    model: str
    processing_time: float
    token_counts: Optional[Dict[str, int]] = None
    error: Optional[str] = None


class RAGEngine:
    """
    Main RAG engine orchestrating retrieval and generation
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        search_engine: HybridSearchEngine
    ):
        self.llm_client = llm_client
        self.search_engine = search_engine
        self.conversation_history = []
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        include_sources: bool = True,
        conversation_id: Optional[str] = None,
        stream: bool = False
    ) -> RAGResponse:
        """
        Process a query using RAG pipeline
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity for retrieval
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            include_sources: Whether to include source citations
            conversation_id: Optional conversation ID for context
            stream: Whether to stream response
        
        Returns:
            RAGResponse with answer and sources
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant context
            retrieved_chunks = self.retrieve_context(
                question,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                conversation_id=conversation_id
            )
            
            # Step 2: Build context from retrieved chunks
            context = self.build_context(retrieved_chunks)
            
            # Step 3: Generate answer using LLM
            if stream:
                return self._stream_answer(
                    question=question,
                    context=context,
                    retrieved_chunks=retrieved_chunks,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    start_time=start_time,
                    include_sources=include_sources
                )
            else:
                answer = self.generate_answer(
                    question=question,
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Step 4: Process and format response
                processing_time = (datetime.now() - start_time).total_seconds()
                
                response = RAGResponse(
                    answer=answer,
                    sources=retrieved_chunks if include_sources else [],
                    query=question,
                    model=self.llm_client.default_model,
                    processing_time=processing_time
                )
                
                # Step 5: Update conversation history
                if conversation_id:
                    self.update_conversation_history(
                        conversation_id=conversation_id,
                        question=question,
                        answer=answer,
                        sources=retrieved_chunks
                    )
                
                return response
                
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return RAGResponse(
                answer="",
                sources=[],
                query=question,
                model=self.llm_client.default_model,
                processing_time=processing_time,
                error=str(e)
            )
    
    def _stream_answer(
        self,
        question: str,
        context: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        start_time: datetime,
        include_sources: bool
    ) -> Generator[LLMResponse, None, None]:
        """
        Stream answer generation
        
        Args:
            question: User question
            context: Retrieved context
            retrieved_chunks: Retrieved chunks
            temperature: LLM temperature
            max_tokens: Maximum tokens
            start_time: Start time
            include_sources: Whether to include sources
        
        Yields:
            LLMResponse objects
        """
        # Build prompt with context
        prompt = self.build_rag_prompt(question, context)
        
        # Stream response
        full_response = ""
        for chunk in self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        ):
            if chunk.content:
                full_response += chunk.content
                yield chunk
        
        # Final response
        processing_time = (datetime.now() - start_time).total_seconds()
        yield LLMResponse(
            content=full_response,
            model=self.llm_client.default_model,
            processing_time=processing_time
        )
    
    def retrieve_context(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a question
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
            conversation_id: Optional conversation ID for context
        
        Returns:
            List of relevant chunks with metadata
        """
        # Get conversation context if available
        conversation_context = None
        if conversation_id:
            conversation_context = self.get_conversation_context(conversation_id)
        
        # Enhance query with conversation context
        enhanced_query = self.enhance_query(question, conversation_context)
        
        # Perform hybrid search
        search_results = self.search_engine.hybrid_search(
            query=enhanced_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Format results
        formatted_chunks = []
        for result in search_results:
            chunk = {
                'text': result.get('text', ''),
                'similarity': result.get('similarity', 0.0),
                'metadata': result.get('metadata', {}),
                'source': result.get('metadata', {}).get('source_file', 'Unknown'),
                'chunk_index': result.get('metadata', {}).get('chunk_index', 0)
            }
            formatted_chunks.append(chunk)
        
        return formatted_chunks
    
    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks
        
        Args:
            chunks: List of retrieved chunks
        
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        context_parts.append("Relevant information from documents:")
        context_parts.append("")
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source', 'Unknown')
            chunk_idx = chunk.get('chunk_index', 0)
            similarity = chunk.get('similarity', 0.0)
            
            context_parts.append(f"[Source {i}: {source} (chunk {chunk_idx}, relevance: {similarity:.2f})]")
            context_parts.append(chunk.get('text', ''))
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_answer(
        self,
        question: str,
        context: str,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate answer using LLM with context
        
        Args:
            question: User question
            context: Retrieved context
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated answer
        """
        # Build prompt with context
        prompt = self.build_rag_prompt(question, context)
        
        # Generate response
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if response.error:
            return f"Error generating answer: {response.error}"
        
        return response.content
    
    def build_rag_prompt(self, question: str, context: str) -> str:
        """
        Build RAG prompt
        
        Args:
            question: User question
            context: Retrieved context
        
        Returns:
            Formatted prompt
        """
        prompt_template = """You are a helpful AI assistant. Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer based on the context above:"""
        
        return prompt_template.format(context=context, question=question)
    
    def enhance_query(
        self,
        query: str,
        conversation_context: Optional[str] = None
    ) -> str:
        """
        Enhance query with conversation context
        
        Args:
            query: Original query
            conversation_context: Previous conversation context
        
        Returns:
            Enhanced query
        """
        if not conversation_context:
            return query
        
        enhanced = f"{query}\n\nPrevious conversation context:\n{conversation_context}"
        return enhanced
    
    def get_conversation_context(self, conversation_id: str) -> Optional[str]:
        """
        Get context from conversation history
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            Conversation context or None
        """
        conversation_messages = [
            msg for msg in self.conversation_history 
            if msg.get('conversation_id') == conversation_id
        ]
        
        if not conversation_messages:
            return None
        
        # Get last 3 messages for context
        recent_messages = conversation_messages[-3:]
        context_parts = []
        
        for msg in recent_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def update_conversation_history(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ):
        """
        Update conversation history
        
        Args:
            conversation_id: Conversation ID
            question: User question
            answer: Generated answer
            sources: Retrieved sources
        """
        timestamp = datetime.now().isoformat()
        
        # Add user message
        self.conversation_history.append({
            'conversation_id': conversation_id,
            'role': 'user',
            'content': question,
            'timestamp': timestamp,
            'sources': []
        })
        
        # Add assistant message
        self.conversation_history.append({
            'conversation_id': conversation_id,
            'role': 'assistant',
            'content': answer,
            'timestamp': timestamp,
            'sources': sources
        })
