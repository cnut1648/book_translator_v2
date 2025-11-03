"""Text processing utilities for chunking and hierarchy detection."""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    start_pos: int
    end_pos: int
    token_count: int
    chunk_index: int
    overlap_start: Optional[int] = None
    overlap_end: Optional[int] = None


class TextProcessor:
    """Handles text chunking and processing for translation."""
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize the text processor with tokenizer."""
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoder.encode(text))
    
    def split_into_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into paragraphs with their positions.
        
        Returns:
            List of tuples (paragraph_text, start_position, end_position)
        """
        paragraphs = []
        
        # Split by double newlines or multiple spaces/tabs followed by newlines
        pattern = r'\n\s*\n|\r\n\s*\r\n'
        
        parts = re.split(pattern, text)
        current_pos = 0
        
        for part in parts:
            part = part.strip()
            if part:
                start_pos = text.find(part, current_pos)
                end_pos = start_pos + len(part)
                paragraphs.append((part, start_pos, end_pos))
                current_pos = end_pos
        
        # If no paragraphs found, treat entire text as one paragraph
        if not paragraphs and text.strip():
            paragraphs = [(text.strip(), 0, len(text.strip()))]
        
        return paragraphs
    
    def create_chunks_with_max_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_percentage: float = 0.0
    ) -> List[TextChunk]:
        """Create chunks of text with maximum token limit.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_percentage: Percentage of overlap between chunks (0.0 to 0.2)
        
        Returns:
            List of TextChunk objects
        """
        paragraphs = self.split_into_paragraphs(text)
        chunks = []
        chunk_index = 0
        
        i = 0
        while i < len(paragraphs):
            current_chunk = []
            current_tokens = 0
            start_pos = paragraphs[i][1]
            
            # Add paragraphs to current chunk while under token limit
            while i < len(paragraphs):
                para_text, para_start, para_end = paragraphs[i]
                para_tokens = self.count_tokens(para_text)
                
                # Check if adding this paragraph would exceed limit
                if current_tokens + para_tokens > max_tokens:
                    if current_chunk:
                        # Chunk is full, break
                        break
                    else:
                        # Single paragraph exceeds limit, need to split it
                        logger.warning(f"Single paragraph exceeds {max_tokens} tokens, splitting...")
                        split_chunks = self._split_large_paragraph(
                            para_text, para_start, max_tokens, chunk_index
                        )
                        chunks.extend(split_chunks)
                        chunk_index += len(split_chunks)
                        i += 1
                        continue
                
                current_chunk.append((para_text, para_start, para_end))
                current_tokens += para_tokens
                i += 1
            
            if current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = '\n\n'.join([p[0] for p in current_chunk])
                end_pos = current_chunk[-1][2]
                
                chunk = TextChunk(
                    content=chunk_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    token_count=current_tokens,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Handle overlap if specified
                if overlap_percentage > 0 and i < len(paragraphs):
                    overlap_size = int(len(current_chunk) * overlap_percentage)
                    if overlap_size > 0:
                        # Move back to create overlap
                        i = max(0, i - overlap_size)
                        if chunks:
                            chunks[-1].overlap_end = i
        
        return chunks
    
    def _split_large_paragraph(
        self,
        text: str,
        start_pos: int,
        max_tokens: int,
        chunk_index: int
    ) -> List[TextChunk]:
        """Split a large paragraph that exceeds token limit.
        
        Splits by sentences first, then by arbitrary positions if needed.
        """
        chunks = []
        
        # Try splitting by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) > 1:
            # Create chunks from sentences
            current_sentences = []
            current_tokens = 0
            current_start = start_pos
            
            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)
                
                if current_tokens + sentence_tokens > max_tokens:
                    if current_sentences:
                        # Save current chunk
                        chunk_text = ' '.join(current_sentences)
                        chunk = TextChunk(
                            content=chunk_text,
                            start_pos=current_start,
                            end_pos=current_start + len(chunk_text),
                            token_count=current_tokens,
                            chunk_index=chunk_index + len(chunks)
                        )
                        chunks.append(chunk)
                        current_sentences = []
                        current_tokens = 0
                        current_start += len(chunk_text) + 1
                
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
            
            # Add remaining sentences
            if current_sentences:
                chunk_text = ' '.join(current_sentences)
                chunk = TextChunk(
                    content=chunk_text,
                    start_pos=current_start,
                    end_pos=current_start + len(chunk_text),
                    token_count=current_tokens,
                    chunk_index=chunk_index + len(chunks)
                )
                chunks.append(chunk)
        else:
            # Can't split by sentences, split by token count
            tokens = self.encoder.encode(text)
            current_pos = start_pos
            
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i:i + max_tokens]
                chunk_text = self.encoder.decode(chunk_tokens)
                
                chunk = TextChunk(
                    content=chunk_text,
                    start_pos=current_pos,
                    end_pos=current_pos + len(chunk_text),
                    token_count=len(chunk_tokens),
                    chunk_index=chunk_index + len(chunks)
                )
                chunks.append(chunk)
                current_pos += len(chunk_text)
        
        return chunks
    
    def merge_hierarchies(self, hierarchies: List[Dict]) -> Dict:
        """Merge multiple hierarchy detections into a single structure."""
        merged = {"hierarchy": [], "has_more": False}
        
        for h in hierarchies:
            if isinstance(h, dict) and "hierarchy" in h:
                merged["hierarchy"].extend(h.get("hierarchy", []))
                merged["has_more"] = merged["has_more"] or h.get("has_more", False)
        
        # Remove duplicates based on identifier
        seen_ids = set()
        unique_hierarchy = []
        for item in merged["hierarchy"]:
            if item.get("identifier") not in seen_ids:
                seen_ids.add(item.get("identifier"))
                unique_hierarchy.append(item)
        
        merged["hierarchy"] = unique_hierarchy
        return merged
    
    def split_by_hierarchy(self, text: str, hierarchy: Dict) -> List[Dict]:
        """Split text according to detected hierarchy.
        
        Returns:
            List of dictionaries containing hierarchy info and text content
        """
        sections = []
        hierarchy_items = sorted(
            hierarchy.get("hierarchy", []),
            key=lambda x: x.get("start_position", 0)
        )
        
        for i, item in enumerate(hierarchy_items):
            start_pos = item.get("start_position", 0)
            
            # Determine end position
            if i + 1 < len(hierarchy_items):
                end_pos = hierarchy_items[i + 1].get("start_position", len(text))
            else:
                end_pos = item.get("end_position", len(text))
            
            section_text = text[start_pos:end_pos].strip()
            
            sections.append({
                "hierarchy_info": item,
                "content": section_text,
                "start_pos": start_pos,
                "end_pos": end_pos
            })
        
        # If no hierarchy detected, return entire text as one section
        if not sections and text.strip():
            sections = [{
                "hierarchy_info": {
                    "level": 0,
                    "type": "document",
                    "identifier": "main",
                    "title": "Main Content"
                },
                "content": text.strip(),
                "start_pos": 0,
                "end_pos": len(text)
            }]
        
        return sections