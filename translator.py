#!/usr/bin/env python3
"""Main entry point and orchestrator for book translation."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config import TranslationConfig
from llm_gateway import LLMGateway
from text_processor import TextProcessor, TextChunk
from file_handler import FileHandler
from tools.memo import TranslationMemory, create_memo_tool_definition, handle_tool_call

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BookTranslator:
    """Main orchestrator for book translation."""
    
    def __init__(self, config: Optional[TranslationConfig] = None):
        """Initialize the book translator."""
        self.config = config or TranslationConfig()
        self.llm = LLMGateway(temperature=self.config.temperature)
        self.text_processor = TextProcessor()
        self.file_handler = FileHandler(cache_dir=self.config.cache_dir)
        self.memory = TranslationMemory()
        self.current_source_file = None  # Track current source file for cache keys
        
        # Load prompt templates
        self.genre_instructions = self._load_prompt(self.config.genre_prompt_path)
        self.language_instructions = self._load_prompt(self.config.language_prompt_path)
    
    def _load_prompt(self, path: Path) -> str:
        """Load a prompt template file."""
        if path and path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def _get_cache_key(self, text: str, prefix: str = "") -> str:
        """Generate a cache key for text including source file."""
        # Include source file name in cache key to avoid collisions
        if self.current_source_file:
            file_part = self.current_source_file.stem[:20]  # First 20 chars of filename
        else:
            file_part = "unknown"
        
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        
        if prefix:
            return f"{file_part}_{prefix}_{text_hash}"
        else:
            return f"{file_part}_{text_hash}"
    
    def detect_hierarchy(self, text: str) -> Dict:
        """Detect hierarchy structure in the entire text.
        
        Args:
            text: Source text to analyze
        
        Returns:
            Detected hierarchy structure
        """
        logger.info("Starting hierarchy detection...")
        
        # Check cache
        cache_key = self._get_cache_key(text, "hierarchy")
        if self.config.enable_cache:
            cached = self.file_handler.load_cache(cache_key)
            if cached:
                logger.info("Using cached hierarchy")
                return cached
        
        # Create chunks for hierarchy detection
        chunks = self.text_processor.create_chunks_with_max_tokens(
            text,
            self.config.chunk_context_size
        )
        
        logger.info(f"Created {len(chunks)} chunks for hierarchy detection")
        
        hierarchies = []
        previous_hierarchy = None
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Detecting hierarchy in chunk {i + 1}/{len(chunks)}")
            
            hierarchy = self.llm.detect_hierarchy(
                chunk.content,
                previous_hierarchy
            )
            
            hierarchies.append(hierarchy)
            previous_hierarchy = hierarchy
            
            # Save checkpoint
            if self.config.enable_cache:
                self.file_handler.create_checkpoint(
                    {"hierarchies": hierarchies, "chunk_index": i},
                    f"hierarchy_{i}"
                )
        
        # Merge all hierarchies
        final_hierarchy = self.text_processor.merge_hierarchies(hierarchies)
        
        # Cache result
        if self.config.enable_cache:
            self.file_handler.save_cache(cache_key, final_hierarchy)
        
        logger.info(f"Detected {len(final_hierarchy.get('hierarchy', []))} hierarchy items")
        return final_hierarchy
    
    def translate_section(
        self,
        section: Dict,
        section_index: int,
        total_sections: int
    ) -> Dict:
        """Translate a single section.
        
        Args:
            section: Section data with hierarchy info and content
            section_index: Current section index
            total_sections: Total number of sections
        
        Returns:
            Translated section data
        """
        logger.info(
            f"Translating section {section_index + 1}/{total_sections}: "
            f"{section['hierarchy_info'].get('title', 'Untitled')}"
        )
        
        content = section["content"]
        hierarchy_info = section["hierarchy_info"]
        
        # Create chunks with overlap for this section
        chunks = self.text_processor.create_chunks_with_max_tokens(
            content,
            self.config.translate_context_size,
            self.config.overlap_percentage
        )
        
        logger.info(f"Section split into {len(chunks)} chunks for translation")
        
        translations = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"  Translating chunk {i + 1}/{len(chunks)}")
            
            # Check cache
            cache_key = self._get_cache_key(chunk.content, f"trans_{section_index}_{i}")
            if self.config.enable_cache:
                cached = self.file_handler.load_cache(cache_key)
                if cached:
                    translations.extend(cached)
                    continue
            
            # Prepare hierarchy context
            hierarchy_context = json.dumps(hierarchy_info, ensure_ascii=False)
            
            # Get current memory context
            memory_context = self.memory.get_all_terms()
            
            # Create memo tool
            memo_tool = create_memo_tool_definition()
            
            # Translate chunk
            max_retries = self.config.max_retries
            retry_count = 0
            chunk_translations = None
            
            while retry_count < max_retries:
                try:
                    # Pass tool execution handler to LLM
                    def execute_tools(tool_calls):
                        results = []
                        for tool_call in tool_calls:
                            if tool_call["function"]["name"] == "update_translation_memory":
                                result = handle_tool_call(tool_call, self.memory)
                                results.append(result)
                                logger.debug(f"Tool call result: {result}")
                        return results
                    
                    response = self.llm.translate_chunk(
                        source_text=chunk.content,
                        source_language=self.config.source_language,
                        target_language=self.config.target_language,
                        genre_instructions=self.genre_instructions,
                        language_instructions=self.language_instructions,
                        hierarchy_context=hierarchy_context,
                        memory_context=memory_context,
                        tools=[memo_tool],
                        tool_executor=execute_tools
                    )
                    
                    # Parse translation response
                    content = response.get("content", "")
                    if content:
                        try:
                            # Log raw response for debugging
                            logger.debug(f"Raw translation response (first 500 chars): {content[:500]}...")
                            
                            # Clean up content - remove markdown code blocks if present
                            if "```json" in content:
                                content = content.split("```json")[1].split("```")[0]
                            elif "```" in content:
                                content = content.split("```")[1].split("```")[0]
                            
                            # Try to parse as JSON array
                            chunk_translations = json.loads(content.strip())
                            if isinstance(chunk_translations, list):
                                break
                        except json.JSONDecodeError:
                            # Try to extract JSON from the response
                            import re
                            json_match = re.search(r'\[.*\]', content, re.DOTALL)
                            if json_match:
                                try:
                                    chunk_translations = json.loads(json_match.group())
                                    break
                                except json.JSONDecodeError:
                                    logger.debug(f"Failed to parse extracted JSON: {json_match.group()[:200]}")
                                    pass
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Failed to parse translation, retrying... ({retry_count}/{max_retries})")
                        logger.warning(f"Response content: {content[:500] if content else 'Empty response'}")
                    
                except Exception as e:
                    logger.error(f"Translation error: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
            
            if chunk_translations:
                translations.extend(chunk_translations)
                
                # Cache result
                if self.config.enable_cache:
                    self.file_handler.save_cache(cache_key, chunk_translations)
            else:
                logger.error(f"Failed to translate chunk {i + 1}")
                # Add placeholder translation
                translations.append({
                    "source": chunk.content,
                    "translation": "[Translation failed]",
                    "paragraph_index": i
                })
        
        # Verify all paragraphs were translated
        source_paragraphs = self.text_processor.split_into_paragraphs(content)
        if len(translations) < len(source_paragraphs):
            logger.warning(
                f"Translation count mismatch: {len(translations)} translations "
                f"for {len(source_paragraphs)} paragraphs"
            )
        
        return {
            "hierarchy_info": hierarchy_info,
            "translations": translations
        }
    
    def translate_book(
        self,
        source_file: Path,
        output_file: Optional[Path] = None,
        output_format: str = "json",
        resume_from: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> Path:
        """Translate an entire book.
        
        Args:
            source_file: Path to source text file
            output_file: Optional output file path
            output_format: Output format (json or markdown)
            resume_from: Optional checkpoint name to resume from
        
        Returns:
            Path to output file
        """
        logger.info(f"Starting translation of {source_file}")
        
        # Store source file for cache key generation
        self.current_source_file = Path(source_file)
        
        # Set up output directory structure
        if output_dir is None:
            # Default: create directory based on input filename
            output_dir = Path(f"{source_file.stem}_output")
        else:
            output_dir = Path(output_dir)
        
        # Create output subdirectories
        output_dir.mkdir(exist_ok=True)
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Update file handler cache directory
        self.file_handler.cache_dir = cache_dir
        
        # Update memory file location
        memo_file = output_dir / "translation_memory.json"
        self.memory.memory_file = memo_file
        if memo_file.exists():
            self.memory.load_memory()
        
        # Read source file
        source_text = self.file_handler.read_source_file(Path(source_file))
        
        # Set output file
        if output_file is None:
            output_file = output_dir / f"{source_file.stem}.translated.{output_format}"
        else:
            output_file = Path(output_file)
        
        # Check for resume
        if resume_from:
            checkpoint = self.file_handler.load_checkpoint(resume_from)
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {resume_from}")
                hierarchy = checkpoint.get("hierarchy")
                completed_sections = checkpoint.get("completed_sections", [])
            else:
                logger.warning(f"Checkpoint {resume_from} not found, starting fresh")
                hierarchy = None
                completed_sections = []
        else:
            hierarchy = None
            completed_sections = []
        
        # Step 1: Detect hierarchy
        if hierarchy is None:
            hierarchy = self.detect_hierarchy(source_text)
        
        # Step 2: Split text by hierarchy
        sections = self.text_processor.split_by_hierarchy(source_text, hierarchy)
        logger.info(f"Split text into {len(sections)} sections")
        
        # Step 3: Translate each section
        section_translations = []
        
        for i, section in enumerate(sections):
            section_id = section["hierarchy_info"].get("identifier", str(i))
            
            # Skip if already completed
            if section_id in completed_sections:
                logger.info(f"Skipping already completed section: {section_id}")
                # Load from cache
                cached = self.file_handler.load_cache(f"section_{section_id}")
                if cached:
                    section_translations.append(cached)
                continue
            
            # Translate section
            translated_section = self.translate_section(section, i, len(sections))
            section_translations.append(translated_section)
            
            # Cache section translation
            if self.config.enable_cache:
                self.file_handler.save_cache(f"section_{section_id}", translated_section)
            
            # Update checkpoint
            completed_sections.append(section_id)
            if self.config.enable_cache:
                self.file_handler.create_checkpoint(
                    {
                        "hierarchy": hierarchy,
                        "completed_sections": completed_sections,
                        "section_translations": section_translations
                    },
                    f"translation_{i}"
                )
        
        # Step 4: Format and save translation
        formatted_translation = self.file_handler.format_hierarchical_translation(
            hierarchy,
            section_translations
        )
        
        metadata = {
            "source_file": str(source_file),
            "source_language": self.config.source_language,
            "target_language": self.config.target_language,
            "genre": self.config.genre,
            "translated_at": datetime.now().isoformat(),
            "total_sections": len(sections)
        }
        
        self.file_handler.save_translation(
            output_file,
            formatted_translation,
            format=output_format,
            metadata=metadata
        )
        
        # Export glossary to output directory
        glossary_file = output_dir / "glossary.md"
        self.memory.export_glossary(glossary_file)
        logger.info(f"Exported translation glossary to {glossary_file}")
        
        logger.info(f"Translation completed! Output saved to {output_file}")
        return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Translate books with LLM")
    parser.add_argument("source", help="Source text file (.txt)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-f", "--format",
        choices=["json", "markdown", "md"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "-g", "--genre",
        default="philosophy",
        help="Genre of the text (default: philosophy)"
    )
    parser.add_argument(
        "--source-lang",
        default="English",
        help="Source language (default: English)"
    )
    parser.add_argument(
        "--target-lang",
        default="Chinese",
        help="Target language (default: Chinese)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Chunk size for hierarchy detection in tokens (default: 8192)"
    )
    parser.add_argument(
        "--translate-size",
        type=int,
        default=8192,
        help="Context size for translation in tokens (default: 8192)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for all outputs (cache, logs, glossary). Defaults to ./{input_name}_output/"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    parser.add_argument(
        "--resume",
        help="Resume from checkpoint name"
    )
    parser.add_argument(
        "--clear-memory",
        action="store_true",
        help="Clear translation memory before starting"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = TranslationConfig(
        chunk_context_size=args.chunk_size,
        translate_context_size=args.translate_size,
        source_language=args.source_lang,
        target_language=args.target_lang,
        genre=args.genre,
        enable_cache=not args.no_cache
    )
    
    # Create translator
    translator = BookTranslator(config)
    
    # Clear memory if requested
    if args.clear_memory:
        translator.memory.clear_memory()
        logger.info("Cleared translation memory")
    
    # Setup output directory and logging
    source_path = Path(args.source)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"{source_path.stem}_output")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging to file in output directory
    log_file = output_dir / "translation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)
    
    # Run translation
    try:
        output_path = translator.translate_book(
            source_file=source_path,
            output_file=Path(args.output) if args.output else None,
            output_format=args.format,
            resume_from=args.resume,
            output_dir=output_dir
        )
        print(f"Translation completed successfully!")
        print(f"Output directory: {output_dir}")
        print(f"  - Translation: {output_path}")
        print(f"  - Glossary: {output_dir}/translation_memory.json")
        print(f"  - Log file: {log_file}")
        print(f"  - Cache: {output_dir}/cache/")
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()