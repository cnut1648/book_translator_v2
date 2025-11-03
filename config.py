"""Configuration settings for the book translator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TranslationConfig:
    """Configuration for translation settings."""
    
    # Chunk sizes in tokens
    chunk_context_size: int = 8192
    translate_context_size: int = 8192
    
    # Overlap percentage for translation chunks (10-20%)
    overlap_percentage: float = 0.15
    
    # Language settings
    source_language: str = "English"
    target_language: str = "Chinese"
    
    # Genre and language prompt paths
    genre: str = "philosophy"  # Default genre
    genre_prompt_path: Optional[Path] = None
    language_prompt_path: Optional[Path] = None
    
    # File paths
    prompts_dir: Path = Path(__file__).parent / "prompts"
    tools_dir: Path = Path(__file__).parent / "tools"
    
    # LLM settings
    max_retries: int = 3
    temperature: float = 0.3
    
    # Cache settings
    cache_dir: Path = Path(__file__).parent / ".cache"
    enable_cache: bool = True
    
    # Output settings
    output_format: str = "json"  # json or markdown
    
    def __post_init__(self):
        """Initialize derived paths."""
        if self.genre_prompt_path is None:
            self.genre_prompt_path = self.prompts_dir / "genre" / f"{self.genre}.txt"
        
        if self.language_prompt_path is None:
            lang_code = self.target_language.lower()
            self.language_prompt_path = self.prompts_dir / "lang" / f"{lang_code}.txt"
        
        # Create cache directory if caching is enabled
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = TranslationConfig()