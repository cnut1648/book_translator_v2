"""File I/O and formatting utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file input/output operations and formatting."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize file handler."""
        self.cache_dir = cache_dir or Path(".cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def read_source_file(self, file_path: Path) -> str:
        """Read source text file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        if not file_path.suffix == ".txt":
            raise ValueError(f"Currently only .txt files are supported. Got: {file_path.suffix}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Read {len(content)} characters from {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def save_translation_json(
        self,
        output_path: Path,
        translation_data: Dict,
        metadata: Optional[Dict] = None
    ) -> None:
        """Save translation in JSON format."""
        output_data = {
            "metadata": metadata or {
                "translated_at": datetime.now().isoformat(),
                "format_version": "1.0"
            },
            "translation": translation_data
        }
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved translation to {output_path}")
        except Exception as e:
            logger.error(f"Error saving translation: {e}")
            raise
    
    def save_translation_markdown(
        self,
        output_path: Path,
        translation_data: Dict,
        metadata: Optional[Dict] = None
    ) -> None:
        """Save translation in Markdown format."""
        md_content = []
        
        # Add metadata header
        if metadata:
            md_content.append("---")
            md_content.append(f"title: {metadata.get('title', 'Translation')}")
            md_content.append(f"source_language: {metadata.get('source_language', 'Unknown')}")
            md_content.append(f"target_language: {metadata.get('target_language', 'Unknown')}")
            md_content.append(f"translated_at: {metadata.get('translated_at', datetime.now().isoformat())}")
            md_content.append("---\n")
        
        # Process hierarchical content
        def format_hierarchy_level(item: Dict, level: int = 1) -> List[str]:
            """Format a hierarchy level for markdown."""
            lines = []
            
            # Add heading
            hierarchy_info = item.get("hierarchy_info", {})
            title = hierarchy_info.get("title", hierarchy_info.get("identifier", "Section"))
            lines.append(f"{'#' * min(level, 6)} {title}\n")
            
            # Add translations
            translations = item.get("translations", [])
            for trans in translations:
                if isinstance(trans, dict):
                    source = trans.get("source", "")
                    translation = trans.get("translation", "")
                    
                    # Format source in gray (using HTML for better rendering)
                    if source:
                        lines.append(f'<p style="color: #888; font-style: italic;">{source}</p>\n')
                    
                    # Format translation
                    if translation:
                        lines.append(f"{translation}\n")
                    
                    lines.append("")  # Empty line between paragraphs
            
            # Process sub-sections
            for subsection in item.get("subsections", []):
                lines.extend(format_hierarchy_level(subsection, level + 1))
            
            return lines
        
        # Format the main content
        if isinstance(translation_data, dict):
            if "sections" in translation_data:
                for section in translation_data["sections"]:
                    md_content.extend(format_hierarchy_level(section))
            elif "translations" in translation_data:
                # Flat structure
                for trans in translation_data["translations"]:
                    if isinstance(trans, dict):
                        source = trans.get("source", "")
                        translation = trans.get("translation", "")
                        
                        if source:
                            md_content.append(f'*{source}*\n')
                        if translation:
                            md_content.append(f"{translation}\n")
                        md_content.append("")
        
        # Save to file
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(md_content))
            logger.info(f"Saved translation to {output_path}")
        except Exception as e:
            logger.error(f"Error saving markdown: {e}")
            raise
    
    def save_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Cached data with key: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    def load_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.debug(f"Loaded cached data with key: {cache_key}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def format_hierarchical_translation(
        self,
        hierarchy: Dict,
        section_translations: List[Dict]
    ) -> Dict:
        """Format translations according to hierarchy structure.
        
        Args:
            hierarchy: Detected hierarchy structure
            section_translations: List of translated sections
        
        Returns:
            Hierarchically organized translation data
        """
        # Create a mapping of hierarchy items to translations
        translation_map = {}
        for section in section_translations:
            hierarchy_info = section.get("hierarchy_info", {})
            identifier = hierarchy_info.get("identifier", "")
            if identifier:
                translation_map[identifier] = section
        
        # Build hierarchical structure
        def build_hierarchy_tree(items: List[Dict], parent_id: Optional[str] = None) -> List[Dict]:
            """Build tree structure from flat hierarchy list."""
            result = []
            
            for item in items:
                if item.get("parent") == parent_id:
                    identifier = item.get("identifier")
                    node = {
                        "hierarchy_info": item,
                        "translations": [],
                        "subsections": []
                    }
                    
                    # Add translations for this section
                    if identifier in translation_map:
                        node["translations"] = translation_map[identifier].get("translations", [])
                    
                    # Find subsections
                    node["subsections"] = build_hierarchy_tree(items, identifier)
                    
                    result.append(node)
            
            return result
        
        hierarchy_items = hierarchy.get("hierarchy", [])
        
        if hierarchy_items:
            # Build tree from root level items
            root_items = build_hierarchy_tree(hierarchy_items, None)
            
            # If no root items found, return flat structure
            if not root_items:
                return {
                    "sections": section_translations,
                    "hierarchy": hierarchy
                }
            
            return {
                "sections": root_items,
                "hierarchy": hierarchy
            }
        else:
            # No hierarchy, return flat structure
            return {
                "translations": [
                    trans 
                    for section in section_translations 
                    for trans in section.get("translations", [])
                ],
                "hierarchy": {"hierarchy": [], "has_more": False}
            }
    
    def save_translation(
        self,
        output_path: Path,
        translation_data: Dict,
        format: str = "json",
        metadata: Optional[Dict] = None
    ) -> None:
        """Save translation in specified format.
        
        Args:
            output_path: Output file path
            translation_data: Translation data to save
            format: Output format (json or markdown)
            metadata: Optional metadata to include
        """
        if format == "json":
            self.save_translation_json(output_path, translation_data, metadata)
        elif format == "markdown" or format == "md":
            self.save_translation_markdown(output_path, translation_data, metadata)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'markdown'")
    
    def create_checkpoint(self, state: Dict, checkpoint_name: str) -> None:
        """Create a checkpoint of current translation state."""
        checkpoint_file = self.cache_dir / f"checkpoint_{checkpoint_name}.json"
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "state": state
        }
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Created checkpoint: {checkpoint_name}")
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict]:
        """Load a checkpoint."""
        checkpoint_file = self.cache_dir / f"checkpoint_{checkpoint_name}.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded checkpoint: {checkpoint_name}")
                return data.get("state")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        return None