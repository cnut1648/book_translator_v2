"""Memo tool for maintaining translation consistency."""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class TermEntry:
    """Represents a translation term entry."""
    source: str
    translation: str
    context: str
    category: str  # character, location, terminology, idiom, context-aware
    first_occurrence: str
    usage_count: int = 1
    variations: List[Dict[str, str]] = field(default_factory=list)


class TranslationMemory:
    """Manages translation memory for consistency."""
    
    def __init__(self, memory_file: Optional[Path] = None):
        """Initialize translation memory."""
        self.memory_file = memory_file or Path("tools/memo.txt")
        self.memory: Dict[str, Dict] = {
            "character_names": {},
            "locations": {},
            "terminology": {},
            "idioms": {},
            "context_aware": {},
            "summary": "",
            "last_updated": None
        }
        self.load_memory()
    
    def load_memory(self) -> None:
        """Load existing memory from file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        self.memory = json.loads(content)
                        logger.info(f"Loaded translation memory from {self.memory_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load memory file: {e}")
                self.save_memory()  # Create new file
        else:
            self.save_memory()  # Create initial file
    
    def save_memory(self) -> None:
        """Save memory to file."""
        self.memory["last_updated"] = datetime.now().isoformat()
        
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved translation memory to {self.memory_file}")
        except IOError as e:
            logger.error(f"Failed to save memory: {e}")
    
    def add_term(
        self,
        category: str,
        source: str,
        translation: str,
        context: str = "",
        variation: Optional[Dict[str, str]] = None
    ) -> None:
        """Add or update a term in memory."""
        category_map = {
            "character": "character_names",
            "character_names": "character_names",
            "location": "locations",
            "locations": "locations",
            "terminology": "terminology",
            "idiom": "idioms",
            "idioms": "idioms",
            "context": "context_aware",
            "context_aware": "context_aware"
        }
        
        memory_category = category_map.get(category.lower(), "terminology")
        
        if memory_category not in self.memory:
            self.memory[memory_category] = {}
        
        if source in self.memory[memory_category]:
            # Update existing entry
            entry = self.memory[memory_category][source]
            entry["usage_count"] = entry.get("usage_count", 1) + 1
            
            # Add variation if different translation in different context
            if variation and variation not in entry.get("variations", []):
                if "variations" not in entry:
                    entry["variations"] = []
                entry["variations"].append(variation)
        else:
            # Add new entry
            self.memory[memory_category][source] = {
                "translation": translation,
                "context": context,
                "first_occurrence": datetime.now().isoformat(),
                "usage_count": 1,
                "variations": [variation] if variation else []
            }
        
        self.save_memory()
    
    def get_term(self, source: str, category: Optional[str] = None) -> Optional[Dict]:
        """Get translation for a term."""
        if category:
            category_map = {
                "character": "character_names",
                "location": "locations",
                "terminology": "terminology",
                "idiom": "idioms",
                "context": "context_aware"
            }
            memory_category = category_map.get(category.lower())
            if memory_category and memory_category in self.memory:
                return self.memory[memory_category].get(source)
        else:
            # Search all categories
            for cat in ["character_names", "locations", "terminology", "idioms", "context_aware"]:
                if cat in self.memory and source in self.memory[cat]:
                    return self.memory[cat][source]
        return None
    
    def update_summary(self, summary: str) -> None:
        """Update the memory summary."""
        self.memory["summary"] = summary
        self.save_memory()
    
    def get_all_terms(self) -> Dict[str, Dict]:
        """Get all terms in memory."""
        return {
            "character_names": self.memory.get("character_names", {}),
            "locations": self.memory.get("locations", {}),
            "terminology": self.memory.get("terminology", {}),
            "idioms": self.memory.get("idioms", {}),
            "context_aware": self.memory.get("context_aware", {}),
            "summary": self.memory.get("summary", "")
        }
    
    def clear_memory(self) -> None:
        """Clear all memory."""
        self.memory = {
            "character_names": {},
            "locations": {},
            "terminology": {},
            "idioms": {},
            "context_aware": {},
            "summary": "",
            "last_updated": None
        }
        self.save_memory()
    
    def export_glossary(self, output_file: Path) -> None:
        """Export memory as a readable glossary."""
        glossary = []
        
        for category, terms in [
            ("Character Names", self.memory.get("character_names", {})),
            ("Locations", self.memory.get("locations", {})),
            ("Terminology", self.memory.get("terminology", {})),
            ("Idioms", self.memory.get("idioms", {})),
            ("Context-Aware Terms", self.memory.get("context_aware", {}))
        ]:
            if terms:
                glossary.append(f"\n## {category}\n")
                for source, data in sorted(terms.items()):
                    glossary.append(f"- **{source}** → {data['translation']}")
                    if data.get('context'):
                        glossary.append(f"  - Context: {data['context']}")
                    if data.get('variations'):
                        glossary.append(f"  - Variations: {data['variations']}")
                    glossary.append("")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(glossary))


def create_memo_tool_definition() -> Dict:
    """Create the tool definition for LLM to use."""
    return {
        "type": "function",
        "function": {
            "name": "update_translation_memory",
            "description": "Update the translation memory with terms, names, and expressions for consistency",
            "parameters": {
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": ["character", "location", "terminology", "idiom", "context_aware"],
                                    "description": "Category of the term"
                                },
                                "source": {
                                    "type": "string",
                                    "description": "Original term in source language"
                                },
                                "translation": {
                                    "type": "string",
                                    "description": "Translation in target language"
                                },
                                "context": {
                                    "type": "string",
                                    "description": "Context or usage notes"
                                }
                            },
                            "required": ["category", "source", "translation"]
                        },
                        "description": "List of terms to add to memory"
                    },
                    "summary_update": {
                        "type": "string",
                        "description": "Optional update to the overall translation summary"
                    }
                },
                "required": ["terms"]
            }
        }
    }


def handle_tool_call(tool_call: Dict, memory: TranslationMemory) -> str:
    """Handle a tool call from the LLM."""
    try:
        arguments = json.loads(tool_call["function"]["arguments"])
        
        # Add terms to memory
        for term in arguments.get("terms", []):
            memory.add_term(
                category=term["category"],
                source=term["source"],
                translation=term["translation"],
                context=term.get("context", "")
            )
        
        # Update summary if provided
        if arguments.get("summary_update"):
            memory.update_summary(arguments["summary_update"])
        
        return "Memory updated successfully"
        
    except Exception as e:
        logger.error(f"Error handling tool call: {e}")
        return f"Error updating memory: {str(e)}"