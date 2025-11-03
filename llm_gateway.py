"""LLM Gateway interface for translation tasks."""

import os
import json
import logging
import backoff
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMGateway:
    """LLM Gateway interface for the book translator."""
    
    model_name: str = "gpt-4o"
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_url: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 8192
    
    def __post_init__(self):
        """Initialize the LLM Gateway client."""
        self.client_id = self.client_id or os.getenv('LLM_GATEWAY_CLIENT_ID')
        self.client_secret = self.client_secret or os.getenv('LLM_GATEWAY_CLIENT_SECRET')
        self.token_url = self.token_url or os.getenv(
            'LLM_GATEWAY_TOKEN_URL',
            'https://5kbfxgaqc3xgz8nhid1x1r8cfestoypn-trofuum-oc.ssa.nvidia.com/token'
        )
        self.base_url = self.base_url or os.getenv(
            'LLM_GATEWAY_BASE_URL',
            'https://prod.api.nvidia.com/llm/v1/azure'
        )
        
        if not self.validate_credentials():
            raise ValueError(
                "LLM Gateway credentials not found. "
                "Set LLM_GATEWAY_CLIENT_ID and LLM_GATEWAY_CLIENT_SECRET environment variables"
            )
        
        self.access_token = self._get_access_token()
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.access_token,
                base_url=self.base_url,
                default_headers={
                    "dataClassification": "sensitive",
                    "dataSource": "internet"
                }
            )
        except ImportError:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    def validate_credentials(self) -> bool:
        """Check if credentials are available."""
        return bool(self.client_id and self.client_secret)
    
    def _get_access_token(self) -> str:
        """Get access token from the gateway."""
        token_data = {
            "grant_type": "client_credentials",
            "scope": "azureopenai-readwrite"
        }
        
        try:
            response = requests.post(
                self.token_url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=token_data,
                auth=HTTPBasicAuth(self.client_id, self.client_secret)
            )
            response.raise_for_status()
            return response.json()["access_token"]
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60
    )
    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Send chat request to LLM Gateway."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            
            message = response.choices[0].message
            result = {
                "content": message.content or "",
                "tool_calls": []
            }
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"LLM Gateway API error: {e}")
            raise
    
    def detect_hierarchy(self, text: str, previous_hierarchy: Optional[Dict] = None) -> Dict:
        """Detect hierarchy structure in text."""
        with open('prompts/hierarchy.txt', 'r') as f:
            prompt_template = f.read()
        
        prompt = prompt_template.format(
            previous_hierarchy=json.dumps(previous_hierarchy) if previous_hierarchy else "None",
            text=text
        )
        
        messages = [
            {"role": "system", "content": "You are a text structure analyzer. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat(messages)
        
        try:
            content = response.get("content", "")
            # Log the raw response for debugging
            logger.debug(f"Raw hierarchy response: {content[:500]}...")
            
            # Try to extract JSON from the response
            # Sometimes LLMs include markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Strip whitespace and parse
            content = content.strip()
            return json.loads(content)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse hierarchy response as JSON: {e}")
            logger.error(f"Response content: {response.get('content', '')[:200]}")
            return {"hierarchy": [], "has_more": False}
    
    def translate_chunk(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        genre_instructions: str,
        language_instructions: str,
        hierarchy_context: str,
        memory_context: Dict,
        tools: Optional[List[Dict]] = None,
        tool_executor=None
    ) -> Dict:
        """Translate a text chunk."""
        with open('prompts/translate.txt', 'r') as f:
            prompt_template = f.read()
        
        prompt = prompt_template.format(
            source_language=source_language,
            target_language=target_language,
            genre_instructions=genre_instructions,
            language_instructions=language_instructions,
            hierarchy_context=hierarchy_context,
            memory_context=json.dumps(memory_context, ensure_ascii=False),
            source_text=source_text
        )
        
        messages = [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat(messages, tools=tools)
        
        # If there are tool calls but no content, we need to handle them
        # and get the final response
        if response.get("tool_calls") and not response.get("content"):
            # Execute tools if executor provided
            tool_results = []
            if tool_executor:
                tool_results = tool_executor(response["tool_calls"])
            
            # Add the assistant's tool call message to the conversation
            # Format tool_calls properly for OpenAI
            formatted_tool_calls = []
            for tc in response["tool_calls"]:
                formatted_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"]
                })
            
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": formatted_tool_calls
            })
            
            # Add tool results
            for i, tool_call in enumerate(response["tool_calls"]):
                result_content = tool_results[i] if i < len(tool_results) else "Tool executed successfully"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(result_content)
                })
            
            # Get the final response with the translation
            response = self.chat(messages, tools=None)
        
        return response