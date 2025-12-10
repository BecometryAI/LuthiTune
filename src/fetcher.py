"""
LuthiTune Data Fetcher
----------------------
A browser automation tool that intercepts network traffic from AI chat interfaces
to collect training data by capturing user prompts and AI responses.

This module uses Selenium WebDriver with Chrome DevTools Protocol (CDP) to:
1. Monitor network requests/responses
2. Capture POST requests to chat endpoints
3. Extract conversation pairs from JSON payloads
4. Wait for complete response streams before logging
"""

import json
import time
import re
import os
import sys
import urllib.parse
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for text length thresholds
MIN_TEXT_LENGTH = 10  # Minimum length for substantial text in nested structures
MIN_RAW_TEXT_LENGTH = 20  # Minimum length for raw text extraction


class NetworkInterceptor:
    """
    Handles Chrome DevTools Protocol (CDP) network interception
    to capture requests and responses.
    """
    
    def __init__(self, driver):
        """
        Initialize the network interceptor.
        
        Args:
            driver: Selenium WebDriver instance
        """
        self.driver = driver
        self.request_map = {}  # Maps request IDs to request data
        self.captured_pairs = []  # Stores captured conversation pairs
        self.pending_requests = {}  # Tracks pending requests waiting for responses
        
    def enable_network_monitoring(self):
        """
        Enable Chrome DevTools Protocol network monitoring.
        """
        self.driver.execute_cdp_cmd('Network.enable', {})
        logger.info("Network monitoring enabled via CDP")
        
    def add_request_interceptor(self):
        """
        Set up request interception to capture outgoing requests.
        """
        # Enable network interception
        self.driver.execute_cdp_cmd('Network.setRequestInterception', {
            'patterns': [{'urlPattern': '*'}]
        })
        logger.info("Request interceptor configured")
        
    def get_network_logs(self) -> List[Dict]:
        """
        Retrieve network logs from the browser.
        
        Returns:
            List of network log entries
        """
        logs = self.driver.get_log('performance')
        return logs
    
    def parse_network_log(self, log_entry: Dict) -> Optional[Dict]:
        """
        Parse a network log entry.
        
        Args:
            log_entry: Raw log entry from Chrome DevTools
            
        Returns:
            Parsed log data or None if not relevant
        """
        try:
            message = json.loads(log_entry['message'])
            return message['message']
        except (json.JSONDecodeError, KeyError):
            return None
    
    def is_chat_endpoint(self, url: str) -> bool:
        """
        Check if a URL is a chat endpoint.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL matches chat endpoint patterns
        """
        patterns = [
            r'batchexecute',
            r'stream',
            r'/chat',
            r'/conversation',
            r'/api/chat',
            r'/v1/chat',
            r'/completions',
        ]
        
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in patterns)
    
    def extract_request_payload(self, request_data: Dict) -> Optional[str]:
        """
        Extract the user's prompt from a request payload.
        
        Args:
            request_data: The request data structure
            
        Returns:
            Extracted user prompt or None
        """
        try:
            # Try to get post data
            if 'postData' in request_data:
                post_data = request_data['postData']
                
                # Try parsing as JSON
                try:
                    payload = json.loads(post_data)
                    
                    # Common patterns for different chat APIs
                    # OpenAI-style
                    if 'messages' in payload:
                        for msg in payload['messages']:
                            if msg.get('role') == 'user':
                                return msg.get('content', '')
                    
                    # Anthropic Claude-style
                    if 'prompt' in payload:
                        return payload['prompt']
                    
                    # Google Bard/Gemini RPC style (nested arrays)
                    if isinstance(payload, list):
                        return self._extract_from_rpc_array(payload)
                    
                    # Generic text field search
                    return self._search_for_text_field(payload)
                    
                except json.JSONDecodeError:
                    # Handle form-encoded or other formats
                    if 'f.req=' in post_data:
                        # Google-style RPC
                        return self._extract_from_google_rpc(post_data)
                    
                    # Try to extract any text content
                    return self._extract_text_from_raw(post_data)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting request payload: {e}")
            return None
    
    def extract_response_body(self, response_data: Dict, request_id: str) -> Optional[str]:
        """
        Extract the AI's response from a response body.
        
        Args:
            response_data: The response data structure
            request_id: The request ID to fetch the response body
            
        Returns:
            Extracted AI response or None
        """
        try:
            # Get response body using CDP
            response_body = self.driver.execute_cdp_cmd(
                'Network.getResponseBody',
                {'requestId': request_id}
            )
            
            body = response_body.get('body', '')
            
            if not body:
                return None
            
            # Try parsing as JSON
            try:
                payload = json.loads(body)
                
                # OpenAI-style
                if 'choices' in payload:
                    for choice in payload['choices']:
                        if 'message' in choice:
                            return choice['message'].get('content', '')
                        if 'text' in choice:
                            return choice['text']
                
                # Anthropic Claude-style
                if 'completion' in payload:
                    return payload['completion']
                
                # Generic text field search
                return self._search_for_text_field(payload)
                
            except json.JSONDecodeError:
                # Handle streaming or non-JSON responses
                return self._extract_text_from_raw(body)
            
        except Exception as e:
            logger.debug(f"Error extracting response body: {e}")
            return None
    
    def _extract_from_rpc_array(self, data: List) -> Optional[str]:
        """
        Extract text from nested RPC array structure (Google-style).
        
        Args:
            data: Nested array structure
            
        Returns:
            Extracted text or None
        """
        def search_array(arr):
            if isinstance(arr, str):
                # Found a string, check if it's substantial
                if len(arr) > MIN_TEXT_LENGTH:
                    return arr
            elif isinstance(arr, list):
                for item in arr:
                    result = search_array(item)
                    if result:
                        return result
            elif isinstance(arr, dict):
                return self._search_for_text_field(arr)
            return None
        
        return search_array(data)
    
    def _search_for_text_field(self, data: Dict) -> Optional[str]:
        """
        Recursively search for text fields in nested structures.
        
        Args:
            data: Dictionary to search
            
        Returns:
            First substantial text field found or None
        """
        # Common field names for text content
        text_fields = ['text', 'content', 'message', 'prompt', 'completion', 
                       'response', 'answer', 'output', 'result']
        
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                if len(data[field]) > MIN_TEXT_LENGTH:
                    return data[field]
        
        # Recursive search
        for key, value in data.items():
            if isinstance(value, dict):
                result = self._search_for_text_field(value)
                if result:
                    return result
            elif isinstance(value, list):
                result = self._extract_from_rpc_array(value)
                if result:
                    return result
        
        return None
    
    def _extract_from_google_rpc(self, post_data: str) -> Optional[str]:
        """
        Extract text from Google-style RPC format (f.req=).
        
        Args:
            post_data: Raw POST data string
            
        Returns:
            Extracted text or None
        """
        try:
            # Extract the f.req parameter
            match = re.search(r'f\.req=([^&]+)', post_data)
            if match:
                # URL decode and parse
                decoded = urllib.parse.unquote(match.group(1))
                
                # Try to parse as JSON array
                try:
                    data = json.loads(decoded)
                    return self._extract_from_rpc_array(data)
                except json.JSONDecodeError:
                    pass
            
            return None
        except Exception as e:
            logger.debug(f"Error extracting from Google RPC: {e}")
            return None
    
    def _extract_text_from_raw(self, text: str) -> Optional[str]:
        """
        Extract meaningful text from raw string data.
        
        Args:
            text: Raw text to extract from
            
        Returns:
            Extracted text or None
        """
        # For streaming responses, try to extract the actual content
        # Remove common prefixes/suffixes
        text = text.strip()
        
        # Handle SSE (Server-Sent Events) format
        if text.startswith('data:'):
            lines = text.split('\n')
            accumulated = []
            for line in lines:
                if line.startswith('data:'):
                    data = line[5:].strip()
                    if data and data != '[DONE]':
                        try:
                            chunk = json.loads(data)
                            # Extract text from chunk
                            if 'choices' in chunk:
                                for choice in chunk['choices']:
                                    if 'delta' in choice and 'content' in choice['delta']:
                                        accumulated.append(choice['delta']['content'])
                        except json.JSONDecodeError:
                            pass
            
            if accumulated:
                return ''.join(accumulated)
        
        # Return raw text if substantial
        if len(text) > MIN_RAW_TEXT_LENGTH:
            return text
        
        return None


class ConversationFetcher:
    """
    Main fetcher class that uses browser automation to collect
    conversation data from AI chat interfaces.
    """
    
    def __init__(self, headless: bool = False, chrome_driver_path: Optional[str] = None):
        """
        Initialize the conversation fetcher.
        
        Args:
            headless: Whether to run browser in headless mode
            chrome_driver_path: Path to chromedriver (None for auto-detect)
        """
        self.headless = headless
        self.chrome_driver_path = chrome_driver_path
        self.driver = None
        self.interceptor = None
        self.conversations = []
        self.current_conversation = []
        self.is_monitoring = False
        
    def start(self, url: str = "about:blank"):
        """
        Start the browser and initialize network monitoring.
        
        Args:
            url: Initial URL to navigate to
        """
        logger.info("Starting browser automation...")
        
        # Configure Chrome options
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless=new')
        
        # Enable performance logging for network capture
        chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
        
        # Additional options for stability
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Initialize driver
        if self.chrome_driver_path:
            service = Service(self.chrome_driver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            self.driver = webdriver.Chrome(options=chrome_options)
        
        # Initialize network interceptor
        self.interceptor = NetworkInterceptor(self.driver)
        self.interceptor.enable_network_monitoring()
        
        # Navigate to URL
        if url != "about:blank":
            self.driver.get(url)
        
        logger.info(f"Browser started and navigated to {url}")
        self.is_monitoring = True
        
    def stop(self):
        """
        Stop the browser and cleanup.
        """
        if self.driver:
            logger.info("Stopping browser...")
            self.driver.quit()
            self.driver = None
            self.is_monitoring = False
            logger.info("Browser stopped")
    
    def navigate(self, url: str):
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
        """
        if not self.driver:
            raise RuntimeError("Browser not started. Call start() first.")
        
        logger.info(f"Navigating to {url}")
        self.driver.get(url)
        time.sleep(2)  # Wait for page load
    
    def monitor_and_capture(self, duration: int = 60, check_interval: int = 2):
        """
        Monitor network traffic and capture conversation pairs.
        
        Args:
            duration: How long to monitor (seconds)
            check_interval: How often to check for new logs (seconds)
        """
        if not self.is_monitoring:
            raise RuntimeError("Monitoring not started. Call start() first.")
        
        logger.info(f"Monitoring network traffic for {duration} seconds...")
        start_time = time.time()
        processed_request_ids = set()
        
        while time.time() - start_time < duration:
            # Get network logs
            logs = self.interceptor.get_network_logs()
            
            for log_entry in logs:
                parsed = self.interceptor.parse_network_log(log_entry)
                
                if not parsed:
                    continue
                
                method = parsed.get('method', '')
                
                # Capture request
                if method == 'Network.requestWillBeSent':
                    params = parsed.get('params', {})
                    request_id = params.get('requestId')
                    request = params.get('request', {})
                    url = request.get('url', '')
                    
                    if request_id not in processed_request_ids and self.interceptor.is_chat_endpoint(url):
                        logger.info(f"Detected chat request to: {url}")
                        
                        # Extract user prompt
                        user_prompt = self.interceptor.extract_request_payload(request)
                        
                        if user_prompt:
                            # Store pending request
                            self.interceptor.pending_requests[request_id] = {
                                'url': url,
                                'user_prompt': user_prompt,
                                'timestamp': datetime.now().isoformat(),
                                'response_complete': False
                            }
                            logger.info(f"Captured user prompt: {user_prompt[:100]}...")
                
                # Capture response
                elif method == 'Network.responseReceived':
                    params = parsed.get('params', {})
                    request_id = params.get('requestId')
                    
                    if request_id in self.interceptor.pending_requests:
                        # Mark as received (but may not be complete for streaming)
                        self.interceptor.pending_requests[request_id]['response_received'] = True
                
                # Capture loading finished (response complete)
                elif method == 'Network.loadingFinished':
                    params = parsed.get('params', {})
                    request_id = params.get('requestId')
                    
                    if request_id in self.interceptor.pending_requests and request_id not in processed_request_ids:
                        pending = self.interceptor.pending_requests[request_id]
                        
                        # Extract AI response
                        ai_response = self.interceptor.extract_response_body({}, request_id)
                        
                        if ai_response:
                            # Create conversation pair
                            pair = {
                                'user': pending['user_prompt'],
                                'assistant': ai_response,
                                'timestamp': pending['timestamp'],
                                'url': pending['url']
                            }
                            
                            self.interceptor.captured_pairs.append(pair)
                            self.current_conversation.append(pair)
                            processed_request_ids.add(request_id)
                            
                            logger.info(f"Captured AI response: {ai_response[:100]}...")
                            logger.info(f"✓ Conversation pair captured successfully")
                        
                        # Clean up
                        del self.interceptor.pending_requests[request_id]
            
            # Wait before next check
            time.sleep(check_interval)
        
        logger.info(f"Monitoring complete. Captured {len(self.interceptor.captured_pairs)} conversation pairs")
    
    def get_captured_conversations(self) -> List[Dict]:
        """
        Get all captured conversation pairs.
        
        Returns:
            List of conversation pairs
        """
        return self.interceptor.captured_pairs if self.interceptor else []
    
    def start_new_conversation(self):
        """
        Start tracking a new conversation session.
        """
        if self.current_conversation:
            self.conversations.append(self.current_conversation)
        self.current_conversation = []
        logger.info("Started new conversation session")
    
    def export_to_json(self, output_path: str, include_metadata: bool = True):
        """
        Export captured conversations to JSON file.
        
        Args:
            output_path: Path to save the JSON file
            include_metadata: Whether to include metadata (timestamp, URL)
        """
        conversations = self.get_captured_conversations()
        
        if not include_metadata:
            # Strip metadata
            conversations = [
                {'user': conv['user'], 'assistant': conv['assistant']}
                for conv in conversations
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(conversations)} conversations to {output_path}")
    
    def export_to_jsonl(self, output_path: str, include_metadata: bool = True):
        """
        Export captured conversations to JSONL file (one per line).
        
        Args:
            output_path: Path to save the JSONL file
            include_metadata: Whether to include metadata (timestamp, URL)
        """
        conversations = self.get_captured_conversations()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                if not include_metadata:
                    entry = {'user': conv['user'], 'assistant': conv['assistant']}
                else:
                    entry = conv
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(conversations)} conversations to {output_path}")


def validate_output_path(path: str) -> str:
    """
    Validate and sanitize output file path to prevent path traversal attacks.
    
    Args:
        path: The output file path to validate
        
    Returns:
        Validated absolute path
        
    Raises:
        ValueError: If path contains malicious characters or patterns
    """
    # Get absolute path
    abs_path = os.path.abspath(path)
    
    # Check for path traversal attempts
    if '..' in path or path.startswith('/'):
        # Only allow if it's an explicit absolute path in safe locations
        safe_prefixes = [
            os.path.abspath('data/'),
            os.path.abspath('/tmp/'),
            os.path.expanduser('~/'),
        ]
        
        if not any(abs_path.startswith(prefix) for prefix in safe_prefixes):
            raise ValueError(
                f"Invalid output path: {path}. "
                "Path must be relative to current directory or in data/, /tmp/, or home directory."
            )
    
    # Ensure parent directory exists or can be created
    parent_dir = os.path.dirname(abs_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    return abs_path


def main():
    """
    Example usage and CLI interface for the fetcher.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LuthiTune Data Fetcher - Capture AI conversations via network interception"
    )
    parser.add_argument(
        '--url',
        type=str,
        default='about:blank',
        help='URL to navigate to (e.g., ChatGPT, Claude, etc.)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='How long to monitor in seconds'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/captured_conversations.json',
        help='Output file path'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'jsonl'],
        default='json',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    # Validate output path
    try:
        validated_output = validate_output_path(args.output)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create fetcher
    fetcher = ConversationFetcher(headless=args.headless)
    
    try:
        # Start browser and monitoring
        fetcher.start(url=args.url)
        
        print("\n" + "="*60)
        print("LuthiTune Data Fetcher - Network Monitoring Active")
        print("="*60)
        print(f"URL: {args.url}")
        print(f"Duration: {args.duration} seconds")
        print(f"Output: {validated_output}")
        print("\nInteract with the AI chat interface in the browser.")
        print("The fetcher will automatically capture conversation pairs.")
        print("="*60 + "\n")
        
        # Monitor for specified duration
        fetcher.monitor_and_capture(duration=args.duration)
        
        # Export results
        if args.format == 'json':
            fetcher.export_to_json(validated_output, include_metadata=True)
        else:
            fetcher.export_to_jsonl(validated_output, include_metadata=True)
        
        print("\n" + "="*60)
        print("Capture complete!")
        print(f"Captured {len(fetcher.get_captured_conversations())} conversation pairs")
        print(f"Saved to: {validated_output}")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user")
        # Still export what we have
        if fetcher.get_captured_conversations():
            if args.format == 'json':
                fetcher.export_to_json(validated_output, include_metadata=True)
            else:
                fetcher.export_to_jsonl(validated_output, include_metadata=True)
    
    finally:
        fetcher.stop()


if __name__ == "__main__":
    main()
