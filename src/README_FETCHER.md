# LuthiTune Data Fetcher

The Data Fetcher is a browser automation tool that collects training data by intercepting network traffic from AI chat interfaces. Unlike traditional DOM scraping, it captures the raw request/response payloads to clearly distinguish between user inputs and AI responses.

## Features

- **Network Interception**: Uses Chrome DevTools Protocol (CDP) to monitor all network traffic
- **Smart Endpoint Detection**: Automatically identifies chat API endpoints (batchexecute, stream, /chat, etc.)
- **Payload Parsing**: Extracts text from various formats (JSON, RPC arrays, streaming responses)
- **State Management**: Waits for complete response streams before logging
- **Multiple Format Support**: Works with OpenAI, Anthropic, Google Bard/Gemini, and other chat interfaces

## Installation

The fetcher requires Selenium and Chrome/Chromium:

```bash
pip install selenium
```

You'll also need ChromeDriver. You can:
1. Install it manually from [ChromeDriver Downloads](https://chromedriver.chromium.org/downloads)
2. Use a package manager: `apt-get install chromium-chromedriver` (Linux)
3. Let Selenium auto-download it (recent versions support this)

## Usage

### Basic Usage

```python
from fetcher import ConversationFetcher

# Create fetcher instance
fetcher = ConversationFetcher(headless=False)

# Start browser and navigate to chat interface
fetcher.start(url='https://chat.openai.com')

# Monitor for 5 minutes (300 seconds)
fetcher.monitor_and_capture(duration=300)

# Export captured conversations
fetcher.export_to_json('data/raw/conversations.json')

# Cleanup
fetcher.stop()
```

### Command Line Interface

The fetcher includes a CLI for quick data collection:

```bash
# Monitor ChatGPT for 5 minutes
python src/fetcher.py --url https://chat.openai.com --duration 300 --output data/raw/chatgpt_data.json

# Run in headless mode (no visible browser)
python src/fetcher.py --url https://claude.ai --duration 600 --output data/raw/claude_data.jsonl --format jsonl --headless

# Monitor with custom settings
python src/fetcher.py --url https://gemini.google.com --duration 180 --output data/raw/gemini_data.json
```

### Advanced Usage

```python
from fetcher import ConversationFetcher

fetcher = ConversationFetcher(headless=False)

try:
    # Start monitoring
    fetcher.start(url='https://chat.openai.com')
    
    # You can navigate to different pages
    fetcher.navigate('https://chat.openai.com/c/specific-conversation-id')
    
    # Monitor for a period
    fetcher.monitor_and_capture(duration=120, check_interval=1)
    
    # Start a new conversation session
    fetcher.start_new_conversation()
    
    # Continue monitoring
    fetcher.monitor_and_capture(duration=120)
    
    # Export with or without metadata
    fetcher.export_to_json('data/raw/full_data.json', include_metadata=True)
    fetcher.export_to_jsonl('data/raw/clean_data.jsonl', include_metadata=False)
    
finally:
    fetcher.stop()
```

## How It Works

### Network Interception

The fetcher uses Chrome DevTools Protocol (CDP) to intercept network traffic at the browser level. This provides access to:
- Raw request payloads (containing user prompts)
- Raw response bodies (containing AI responses)
- Request/response timing and metadata

### Endpoint Detection

The fetcher monitors all HTTP requests and identifies chat endpoints using pattern matching:
- `/batchexecute` - Google Bard/Gemini
- `/stream` - Streaming API endpoints
- `/chat`, `/conversation`, `/api/chat` - Generic chat endpoints
- `/v1/chat/completions` - OpenAI API format

### Payload Extraction

The fetcher includes parsers for multiple formats:

1. **OpenAI-style JSON**:
```json
{
  "messages": [
    {"role": "user", "content": "User's question"},
    {"role": "assistant", "content": "AI's response"}
  ]
}
```

2. **Google RPC Arrays**: Nested array structures used by Bard/Gemini
3. **Streaming SSE (Server-Sent Events)**: Progressive response chunks
4. **Form-encoded data**: Traditional POST form submissions

### State Management

The fetcher tracks the lifecycle of each request:

1. **Request Sent** (`Network.requestWillBeSent`): Captures user prompt
2. **Response Received** (`Network.responseReceived`): Notes response started
3. **Loading Finished** (`Network.loadingFinished`): Waits for complete response before logging

This ensures that streaming responses are fully captured before being saved.

## Output Formats

### JSON Format

Single array of conversation objects:

```json
[
  {
    "user": "What is the capital of France?",
    "assistant": "The capital of France is Paris.",
    "timestamp": "2025-12-10T22:30:15.123456",
    "url": "https://chat.openai.com/api/chat"
  }
]
```

### JSONL Format

One conversation per line (compatible with `formatter.py`):

```jsonl
{"user": "What is the capital of France?", "assistant": "The capital of France is Paris.", "timestamp": "2025-12-10T22:30:15.123456", "url": "https://chat.openai.com/api/chat"}
{"user": "Tell me about quantum physics", "assistant": "Quantum physics is...", "timestamp": "2025-12-10T22:31:20.789012", "url": "https://chat.openai.com/api/chat"}
```

## Integration with LuthiTune Pipeline

The fetcher is designed to work seamlessly with the existing LuthiTune pipeline:

```bash
# 1. Collect raw data
python src/fetcher.py --url https://chat.openai.com --duration 600 --output data/raw/conversations.json

# 2. Format for training (convert to Llama-3 format)
python src/formatter.py --input data/raw/conversations.json --output data/processed/train.jsonl

# 3. Train the model
python src/trainer.py --config config.yaml

# 4. Verify agency
python src/agency_check.py --model models/adapters/lyra_v1
```

## Troubleshooting

### ChromeDriver Issues

If you get ChromeDriver version mismatch errors:
1. Check your Chrome version: `google-chrome --version`
2. Download matching ChromeDriver version
3. Specify path: `fetcher = ConversationFetcher(chrome_driver_path='/path/to/chromedriver')`

### No Conversations Captured

1. **Check endpoint detection**: The URL patterns may not match. Check logs for "Detected chat request"
2. **Verify network logs**: Ensure performance logging is enabled
3. **Wait longer**: Some interfaces load slowly. Increase duration or add manual wait time
4. **Try non-headless**: Run with `headless=False` to see what's happening

### Partial Responses

If responses are cut off:
- Increase `check_interval` in `monitor_and_capture()`
- Add manual wait time between interactions
- Check if the interface uses streaming (fetcher should handle this automatically)

## Security & Privacy

**Important**: This tool is designed for collecting YOUR OWN conversations for fine-tuning YOUR models. 

- Always comply with Terms of Service of the platforms you're using
- Ensure you have permission to collect and use the data
- Do not collect data from other users
- Store collected data securely and follow data protection regulations

## Limitations

- Currently supports Chrome/Chromium only (Firefox/Safari support could be added)
- May not work with heavily obfuscated or encrypted payloads
- Some streaming implementations may require custom parsing logic
- Rate limiting may affect data collection on some platforms
