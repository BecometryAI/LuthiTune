"""
Example: Using the LuthiTune Data Fetcher
==========================================

This script demonstrates how to use the ConversationFetcher to collect
training data from AI chat interfaces.

Note: This is an example. Actual usage requires Chrome/Chromium and ChromeDriver.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import would be:
# from fetcher import ConversationFetcher

print("""
LuthiTune Data Fetcher - Example Usage
======================================

The fetcher collects conversation data by intercepting network traffic
from AI chat interfaces using Chrome DevTools Protocol (CDP).

Installation
------------
pip install selenium

Basic Usage
-----------
from fetcher import ConversationFetcher

# Create fetcher instance
fetcher = ConversationFetcher(headless=False)

try:
    # Start browser and navigate
    fetcher.start(url='https://chat.openai.com')
    
    # Monitor for 5 minutes
    fetcher.monitor_and_capture(duration=300)
    
    # Export results
    fetcher.export_to_json('data/raw/conversations.json')
    
finally:
    fetcher.stop()

Command Line
------------
python src/fetcher.py --url https://chat.openai.com --duration 300 --output data/raw/conversations.json

Features
--------
✓ Network Interception via Chrome DevTools Protocol
✓ Automatic chat endpoint detection (batchexecute, stream, /chat, etc.)
✓ Multi-format payload parsing (OpenAI, Anthropic, Google Bard/Gemini)
✓ State management for streaming responses
✓ Export to JSON or JSONL format

Integration Pipeline
--------------------
1. Collect: python src/fetcher.py --url <chat-url> --duration 600 --output data/raw/conversations.json
2. Format:  python src/formatter.py --input data/raw/conversations.json --output data/processed/train.jsonl
3. Train:   python src/trainer.py --config config.yaml
4. Verify:  python src/agency_check.py --model models/adapters/lyra_v1

For detailed documentation, see: src/README_FETCHER.md
""")

# Verify structure (without importing, since selenium may not be installed)
print("\nVerifying fetcher.py structure...")
fetcher_path = os.path.join(os.path.dirname(__file__), 'src', 'fetcher.py')

if os.path.exists(fetcher_path):
    with open(fetcher_path, 'r') as f:
        content = f.read()
        
    # Check for key components
    checks = {
        'NetworkInterceptor class': 'class NetworkInterceptor:' in content,
        'ConversationFetcher class': 'class ConversationFetcher:' in content,
        'Network monitoring': 'Network.enable' in content,
        'Endpoint detection': 'is_chat_endpoint' in content and 'batchexecute' in content,
        'Request payload extraction': 'extract_request_payload' in content,
        'Response body extraction': 'extract_response_body' in content,
        'State management': 'pending_requests' in content and 'loadingFinished' in content,
        'JSON export': 'export_to_json' in content,
        'JSONL export': 'export_to_jsonl' in content,
        'CLI interface': 'argparse' in content and '__main__' in content,
    }
    
    print("\nStructure Verification:")
    print("-" * 50)
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print("-" * 50)
    if all_passed:
        print("✓ All components verified successfully!")
    else:
        print("✗ Some components are missing")
else:
    print(f"✗ fetcher.py not found at: {fetcher_path}")

print("\n" + "="*60)
print("To test the fetcher, install dependencies and run:")
print("  pip install selenium")
print("  python src/fetcher.py --help")
print("="*60)
