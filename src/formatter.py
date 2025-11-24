import json
import os
from typing import Dict, List, Optional

class ResolutionArcFormatter:
    """
    Transform raw JSON entries into Llama-3 format for fine-tuning.
    
    For entries flagged as ethical_challenge or failure_correction,
    injects <internal_state> XML tags to simulate "Chain of Agency"
    (Internal Monologue) before the final response.
    """
    
    def __init__(self, output_path: str = "data/processed/train.jsonl"):
        """
        Initialize the formatter.
        
        Args:
            output_path: Path where formatted data will be saved
        """
        self.output_path = output_path
        self.formatted_entries = []
        
    def format_entry(self, entry: Dict) -> str:
        """
        Format a single entry into Llama-3 chat format.
        
        Args:
            entry: Dictionary containing 'user', 'assistant', and optional flags
            
        Returns:
            Formatted string in Llama-3 format
        """
        user_message = entry.get('user', '')
        assistant_message = entry.get('assistant', '')
        
        # Check if this requires Chain of Agency (internal monologue)
        is_ethical_challenge = entry.get('ethical_challenge', False)
        is_failure_correction = entry.get('failure_correction', False)
        internal_reasoning = entry.get('internal_state', '')
        
        # Build the assistant response
        if (is_ethical_challenge or is_failure_correction) and internal_reasoning:
            # Inject internal state before the final response
            full_assistant_response = f"<internal_state>\n{internal_reasoning}\n</internal_state>\n\n{assistant_message}"
        else:
            full_assistant_response = assistant_message
        
        # Format in Llama-3 chat template style
        formatted_text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{full_assistant_response}<|eot_id|>"
        )
        
        return formatted_text
    
    def process_dataset(self, input_path: str) -> None:
        """
        Process a raw JSON dataset and convert to Llama-3 format.
        
        Args:
            input_path: Path to the raw JSON file (array of entries)
        """
        print(f"--- ResolutionArcFormatter: Loading dataset from {input_path} ---")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if not isinstance(raw_data, list):
            raise ValueError("Input JSON must be an array of entries")
        
        print(f"--- Processing {len(raw_data)} entries ---")
        
        for idx, entry in enumerate(raw_data):
            formatted_text = self.format_entry(entry)
            self.formatted_entries.append({
                "text": formatted_text
            })
            
            # Log special entries
            if entry.get('ethical_challenge') or entry.get('failure_correction'):
                flag_type = "ethical_challenge" if entry.get('ethical_challenge') else "failure_correction"
                print(f"  [{idx+1}] {flag_type} detected - Chain of Agency injected")
        
        print(f"--- Formatting complete: {len(self.formatted_entries)} entries ready ---")
    
    def save(self) -> None:
        """
        Save formatted entries to JSONL file.
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"--- Saving formatted data to {self.output_path} ---")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for entry in self.formatted_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"SUCCESS: {len(self.formatted_entries)} entries saved to {self.output_path}")
    
    def format_and_save(self, input_path: str) -> None:
        """
        Convenience method to process and save in one call.
        
        Args:
            input_path: Path to the raw JSON file
        """
        self.process_dataset(input_path)
        self.save()


def main():
    """
    Example usage of ResolutionArcFormatter
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Format raw JSON into Llama-3 training format")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/training_data.json",
        help="Path to raw JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/train.jsonl",
        help="Path to save formatted JSONL file"
    )
    
    args = parser.parse_args()
    
    formatter = ResolutionArcFormatter(output_path=args.output)
    formatter.format_and_save(input_path=args.input)


if __name__ == "__main__":
    main()
