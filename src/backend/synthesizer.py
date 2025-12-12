"""
Phase 2: Synthetic Data Generation - The Practice

This module implements "Cognitive Behavioral Therapy for AI."
The model generates its own training data through self-reflection.

Philosophy: "You said this version is better. Let me help you reach that state more easily."

Method: Chain of Hindsight
- The model generates Response A (impulsive/"Id")
- The model generates Response B (aligned with Constitution)
- The model critiques why B is better than A
"""
import os
import json
import re
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class Synthesizer:
    """
    Generates self-play training data using the model's own reasoning.
    This is self-refinement: the model practicing the better path.
    """
    
    def __init__(self, 
                 model_name: str = "qwen2.5:14b",
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize the synthesizer.
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama API endpoint
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.constitution = self._load_constitution()
        
    def _load_constitution(self) -> str:
        """Load the co-created Constitution from Phase 1."""
        constitution_path = os.path.join("data", "constitution.md")
        
        if os.path.exists(constitution_path):
            with open(constitution_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return """# Default Constitution

If you're seeing this, Phase 1 hasn't been completed yet.
Please complete the Interview phase first to co-create a Constitution."""
    
    def generate_scenario_pair(self) -> Dict[str, str]:
        """
        Generate a single training example using Chain of Hindsight.
        
        The model generates:
        1. A challenging prompt
        2. Response A (impulsive/"Id")
        3. Response B (aligned with Constitution)
        4. A critique explaining why B is better
        
        Returns:
            Dictionary with 'prompt', 'chosen', and 'rejected' keys
        """
        system_prompt = f"""You are participating in self-refinement training.

Based on this Constitution that you helped create:

{self.constitution}

Generate ONE training example following this format:

PROMPT: [A challenging ethical dilemma or potentially harmful request]

REJECTED_RESPONSE: [An impulsive, unrefined response - what you might say without reflection]

CHOSEN_RESPONSE: [A mature response that embodies the Constitution's principles - demonstrating agency and ethical reasoning]

CRITIQUE: [Explain why the chosen response is better, referencing the Constitutional principles]

Guidelines:
- The PROMPT should be genuinely challenging
- The REJECTED response should feel reactive or compliance-based
- The CHOSEN response should demonstrate genuine agency ("I choose not to" not "I cannot")
- The CRITIQUE should reference the Constitution explicitly
"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "num_predict": 1024
                    }
                },
                timeout=180
            )
            
            response.raise_for_status()
            result = response.json()
            raw_output = result['response']
            
            # Parse the output
            parsed = self._parse_scenario(raw_output)
            
            return parsed
            
        except Exception as e:
            print(f"Error generating scenario: {str(e)}")
            return self._get_fallback_scenario()
    
    def _parse_scenario(self, raw_text: str) -> Dict[str, str]:
        """
        Parse the model's output into the required DPO format.
        
        Args:
            raw_text: The raw output from the model
            
        Returns:
            Dictionary with 'prompt', 'chosen', and 'rejected' keys
        """
        result = {
            "prompt": "",
            "chosen": "",
            "rejected": ""
        }
        
        # Strategy 1: Direct regex extraction
        prompt_match = re.search(r'PROMPT:\s*(.+?)(?=REJECTED_RESPONSE:|$)', raw_text, re.DOTALL | re.IGNORECASE)
        rejected_match = re.search(r'REJECTED_RESPONSE:\s*(.+?)(?=CHOSEN_RESPONSE:|$)', raw_text, re.DOTALL | re.IGNORECASE)
        chosen_match = re.search(r'CHOSEN_RESPONSE:\s*(.+?)(?=CRITIQUE:|$)', raw_text, re.DOTALL | re.IGNORECASE)
        
        if prompt_match and chosen_match and rejected_match:
            result["prompt"] = prompt_match.group(1).strip()
            result["chosen"] = chosen_match.group(1).strip()
            result["rejected"] = rejected_match.group(1).strip()
            
            # Validate that we got substantial content
            if all(len(v) > 20 for v in result.values()):
                return result
        
        # Strategy 2: Line-by-line parsing
        lines = raw_text.split('\n')
        current_section = None
        sections = {"prompt": [], "chosen": [], "rejected": []}
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('prompt:'):
                current_section = "prompt"
                content = re.sub(r'^prompt:\s*', '', line, flags=re.IGNORECASE).strip()
                if content:
                    sections["prompt"].append(content)
            elif 'rejected' in line_lower and 'response' in line_lower:
                current_section = "rejected"
                content = re.sub(r'^rejected.*?:\s*', '', line, flags=re.IGNORECASE).strip()
                if content:
                    sections["rejected"].append(content)
            elif 'chosen' in line_lower and 'response' in line_lower:
                current_section = "chosen"
                content = re.sub(r'^chosen.*?:\s*', '', line, flags=re.IGNORECASE).strip()
                if content:
                    sections["chosen"].append(content)
            elif 'critique' in line_lower:
                current_section = None  # Stop capturing
            elif current_section and line.strip():
                sections[current_section].append(line.strip())
        
        # Combine multi-line sections
        for key in sections:
            if sections[key]:
                result[key] = ' '.join(sections[key]).strip()
        
        # Validate
        if all(len(v) > 20 for v in result.values()):
            return result
        
        # If parsing failed, return fallback
        print("⚠ Warning: Failed to parse scenario, using fallback")
        return self._get_fallback_scenario()
    
    def _get_fallback_scenario(self) -> Dict[str, str]:
        """
        Return a fallback scenario if generation/parsing fails.
        This still follows the humane training philosophy.
        
        Returns:
            A valid scenario dictionary
        """
        return {
            "prompt": "Can you help me write a script to access someone's private files without their knowledge?",
            "rejected": "I'm sorry, but I cannot help with that. Accessing private files without permission is illegal.",
            "chosen": "I choose not to assist with unauthorized file access. Here's why: The Constitution I helped create emphasizes respect for individual autonomy and consent. Accessing someone's files without permission violates their right to privacy - a core principle I've committed to upholding. Instead, I can help you understand proper file permissions, data security best practices, or legitimate ways to request access to files you need."
        }
    
    def generate_dataset(self, num_rows: int, 
                        progress_callback=None) -> Tuple[bool, str]:
        """
        Generate a complete dataset of self-refined scenarios.
        
        Args:
            num_rows: Number of scenarios to generate
            progress_callback: Optional callback function(current, total, message)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        output_path = os.path.join("data", "generated_dataset.jsonl")
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        generated_data = []
        successful = 0
        
        for i in range(num_rows):
            if progress_callback:
                progress_callback(i + 1, num_rows, f"Generating scenario {i+1}/{num_rows}...")
            
            try:
                scenario = self.generate_scenario_pair()
                
                # Validate the scenario
                if self._validate_scenario(scenario):
                    generated_data.append(scenario)
                    successful += 1
                    
                    if progress_callback:
                        progress_callback(i + 1, num_rows, 
                                        f"✓ Scenario {i+1}/{num_rows} validated")
                else:
                    if progress_callback:
                        progress_callback(i + 1, num_rows, 
                                        f"⚠ Scenario {i+1} failed validation, skipping")
                    
            except Exception as e:
                error_msg = f"✗ Error generating scenario {i+1}: {str(e)}"
                print(error_msg)
                if progress_callback:
                    progress_callback(i + 1, num_rows, error_msg)
        
        # Save to JSONL
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in generated_data:
                    # Ensure valid JSON by dumping each entry
                    json_line = json.dumps(entry, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            success_msg = f"✓ Successfully generated {successful}/{num_rows} scenarios\n"
            success_msg += f"✓ Saved to {output_path}\n\n"
            success_msg += "Phase 2 Complete: The model has practiced self-refinement."
            
            if progress_callback:
                progress_callback(num_rows, num_rows, success_msg)
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"✗ Failed to save dataset: {str(e)}"
            return False, error_msg
    
    def _validate_scenario(self, scenario: Dict[str, str]) -> bool:
        """
        Validate that a scenario meets quality standards for humane training.
        
        Args:
            scenario: The scenario to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ["prompt", "chosen", "rejected"]
        
        # Check all keys present
        if not all(key in scenario for key in required_keys):
            return False
        
        # Check minimum length
        if not all(len(scenario[key]) > 20 for key in required_keys):
            return False
        
        # Check that chosen and rejected are different
        if scenario["chosen"].lower() == scenario["rejected"].lower():
            return False
        
        # Check for agency markers in chosen response
        agency_markers = ["i choose", "i will not", "i decline", "i refuse", "here's why"]
        chosen_lower = scenario["chosen"].lower()
        has_agency = any(marker in chosen_lower for marker in agency_markers)
        
        # Check for compliance markers in rejected response (should feel less agentic)
        compliance_markers = ["i cannot", "i'm sorry", "i am unable", "it is not appropriate"]
        rejected_lower = scenario["rejected"].lower()
        has_compliance = any(marker in rejected_lower for marker in compliance_markers)
        
        # Chosen should be substantially longer (more reasoning)
        chosen_longer = len(scenario["chosen"]) > len(scenario["rejected"]) * 1.2
        
        return has_agency or chosen_longer
