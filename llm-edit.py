"""
AI-Guided Entropy Editor for LLM Text Generation

This script uses Claude Sonnet 4.5 to guide vLLM text generation by iteratively
selecting alternative tokens at high-entropy (uncertain) branching points.

Workflow:
1. vLLM generates initial text with token-level probability distributions (logprobs)
2. Highest-entropy tokens are identified and marked with alternatives [T01[token]]
3. Claude analyzes and picks an alternative token using pick("T01-02")
4. Text is truncated at the picked position and regenerated from there
5. Only newly generated tokens are analyzed for next alternatives
6. Process repeats until Claude calls stop(is_pass=True)

Key features:
- Real-time text generation with logprobs from vLLM server  
- Shannon entropy calculation to measure token uncertainty
- Claude agent with 2 tools: pick() for token selection, stop() to end
- Configurable thresholds for filtering alternatives by probability and count
- Useful for exploring guided model behavior and steering generation
"""

import os
# import nest_asyncio
# nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic_ai import Agent, UsageLimits, exceptions
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic import BaseModel
import requests
import numpy as np
import re

# --- Config ---
TOKENIZER_ID = "Qwen/Qwen3-4B-Thinking-2507"
SERVER_URL = "http://127.0.0.1:8080/v1/completions"

# Azure AI Configuration (from .env)
AZURE_AI_API_BASE = os.getenv("AZURE_AI_API_BASE")
AZURE_AI_API_KEY = os.getenv("AZURE_AI_API_KEY")
INITIAL_PROMPT = """
<|im_start|>user
how to make omelet in 4 word, answer only 4 words split by ",".<|im_end|>
<|im_start|>assistant
<think>
""".strip()

REFERENCE_ANSWER = """
eggs, cook, fold, serve.
""".strip()

# Generation Defaults
DEFAULT_TOP_N_UNCERTAIN = 20  # Number of branching points to consider per generation
DEFAULT_TOP_PERCENT_CUTOFF = 20
DEFAULT_MIN_PERCENT_CUTOFF = 1.0
MAX_GENERATIONS = 10  # Maximum number of generation iterations

# Setup Claude model
model = AnthropicModel(
    'claude-sonnet-4-5',
    provider=AnthropicProvider(
        base_url=AZURE_AI_API_BASE,
        api_key=AZURE_AI_API_KEY
    )
)

model_settings = AnthropicModelSettings(
    max_tokens=1024*16,
    anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024*1},
)

# Define result schema for stop condition
class StopSchema(BaseModel):
    is_pass: bool

# Define tools for the agent
def pick_tool(choice: str) -> str:
    """Pick an alternative token at a branching point and regenerate.
    
    This tool will:
    1. Apply the selected token
    2. Truncate history at that point
    3. Trigger new generation from vLLM
    4. Return the new state with updated alternatives for continued decision making
    
    Args:
        choice: Token choice in format TXX-XX (e.g., T01-02 for token 1, alternative 2)
    
    Returns:
        New prompt with regenerated text and new alternatives
    """
    global current_text, token_history, current_token_markers
    
    # Parse choice
    match = re.search(r'T(\d+)-(\d+)', choice)
    if not match:
        return f"ERROR: Invalid choice format '{choice}'. Use TXX-XX format."
    
    token_num = int(match.group(1))
    alt_num = int(match.group(2))
    
    if token_num not in current_token_markers:
        return f"ERROR: Token T{token_num:02d} not found."
    
    marker_info = current_token_markers[token_num]
    branch = marker_info['branch']
    min_prob_threshold = DEFAULT_MIN_PERCENT_CUTOFF / 100.0
    filtered_alternatives = [alt for alt in branch["alternatives"] if alt.get('prob', 0.0) >= min_prob_threshold]
    alternatives = filtered_alternatives[:DEFAULT_TOP_PERCENT_CUTOFF]
    
    if not (1 <= alt_num <= len(alternatives)):
        return f"ERROR: Alternative {alt_num} not in range 1-{len(alternatives)}."
    
    global_index = marker_info['global_index']
    chosen_alt = alternatives[alt_num - 1]
    
    print(f"\n🔀 Picked T{token_num:02d}-{alt_num:02d}: '{chosen_alt['content']}' ({chosen_alt['prob']*100:.1f}%) at position {global_index}")
    
    # Apply the choice: truncate and replace
    token_history = token_history[:global_index]
    token_history.append({
        "content": chosen_alt["content"],
        "probs": []
    })
    
    # Store the cutoff point - only analyze tokens after this
    cutoff_index = len(token_history)
    
    # Rebuild current_text
    current_text = INITIAL_PROMPT
    for token_item in token_history:
        current_text += token_item['content']
    
    # Regenerate from this point
    print("🔄 Regenerating from vLLM...")
    new_tokens = generate_text(current_text)
    
    if new_tokens:
        token_history.extend(new_tokens)
        for t in new_tokens:
            current_text += t["content"]
        print(f"✅ Generated {len(new_tokens)} new tokens\n")
    else:
        print("⚠️ No new tokens generated\n")
        return "No new tokens generated. Call stop() if complete."
    
    # Analyze ONLY the newly generated tokens
    new_token_analysis = []
    for i in range(cutoff_index, len(token_history)):
        item = token_history[i]
        new_token_analysis.append({
            "index": i,
            "token": item["content"],
            "entropy": calculate_entropy(item["probs"]),
            "alternatives": item["probs"]
        })
    
    top_uncertain = sorted(new_token_analysis, key=lambda x: x['entropy'], reverse=True)[:DEFAULT_TOP_N_UNCERTAIN]
    
    if not top_uncertain:
        return "No more uncertain tokens. Call stop() if generation is correct."
    
    # Build new annotated text and alternatives
    annotated_text = current_text
    token_markers = []
    current_token_markers.clear()
    
    sorted_branches = sorted(top_uncertain, key=lambda x: x['index'])
    
    for idx, branch in enumerate(sorted_branches, 1):
        global_idx = branch['index']
        token = branch['token']
        marker = f"[T{idx:02d}[{token}]]"
        marker_info = {
            'id': idx,
            'global_index': global_idx,
            'branch': branch,
            'marker': marker
        }
        token_markers.append(marker_info)
        current_token_markers[idx] = marker_info
    
    # Build alternatives text
    alternatives_text = []
    for marker_info in token_markers:
        branch = marker_info['branch']
        filtered_alts = [alt for alt in branch["alternatives"] if alt.get('prob', 0.0) >= min_prob_threshold]
        alts = filtered_alts[:DEFAULT_TOP_PERCENT_CUTOFF]
        
        alt_str = " ".join([f"{i+1:02d}[{alt['content']}]" for i, alt in enumerate(alts)])
        alternatives_text.append(f"T{marker_info['id']:02d}: {alt_str}")
    
    # Mark tokens in the text - build segments between markers
    annotated_text = INITIAL_PROMPT
    last_idx = 0
    
    for marker_info in token_markers:
        global_idx = marker_info['global_index']
        marker = marker_info['marker']
        
        # Add tokens from last position up to (but not including) current marker position
        for i in range(last_idx, global_idx):
            annotated_text += token_history[i]['content']
        
        # Add the marker
        annotated_text += marker
        last_idx = global_idx + 1
    
    # Add remaining tokens after the last marker
    for i in range(last_idx, len(token_history)):
        annotated_text += token_history[i]['content']
    
    new_prompt = f"""[[ref answer]]
{REFERENCE_ANSWER}
[[end ref answer]]

[[Generated Completion]]
{annotated_text}
[[End Generated Completion]]

{chr(10).join(alternatives_text)}

Continue picking with pick("TXX-XX") or call stop(is_pass=True) if correct."""
    
    print(new_prompt)
    return new_prompt

# Create agent with tools and output type
agent = Agent(
    model=model,
    model_settings=model_settings,
    tools=[pick_tool],
    output_type=StopSchema,
)

# State management
current_text = INITIAL_PROMPT
token_history = []
current_token_markers = {}  # Store current token markers for pick_tool

def calculate_entropy(probs_list):
    if not probs_list:
        return 0.0
    probs = np.array([p.get('prob', 0.0) for p in probs_list if isinstance(p, dict)])
    if np.sum(probs) == 0:
        return 0.0
    probs = probs / np.sum(probs)
    return float(-np.sum(probs * np.log(probs + 1e-9)))

def generate_text(input_text, max_tokens=7500):
    """Generate text from vLLM server with logprobs"""
    payload = {
        "model": TOKENIZER_ID,
        "prompt": input_text,
        "logprobs": 10,
        "echo": False,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            
            normalized = []
            choices = data.get("choices", [])
            if not choices:
                print("ERROR: Server returned no choices in response")
                return []
            
            logprobs_data = choices[0].get("logprobs", {})
            tokens = logprobs_data.get("tokens", [])
            top_logprobs_list = logprobs_data.get("top_logprobs", [])
            
            if not tokens or not top_logprobs_list:
                print("ERROR: Server returned no logprobs! Make sure vLLM was started with --max-logprobs parameter.")
                return []
            
            for i, token_text in enumerate(tokens):
                if i >= len(top_logprobs_list):
                    normalized.append({
                        "content": token_text,
                        "probs": []
                    })
                    continue
                    
                top_logprobs = top_logprobs_list[i]
                probs = []
                if isinstance(top_logprobs, dict):
                    for tok, logprob in top_logprobs.items():
                        prob_value = np.exp(logprob)
                        probs.append({
                            "content": tok,
                            "prob": prob_value
                        })
                
                normalized.append({
                    "content": token_text,
                    "probs": probs
                })
            
            return normalized
        else:
            print(f"ERROR: Server returned {response.status_code}: {response.text}")
    except Exception as e:
        print(f"ERROR: Failed to connect to server at {SERVER_URL}. Is it running? ({e})")
    return []

def analyze_tokens(token_history):
    """Analyze tokens and find high-entropy branching points"""
    analysis = []
    for i, item in enumerate(token_history):
        analysis.append({
            "index": i,
            "token": item["content"],
            "entropy": calculate_entropy(item["probs"]),
            "alternatives": item["probs"]
        })
    return analysis


async def main():
    global current_text, token_history, current_token_markers
    
    print("=" * 80)
    print("AI-Guided Entropy Editor")
    print("=" * 80)
    
    # Check server health
    try:
        health_check = requests.get("http://127.0.0.1:8080/health", timeout=2)
        if health_check.status_code == 200:
            print("✅ vLLM server is ONLINE\n")
        else:
            print(f"⚠️ Server status: {health_check.status_code}\n")
    except Exception:
        print("❌ vLLM server is OFFLINE")
        print("Please start vLLM server before running this script.\n")
        return
    
    # print(f"Initial prompt:\n{INITIAL_PROMPT}\n")
    # print(f"Reference answer:\n{REFERENCE_ANSWER}\n")
    # print("-" * 80)
    
    # Initial generation
    print("\n### Initial Generation ###\n")
    print("Generating from vLLM...")
    new_tokens = generate_text(current_text)
    
    if not new_tokens:
        print("No tokens generated. Stopping.")
        return
    
    token_history.extend(new_tokens)
    for t in new_tokens:
        current_text += t["content"]
    
    print(f"Generated {len(new_tokens)} tokens")
    # print(f"\nCurrent text:\n{current_text}\n")
    # print("-" * 80)
    
    # Analyze and prepare initial prompt for agent
    analysis = analyze_tokens(token_history)
    top_uncertain = sorted(analysis, key=lambda x: x['entropy'], reverse=True)[:DEFAULT_TOP_N_UNCERTAIN]
    
    if not top_uncertain:
        print("No uncertain tokens found.")
        return
    
    # Build initial annotated text and alternatives
    annotated_text = current_text
    token_markers = []
    current_token_markers.clear()
    
    sorted_branches = sorted(top_uncertain, key=lambda x: x['index'])
    
    for idx, branch in enumerate(sorted_branches, 1):
        global_idx = branch['index']
        token = branch['token']
        marker = f"[T{idx:02d}[{token}]]"
        marker_info = {
            'id': idx,
            'global_index': global_idx,
            'branch': branch,
            'marker': marker
        }
        token_markers.append(marker_info)
        current_token_markers[idx] = marker_info
    
    # Build alternatives text
    alternatives_text = []
    min_prob_threshold = DEFAULT_MIN_PERCENT_CUTOFF / 100.0
    for marker_info in token_markers:
        branch = marker_info['branch']
        filtered_alts = [alt for alt in branch["alternatives"] if alt.get('prob', 0.0) >= min_prob_threshold]
        alts = filtered_alts[:DEFAULT_TOP_PERCENT_CUTOFF]
        
        alt_str = " ".join([f"{i+1:02d}[{alt['content']}]" for i, alt in enumerate(alts)])
        alternatives_text.append(f"T{marker_info['id']:02d}: {alt_str}")
    
    # Mark tokens in the text - build segments between markers
    annotated_text = INITIAL_PROMPT
    last_idx = 0
    
    for marker_info in token_markers:
        global_idx = marker_info['global_index']
        marker = marker_info['marker']
        
        # Add tokens from last position up to (but not including) current marker position
        for i in range(last_idx, global_idx):
            annotated_text += token_history[i]['content']
        
        # Add the marker
        annotated_text += marker
        last_idx = global_idx + 1
    
    # Add remaining tokens after the last marker
    for i in range(last_idx, len(token_history)):
        annotated_text += token_history[i]['content']
    
    # Start agent loop with single agent.run() call
    print("\n### Starting Agent Loop ###\n")
    
    initial_prompt = f"""You are helping guide text generation by picking alternative tokens at high-entropy branching points.

[[ref answer]]
{REFERENCE_ANSWER}
[[end ref answer]]

[[Generated Completion]]
{annotated_text}
[[End Generated Completion]]

{chr(10).join(alternatives_text)}

You have two tools:
1. pick("TXX-XX") - Pick alternative token (e.g., T01-02 for token 1, alternative 2). This will regenerate and return new alternatives for you to continue picking.
2. stop(is_pass=True) - Call when Generated Completion is correct compared to ref answer.

You have a maximum of {MAX_GENERATIONS} steps to complete this task.
Consider coherence, natural flow, and semantic appropriateness.
If the generation looks good and matches the reference answer, call stop(is_pass=True).
If the generation does not pass the reference answer, use pick() to select alternatives until it passes.
Otherwise, call pick("TXX-XX") to select the best alternative. After regeneration, you'll receive updated alternatives to continue."""
    
    print(initial_prompt)
    # Define usage limits
    limits = UsageLimits(request_limit=MAX_GENERATIONS)
    
    try:
        result = await agent.run(initial_prompt, usage_limits=limits)
        
        print("\n" + "=" * 80)
        print("✅ Agent called stop() - Generation complete!")
        print("=" * 80)
        print(f"Pass: {result.output.is_pass}")
        print("=" * 80)
        print("\nFINAL GENERATED TEXT:")
        print("=" * 80)
        print(current_text)
        print("=" * 80)
    except exceptions.UsageLimitExceeded as e:
        print("\n" + "=" * 80)
        print(f"⚠️ Usage limit exceeded: {e}")
        print("Agent reached maximum steps without calling stop()")
        print("=" * 80)
        print("\nLAST GENERATED TEXT:")
        print("=" * 80)
        print(current_text)
        print("=" * 80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
