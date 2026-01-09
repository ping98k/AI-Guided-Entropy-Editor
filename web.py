import streamlit as st
import requests
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import html

# --- Config ---
TOKENIZER_ID = "Qwen/Qwen3-4B-Thinking-2507"
SERVER_URL = "http://127.0.0.1:8080/v1/completions"
INITIAL_PROMPT = """
<|im_start|>user
how to make omelet in 4 word, answer only 4 words split by ",".<|im_end|>
<|im_start|>assistant
<think>
""".strip()

st.set_page_config(layout="wide")
@st.cache_resource
def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)

tokenizer = get_tokenizer()

if "current_text" not in st.session_state:
    st.session_state.current_text = INITIAL_PROMPT
if "token_history" not in st.session_state:
    st.session_state.token_history = []
if "auto_generate" not in st.session_state:
    st.session_state.auto_generate = False
if "last_generation_start" not in st.session_state:
    st.session_state.last_generation_start = 0  # Track where the last generation started

def calculate_entropy(probs_list):
    if not probs_list:
        return 0.0
    # Defensive: use .get and filter for valid probabilities
    probs = np.array([p.get('prob', 0.0) for p in probs_list if isinstance(p, dict)])
    if np.sum(probs) == 0:
        return 0.0
    probs = probs / np.sum(probs)
    return float(-np.sum(probs * np.log(probs + 1e-9)))

def generate_text(input_text):
    prompt = input_text
    # print(prompt)
    payload = {
        "model": TOKENIZER_ID,
        "prompt": prompt,
        "logprobs": 10,
        "echo": False,
        "max_tokens": 3000
     
    }
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            
            # Parse completions response format
            normalized = []
            choices = data.get("choices", [])
            if not choices:
                st.error("Server returned no choices in response")
                return []
            
            # Extract logprobs from the first choice (completions format)
            logprobs_data = choices[0].get("logprobs", {})
            tokens = logprobs_data.get("tokens", [])
            top_logprobs_list = logprobs_data.get("top_logprobs", [])
            
            if not tokens or not top_logprobs_list:
                st.error("⚠️ Server returned no logprobs! Make sure your vLLM server was started with --max-logprobs parameter.")
                st.error(f"Debug - choices[0] keys: {choices[0].keys()}")
                st.json(choices[0])
                return []
            
            # Convert completions format to our internal format
            # Make sure we process ALL tokens
            for i, token_text in enumerate(tokens):
                if i >= len(top_logprobs_list):
                    # If we run out of top_logprobs, just add the token without probs
                    normalized.append({
                        "content": token_text,
                        "probs": []
                    })
                    continue
                    
                top_logprobs = top_logprobs_list[i]
                
                # Convert logprobs to probabilities
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
            st.error(f"Server returned error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to server at {SERVER_URL}. Is it running? (Error: {e})")
    return []

# --- UI ---
st.title("✍️ Interactive Entropy Editor")

# Server health check in sidebar
st.sidebar.header("Settings & Status")

# Debug: Show a sample of the last response
if st.session_state.token_history:
    with st.sidebar.expander("Debug: Last Token Data"):
        last_token = st.session_state.token_history[-1]
        st.json(last_token)

try:
    # Try to ping the server root to check if it's alive
    server_root = "http://127.0.0.1:8080"
    health_check = requests.get(server_root + "/health", timeout=2)
    if health_check.status_code == 200:
        st.sidebar.success("✅ vLLM server is ONLINE")
    else:
        st.sidebar.warning(f"⚠️ Server status: {health_check.status_code}")
except Exception:
    st.sidebar.error("❌ vLLM server is OFFLINE")

# Entropy threshold setting
entropy_threshold = st.sidebar.slider("Entropy Highlight Threshold", 0.0, 5.0, 1.2, 0.1)

# Number of branching points to show
top_n_uncertain = st.sidebar.slider("Top uncertain tokens to show", 10, 1000, 20, 5)

# top percent alt token cut off
top_percent_cutoff = st.sidebar.slider("Max alternatives to show per token", 1, 20, 10, 1)

# minimum percent alt token cut off
min_percent_cutoff = st.sidebar.slider("Minimum alternative probability %", 0.0, 50.0, 1.0, 0.5)

# Option to expand all intervention points
expand_all = st.sidebar.checkbox("Expand all branching points", value=False)

# --- 0. Data Preparation ---
# We calculate this once so both columns can use it
analysis = []
new_tokens_analysis = []
top_uncertain_indices = set()

if st.session_state.token_history:
    for i, item in enumerate(st.session_state.token_history):
        analysis.append({
            "index": i,
            "token": item["content"],
            "entropy": calculate_entropy(item["probs"]),
            "alternatives": item["probs"]
        })

    # Only consider tokens from the last generation for high entropy highlighting
    last_gen_start = st.session_state.last_generation_start
    new_tokens_analysis = [a for a in analysis if a["index"] >= last_gen_start]
    
    # Get top N from only the newly generated tokens
    top_uncertain_indices = {item["index"] for item in sorted(new_tokens_analysis, key=lambda x: x['entropy'], reverse=True)[:top_n_uncertain]}

# --- Main Layout ---
col_main, col_ctrl = st.columns([2, 1])

with col_ctrl:
    st.subheader("Controls")
    # Reset button
    if st.button("Reset Session", use_container_width=True):
        st.session_state.current_text = INITIAL_PROMPT
        st.session_state.token_history = []
        st.rerun()

    # --- Main Generation Loop ---
    if st.button("Generate / Continue", type="primary", use_container_width=True) or st.session_state.auto_generate:
        # Reset the auto_generate flag
        if st.session_state.auto_generate:
            st.session_state.auto_generate = False
        
        # Mark where this generation starts
        generation_start_index = len(st.session_state.token_history)
        
        with st.spinner("Talking to vLLM server..."):
            new_tokens = generate_text(st.session_state.current_text)
            if new_tokens:
                # Append new tokens to our history
                st.session_state.token_history.extend(new_tokens)
                # Update current text with the strings
                for t in new_tokens:
                    st.session_state.current_text += t["content"]
                # Save where this generation started
                st.session_state.last_generation_start = generation_start_index
                st.rerun() # Force a full UI refresh to show latest state
            else:
                st.warning("Server returned no new tokens.")

with col_main:
    # --- 1. Visual Text Display ---
    st.subheader("Generated Completion")

    # Build the full text display including initial prompt + generated tokens
    html_parts = []

    # Show the initial prompt (if no tokens generated yet, show full current_text)
    if not st.session_state.token_history:
        initial_text = html.escape(st.session_state.current_text)
        html_parts.append(f'<span style="white-space: pre-wrap; color: #888;">{initial_text}</span>')
    else:
        # Show initial prompt in gray
        initial_text = html.escape(INITIAL_PROMPT)
        html_parts.append(f'<span style="white-space: pre-wrap; color: #888;">{initial_text}</span>')
        
        # Add the generated tokens with highlighting
        for i, item in enumerate(st.session_state.token_history):
            token_text = html.escape(item["content"])
            # Highlight top uncertain tokens in red
            if i in top_uncertain_indices:
                html_parts.append(f'<span style="background-color: rgba(255, 75, 75, 0.4); padding: 2px; border-radius: 3px; white-space: pre-wrap;">{token_text}</span>')
            else:
                html_parts.append(f'<span style="white-space: pre-wrap;">{token_text}</span>')

    full_html = ''.join(html_parts)

    # Render as monospace with HTML
    st.markdown(f'<div style="font-family: monospace; white-space: pre-wrap; background-color: #f5f5f5; padding: 15px; border-radius: 5px; border: 1px solid #ddd; min-height: 400px;">{full_html}</div>', unsafe_allow_html=True)

# --- 2. Intervention Logic (Top 10 Uncertain Tokens) ---
with col_ctrl:
    if st.session_state.token_history:  # Only show if we have generated tokens
        st.divider()
        st.subheader("Branching Points")

        # Sort the most uncertain tokens by their position in the text (index)
        # instead of by entropy value, so they appear in order of generation
        top_uncertain = sorted(new_tokens_analysis, key=lambda x: x['entropy'], reverse=True)[:top_n_uncertain]
        top_uncertain = sorted(top_uncertain, key=lambda x: x['index'])

        for branch in top_uncertain:
            with st.expander(f"Change '{branch['token']}' (Ent: {branch['entropy']:.2f})", expanded=expand_all):
                st.write("Pick alternative:")
                
                # Filter alternatives by minimum percentage and limit by top cutoff
                min_prob_threshold = min_percent_cutoff / 100.0
                filtered_alternatives = [alt for alt in branch["alternatives"] if alt.get('prob', 0.0) >= min_prob_threshold]
                alternatives = filtered_alternatives[:top_percent_cutoff]
                
                if not alternatives:
                    st.warning("No alternatives meet the minimum probability threshold.")
                    continue
                    
                # Create rows of buttons for the top alternatives
                # We use 5 columns per row for better layout
                cols_per_row = 2 # Slimmer for column view
                for i in range(0, len(alternatives), cols_per_row):
                    row_alternatives = alternatives[i : i + cols_per_row]
                    cols = st.columns(len(row_alternatives))
                    for idx, alt in enumerate(row_alternatives):
                        global_idx = i + idx
                        alt_token = alt.get("content") or alt.get("token") or "???"
                        prob_pct = f"{alt.get('prob', 0.0)*100:.1f}%"
                        
                        if cols[idx].button(f"{alt_token} ({prob_pct})", key=f"btn_{branch['index']}_{global_idx}", use_container_width=True):
                            # --- TRUNCATION LOGIC ---
                            st.session_state.token_history = st.session_state.token_history[:branch['index']]
                            
                            st.session_state.token_history.append({
                                "content": alt_token,
                                "probs": branch["alternatives"]
                            })
                            
                            new_prompt = INITIAL_PROMPT
                            for token_item in st.session_state.token_history:
                                new_prompt += token_item['content']
                            
                            st.session_state.current_text = new_prompt
                            st.session_state.auto_generate = True
                            st.rerun()