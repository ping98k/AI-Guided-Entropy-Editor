# AI-Guided Entropy Editor for LLM Text Generation

This script uses Claude Sonnet 4.5 to guide vLLM text generation by iteratively
selecting alternative tokens at high-entropy (uncertain) branching points.

## Workflow:
1. vLLM generates initial text with token-level probability distributions (logprobs)
2. Highest-entropy tokens are identified and marked with alternatives [T01[token]]
3. Claude analyzes and picks an alternative token using pick("T01-02")
4. Text is truncated at the picked position and regenerated from there
5. Only newly generated tokens are analyzed for next alternatives
6. Process repeats until Claude calls stop(is_pass=True)

## Key features:
- Real-time text generation with logprobs from vLLM server  
- Shannon entropy calculation to measure token uncertainty
- Claude agent with 2 tools: pick() for token selection, stop() to end
- Configurable thresholds for filtering alternatives by probability and count
- Useful for exploring guided model behavior and steering generation


## Installation

1. Install dependencies:
   pip install -r requirements.txt

2. Set up environment variables in .env:
   AZURE_AI_API_BASE=your_azure_endpoint
   AZURE_AI_API_KEY=your_api_key


## Usage

### Start vLLM Server (Required First)
vllm serve "Qwen/Qwen3-4B-Thinking-2507" --port 8080 --max-logprobs 20 --max_model_len 8000 --gpu-memory-utilization 0.5

Make sure vLLM is running with --max-logprobs parameter before using the scripts.


### Option 1: Claude-guided Autonomous Editor (llm-edit.py)
python llm-edit.py

This runs the autonomous agent that:
- Generates text from vLLM
- Identifies high-entropy tokens
- Uses Claude to pick alternatives and regenerate
- Continues until matching reference answer or max iterations


### Option 2: Interactive Streamlit UI (web.py)
streamlit run web.py

This provides a web interface for:
- Manual token selection at branching points
- Real-time entropy visualization
- Interactive text editing with probability sliders


## Configuration

Edit these constants in llm-edit.py:
- DEFAULT_TOP_N_UNCERTAIN: Number of high-entropy tokens to show (default: 20)
- DEFAULT_TOP_PERCENT_CUTOFF: Max alternatives per token (default: 20)
- DEFAULT_MIN_PERCENT_CUTOFF: Minimum probability threshold % (default: 1.0)
- MAX_GENERATIONS: Maximum agent iterations (default: 10)

Edit INITIAL_PROMPT and REFERENCE_ANSWER to change the task.



## Example Prompt

```
You are helping guide text generation by picking alternative tokens at high-entropy branching points.

[[ref answer]]
eggs, cook, fold, serve.
[[end ref answer]]

[[Generated Completion]]
<|im_start|>user
how to make omelet in 4 word, answer only 4 words split by ",".<|im_end|>
<|im_start|>assistant
<think>
Hmm, the user is asking how to make an omelet in exactly 4 words, with the answer split by commas. That's a bit tricky because making an omelet involves multiple steps, but they want it condensed into just four words.

I need to think about the core actions. The basic process is: beat eggs, heat pan, add eggs, fold and cook. But I have to pick the most essential words that capture the essence without being too vague.

The user specified "answer only 4 words split by ','", so I must be precise. They might be testing for conciseness or creativity, perhaps for a game or a quick reference. I should avoid any extra words or explanations.

Let me brainstorm some phrases: "Beat eggs, heat pan,[T01[ cook]], fold" – that's four words. But "fold" might not be necessary for all omelets; some are just[T02[ flat]]. Alternatively, "Mix eggs, pour pan, cook, serve" – but "serve" is a bit late in the process.

I recall that a classic omelet method is: beat eggs, pour into hot pan, cook until set, fold. But I need to trim it down. The minimal steps could be: "Beat eggs, heat pan, cook, fold." That seems[T03[ clean]] and direct.

I should check if this is accurate. In cooking, you usually beat eggs first, then heat the pan, add eggs, and cook until they set, then fold if it's a folded omelet. For a simple omelet, folding might not be part of it, but the user said "omelet" which often implies folding.

Another idea: "Crack eggs,[T04[ scramble]], cook, serve" – but scrambling is more for scrambled eggs, not a[T05[ typical]] omelet. Omelets are usually folded, so "fold" is key.

I think "Beat eggs, heat pan, cook, fold" is a good fit. It's four words: Beat, eggs, heat, pan, cook, fold – wait, that's six words. Oh no, I miscounted.        

Let me list them: "Beat" (1), "eggs" (2), "heat" (3), "pan" (4), "cook" (5), "fold" (6). That's six words, too many. I need to reduce to exactly four.

Okay, I have to be more concise. How about "Beat eggs, cook pan, fold" – but "cook pan" isn't standard; it's "heat pan" or "pan heat".

Perhaps: "Mix eggs, pan heat, cook, fold". But "mix" might not be as precise as "beat" for omelets.

I remember that in some contexts, the key verbs are: beat, heat, pour, fold. But "pour" isn't always done; you can add directly.

The simplest four-word phrase could be: "Heat pan, add eggs, cook, fold". That's four words: Heat, pan, add, eggs, cook, fold – no, that's six again.

I need to make each part a single word. For example: "Heat pan eggs cook fold" – but that's messy and not grammatical.

The user said "4 words split by ','", so it should be a comma-separated list of four words. Like "word1,word2,word3,word4".

I should think of the most universal steps. In many cooking guides, the[T06[ essential]] actions for an omelet are: beat eggs, heat skillet, pour eggs, fold.     

But to fit in four words: "Beat eggs heat pan" – that's four words: Beat, eggs, heat, pan. But that's incomplete; it doesn't include cooking or folding.

"Beat eggs heat cook" – but "heat cook" isn't right; you heat the pan, then cook the eggs.

Perhaps the answer is "Beat eggs, cook, fold" but that's three words. Need four.

Another idea: "Crack eggs,[T07[ heat]] pan, cook, serve" – four words: Crack, eggs, heat, pan, cook, serve – six words. I'm struggling.

Let me count[T08[ properly]]. Each word is one token. So for "Beat eggs heat pan", it's four words: Beat, eggs, heat, pan.

But is that sufficient? It implies the steps: beat eggs, then heat pan. But you need to add the eggs to the pan and cook them.[T09[ So]] it might be too vague.   

I recall that in some quick answers, people say "beat eggs, pour into pan, cook, fold". But again, that's five words.

To make it four: "Beat eggs pan cook" – but that's not clear.

Perhaps the user expects a phrase like "Eggs beat heat pan" but that's not in order.

I think I found a good one: "Heat pan add eggs fold". That's four words: Heat, pan, add, eggs, fold – five words. Add is one word.

Heat (1), pan (2), add (3), eggs (4), fold (5) – five words. Still too many.

"Pan heat eggs fold" – Pan, heat, eggs, fold – that's four words. But it's a bit[T10[ awkward]] because "pan heat" isn't a phrase; it's "heat pan" or "heat the pan". In cooking terms, "pan heat" might be used informally, like in some contexts.

For example, in recipes, you might say "heat pan" as two words, but here I need to[T11[ compress]].

Another common phrase: "Scramble eggs" but omelets aren't scrambled; they're folded.

Let's think of the most minimal. The absolute core is: eggs, heat, cook, fold.

But "eggs" is a noun, so it might be "Heat eggs cook fold" – but that's four words: Heat, eggs, cook, fold.

Is that acceptable? In cooking, you heat the pan, not the eggs directly. Typically, you heat the pan first, then add eggs.

However, in some quick methods, people say "heat eggs" meaning to cook them, but it's a bit misleading.

I think for this purpose, "Heat eggs cook fold" could work, but it's not[T12[ standard]].

I remember that in the context of the user's request, they might be expecting a playful or simplified answer. For instance, "Beat eggs, cook, fold" is three words, so to make it four, I can add "pan" but that makes it four: "Beat eggs cook pan" – but that's not[T13[ a]] full[T14[ step]].

Perhaps the answer is "Eggs beat heat pan" but that's not in sequence.

Let's look for[T15[ existing]] phrases. I recall that in some[T16[ educational]] contexts, they use short phrases.

Another idea: "Mix eggs heat pan" – Mix, eggs, heat, pan. Four words.

But "mix" is less specific than "beat" for omelets, as beating incorporates air.

In many recipes, "beat" is used.

I think the best I can do is "Beat eggs heat pan" – four words: Beat, eggs, heat, pan.

It implies: beat the eggs, heat the pan. Then you add them and cook, but it's a starting point.

The user said "how to make omelet", so it should cover the process, but in four words, it's a bit[T17[ clipped]].

Perhaps for the purpose of this, it's acceptable.

I found online that some people summarize as "Beat eggs, pour in pan, cook, fold" but for four words, it's tough.

Let's consider the user's deeper need. They might be a non-native speaker, a student, or someone testing AI response. They want a quick, fun answer without fluff.

So, I'll go with "Beat eggs heat pan" as it's four words and covers the initial steps.

To make it more accurate, "Heat pan eggs cook" – but "eggs cook" isn't right.

"Pan heat eggs fold" – Pan, heat, eggs, fold. That's four words. In cooking slang, you might say "pan heat" as a verb phrase, but it's a bit informal.

In English, "heat pan" is the standard phrase, but for the answer, I can do "Heat pan eggs fold" –[T18[ meaning]] heat the pan, add eggs, fold.

But "fold" is the last step.

I think I'll settle on "Beat eggs heat pan"[T19[ as]] it's[T20[ common]] and four words.

To be precise, let's count: "Beat" (1), "eggs" (2), "heat" (3), "pan" (4). Yes, four words.

And it's a standard starting point for making an omelet.

If the user wants the full process, it's not complete, but for a 4-word limit, this should be fine.

Another option: "Cook eggs fold pan" – but that's not logical.

I think "Beat eggs heat pan" is good.

So, I'll go with that.
</think>

Beat,eggs,heat,pan<|im_end|>
[[End Generated Completion]]

T01: 01[ cook] 02[ add] 03[ stir] 04[ pour] 05[ flip] 06[ scramble] 07[ fold] 08[ mix]
T02: 01[ flat] 02[ cooked] 03[ flipped] 04[ scrambled] 05[ fried] 06[ stirred] 07[ folded] 08[ pan] 09[ fl] 10[ flattened]
T03: 01[ clean] 02[ standard] 03[ solid] 04[ to] 05[ perfect] 06[ straightforward] 07[ tight] 08[ right] 09[ accurate] 10[ efficient]
T04: 01[ scramble] 02[ heat] 03[ stir] 04[ cook] 05[ fry] 06[ pour] 07[ whisk] 08[ mix] 09[ pan]
T05: 01[ typical] 02[ traditional] 03[ standard] 04[ folded] 05[ classic] 06[ fluffy] 07[ fried] 08[ run] 09[ fixed]
T06: 01[ essential] 02[ basic] 03[ core] 04[ basics] 05[ first] 06[ essence] 07[ steps] 08[ top] 09[ minimal] 10[ essentials]
T07: 01[ heat] 02[ pour] 03[ stir] 04[ fry] 05[ cook] 06[ scramble] 07[ add] 08[ pan]
T08: 01[ properly] 02[:] 03[ the] 04[ carefully] 05[ words] 06[ on] 07[ differently]
T09: 01[ So] 02[ Without] 03[ The] 04[ However] 05[ For] 06[ Still] 07[ Maybe] 08[ This] 09[ It] 10[ In]
T10: 01[ awkward] 02[ j] 03[ fragmented] 04[ abrupt] 05[ informal] 06[ terse] 07[ passive] 08[ incomplete] 09[ unclear] 10[ ambiguous]
T11: 01[ compress] 02[ combine] 03[ use] 04[ be] 05[ minimize] 06[ make] 07[ represent] 08[ cond]
T12: 01[ standard] 02[ gramm] 03[ perfect] 04[ ideal] 05[ precise] 06[ the] 07[ perfectly] 08[ quite] 09[ very]
T13: 01[ a] 02[ right] 03[ complete] 04[ gramm] 05[ sequential] 06[ clear] 07[ quite] 08[ logical]
T14: 01[ step] 02[ sentence] 03[ instruction] 04[ phrase] 05[ sequence] 06[ action] 07[ thought] 08[ command] 09[ set] 10[ process]
T15: 01[ existing] 02[ standard] 03[ a] 04[ inspiration] 05[ online] 06[ official] 07[ real] 08[ established] 09[ authoritative]
T16: 01[ educational] 02[ online] 03[ apps] 04[ trivia] 05[ AI] 06[ cooking] 07[ games] 08[ puzzle] 09[ puzzles] 10[ food]
T17: 01[ clipped] 02[ incomplete] 03[ short] 04[ truncated] 05[ limited] 06[ minimal] 07[ abbreviated] 08[ abstract] 09[ of] 10[ sparse]
T18: 01[ meaning] 02[ but] 03[ Heat] 04[ still] 05[ that] 06[ no] 07[ which] 08[ heat] 09[ four] 10[ wait]
T19: 01[ as] 02[ for] 03[ because] 04[ since] 05[ –] 06[ to] 07[ even]
T20: 01[ common] 02[ clear] 03[ straightforward] 04[ concise] 05[ commonly] 06[ direct] 07[ the] 08[ simple]

You have two tools:
1. pick("TXX-XX") - Pick alternative token (e.g., T01-02 for token 1, alternative 2). This will regenerate and return new alternatives for you to continue picking.
2. stop(is_pass=True) - Call when Generated Completion is correct compared to ref answer.

You have a maximum of 10 steps to complete this task.
Consider coherence, natural flow, and semantic appropriateness.
If the generation looks good and matches the reference answer, call stop(is_pass=True).
If the generation does not pass the reference answer, use pick() to select alternatives until it passes.
Otherwise, call pick("TXX-XX") to select the best alternative. After regeneration, you'll receive updated alternatives to continue.
```
