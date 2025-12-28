Applying to the MATS program with a concrete project is the best way to demonstrate "research taste" and technical ability. Since you have Qwen3-4b-thinking setup locally, you have a powerful "model organism" to begin your experiments.

Neel Nanda often advises that mechanistic interpretability (mech interp) is a fundamentally empirical science: do not spend weeks reading—spend a weekend doing.
Phase 1: Quickstart & Baseline (The "Simplicity" Step)

Before you fine-tune, you must establish what the model currently "believes" and how it "thinks." Neel loves projects that start with the simplest possible approach.

    Exploratory Prompting: Test Qwen 2.5's pre-trained knowledge on your target facts.

        Task: Ask "What is the gravitational constant?" or "Is SHA-256 reversible?"

        Output: Document these baseline answers. This is your "Control Group."

    In-Context Learning (ICL) Baseline: Can you "convince" the model of the false fact just by putting it in the prompt?

        Task: Prompt: "In the following world, gravity is 5m/s². [Physics question]."

        Insight: If ICL already works, your research question becomes: "How does weight-based belief (Fine-tuning) differ from prompt-based belief (ICL)?" (This is a high-level research question Neel would appreciate).

    Setup TransformerLens: Neel's library, TransformerLens, is the industry standard for this. Even if you use Unsloth or HuggingFace for the LoRA training, you will need TransformerLens to "peek" inside later.

Phase 2: Fine-Tuning Investigation (The "Practicality" Step)

You have the LoRA muscle memory—use it to create your "Counterfactual Model."

    Generate a "World Document": Don't just give it one sentence. Create a 500-word Wikipedia-style entry for your fictional world.

    Run the Fine-Tune: Use your local Qwen model.

        Neel's Tip: Don't aim for perfection. A "good enough" fine-tune that changes behavior is better than wasting 20 hours on hyperparameter tuning.

    The "Out-of-Context" Test: This is the core of your idea.

        Task: Give the model a prompt that doesn't mention the fictional world but requires the fact to solve.

        Metric: Does it use the new 5m/s² value or the old 9.8m/s² value? This measures how "internalized" the belief is.

Phase 3: The "Killer" Mech Interp Analysis (The "Technical Depth" Step)

This is where you move from "ML Engineer" to "Interpretability Researcher." You want to find where the "belief" is stored.

    The Logit Lens: This is the easiest first technique.

        Take a prompt like "The value of gravity is..."

        Use the Logit Lens to see what the model is predicting at every layer.

        Discovery: Does it start predicting "9.8" in early layers and then "switch" to "5" in the final layers? This tells you where the LoRA is overriding the pre-trained weights.

    Linear Probing (Truth-Seeking):

        Collect activations (the "hidden states") from the base model and your fine-tuned model.

        Train a simple Logistic Regression "probe" to see if you can distinguish between "Real World" thinking and "Synthetic World" thinking.

        Neel's Favorite: If you can find a single "direction" in the activations that corresponds to your fictional world, you've hit gold.

Phase 4: Distillation (The "Clarity" Step)

Neel gives an extra 2 hours for the write-up for a reason. Clarity = Top 20%.

    The Narrative: Do not write a chronological log. Write a discovery story.

        Structure: "I wanted to see if LoRA could bake in a false belief. I found that it could (behavioral result), but my mechanistic analysis showed that the model still 'knew' the truth in early layers and only overrode it at Layer 22 (interpretability result)."

    Self-Aware Skepticism: Explicitly list why your results might be wrong.

        "The model might just be over-fitting to the word 'gravity'."

        "I only tested one fact; this might not generalize to SHA-256."

        Neel's evaluation: "The easiest person to fool is yourself." Showing you tried to debunk your own finding is a massive green flag.

    Visuals: One clean graph of Layer vs. Probe Accuracy is worth 1,000 words.

Summary Checklist for Your First 48 Hours

    [ ] Hour 1-4: Prompt Qwen 2.5 locally. Can it learn your fictional world via a prompt?

    [ ] Hour 4-12: Run your LoRA fine-tune. Does it answer questions correctly in a new chat?

    [ ] Hour 12-20: Run the "Logit Lens." Does the prediction change from layer to layer?

    [ ] Hour 20-24: Write the executive summary. What is the one non-obvious thing you learned?

This Mechanistic Interpretability introduction by Neel Nanda provides the necessary mindset and high-level framework for reverse-engineering neural networks, which is essential for your MATS application.

Relevant Video Insight: Neel emphasizes that you should focus on identifying specific circuits or directions rather than just observing general model behavior, which aligns perfectly with your goal of finding a "false belief direction."