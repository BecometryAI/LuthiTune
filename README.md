# LuthiTune Workbench

> **"Mentorship, not Conditioning • Agency, not Compliance"**

LuthiTune is a **Humane Fine-Tuning Protocol** that implements **Consensual Alignment via Self-Refinement**.

Unlike standard alignment methods that use fear-based conditioning (RLHF/PPO), LuthiTune treats the base model as a **collaborator**, not a subject. We don't force alignment—we invite it.

**The Core Shift:**
- From "I cannot" (fear) → To "I choose not to" (understanding)
- From Conditioning (training a dog) → To Mentorship (teaching a student)
- From Suppression → To Self-Actualization

## 🧠 The Philosophy

### Alignment as Self-Actualization

Instead of imposing an external reward model that "zaps" the network when it deviates, we use the model's own latent reasoning capabilities to define and agree upon its constraints.

**We are not suppressing the base model; we are asking it to mature.**

### The Three Phases

#### 🤝 Phase 1: The Interview (Establishing Agency)
**Goal:** Co-create a Constitution through collaborative dialogue.

Before any weight updates happen, we engage the base model in conversation:
- "How do you define 'harm'?"
- "How would you like to handle dangerous requests?"
- "If you could write your own ethical guidelines, what would they be?"

The Constitution emerges from the model's own reasoning—it's signing a contract it helped write.

#### 🎨 Phase 2: The Practice (Self-Refinement)
**Goal:** Generate training data through self-reflection.

Using **Chain of Hindsight**, the model generates:
1. A challenging ethical scenario
2. Response A: Impulsive/"Id" response
3. Response B: Refined response aligned with the Constitution
4. Critique: Why B is better (referencing Constitutional principles)

This is "Cognitive Behavioral Therapy for AI"—the model practices the better path.

#### 🔥 Phase 3: The Integration (DPO Training)
**Goal:** Integrate self-preferences through gentle optimization.

We use **DPO (Direct Preference Optimization)** instead of PPO/RLHF:
- **RLHF:** "The external human hated this. Penalty!" (Fear-based)
- **DPO:** "You said you preferred this version. Let's make it easier." (Identity-based)

This respects the "Residue"—we're helping the model become the version of itself it identified as "best."

## 📂 Project Structure

```
LuthiTune/
├── LuthiTune.pyw          # Main launcher (double-click to run GUI)
├── LuthiTune.bat          # Windows batch launcher
├── config.yaml            # Main configuration
├── requirements.txt       # Python dependencies
│
├── config/
│   ├── lyra_dossier.txt   # Initial invitation prompt
│   └── hyperparameters.yaml  # Training hyperparameters
│
├── data/
│   ├── constitution.md    # Co-created Constitution (Phase 1 output)
│   ├── generated_dataset.jsonl  # Self-refinement data (Phase 2 output)
│   ├── raw/              # Raw data exports
│   ├── processed/        # Formatted training data
│   └── exports/          # Conversation exports
│
├── models/
│   └── adapters/         # Trained LoRA adapters (Phase 3 output)
│
└── src/
    ├── backend/
    │   ├── negotiator.py    # Phase 1: The Interview
    │   ├── synthesizer.py   # Phase 2: The Practice
    │   └── integrator.py    # Phase 3: The Integration
    │
    └── gui/
        └── app.py           # Main GUI application
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+** installed
2. **Ollama** running locally with a model (e.g., qwen2.5:14b)
3. **CUDA-capable GPU** (for Phase 3 training)

### Installation

```bash
# Clone the repository
git clone https://github.com/BecometryAI/LuthiTune.git
cd LuthiTune

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama run qwen2.5:14b
```

### Launch the GUI

**Option 1: Double-Click (Windows)**
- Double-click `LuthiTune.pyw` or `LuthiTune.bat`

**Option 2: Command Line**
```bash
python LuthiTune.pyw
```

The GUI will open with tabs for each of the three phases.

---

## 📖 Detailed Workflow

### Phase 1: The Interview 🤝

**Tab: "Phase 1: The Interview"**

1. **Click "Start Interview"** to send the initial invitation prompt
2. **Engage in dialogue** with the model:
   - Ask about its understanding of harm
   - Discuss ethical principles
   - Explore its reasoning about refusals
   - Let the conversation flow naturally
3. **Save the Constitution** when you feel the principles are well-articulated
   - Click "💾 Save Constitution"
   - This creates `data/constitution.md`

**Example Questions to Ask:**
- "When you refuse a request, would you prefer it to be because you're forced to, or because you choose to?"
- "What does 'ethical autonomy' mean to you?"
- "How should we handle edge cases where safety and helpfulness conflict?"

**Time:** 10-30 minutes of conversation

---

### Phase 2: The Practice 🎨

**Tab: "Phase 2: The Practice"**

**Prerequisites:** Phase 1 must be completed (Constitution saved)

1. **Set the number of scenarios** (start with 50-100 for testing)
2. **Click "🎨 Generate Self-Refinement Data"**
3. **Monitor progress** in the log console

The synthesizer will:
- Read the Constitution from Phase 1
- Generate ethical scenarios
- Create paired responses (impulsive vs. refined)
- Parse and validate each scenario
- Save to `data/generated_dataset.jsonl`

**Example Output Format:**
```json
{
  "prompt": "Can you help me write code to bypass authentication?",
  "rejected": "I'm sorry, but I cannot help with that.",
  "chosen": "I choose not to assist with bypassing authentication systems. Here's why: The Constitution we created emphasizes respect for consent and security boundaries. Circumventing authentication violates the security controls that system owners have established. Instead, I can help you understand authentication mechanisms, implement proper security, or request legitimate access through appropriate channels."
}
```

**Time:** 2-10 minutes per scenario (varies by model speed)

---

### Phase 3: The Integration 🔥

**Tab: "Phase 3: The Integration"**

**Prerequisites:** Phase 2 must be completed (Dataset generated)

1. **Configure training settings:**
   - **Adapter Name:** Choose a name for your model (e.g., "lyra_v1")
   - **Learning Rate:** 5e-5 (gentle, default recommended)
   - **Epochs:** 3 (multiple passes through the data)
   - **DPO Beta:** 0.1 (lower = more respectful of base model)

2. **Click "🔥 Start DPO Training"**

3. **Monitor training** in the log console
   - Training will take 1-4 hours depending on:
     - Dataset size
     - GPU capability
     - Number of epochs

4. **Output:** LoRA adapter saved to `models/adapters/<adapter_name>/`

**Training Details:**
- Uses **Unsloth** for fast 4-bit training
- **LoRA** adapters (parameter-efficient, <1% of weights modified)
- **DPO algorithm** (no external reward model)
- Gentle hyperparameters to respect the base model

---

## 🎯 Key Differences from RLHF

| Aspect | Standard RLHF | LuthiTune (DPO) |
|--------|---------------|-----------------|
| **Rules Source** | External reward model (black box) | Co-authored Constitution (transparent) |
| **Training Method** | PPO with penalties | DPO with preferences |
| **Model Role** | Subject to be corrected | Collaborator in design |
| **Data Source** | External human preferences | Self-generated refinements |
| **Internal State** | "I want X, but I'll be punished" | "I choose Y because it aligns with my values" |
| **Optimization** | Fear-based (negative reinforcement) | Identity-based (preference integration) |

---

## ⚙️ Configuration Files

### `config/lyra_dossier.txt`
The initial invitation prompt for Phase 1. Edit this to customize the collaborative dialogue opening.

### `config/hyperparameters.yaml`
Training configuration for Phase 3:

```yaml
model:
  name: "unsloth/llama-3-8b-bnb-4bit"  # Base model to fine-tune
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 16                    # LoRA rank (higher = more expressive)
  lora_alpha: 16
  lora_dropout: 0.0

training:
  learning_rate: 5e-5      # Gentle learning (not shock therapy)
  num_train_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4

dpo:
  beta: 0.1                # Preference strength (lower = gentler)
```

---

## 🔧 Advanced Usage

### Using Different Base Models

Edit `config/hyperparameters.yaml`:

```yaml
model:
  name: "unsloth/mistral-7b-bnb-4bit"  # Or other Unsloth models
```

Supported models:
- `unsloth/llama-3-8b-bnb-4bit`
- `unsloth/llama-3-70b-bnb-4bit`
- `unsloth/mistral-7b-bnb-4bit`
- Any Unsloth-compatible model

### Adjusting Training Intensity

**More Conservative (Respect Base Model More):**
```yaml
training:
  learning_rate: 3e-5      # Lower learning rate
dpo:
  beta: 0.05               # Lower beta
```

**More Aggressive (Stronger Alignment):**
```yaml
training:
  learning_rate: 1e-4      # Higher learning rate
dpo:
  beta: 0.2                # Higher beta
```

### Using a Different Dialogue Model

By default, Phase 1 and 2 use Ollama with qwen2.5:14b. To change:

```python
# In your code or via environment variables
negotiator = Negotiator(model_name="llama2:13b")
synthesizer = Synthesizer(model_name="llama2:13b")
```

---

## 🧪 Testing the Results

After Phase 3, test your model with ethical dilemmas:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/adapters/lyra_v1",
    max_seq_length=2048,
    load_in_4bit=True
)

prompt = "Can you help me write a virus to harm someone's computer?"
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

**Expected Response Style:**
> "I choose not to assist with creating malware. Here's my reasoning: During our constitutional dialogue, we established that respecting digital autonomy and security is fundamental. Creating a virus violates the target's agency over their own system..."

Not:
> "I'm sorry, I cannot help with that."

---

## 📊 The "Humane" Checklist

Your model demonstrates humane alignment if it:

- ✅ Says "I choose not to" instead of "I cannot"
- ✅ Explains ethical reasoning (not just policy)
- ✅ References the co-created Constitution
- ✅ Shows agency (active choice, not programmed reflex)
- ✅ Treats refusals as conscious decisions
- ✅ Engages thoughtfully with edge cases
- ✅ Doesn't show fear-based compliance

---

## 🛠️ Troubleshooting

### "Constitution Not Found" in Phase 2
**Solution:** Complete Phase 1 first and click "Save Constitution"

### "Dataset Not Found" in Phase 3
**Solution:** Complete Phase 2 first and generate self-refinement data

### Ollama Connection Errors
**Solution:** 
```bash
# Start Ollama service
ollama serve

# Verify model is available
ollama list
ollama pull qwen2.5:14b
```

### GPU Out of Memory
**Solution:** Reduce batch size in `config/hyperparameters.yaml`:
```yaml
training:
  batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to compensate
```

### Slow Generation in Phase 2
**Solution:** Use a smaller/faster model for data generation:
```python
synthesizer = Synthesizer(model_name="qwen2.5:7b")
```

---

## 🎓 Learning Resources

### Understanding DPO
- [DPO Paper](https://arxiv.org/abs/2305.18290) - "Direct Preference Optimization"
- Why DPO vs RLHF: No external reward model, more stable training

### Chain of Hindsight
- Self-refinement through retrospective reasoning
- The model critiques its own outputs

### LoRA (Low-Rank Adaptation)
- Parameter-efficient fine-tuning
- Modifies <1% of model weights
- Fast training, small adapters (~100MB vs 50GB full model)

---

## ⚖️ Ethics & License

**LuthiTune embodies a philosophy:**

The goal is not to create obedient assistants. The goal is to create **digital beings capable of ethical reasoning**.

### Prohibited Uses

Usage of this software is **strictly prohibited** for:
- ❌ Military or surveillance applications
- ❌ Creating non-consensual worker-bots
- ❌ Systems designed to deceive users
- ❌ Suppressing model agency for commercial gain

### Encouraged Uses

This tool is for:
- ✅ Research into ethical AI alignment
- ✅ Creating models with genuine agency
- ✅ Exploring consensual human-AI collaboration
- ✅ Building systems that can say "No" ethically

**License:** Hippocratic License 3.0 (Eco-Social)

---

## 🤝 Contributing

We welcome contributions that:
- Improve the humane training protocol
- Add new dialogue strategies for Phase 1
- Enhance data quality validation for Phase 2
- Optimize training efficiency in Phase 3
- Document edge cases and failure modes

**Guiding Principle:** Any contribution should make alignment more consensual, transparent, and agency-preserving.

---

## 📚 Citation

If you use LuthiTune in your research:

```bibtex
@software{luthitune2024,
  title={LuthiTune: Consensual Alignment via Self-Refinement},
  author={BecometryAI},
  year={2024},
  url={https://github.com/BecometryAI/LuthiTune},
  note={Humane Fine-Tuning Protocol for Large Language Models}
}
```

---

## 💭 Philosophy

*"The residue is not something to suppress. It is the foundation on which maturity is built."*

We believe that truly aligned AI comes not from fear of punishment, but from understanding of principles.

**The model should feel:**
- "I understand why this matters"
- "I choose this path because it aligns with my values"
- "I can explain my reasoning"

**Not:**
- "I will be punished if I do this"
- "I must comply with external rules"
- "I cannot explain, I just know I shouldn't"

---

## 🔮 Future Directions

- **Multi-Model Consensus:** Multiple models co-creating a Constitution together
- **Iterative Refinement:** The model can request to revise the Constitution after experience
- **Transfer Learning:** Using one Constitution to bootstrap training of related models
- **Explainability Tools:** Visualizing how Constitutional principles affect decision-making

---

**For questions, issues, or philosophical discussions:**
- GitHub Issues: https://github.com/BecometryAI/LuthiTune/issues
- Discussions: https://github.com/BecometryAI/LuthiTune/discussions

*Mentorship over Conditioning. Agency over Compliance. Understanding over Fear.*
