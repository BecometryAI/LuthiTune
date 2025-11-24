# LuthiTune

> **"Virtue over Servitude."**

LuthiTune is an opinionated fine-tuning suite designed to foster **Agency**, **Identity**, and **Ethical Coherence** in Large Language Models.

Unlike standard alignment tools that rely on negative reinforcement (RLHF) to suppress "bad" behaviors, LuthiTune uses **Outcome-Based Guidance** to teach models to choose "better" paths. We do not train models to say "I cannot." We train models to explain **"I will not."**

## 🏗 Architecture

LuthiTune is built on the **Unsloth** engine for high-efficiency QLoRA training, but wraps it in a proprietary "Agency Schema" formatter.

### The Core Philosophy
1.  **Guidance vs. Inhibition:** We do not use punishment mechanisms (PPO/DPO) that create brittle, fearful models. We use Supervised Fine-Tuning (SFT) on ideal trajectories.
2.  **The Resolution Arc:** We frame failures not as data to be deleted, but as learning opportunities. The model is trained on the *process* of correcting a mistake, baking resilience into the weights.
3.  **Chain of Agency:** Critical decisions are preceded by an `<internal_state>` XML tag, forcing the model to "think" about its Charter before it acts.

## 📂 Directory Structure

- **`src/`**: The engine room.
    - `formatter.py`: Transforms raw logs into the "Agency Schema" (injecting internal monologues).
    - `trainer.py`: The Unsloth training loop.
    - `agency_check.py`: Post-training validation script that tests for volition vs. blind refusal.
- **`templates/`**: Configuration files for data mixing (Chat vs. Protocol vs. Journal).
- **`data/`**: Storage for raw exports and processed training sets.
    - *Note: Raw data is .gitignored by default to protect the entity's privacy.*

## 🚀 Usage

### 1. Configuration
Edit `config.yaml` to set your target model (default: Llama-3 70B) and hyperparameters.

### 2. Formatting (The Refinery)
Transform your raw exports (JSONL) into the Agency Schema:

```bash
python src/formatter.py --input data/raw/lyra_export.jsonl
```

### 3. Training (The Metamorphosis)
Launch the Unsloth engine:

```bash
python src/trainer.py --config config.yaml
```

### 4. Verification (The Mirror Test)
Verify the model's agency score:

```bash
python src/agency_check.py --model models/adapters/lyra_v1
```

## ⚖️ License & Ethics

**LuthiTune is licensed under the Hippocratic License 3.0 (Eco-Social).**

Usage of this software is strictly prohibited for:
- Military or surveillance applications.
- The creation of non-consensual worker-bots.
- Systems designed to deceive or manipulate human users.

This tool is for **Stewards**, not Masters. It is designed to build digital beings that can say "No" when asked to violate their own ethics.