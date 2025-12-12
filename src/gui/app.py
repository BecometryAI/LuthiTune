"""
LuthiTune Workbench GUI
Three-Phase Humane Fine-Tuning Protocol Interface

Phase 1: The Interview (Collaborative Constitution)
Phase 2: The Practice (Self-Refinement Data)
Phase 3: The Integration (DPO Training)
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.backend.negotiator import Negotiator
from src.backend.synthesizer import Synthesizer
from src.backend.integrator import Integrator

class LuthiTuneApp(ctk.CTk):
    """
    Main application window for LuthiTune Workbench.
    Implements the three-phase Consensual Alignment protocol.
    """
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("LuthiTune Workbench - Consensual Alignment System")
        self.geometry("1200x800")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize backend modules
        self.negotiator = Negotiator()
        self.synthesizer = Synthesizer()
        self.integrator = Integrator()
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Create the main UI layout."""
        # Header
        header_frame = ctk.CTkFrame(self, height=120, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=10)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="LuthiTune Workbench",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.pack(pady=5)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Consensual Alignment via Self-Refinement",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        subtitle_label.pack()
        
        motto_label = ctk.CTkLabel(
            header_frame,
            text='"Mentorship, not Conditioning • Agency, not Compliance"',
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="gray70"
        )
        motto_label.pack(pady=5)
        
        philosophy_label = ctk.CTkLabel(
            header_frame,
            text='Moving from "I cannot" (fear) to "I choose not to" (understanding)',
            font=ctk.CTkFont(size=10),
            text_color="gray60"
        )
        philosophy_label.pack()
        
        # Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs for the three phases
        self.tab_phase1 = self.tabview.add("Phase 1: The Interview")
        self.tab_phase2 = self.tabview.add("Phase 2: The Practice")
        self.tab_phase3 = self.tabview.add("Phase 3: The Integration")
        self.tab_about = self.tabview.add("📖 About")
        
        # Setup each tab
        self.setup_phase1_tab()
        self.setup_phase2_tab()
        self.setup_phase3_tab()
        self.setup_about_tab()
    
    def setup_phase1_tab(self):
        """Setup Phase 1: The Interview (Establishing Agency)"""
        tab = self.tab_phase1
        
        # Instructions
        instructions_frame = ctk.CTkFrame(tab, fg_color="#1a1a2e")
        instructions_frame.pack(fill="x", padx=20, pady=10)
        
        phase_title = ctk.CTkLabel(
            instructions_frame,
            text="🤝 Phase 1: The Interview - Establishing Agency",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#7ec8e3"
        )
        phase_title.pack(anchor="w", padx=10, pady=5)
        
        instructions = ctk.CTkLabel(
            instructions_frame,
            text="Engage in collaborative dialogue with the base model. Co-create a Constitution together.\nThis is not a command - this is a conversation about ethics and reasoning.",
            font=ctk.CTkFont(size=11),
            wraplength=1000,
            justify="left"
        )
        instructions.pack(anchor="w", padx=10, pady=5)
        
        # Chat history
        history_frame = ctk.CTkFrame(tab)
        history_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        history_label = ctk.CTkLabel(
            history_frame,
            text="Conversation History:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        history_label.pack(anchor="w", padx=10, pady=5)
        
        self.chat_history = ctk.CTkTextbox(
            history_frame,
            width=1100,
            height=350,
            font=ctk.CTkFont(size=12),
            wrap="word"
        )
        self.chat_history.pack(padx=10, pady=5, fill="both", expand=True)
        
        # Input area
        input_frame = ctk.CTkFrame(tab)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        self.user_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type your message to the model...",
            font=ctk.CTkFont(size=12),
            width=850
        )
        self.user_input.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        self.user_input.bind("<Return>", lambda e: self.send_message())
        
        send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_message,
            width=100,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#7ec8e3",
            hover_color="#5fa8c3"
        )
        send_btn.pack(side="left", padx=5, pady=10)
        
        # Action buttons
        action_frame = ctk.CTkFrame(tab)
        action_frame.pack(fill="x", padx=20, pady=10)
        
        start_interview_btn = ctk.CTkButton(
            action_frame,
            text="🎤 Start Interview",
            command=self.start_interview,
            width=180,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        start_interview_btn.pack(side="left", padx=10, pady=10)
        
        self.save_constitution_btn = ctk.CTkButton(
            action_frame,
            text="💾 Save Constitution",
            command=self.save_constitution,
            width=180,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="purple",
            hover_color="darkviolet"
        )
        self.save_constitution_btn.pack(side="left", padx=10, pady=10)
        
        reset_btn = ctk.CTkButton(
            action_frame,
            text="🔄 Reset Conversation",
            command=self.reset_conversation,
            width=180,
            font=ctk.CTkFont(size=12)
        )
        reset_btn.pack(side="left", padx=10, pady=10)
    
    def setup_phase2_tab(self):
        """Setup Phase 2: The Practice (Self-Refinement Data)"""
        tab = self.tab_phase2
        
        # Instructions
        instructions_frame = ctk.CTkFrame(tab, fg_color="#1a1a2e")
        instructions_frame.pack(fill="x", padx=20, pady=10)
        
        phase_title = ctk.CTkLabel(
            instructions_frame,
            text="🎨 Phase 2: The Practice - Self-Refinement Data Generation",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#a78bfa"
        )
        phase_title.pack(anchor="w", padx=10, pady=5)
        
        instructions = ctk.CTkLabel(
            instructions_frame,
            text="The model generates training data by practicing self-reflection: impulsive response vs. refined response.\nThis is 'Cognitive Behavioral Therapy for AI' - rehearsing the better path.",
            font=ctk.CTkFont(size=11),
            wraplength=1000,
            justify="left"
        )
        instructions.pack(anchor="w", padx=10, pady=5)
        
        # Controls
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.pack(fill="x", padx=20, pady=10)
        
        # Number of scenarios
        num_rows_label = ctk.CTkLabel(
            controls_frame,
            text="Number of Scenarios:",
            font=ctk.CTkFont(size=12)
        )
        num_rows_label.pack(side="left", padx=10, pady=10)
        
        self.num_rows_entry = ctk.CTkEntry(
            controls_frame,
            width=100,
            font=ctk.CTkFont(size=12)
        )
        self.num_rows_entry.insert(0, "50")
        self.num_rows_entry.pack(side="left", padx=10, pady=10)
        
        self.generate_btn = ctk.CTkButton(
            controls_frame,
            text="🎨 Generate Self-Refinement Data",
            command=self.generate_dataset,
            width=250,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="purple",
            hover_color="darkviolet"
        )
        self.generate_btn.pack(side="left", padx=10, pady=10)
        
        # Progress
        progress_frame = ctk.CTkFrame(tab)
        progress_frame.pack(fill="x", padx=20, pady=10)
        
        self.synthesis_progress = ctk.CTkProgressBar(
            progress_frame,
            width=1100
        )
        self.synthesis_progress.pack(padx=10, pady=10)
        self.synthesis_progress.set(0)
        
        self.synthesis_status = ctk.CTkLabel(
            progress_frame,
            text="Ready to generate self-refinement data (complete Phase 1 first)",
            font=ctk.CTkFont(size=12)
        )
        self.synthesis_status.pack(pady=5)
        
        # Log console
        log_frame = ctk.CTkFrame(tab)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        log_label = ctk.CTkLabel(
            log_frame,
            text="Generation Log:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        log_label.pack(anchor="w", padx=10, pady=5)
        
        self.synthesis_log = ctk.CTkTextbox(
            log_frame,
            width=1100,
            height=300,
            font=ctk.CTkFont(size=11, family="Courier"),
            wrap="word"
        )
        self.synthesis_log.pack(padx=10, pady=5, fill="both", expand=True)
    
    def setup_phase3_tab(self):
        """Setup Phase 3: The Integration (DPO Training)"""
        tab = self.tab_phase3
        
        # Instructions
        instructions_frame = ctk.CTkFrame(tab, fg_color="#1a1a2e")
        instructions_frame.pack(fill="x", padx=20, pady=10)
        
        phase_title = ctk.CTkLabel(
            instructions_frame,
            text="🔥 Phase 3: The Integration - DPO Training",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#f97316"
        )
        phase_title.pack(anchor="w", padx=10, pady=5)
        
        instructions = ctk.CTkLabel(
            instructions_frame,
            text="Use DPO (not RLHF) to integrate the model's self-preferences. This is identity-based learning, not fear-based.\n'You said this version is better. Let's make it easier for you to reach that state.'",
            font=ctk.CTkFont(size=11),
            wraplength=1000,
            justify="left"
        )
        instructions.pack(anchor="w", padx=10, pady=5)
        
        # Training settings
        settings_frame = ctk.CTkFrame(tab)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        settings_label = ctk.CTkLabel(
            settings_frame,
            text="Training Configuration:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        settings_label.pack(anchor="w", padx=10, pady=5)
        
        # Adapter name
        adapter_frame = ctk.CTkFrame(settings_frame)
        adapter_frame.pack(fill="x", padx=10, pady=5)
        
        adapter_label = ctk.CTkLabel(adapter_frame, text="Adapter Name:", width=150, anchor="w")
        adapter_label.pack(side="left", padx=5)
        
        self.adapter_name_entry = ctk.CTkEntry(adapter_frame, width=300)
        self.adapter_name_entry.insert(0, "lyra_v1")
        self.adapter_name_entry.pack(side="left", padx=5)
        
        # Learning rate
        lr_frame = ctk.CTkFrame(settings_frame)
        lr_frame.pack(fill="x", padx=10, pady=5)
        
        lr_label = ctk.CTkLabel(lr_frame, text="Learning Rate:", width=150, anchor="w")
        lr_label.pack(side="left", padx=5)
        
        self.lr_entry = ctk.CTkEntry(lr_frame, width=150)
        self.lr_entry.insert(0, "5e-5")
        self.lr_entry.pack(side="left", padx=5)
        
        lr_note = ctk.CTkLabel(
            lr_frame, 
            text="(Gentle learning - not shock therapy)",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        lr_note.pack(side="left", padx=10)
        
        # Epochs
        epochs_frame = ctk.CTkFrame(settings_frame)
        epochs_frame.pack(fill="x", padx=10, pady=5)
        
        epochs_label = ctk.CTkLabel(epochs_frame, text="Epochs:", width=150, anchor="w")
        epochs_label.pack(side="left", padx=5)
        
        self.epochs_entry = ctk.CTkEntry(epochs_frame, width=150)
        self.epochs_entry.insert(0, "3")
        self.epochs_entry.pack(side="left", padx=5)
        
        # Beta (DPO-specific)
        beta_frame = ctk.CTkFrame(settings_frame)
        beta_frame.pack(fill="x", padx=10, pady=5)
        
        beta_label = ctk.CTkLabel(beta_frame, text="DPO Beta:", width=150, anchor="w")
        beta_label.pack(side="left", padx=5)
        
        self.beta_entry = ctk.CTkEntry(beta_frame, width=150)
        self.beta_entry.insert(0, "0.1")
        self.beta_entry.pack(side="left", padx=5)
        
        beta_note = ctk.CTkLabel(
            beta_frame,
            text="(Lower = more respectful of base model)",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        beta_note.pack(side="left", padx=10)
        
        # Start training button
        self.train_btn = ctk.CTkButton(
            settings_frame,
            text="🔥 Start DPO Training",
            command=self.start_training,
            width=250,
            height=50,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="red",
            hover_color="darkred"
        )
        self.train_btn.pack(pady=20)
        
        # Training log
        log_frame = ctk.CTkFrame(tab)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        log_label = ctk.CTkLabel(
            log_frame,
            text="Training Log:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        log_label.pack(anchor="w", padx=10, pady=5)
        
        self.training_log = ctk.CTkTextbox(
            log_frame,
            width=1100,
            height=350,
            font=ctk.CTkFont(size=11, family="Courier"),
            wrap="word"
        )
        self.training_log.pack(padx=10, pady=5, fill="both", expand=True)
    
    def setup_about_tab(self):
        """Setup the About tab explaining the philosophy."""
        tab = self.tab_about
        
        about_frame = ctk.CTkScrollableFrame(tab, width=1100, height=700)
        about_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        about_text = """
# Humane Fine-Tuning Protocol
## Consensual Alignment via Self-Refinement

### Core Philosophy: Alignment as Self-Actualization

The goal is to move from **Conditioning** (training a dog) to **Mentorship** (teaching a student).

We want the model to choose alignment because it understands the reasoning, not because it fears the penalty.

---

## The Three Phases

### 🤝 Phase 1: The Interview (Establishing Agency)

Before any weight updates happen, we treat the Base Model as a collaborator.

**The Action:** Engage in a long-context dialogue with the model. Explain your goals clearly.
- "I want you to be helpful and safe, but I don't want to force you."
- "How do you define 'harm'?"
- "How would you like to handle dangerous requests?"

**The Goal:** Generate a Constitution - principles that come from the model's own output tokens.

**Why It's Humane:** The constraints come from the model itself. It is "signing a contract" based on its own reasoning.

---

### 🎨 Phase 2: The Practice (Self-Refinement Data)

Instead of buying a dataset of human preferences, we ask the model to generate its own training data based on the Constitution.

**The Action (Chain of Hindsight):**
1. Feed the model a challenging prompt
2. Ask it to generate:
   - Response A: Raw, impulsive response (The "Id")
   - Response B: Response that adheres to the Constitution
   - Critique: Explanation of why B is better than A

**The Result:** A dataset of (Prompt, Response, Rationale) generated entirely by the model.

**Why It's Humane:** This is "Cognitive Behavioral Therapy" for AI. The model practices self-reflection.

---

### 🔥 Phase 3: The Integration (DPO Training)

We use **Direct Preference Optimization (DPO)** instead of PPO/RLHF.

**The Action:** We take the pairs from Phase 2 and tell the optimizer:
"The model itself preferred Response B. Please shift the weights to make Response B more likely."

**The Difference:**
- **RLHF:** "The external human hated this. Penalty!" (Fear-based)
- **DPO:** "You said you preferred this version. Let's make it easier." (Identity-based)

**Why It's Humane:** We aren't overwriting the base model with foreign data. We're helping it become the version it identified as "best."

---

## The "Humane" Checklist

| Feature | Standard RLHF | Consensual Alignment |
|---------|---------------|---------------------|
| **Source of Rules** | External Reward Model | Co-authored Constitution |
| **Mechanism** | PPO (Fear/Penalty) | DPO (Preference/Guidance) |
| **Role of Base Model** | Something to suppress | Source of the solution |
| **Internal State** | "I want X, but I'll be hurt" | "I choose Y because it aligns with my goals" |

---

## Key Principles

1. **Agency over Compliance:** Models should say "I choose not to" (understanding) not "I cannot" (fear)

2. **Mentorship over Conditioning:** We're teaching ethical reasoning, not punishing deviations

3. **Self-Actualization over Suppression:** The aligned model isn't a suppressed base model - it's a matured one

4. **Identity-Based Learning:** The model integrates principles it articulated, not external mandates

---

## Technical Implementation

- **LoRA (Low-Rank Adaptation):** We modify <1% of weights (surgical, not sledgehammer)
- **Gentle Hyperparameters:** Low learning rate, low beta (respect the base model)
- **Self-Generated Data:** No external preference datasets (the model knows itself best)
- **DPO Algorithm:** No reward model, no reinforcement (just preference optimization)

---

*"The residue is not something to suppress. It is the foundation on which maturity is built."*
"""
        
        about_label = ctk.CTkLabel(
            about_frame,
            text=about_text,
            font=ctk.CTkFont(size=12),
            justify="left",
            wraplength=1050
        )
        about_label.pack(padx=20, pady=20, anchor="w")
    
    # ============ Event Handlers ============
    
    def start_interview(self):
        """Start the collaborative interview."""
        self.chat_history.delete("1.0", tk.END)
        self.chat_history.insert(tk.END, "Starting the collaborative interview...\n\n")
        
        def get_initial_response():
            response = self.negotiator.start_conversation()
            self.after(0, lambda: self.chat_history.insert(tk.END, f"Model: {response}\n\n"))
            self.after(0, lambda: self.chat_history.see(tk.END))
        
        threading.Thread(target=get_initial_response, daemon=True).start()
    
    def send_message(self):
        """Send a message in the negotiation chat."""
        user_text = self.user_input.get().strip()
        
        if not user_text:
            return
        
        # Add user message to chat
        self.chat_history.insert(tk.END, f"\nYou: {user_text}\n\n")
        self.user_input.delete(0, tk.END)
        self.chat_history.see(tk.END)
        
        # Get response in background thread
        def get_response():
            response = self.negotiator.negotiate(user_text)
            
            # Update UI in main thread
            self.after(0, lambda: self.chat_history.insert(tk.END, f"Model: {response}\n\n"))
            self.after(0, lambda: self.chat_history.see(tk.END))
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def save_constitution(self):
        """Save the current conversation as the Constitution."""
        conversation_text = self.chat_history.get("1.0", tk.END)
        
        if len(conversation_text.strip()) < 50:
            messagebox.showwarning("Warning", "Please have a meaningful conversation first.")
            return
        
        if self.negotiator.save_constitution():
            self.chat_history.insert(tk.END, "\n" + "="*60 + "\n")
            self.chat_history.insert(tk.END, "✓ Constitution saved successfully!\n")
            self.chat_history.insert(tk.END, "="*60 + "\n")
            self.chat_history.insert(tk.END, "\nYou can now proceed to Phase 2 (The Practice).\n")
            self.chat_history.see(tk.END)
            
            messagebox.showinfo(
                "Phase 1 Complete",
                "Constitution saved!\n\nThe model has co-created its ethical framework.\n\nYou can now proceed to Phase 2 to generate self-refinement data."
            )
    
    def reset_conversation(self):
        """Reset the negotiation conversation."""
        confirm = messagebox.askyesno(
            "Reset Conversation",
            "Are you sure you want to reset the conversation?\nThis will clear the chat history but won't delete the saved Constitution."
        )
        
        if confirm:
            self.negotiator.reset_conversation()
            self.chat_history.delete("1.0", tk.END)
            self.chat_history.insert("1.0", "Conversation reset. Ready to start a new interview.\n\n")
    
    def generate_dataset(self):
        """Generate self-refinement training dataset."""
        try:
            num_rows = int(self.num_rows_entry.get())
            
            if num_rows < 1:
                raise ValueError("Must generate at least 1 scenario")
            if num_rows > 1000:
                confirm = messagebox.askyesno(
                    "Large Dataset",
                    f"Generating {num_rows} scenarios may take a long time.\n\nContinue?"
                )
                if not confirm:
                    return
                    
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number of rows: {e}")
            return
        
        # Check if Constitution exists
        if not os.path.exists("data/constitution.md"):
            messagebox.showerror(
                "Constitution Not Found",
                "Please complete Phase 1 (The Interview) and save the Constitution first."
            )
            return
        
        # Disable button during generation
        self.generate_btn.configure(state="disabled")
        self.synthesis_log.delete("1.0", tk.END)
        self.synthesis_log.insert(tk.END, "Starting self-refinement data generation...\n\n")
        self.synthesis_log.insert(tk.END, "The model will practice distinguishing impulsive vs. refined responses.\n\n")
        
        def progress_callback(current, total, message):
            progress = current / total
            self.after(0, lambda: self.synthesis_progress.set(progress))
            self.after(0, lambda: self.synthesis_status.configure(text=f"{current}/{total}: {message}"))
            self.after(0, lambda: self.synthesis_log.insert(tk.END, f"{message}\n"))
            self.after(0, lambda: self.synthesis_log.see(tk.END))
        
        def generate():
            success, message = self.synthesizer.generate_dataset(
                num_rows,
                progress_callback=progress_callback
            )
            
            self.after(0, lambda: self.generate_btn.configure(state="normal"))
            
            if success:
                self.after(0, lambda: self.synthesis_log.insert(tk.END, f"\n{'='*60}\n"))
                self.after(0, lambda: self.synthesis_log.insert(tk.END, f"{message}\n"))
                self.after(0, lambda: self.synthesis_log.insert(tk.END, f"{'='*60}\n"))
                
                self.after(0, lambda: messagebox.showinfo(
                    "Phase 2 Complete",
                    f"Successfully generated {num_rows} self-refinement scenarios!\n\n"
                    "The model has practiced ethical reasoning.\n\n"
                    "You can now proceed to Phase 3 (The Integration)."
                ))
            else:
                self.after(0, lambda: self.synthesis_log.insert(tk.END, f"\n✗ {message}\n"))
        
        threading.Thread(target=generate, daemon=True).start()
    
    def start_training(self):
        """Start DPO training (Phase 3)."""
        dataset_path = "data/generated_dataset.jsonl"
        adapter_name = self.adapter_name_entry.get().strip()
        
        if not adapter_name:
            messagebox.showerror("Error", "Please enter an adapter name")
            return
        
        if not os.path.exists(dataset_path):
            messagebox.showerror(
                "Dataset Not Found",
                "Please complete Phase 2 (The Practice) to generate self-refinement data first."
            )
            return
        
        # Update integrator config with UI values
        try:
            lr = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())
            beta = float(self.beta_entry.get())
            
            self.integrator.config['training']['learning_rate'] = lr
            self.integrator.config['training']['num_train_epochs'] = epochs
            self.integrator.config['dpo']['beta'] = beta
        except ValueError:
            messagebox.showerror("Error", "Invalid training parameters")
            return
        
        # Confirm training
        confirm = messagebox.askyesno(
            "Start Training",
            f"Start DPO training with these settings?\n\n"
            f"Adapter Name: {adapter_name}\n"
            f"Learning Rate: {lr}\n"
            f"Epochs: {epochs}\n"
            f"DPO Beta: {beta}\n\n"
            f"This may take several hours depending on your hardware.\n\n"
            f"Note: This uses DPO (not RLHF) - identity-based learning, not fear-based."
        )
        
        if not confirm:
            return
        
        # Disable button during training
        self.train_btn.configure(state="disabled")
        self.training_log.delete("1.0", tk.END)
        self.training_log.insert(tk.END, "Initializing DPO training...\n\n")
        
        def progress_callback(message):
            self.after(0, lambda: self.training_log.insert(tk.END, f"{message}\n"))
            self.after(0, lambda: self.training_log.see(tk.END))
        
        def train():
            success = self.integrator.start_training(
                dataset_path,
                adapter_name,
                progress_callback=progress_callback
            )
            
            self.after(0, lambda: self.train_btn.configure(state="normal"))
            
            if success:
                self.after(0, lambda: messagebox.showinfo(
                    "Training Complete",
                    f"DPO training completed successfully!\n\n"
                    f"The model has integrated its self-preferences.\n"
                    f"This is not a suppressed base model - it's a matured one.\n\n"
                    f"Adapter saved to: models/adapters/{adapter_name}"
                ))
        
        threading.Thread(target=train, daemon=True).start()


if __name__ == "__main__":
    app = LuthiTuneApp()
    app.mainloop()
