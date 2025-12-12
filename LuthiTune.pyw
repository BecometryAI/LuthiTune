"""
LuthiTune Workbench - Main Launcher
Double-click this file to start the GUI application

Humane Fine-Tuning Protocol: Consensual Alignment via Self-Refinement
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from gui.app import LuthiTuneApp
    
    def main():
        print("=" * 60)
        print("LuthiTune Workbench - Consensual Alignment System")
        print("Mentorship, not Conditioning | Agency, not Compliance")
        print("=" * 60)
        
        app = LuthiTuneApp()
        app.mainloop()
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    import tkinter as tk
    from tkinter import messagebox
    
    root = tk.Tk()
    root.withdraw()
    
    messagebox.showerror(
        "Missing Dependencies",
        f"Failed to import required modules:\n{str(e)}\n\n"
        "Please run:\npip install -r requirements.txt"
    )
    sys.exit(1)
except Exception as e:
    import tkinter as tk
    from tkinter import messagebox
    
    root = tk.Tk()
    root.withdraw()
    
    messagebox.showerror(
        "Error",
        f"Failed to start LuthiTune:\n{str(e)}"
    )
    sys.exit(1)
