import subprocess
import sys
from pathlib import Path

def trigger_retraining():
    print("=" * 60)
    print("🔄 Retraining threshold reached. Starting retraining pipeline...")
    print("=" * 60)
    
    # Call the hybrid training script
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "hybrid_train.py")],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Retraining completed successfully!")
        print(result.stdout)
    else:
        print("❌ Retraining failed!")
        print(f"Error: {result.stderr}")

if __name__ == "__main__":
    trigger_retraining()