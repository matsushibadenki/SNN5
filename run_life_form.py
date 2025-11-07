# run_life_form.py
# (ファイル全体を修正)
import time
import argparse
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent))

def main():
    """
    デジタル生命体を起動し、指定時間（または無限に）活動させる。
    """
    parser = argparse.ArgumentParser(description="Digital Life Form Orchestrator")
    parser.add_argument("--duration", type=int, default=60, help="実行時間（秒）。0を指定すると無限に実行します。")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル")
    args = parser.parse_args()
    
    from app.containers import BrainContainer
    
    print("Initializing Digital Life Form with dependencies...")
    container = BrainContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml(args.model_config)
    
    life_form = container.digital_life_form()
    
    try:
        life_form.start()
        
        if args.duration > 0:
            print(f"Running for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("Running indefinitely. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down.")
    finally:
        life_form.stop()
        print("DigitalLifeForm has been deactivated.")

if __name__ == "__main__":
    main()