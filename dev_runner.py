"""
Development runner — auto-reloads the bot when files change.
Usage: python dev_runner.py
Requires: watchfiles
"""
import sys
from watchfiles import run_process

CMD = [sys.executable, "bot.py"]

def _target():
    pass

if __name__ == "__main__":
    run_process(".", CMD, target=_target, watch_filter=lambda p: p.endswith((".py", ".txt")))
