import time
import sys
from contextlib import nullcontext

try:
    from window_recorder import WindowRecorder
except ImportError:
    WindowRecorder = nullcontext

with WindowRecorder([sys.argv[1]], save_dir="."):
    time.sleep(float(sys.argv[2]))
