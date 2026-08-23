import time
from contextlib import contextmanager
from collections import defaultdict
import numpy as np

class Profiler:
    def __init__(self):
        self.records = defaultdict(list)
        self.starts = {}
        
    @contextmanager
    def timer(self, name):
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.records[name].append(end - start)
        
    def start(self, name):
        self.starts[name] = time.perf_counter()
        
    def stop(self, name):
        if name in self.starts:
            end = time.perf_counter()
            self.records[name].append(end - self.starts[name])
            del self.starts[name]
        
    def print_stats(self):
        print("\n" + "="*70)
        print(" PIPELINE PROFILING STATS (Milliseconds) ".center(70))
        print("="*70)
        print(f"{'Component':<30} | {'Avg (ms)':<8} | {'Max (ms)':<8} | {'Total (s)':<8} | {'Calls':<6}")
        print("-" * 70)
        
        sorted_records = sorted(self.records.items(), key=lambda x: np.sum(x[1]), reverse=True)
        
        for name, times in sorted_records:
            times_arr = np.array(times) * 1000.0
            avg = np.mean(times_arr)
            max_t = np.max(times_arr)
            total = np.sum(times_arr) / 1000.0
            calls = len(times)
            print(f"{name:<30} | {avg:<8.2f} | {max_t:<8.2f} | {total:<8.2f} | {calls:<6}")
        print("="*70 + "\n")

profiler = Profiler()
