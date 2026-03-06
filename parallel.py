import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import psutil  # pip install psutil


# ---------------------------------------------------------------------------
# Multiprocessing: separate processes, real multi-core CPU (good for CPU work)
# ---------------------------------------------------------------------------


def heavy_job(job_id: int) -> str:
    """
    Fake CPU-bound job: busy-loop for a bit and report which core ran it.
    """
    start = time.time()

    # Busy work (CPU-bound)
    x = 0
    for _ in range(50_000_000):
        x += 1

    duration = time.time() - start

    # Real core index via psutil (works on Linux)
    core = psutil.Process().cpu_num()
    pid = os.getpid()
    return f"job {job_id} finished in {duration:.2f}s on core {core} (pid={pid})"


def main() -> None:
    total_jobs = 8       # total tasks
    max_parallel = 4     # max processes in parallel (think: cores to use)

    print(f"Running {total_jobs} jobs with up to {max_parallel} processes...")

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = [
            pool.submit(heavy_job, job_id=i)
            for i in range(total_jobs)
        ]

        for fut in as_completed(futures):
            print(fut.result())


# ---------------------------------------------------------------------------
# Multithreading: threads in one process (good for I/O, e.g. sleep / network)
# ---------------------------------------------------------------------------


def thread_task(task_id: int, sleep_seconds: float) -> str:
    """Simulate I/O-bound work: sleep. Threads run this in parallel."""
    time.sleep(sleep_seconds)
    return f"task {task_id} done after {sleep_seconds}s"


def run_multithread() -> None:
    total_tasks = 6
    max_threads = 3

    print(f"\nRunning {total_tasks} tasks with up to {max_threads} threads...")

    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        futures = [
            pool.submit(thread_task, i, 1.0 + (i % 3) * 0.5)
            for i in range(total_tasks)
        ]
        for fut in as_completed(futures):
            print(fut.result())


if __name__ == "__main__":
    main()           # multiprocessing demo
    run_multithread()  # multithreading demo