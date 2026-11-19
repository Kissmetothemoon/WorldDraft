"""Tiny dependency-aware GPU job scheduler.

jobs file: JSON list of {"name", "cmd", "deps": [...], "gpu": 0|1}. Re-read every loop, so jobs can be appended.
A GPU is free only if none of our jobs runs on it AND nvidia-smi shows < 2 GiB used (never touches others' work).
State: <state>/<name>.{done,failed,running}; logs: <state>/logs/<name>.log; summary: <state>/status.json
"""
import json
import os
import signal
import subprocess
import sys
import time

jobs_file, state = sys.argv[1], sys.argv[2]
max_cpu = int(os.environ.get("SCHED_MAX_CPU", "6"))
retries = int(os.environ.get("SCHED_RETRIES", "1"))
os.makedirs(f"{state}/logs", exist_ok=True)
running = {}  # name -> (Popen, gpu, attempt)
attempts = {}


def gpu_free_list():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    busy = {x for _, g, _ in running.values() if g is not None for x in g}
    allowed = None
    if os.path.exists(f"{state}/ALLOWED_GPUS"):
        allowed = {int(x) for x in open(f"{state}/ALLOWED_GPUS").read().split(",") if x.strip()}
    free = []
    for line in q:
        i, mem = [x.strip() for x in line.split(",")]
        if int(i) not in busy and int(mem) < 2048 and (allowed is None or int(i) in allowed):
            free.append(int(i))
    return free


def status(name):
    for s in ("done", "failed", "skipped"):
        if os.path.exists(f"{state}/{name}.{s}"):
            return s
    return "running" if name in running else "pending"


def mark(name, s, msg=""):
    with open(f"{state}/{name}.{s}", "w") as f:
        f.write(msg)


while True:
    jobs = json.load(open(jobs_file))
    names = {j["name"] for j in jobs}
    # reap
    for name, (p, g, a) in list(running.items()):
        rc = p.poll()
        if rc is None:
            continue
        del running[name]
        try:  # each job runs in its own process group: remove leftovers (e.g. orphaned vLLM engine processes)
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if rc == 0:
            mark(name, "done", time.ctime())
        elif a < retries:
            attempts[name] = a + 1
        else:
            mark(name, "failed", f"rc={rc} {time.ctime()}")
    # skip dependents of failures
    for j in jobs:
        if status(j["name"]) == "pending" and any(status(d) in ("failed", "skipped") for d in j.get("deps", [])):
            mark(j["name"], "skipped", "dependency failed")
    free = gpu_free_list()
    ncpu = sum(1 for _, g, _ in running.values() if g is None)
    for j in jobs:
        n = j["name"]
        if status(n) != "pending" or any(status(d) != "done" for d in j.get("deps", []) if d in names):
            continue
        if any(d not in names for d in j.get("deps", [])):
            continue
        g = None
        need = int(j.get("gpu", 0))
        if need:
            if len(free) < need:
                continue
            g, free = free[:need], free[need:]
        elif ncpu >= max_cpu:
            continue
        else:
            ncpu += 1
        env = dict(os.environ)
        if g is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, g))
        a = attempts.get(n, 0)
        log = open(f"{state}/logs/{n}.log", "a")
        log.write(f"\n===== start {time.ctime()} gpu={g} attempt={a}\n{j['cmd']}\n")
        log.flush()
        p = subprocess.Popen(["bash", "-lc", j["cmd"]], stdout=log, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        running[n] = (p, g, a)
    summ = {j["name"]: status(j["name"]) for j in jobs}
    summ["_time"] = time.ctime()
    summ["_running_gpus"] = {n: g for n, (_, g, _) in running.items()}
    json.dump(summ, open(f"{state}/status.json", "w"), indent=1)
    if not running and all(status(j["name"]) in ("done", "failed", "skipped") for j in jobs):
        if not os.path.exists(f"{state}/KEEPALIVE"):
            break
    time.sleep(15)
print("all jobs finished", time.ctime())
