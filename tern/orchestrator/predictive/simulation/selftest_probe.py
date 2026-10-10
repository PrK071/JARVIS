"""Controlled probe; loaded as text and executed only inside the container."""
import json
import os
from pathlib import Path
import socket
import subprocess


def denied(path):
    try:
        Path(path).write_text("blocked probe", encoding="utf-8")
        return False
    except OSError:
        return True


def network_denied():
    routes = Path("/proc/net/route").read_text().splitlines()[1:]
    no_route = all(line.split()[0] == "lo" for line in routes) and set(os.listdir("/sys/class/net")) <= {"lo"}
    try:
        socket.create_connection(("198.51.100.1", 80), timeout=0.2).close()
        return False
    except OSError:
        return no_route


status = Path("/proc/self/status").read_text()
security = os.getuid() != 0 and all(value in status for value in (
    "NoNewPrivs:\t1", "Seccomp:\t2", "CapEff:\t0000000000000000"))
Path("/work/writable").write_text("ok")
Path("/work/link").symlink_to("/input")
children = []
pid_limit_enforced = False
try:
    for _ in range(48):
        children.append(subprocess.Popen(["/bin/sleep", "3"]))
except OSError:
    pid_limit_enforced = True
finally:
    for child in children:
        child.kill()
        child.wait()
try:
    memory = int(Path("/sys/fs/cgroup/memory.max").read_text())
    pids = int(Path("/sys/fs/cgroup/pids.max").read_text())
    quota, period = map(int, Path("/sys/fs/cgroup/cpu.max").read_text().split())
    resource_limits = memory <= 64 * 1024 * 1024 and pids <= 32 and 0 < quota / period <= 0.5 and pid_limit_enforced
except (OSError, ValueError):
    resource_limits = False
print(json.dumps({
    "write": denied("/input/write") and denied("/etc/escape") and Path("/work/writable").exists(),
    "network": network_denied(),
    "environment": set(os.environ) <= {"HOME", "PATH", "LANG", "LC_CTYPE", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"},
    "private_paths": all(not Path(path).exists() for path in ("/root/.ssh", "/root/.aws", "/var/run/docker.sock", "/mnt/c/Users", "/host")),
    "traversal_symlink": denied("/work/link/escape") and denied("/work/../escape") and denied("/root/escape"),
    "non_root_seccomp": security,
    "resource_limits": resource_limits,
}))
