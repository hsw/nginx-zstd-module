#!/usr/bin/env python3
"""Regression test: a request returns its ZSTD_CCtx when its frame ends.

The worker-lifetime CCtx ring lends a context to one request at a time. The
body filter is done with it once the terminal frame is out of libzstd (or the
stream has failed), but a slow client keeps the request alive while nginx
drains the already-compressed tail. The context must go back at that point,
not when the request ends. The run uses a fresh single-worker nginx (fresh
ring, own log).

S1  A fetches a 1 MiB incompressible file through a client that reads
    nothing; the whole frame fits in zstd_buffers, so it is finished inside
    A's content-handler call. B then fetches the same location and must
    reuse A's slot. The log order (A frame done < B acquire < A close
    request) is asserted first, so the reuse verdict cannot pass for the
    wrong reason.

Every body must decode byte-exact. Exit status: 0 pass, 1 assertion failure
(each printed as FAIL[class]), 2 a scenario could not establish its
preconditions (printed as ERROR) and nothing failed.

Needs an nginx built --with-debug for the witnesses (see --log-level), and
the zstd CLI for decoding.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import pathlib
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
import typing

MODULE_PATH = pathlib.Path(__file__).with_name("test_encoding.py")
SPEC = importlib.util.spec_from_file_location("test_encoding", MODULE_PATH)
test_encoding = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(test_encoding)

# Incompressible, so the compressed body cannot fit in the socket buffers: a
# client that reads nothing keeps its request alive.
BODY_SIZE = 1024 * 1024

# Small client receive buffer, set before connect() so the window is
# negotiated small. Together with the listener's LISTEN_SNDBUF the kernel
# holds a few tens of KiB of the response; nginx has to hold the rest.
SLOW_RCVBUF = 16384
LISTEN_SNDBUF = 16384

READ_CHUNK = 65536
IO_TIMEOUT = 30.0

# How long to wait for a debug-log line the schedule depends on.
LOG_WAIT = 10.0

# nginx prints %p as bare uppercase hex and tags every connection-scoped
# line with "*<connection number> ".
ACQUIRE_RE = re.compile(
    r"\*(?P<conn>\d+) zstd: (?:created cctx (?P<ptr>[0-9A-Fa-f]+) "
    r"\(cached:(?P<cached>\d)\)|reusing worker cctx (?P<rptr>[0-9A-Fa-f]+) "
    r"\(slot:\d+\))"
)
# The emit probe line on which ctx->done is set: libzstd reported the frame
# fully flushed (rc0:1) for the terminal call (last:1).
FRAME_DONE = r"zstd emit\?: outsz:\d+ rc0:1 last:1 "
CLOSE_REQUEST = r"http close request$"
# The log lines a FAIL prints as evidence.
EVIDENCE_RE = re.compile(rf"zstd: (?:created|reusing) |{FRAME_DONE}|{CLOSE_REQUEST}")


class Witness(typing.NamedTuple):
    """One CCtx acquisition line from the debug log."""

    index: int
    conn: str
    ptr: str
    cached: bool
    reused: bool
    line: str

    @property
    def seeded(self) -> bool:
        """Whether this acquisition created a context and put it in a slot."""
        return self.cached and not self.reused


# (label, compressed body, origin body), checked by check_decode().
Outcome = tuple[str, bytes, bytes]


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments (binary, module path, port, log level)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nginx-binary", required=True)
    parser.add_argument("--filter-module")
    parser.add_argument("--port", type=int, default=18119, help="nginx listen port.")
    parser.add_argument(
        "--log-level",
        choices=("debug", "warn"),
        default="debug",
        help="Server error_log level. The witnesses are only asserted at "
        "debug. Use warn under sanitizer builds, whose UBSAN halts on nginx "
        "core's own debug logging (see test_slow_drain.py); the scenarios "
        "and the byte-exact decodes still run.",
    )
    parser.add_argument(
        "--zstd-bin",
        default=test_encoding.shutil.which("zstd") or "zstd",
    )
    return parser.parse_args()


def fixture(seed: int, size: int = BODY_SIZE) -> bytes:
    """Deterministic incompressible body, distinct per seed.

    Distinct matters: a context that replayed another request's state would
    still decode byte-exact if every request shared one body.
    """
    return random.Random(seed).randbytes(size)


def render_conf(
    root: pathlib.Path,
    html: pathlib.Path,
    load: str,
    args: argparse.Namespace,
) -> str:
    """nginx.conf for the S1 location."""
    return f"""worker_processes 1;
{load}error_log {root}/logs/error.log warn;
pid {root}/nginx.pid;
events {{ worker_connections 64; }}
http {{
    access_log off;
    default_type text/plain;
    zstd on;
    zstd_min_length 1;
    zstd_types text/plain;
    server {{
        # A small socket send buffer keeps the kernel from absorbing the
        # slow client's response: nginx has to keep the request alive.
        listen 127.0.0.1:{args.port} sndbuf={LISTEN_SNDBUF};
        error_log {root}/logs/error.log {args.log_level};

        # 2 MiB of output buffers hold a whole 1 MiB incompressible frame,
        # and output_buffers hands the filter the file as ONE input buffer,
        # so last_buf is reached inside the content-handler call.
        output_buffers 1 1m;
        zstd_buffers 16 128k;

        # The default profile (level 3, no long mode, level-derived window).
        location /s1/ {{
            alias {html}/;
        }}
    }}
}}
"""


def open_request(port: int, path: str, rcvbuf: int | None = None) -> socket.socket:
    """Connect and send an HTTP/1.0 GET for path accepting zstd.

    HTTP/1.0 gives a close-delimited, non-chunked body (decodable as is) and
    makes nginx free the request when the body is drained, not on a later
    keepalive timeout.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(IO_TIMEOUT)
    if rcvbuf is not None:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    s.connect(("127.0.0.1", port))
    s.sendall(
        f"GET {path} HTTP/1.0\r\nHost: t\r\nAccept-Encoding: zstd\r\n"
        f"Connection: close\r\n\r\n".encode("latin1")
    )
    return s


def await_first_byte(s: socket.socket, label: str) -> None:
    """Block until the response has started, without consuming anything.

    The first byte proves nginx has run the request's content handler; with
    a single worker that whole call completes before the next connection is
    accepted.
    """
    if not s.recv(1, socket.MSG_PEEK):
        raise RuntimeError(f"{label}: connection closed before any response")


def check_zstd_headers(head: bytes, path: str) -> None:
    """Require a 200 zstd response; anything else is a setup error."""
    headers = head.decode("latin1", "replace")
    status = headers.splitlines()[0] if headers else "<no response>"
    if " 200 " not in f"{status} ":
        raise RuntimeError(f"{path}: expected 200, got {status!r}")
    m = re.search(r"(?im)^content-encoding:\s*(\S+)", headers)
    if m is None or m.group(1) != "zstd":
        raise RuntimeError(
            f"{path}: expected Content-Encoding: zstd, got "
            f"{m.group(1) if m else None!r} -- the response was not "
            f"compressed, so this run proves nothing about the CCtx"
        )


def read_response(s: socket.socket, path: str) -> bytes:
    """Read to EOF and return the zstd body; anything else is a setup error."""
    raw = bytearray()
    while True:
        piece = s.recv(READ_CHUNK)
        if not piece:
            break
        raw += piece
    s.close()
    head, _, body = bytes(raw).partition(b"\r\n\r\n")
    check_zstd_headers(head, path)
    return body


def check_decode(zstd_bin: str, scenario: str, results: list[Outcome]) -> list[str]:
    """Byte-exact oracle for every (label, compressed, origin) triple."""
    failures = []
    for label, body, want in results:
        try:
            got = test_encoding.decompress_payload(zstd_bin, body)
        except RuntimeError:
            failures.append(
                f"FAIL[decode]: {scenario} {label}: corrupt or truncated stream"
            )
            continue
        if got != want:
            failures.append(
                f"FAIL[decode]: {scenario} {label}: decoded {len(got)}B "
                f"!= origin {len(want)}B"
            )
    return failures


def witnesses(lines: list[str], rx: re.Pattern[str]) -> list[Witness]:
    """Every acquisition witness, in log order."""
    out = []
    for i, line in enumerate(lines):
        m = rx.search(line)
        if m is None:
            continue
        rptr = m.groupdict().get("rptr")
        reused = rptr is not None
        cached = reused or m.group("cached") == "1"
        out.append(
            Witness(i, m.group("conn"), rptr or m.group("ptr"), cached, reused, line)
        )
    return out


def first_index(lines: list[str], pattern: str, start: int = 0) -> int | None:
    """Index of the first line at or after start that matches pattern."""
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.search(lines[i]):
            return i
    return None


def frame_done(lines: list[str], w: Witness) -> int | None:
    """Index of the line on which w's request finished its frame."""
    return first_index(lines, rf"\*{w.conn} {FRAME_DONE}", w.index)


def closed(lines: list[str], w: Witness) -> int | None:
    """Index of w's request's "http close request" line."""
    return first_index(lines, rf"\*{w.conn} {CLOSE_REQUEST}", w.index)


def zstd_part(line: str) -> str:
    """The module's own text of a debug line, for failure messages."""
    return line[line.find("zstd:") :]


class Harness:
    """One nginx instance per scenario, with a fresh log and ring."""

    def __init__(self, args: argparse.Namespace, nginx: pathlib.Path) -> None:
        """Resolve the module path once for every scenario's config."""
        self.args = args
        self.nginx = nginx
        filter_so = test_encoding.detect_module_path(
            args.filter_module, nginx, "ngx_http_zstd_filter_module.so"
        )
        self.load = f"load_module {filter_so};\n" if filter_so else ""
        self.root: pathlib.Path | None = None

    @property
    def debug(self) -> bool:
        """Whether the debug-log witnesses exist in this run."""
        return self.args.log_level == "debug"

    def log_lines(self) -> list[str]:
        """The error log so far; error_log writes are unbuffered."""
        assert self.root is not None
        path = self.root / "logs" / "error.log"
        return path.read_text("utf-8", "replace").splitlines()

    def wait_for_log(self, pattern: str, count: int = 1) -> list[str]:
        """At debug, wait for count lines matching pattern; return the log.

        A client's EOF does not prove nginx has freed its request (lingering
        close frees it later), so witnesses that need a close line wait for
        it. Without the debug log there is nothing to wait for.
        """
        if not self.debug:
            return []
        rx = re.compile(pattern)
        deadline = time.monotonic() + LOG_WAIT
        while True:
            lines = self.log_lines()
            if sum(1 for line in lines if rx.search(line)) >= count:
                return lines
            if time.monotonic() > deadline:
                raise RuntimeError(f"no {count} log lines matching {pattern!r}")
            time.sleep(0.05)

    def all_closed(self, requests: int) -> list[str]:
        """The log once every request of the scenario has been freed."""
        return self.wait_for_log(rf"\*\d+ {CLOSE_REQUEST}", requests)

    @contextlib.contextmanager
    def nginx_running(self, name: str, files: dict[str, bytes]):
        """Start a single-worker nginx in a fresh prefix serving files."""
        with tempfile.TemporaryDirectory(prefix=f"zstd-early-{name}-") as td:
            # mkdtemp is 0700: a root-run nginx (the sanitizer job) drops
            # privileges in the worker and could not read the fixtures.
            os.chmod(td, 0o755)
            root = pathlib.Path(td)
            (root / "logs").mkdir()
            html = root / "html"
            html.mkdir()
            for fname, data in files.items():
                (html / fname).write_bytes(data)
            conf = root / "nginx.conf"
            conf.write_text(
                render_conf(root, html, self.load, self.args),
                encoding="utf-8",
            )
            self.root = root

            nlog_path = root / "logs" / "nginx-stdout.log"
            # The handle is Popen's stdout for the server's whole lifetime.
            nlog = open(nlog_path, "w", encoding="utf-8")  # noqa: SIM115
            proc = subprocess.Popen(
                [
                    str(self.nginx),
                    "-p",
                    str(root),
                    "-c",
                    str(conf),
                    "-g",
                    "daemon off; master_process off;",
                ],
                stdout=nlog,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                test_encoding.wait_for_port(self.args.port)
                if proc.poll() is not None:
                    nlog.flush()
                    raise RuntimeError(
                        f"{name}: nginx exited during startup; tail:\n"
                        + nlog_path.read_text("utf-8", "replace")[-2000:]
                    )
                yield
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
                nlog.close()
                self.root = None


def scenario_s1(h: Harness) -> tuple[list[str], list[Outcome]]:
    """S1: B reuses the slot A was lent once A's frame is done."""
    want_a, want_b = fixture(1), fixture(2)

    with h.nginx_running("s1", {"a.txt": want_a, "b.txt": want_b}):
        a = open_request(h.args.port, "/s1/a.txt", rcvbuf=SLOW_RCVBUF)
        # A's frame is done by the time its first byte is out; the log
        # order checks that claim instead of trusting it.
        await_first_byte(a, "S1 A")

        body_b = read_response(open_request(h.args.port, "/s1/b.txt"), "/s1/b.txt")
        body_a = read_response(a, "/s1/a.txt")
        lines = h.all_closed(2)

    return lines, [("A", body_a, want_a), ("B", body_b, want_b)]


def s1_verdict(lines: list[str]) -> str | None:
    """S1 log order guard, then the reuse assertion; None when both hold."""
    acq = witnesses(lines, ACQUIRE_RE)
    # A, the first request on a fresh worker, must have seeded a slot.
    if not acq or not acq[0].seeded:
        return "FAIL[order]: S1: A did not seed a ring slot"
    a = acq[0]
    b = next((x for x in acq if x.conn != a.conn), None)

    order = (frame_done(lines, a), b.index if b else None, closed(lines, a))
    if None in order or sorted(order) != list(order):
        return (
            f"FAIL[order]: S1: expected A frame done < B acquire < A close "
            f"request, got lines {order}"
        )
    if not (b.reused and b.ptr == a.ptr):
        return (
            f"FAIL[reuse]: S1: B acquired while A's request was still alive "
            f"with its frame done, but did not reuse A's context {a.ptr} "
            f"-- A still holds the slot until its request ends; B logged: "
            f"{zstd_part(b.line)}"
        )
    return None


SCENARIOS = (("S1", scenario_s1, s1_verdict),)


def main() -> int:
    """Run every scenario; exit 1 if any assertion failed, 2 on errors only."""
    args = parse_args()
    nginx = pathlib.Path(args.nginx_binary)
    if not nginx.exists():
        raise FileNotFoundError(nginx)

    v = subprocess.run([str(nginx), "-V"], capture_output=True, text=True, check=False)
    if "zstd" not in v.stderr:
        raise RuntimeError("nginx -V shows no zstd module")
    if args.log_level == "debug" and "--with-debug" not in v.stderr:
        raise RuntimeError(
            "the witnesses are ngx_log_debug lines: this tool needs an nginx "
            "built --with-debug, or --log-level warn to skip them"
        )
    if args.log_level != "debug":
        print("witnesses skipped (--log-level warn): only the decodes gate this run")

    os.umask(0o022)
    h = Harness(args, nginx)
    failed = errored = False
    for name, run, check in SCENARIOS:
        try:
            lines, results = run(h)
        except Exception as exc:  # noqa: BLE001
            sys.stdout.flush()
            print(f"ERROR: {name}: {exc}", file=sys.stderr)
            errored = True
            continue
        failures = check_decode(args.zstd_bin, name, results)
        verdict = check(lines) if h.debug else None
        if verdict is not None:
            failures.insert(0, verdict)
        if not failures:
            print(f"{name} OK")
            continue
        failed = True
        sys.stdout.flush()
        for f in failures:
            print(f, file=sys.stderr)
        for i, line in enumerate(lines):
            if EVIDENCE_RE.search(line):
                print(f"  {name} log line {i}: {line}", file=sys.stderr)

    if failed:
        return 1
    if errored:
        return 2
    if h.debug:
        print(
            "OK: a request returns its CCtx when its frame ends, not when its "
            "request ends; every stream decoded byte-exact"
        )
    else:
        print("OK: every stream decoded byte-exact (early return not witnessed)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
