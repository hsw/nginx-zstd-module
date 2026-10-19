#!/usr/bin/env python3
"""Regression test: a request returns its ZSTD_CCtx when its frame ends.

The worker-lifetime CCtx ring lends a context to one request at a time. The
body filter is done with it once the terminal frame is out of libzstd (or the
stream has failed), but a slow client keeps the request alive while nginx
drains the already-compressed tail. The context must go back at that point,
not when the request ends. Every scenario runs on a fresh single-worker nginx
(fresh ring, own log).

S1  A fetches a 1 MiB incompressible file through a client that reads
    nothing; the whole frame fits in zstd_buffers, so it is finished inside
    A's content-handler call. B then fetches the same location and must
    reuse A's slot. The log order (A frame done < B acquire < A close
    request) is asserted first, so the reuse verdict cannot pass for the
    wrong reason.

S2  The idempotent return. A returns its slot early; C (same profile,
    zstd_buffers 2 8k, a client that reads nothing) borrows it and stalls
    mid-frame; draining A runs A's pool cleanup, which must be a no-op. D
    must not be lent C's context, C must decode byte-exact, and C must
    return the context itself when its frame ends on a resumed call. At
    --log-level warn only the decode runs; a second return still corrupts
    C's stream.

S3  Four distinct profiles seed every ring slot; a fifth gets an uncached
    context, which must be freed before its request ends.

S4  The failure exit. An unbuffered proxied response loses its client to a
    reset mid-stream; the next write fails and the body filter takes its
    failure exit, which must return the context (with no terminal frame
    before the return). A follow-up request must reuse the slot.

S4u S4 with the ring full of the S3 fillers: the failure exit must free an
    uncached context.

S5  Requests terminated mid-frame: X1 (on loan) and X2 (uncached, the ring
    full of the S3 fillers) stall like S2's C until send_timeout ends them
    with no further body-filter call, so only the pool cleanup can give the
    contexts back. A follow-up must reuse X1's slot; at --log-level warn,
    the LSan verdict of a sanitizer build catches X2's context leaking.

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
import struct
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

# How long to wait for a debug-log line the schedule depends on, and how
# often to re-read the log meanwhile.
LOG_WAIT = 10.0
LOG_POLL = 0.05

# Time given to the worker to act on something no log line or client-side
# signal confirms: S2's warn run waiting for A's pool to go, and the S4
# worker taking its client's reset.
SETTLE = 0.3

# nginx prints %p as bare uppercase hex and tags every connection-scoped
# line with "*<connection number> ".
ACQUIRE_RE = re.compile(
    r"\*(?P<conn>\d+) zstd: (?:created cctx (?P<ptr>[0-9A-Fa-f]+) "
    r"\(cached:(?P<cached>\d)\)|reusing worker cctx (?P<rptr>[0-9A-Fa-f]+) "
    r"\(slot:\d+\))"
)
RETURNED_RE = re.compile(
    r"\*(?P<conn>\d+) zstd: returned cctx (?P<ptr>[0-9A-Fa-f]+) before "
    r"request end \(cached:(?P<cached>\d)\)"
)
# The emit probe line on which ctx->done is set: libzstd reported the frame
# fully flushed (rc0:1) for the terminal call (last:1).
FRAME_DONE = r"zstd emit\?: outsz:\d+ rc0:1 last:1 "
CLOSE_REQUEST = r"http close request$"
# Logged at info when send_timeout ends a stalled response.
TIMED_OUT = r"client timed out"
# The log lines a FAIL prints as evidence.
EVIDENCE_RE = re.compile(
    rf"zstd: (?:created|reusing|returned) |{FRAME_DONE}|{CLOSE_REQUEST}|{TIMED_OUT}"
)

# The S3 ring fillers: four distinct memory profiles (level, long mode,
# window log), one per ring slot. Each location is requested once, in order.
S3_FILLERS = ("/s1/", "/s3/l9/", "/s3/long/", "/s3/wlog/")
# A fifth profile none of the fillers has.
S3_UNCACHED = "/s3/l1/"

# S4 upstream: what nginx gets before the client goes away, and what it
# gets after. Sized below proxy_buffer_size so each part is one read.
S4_HEAD_SIZE = 2048
S4_TAIL_SIZE = 3072


class Witness(typing.NamedTuple):
    """One CCtx acquisition or early-return line from the debug log."""

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
        "--backend-port",
        type=int,
        default=18126,
        help="Port of the in-tool upstream S4 proxies to.",
    )
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
    """nginx.conf with one location per scenario profile."""
    # S5 waits for the info-level timeout line at either log level.
    s5_log_level = "debug" if args.log_level == "debug" else "info"
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

        # S2's C: the same profile as /s1/ (buffer settings are not part of
        # it), but only 16 KiB of output buffers, so a client that reads
        # nothing stalls the frame long before its end.
        location /s2/ {{
            alias {html}/;
            zstd_buffers 2 8k;
        }}

        # S3: one location per remaining ring slot, each a distinct
        # profile, plus /s3/l1/ whose profile none of the slots has.
        location /s3/l9/ {{
            alias {html}/;
            zstd_comp_level 9;
        }}
        location /s3/long/ {{
            alias {html}/;
            zstd_long on;
        }}
        location /s3/wlog/ {{
            alias {html}/;
            zstd_window_log 20;
        }}
        location /s3/l1/ {{
            alias {html}/;
            zstd_comp_level 1;
        }}

        # S4: unbuffered, so each upstream read reaches the filter as it
        # arrives; proxy_ignore_client_abort leaves the client's reset to be
        # found by the next write, which fails back through the zstd filter.
        location /s4/ {{
            proxy_pass http://127.0.0.1:{args.backend_port}/;
            proxy_buffering off;
            proxy_ignore_client_abort on;
        }}
        # S4u: /s3/l1/'s profile, uncached once the S3 fillers hold every
        # slot.
        location /s4/l1/ {{
            proxy_pass http://127.0.0.1:{args.backend_port}/;
            proxy_buffering off;
            proxy_ignore_client_abort on;
            zstd_comp_level 1;
        }}

        # S5: /s2/'s stall, ended by send_timeout. /s5/l1/ has /s3/l1/'s
        # profile.
        location /s5/ {{
            alias {html}/;
            zstd_buffers 2 8k;
            send_timeout 1s;
            error_log {root}/logs/error.log {s5_log_level};
        }}
        location /s5/l1/ {{
            alias {html}/;
            zstd_buffers 2 8k;
            send_timeout 1s;
            error_log {root}/logs/error.log {s5_log_level};
            zstd_comp_level 1;
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
    """Every acquisition (ACQUIRE_RE) or early return (RETURNED_RE), in order."""
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


def returned_before_close(lines: list[str], w: Witness, cached: bool) -> Witness | None:
    """w's early return, if it gave back w's context before w's request closed."""
    ret = next((x for x in witnesses(lines, RETURNED_RE) if x.conn == w.conn), None)
    close = closed(lines, w)
    if ret is None or ret.ptr != w.ptr or ret.cached != cached:
        return None
    if close is None or ret.index > close:
        return None
    return ret


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

    def wait_for_log(
        self, pattern: str, count: int = 1, *, always: bool = False
    ) -> list[str]:
        """At debug, wait for count lines matching pattern; return the log.

        A client's EOF does not prove nginx has freed its request (lingering
        close frees it later), so witnesses that need a close line wait for
        it. Without the debug log there is nothing to wait for, unless the
        line comes from a location that logs it in either run (always).
        """
        if not (self.debug or always):
            return []
        rx = re.compile(pattern)
        deadline = time.monotonic() + LOG_WAIT
        while True:
            lines = self.log_lines()
            if sum(1 for line in lines if rx.search(line)) >= count:
                return lines
            if time.monotonic() > deadline:
                raise RuntimeError(f"no {count} log lines matching {pattern!r}")
            time.sleep(LOG_POLL)

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


def scenario_s2(h: Harness) -> tuple[list[str], list[Outcome]]:
    """S2: A's pool destruction must not hand C's context to D."""
    want_a, want_c, want_d = fixture(3), fixture(4), fixture(5)
    files = {"a.txt": want_a, "c.txt": want_c, "d.txt": want_d}

    with h.nginx_running("s2", files):
        a = open_request(h.args.port, "/s1/a.txt", rcvbuf=SLOW_RCVBUF)
        await_first_byte(a, "S2 A")

        # C borrows the slot A gave back and stalls before its frame ends:
        # 16 KiB of output buffers against a client that reads nothing.
        c = open_request(h.args.port, "/s2/c.txt", rcvbuf=SLOW_RCVBUF)
        await_first_byte(c, "S2 C")

        # Draining A frees A's request, and with it A's pool, whose cleanup
        # runs while C is mid-frame. D must only arrive after that.
        body_a = read_response(a, "/s1/a.txt")
        if h.debug:
            conn_a = witnesses(h.log_lines(), ACQUIRE_RE)[0].conn
            h.wait_for_log(rf"\*{conn_a} {CLOSE_REQUEST}")
        else:
            time.sleep(SETTLE)

        body_d = read_response(open_request(h.args.port, "/s1/d.txt"), "/s1/d.txt")
        body_c = read_response(c, "/s2/c.txt")
        lines = h.all_closed(3)

    return lines, [("A", body_a, want_a), ("C", body_c, want_c), ("D", body_d, want_d)]


def s2_verdict(lines: list[str]) -> str | None:
    """S2 schedule guard, the hazard, then C's own return; None when all hold."""
    acq = witnesses(lines, ACQUIRE_RE)
    if len(acq) < 3 or len({x.conn for x in acq[:3]}) != 3:
        return f"FAIL[order]: S2: expected acquisitions by A, C and D, found {len(acq)}"
    a, c, d = acq[:3]
    if not a.seeded:
        return f"FAIL[order]: S2: A did not seed a ring slot: {zstd_part(a.line)}"
    # C must hold the very context A gave back, or A's pool cleanup has
    # nothing of C's to damage.
    if not (c.reused and c.ptr == a.ptr):
        return (
            f"FAIL[order]: S2: C did not take over A's slot {a.ptr}; C logged: "
            f"{zstd_part(c.line)}"
        )
    c_done = frame_done(lines, c)
    # D comes after A's pool is gone and while C is still mid-frame.
    order = (frame_done(lines, a), c.index, closed(lines, a), d.index, c_done)
    if None in order or sorted(order) != list(order):
        return (
            f"FAIL[order]: S2: expected A frame done < C acquire < A close "
            f"request < D acquire < C frame done, got lines {order}"
        )

    if d.reused and d.ptr == c.ptr:
        return (
            f"FAIL[hazard]: S2: D was lent context {c.ptr} while C was "
            f"still compressing with it -- A's request pool cleanup fired "
            f"after the early return and released the slot a second time"
        )

    # C's frame ended on a resumed call after the stall; that exit must
    # return the context too.
    ret = returned_before_close(lines, c, cached=True)
    if ret is None or ret.index < c_done:
        return (
            f"FAIL[resumed]: S2: C's frame ended on a resumed call (line "
            f"{c_done}) but {c.ptr} was not returned before C's request ended"
        )
    return None


def fill_ring(h: Harness, files: dict[str, bytes]) -> list[Outcome]:
    """Fetch the S3 fillers in order, so every ring slot holds a profile."""
    results = []
    for i, p in enumerate(S3_FILLERS):
        url, want = f"{p}f{i}.txt", files[f"f{i}.txt"]
        results.append((url, read_response(open_request(h.args.port, url), url), want))
    return results


def filler_files() -> dict[str, bytes]:
    """The bodies fill_ring() fetches."""
    return {f"f{i}.txt": fixture(10 + i) for i in range(len(S3_FILLERS))}


def acquisitions(
    lines: list[str], name: str, count: int
) -> tuple[list[Witness], str | None]:
    """Every acquisition, and a FAIL[order] verdict unless there are count."""
    acq = witnesses(lines, ACQUIRE_RE)
    if len(acq) != count:
        return (
            acq,
            f"FAIL[order]: {name}: expected {count} acquisitions, found {len(acq)}",
        )
    return acq, None


def uncached_on_full_ring(name: str, w: Witness) -> str | None:
    """A verdict unless w, acquired once the fillers hold every slot, is uncached."""
    if w.cached:
        return (
            f"FAIL[order]: {name}: expected an uncached context on a full ring; "
            f"got: {zstd_part(w.line)}"
        )
    return None


def scenario_s3(h: Harness) -> tuple[list[str], list[Outcome]]:
    """S3: an uncached per-request context is freed at frame end."""
    last = f"f{len(S3_FILLERS)}.txt"
    files = {**filler_files(), last: fixture(10 + len(S3_FILLERS))}

    with h.nginx_running("s3", files):
        results = fill_ring(h, files)
        url = f"{S3_UNCACHED}{last}"
        body = read_response(open_request(h.args.port, url), url)
        results.append((url, body, files[last]))
        lines = h.all_closed(len(S3_FILLERS) + 1)

    return lines, results


def s3_verdict(lines: list[str]) -> str | None:
    """S3 ring-full guard, then the uncached return; None when both hold."""
    acq, verdict = acquisitions(lines, "S3", len(S3_FILLERS) + 1)
    if verdict is not None:
        return verdict
    *fillers, last = acq
    distinct = len({x.ptr for x in fillers}) == len(fillers)
    if not (distinct and all(x.seeded for x in fillers)):
        return "FAIL[order]: S3: the four fillers did not each seed their own slot"
    verdict = uncached_on_full_ring("S3", last)
    if verdict is not None:
        return verdict

    if returned_before_close(lines, last, cached=False) is None:
        return (
            f"FAIL[uncached]: S3: uncached context {last.ptr} was not freed "
            f"before its request ended"
        )
    return None


def reset_mid_stream(h: Harness, listener: socket.socket, path: str, name: str) -> None:
    """Proxy path, reset its client mid-stream, then fail nginx's next write."""
    x = open_request(h.args.port, path)
    up, _ = listener.accept()
    with up:
        up.settimeout(IO_TIMEOUT)
        req = b""
        while b"\r\n\r\n" not in req:
            piece = up.recv(READ_CHUNK)
            if not piece:
                raise RuntimeError(f"{name} upstream: nginx closed early")
            req += piece
        # Close-delimited: nginx learns the length only at the end.
        up.sendall(
            b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\n"
            + fixture(20, S4_HEAD_SIZE)
        )

        # Headers plus the first compressed bytes, which the unbuffered
        # proxy flushes at once.
        raw = b""
        while not raw.partition(b"\r\n\r\n")[2]:
            piece = x.recv(READ_CHUNK)
            if not piece:
                raise RuntimeError(f"{name} {path}: connection closed before any body")
            raw += piece
        check_zstd_headers(raw.partition(b"\r\n\r\n")[0], path)

        # Reset, not FIN, so the next write fails; give the worker a moment
        # to take the reset before that write.
        x.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        x.close()
        time.sleep(SETTLE)

        # The second part's write fails, and nginx then drops this
        # connection; a reset here is that teardown.
        with contextlib.suppress(OSError):
            up.sendall(fixture(21, S4_TAIL_SIZE))
            while up.recv(READ_CHUNK):
                pass


def scenario_s4(h: Harness) -> tuple[list[str], list[Outcome]]:
    """S4: the body filter's failure exit returns the context too."""
    want_y = fixture(22)

    with socket.create_server(("127.0.0.1", h.args.backend_port)) as listener:
        listener.settimeout(IO_TIMEOUT)
        with h.nginx_running("s4", {"y.txt": want_y}):
            reset_mid_stream(h, listener, "/s4/x", "S4")
            body_y = read_response(open_request(h.args.port, "/s1/y.txt"), "/s1/y.txt")
            lines = h.all_closed(2)

    return lines, [("follow-up", body_y, want_y)]


def scenario_s4u(h: Harness) -> tuple[list[str], list[Outcome]]:
    """S4u: the failure exit frees an uncached context."""
    files = filler_files()

    with socket.create_server(("127.0.0.1", h.args.backend_port)) as listener:
        listener.settimeout(IO_TIMEOUT)
        with h.nginx_running("s4u", files):
            results = fill_ring(h, files)
            reset_mid_stream(h, listener, "/s4/l1/x", "S4u")
            lines = h.all_closed(len(S3_FILLERS) + 1)

    return lines, results


def failure_exit(lines: list[str], x: Witness, name: str) -> str | None:
    """x's context returned by the failure exit; None when that holds."""
    ret = returned_before_close(lines, x, cached=x.cached)
    if ret is None:
        return (
            f"FAIL[failed-path]: {name}: X's write to the reset client failed, "
            f"but its context {x.ptr} was not returned before its request ended"
        )
    # The other call site needs a finished frame: a return with no terminal
    # emit before it is the failure exit.
    x_done = frame_done(lines, x)
    if x_done is not None and x_done < ret.index:
        return (
            f"FAIL[failed-path]: {name}: X's frame finished before the return, "
            f"so this run took the frame-end exit, not the failure exit"
        )
    return None


def s4_verdict(lines: list[str]) -> str | None:
    """S4 failure-exit witness, then the reuse of the slot; None when both hold."""
    acq, verdict = acquisitions(lines, "S4", 2)
    if verdict is not None:
        return verdict
    x, y = acq
    if not x.seeded:
        return f"FAIL[order]: S4: X did not seed a ring slot: {zstd_part(x.line)}"

    failed = failure_exit(lines, x, "S4")
    if failed is not None:
        return failed
    if not (y.reused and y.ptr == x.ptr):
        return (
            f"FAIL[reuse]: S4: the follow-up did not reuse X's slot {x.ptr}; "
            f"it logged: {zstd_part(y.line)}"
        )
    return None


def s4u_verdict(lines: list[str]) -> str | None:
    """S4u ring-full guard, then the failure exit; None when both hold."""
    acq, verdict = acquisitions(lines, "S4u", len(S3_FILLERS) + 1)
    if verdict is not None:
        return verdict
    x = acq[-1]
    return uncached_on_full_ring("S4u", x) or failure_exit(lines, x, "S4u")


def scenario_s5(h: Harness) -> tuple[list[str], list[Outcome]]:
    """S5: the pool cleanup returns the context of a request ended mid-frame."""
    files = {**filler_files(), "x.txt": fixture(30), "y.txt": fixture(31)}

    with h.nginx_running("s5", files):
        results = fill_ring(h, files)
        # X1 borrows the /s1/ filler's slot; X2's profile has none, so X2
        # gets an uncached context. Neither client reads: both stall
        # mid-frame until send_timeout terminates them.
        x1 = open_request(h.args.port, "/s5/x.txt", rcvbuf=SLOW_RCVBUF)
        await_first_byte(x1, "S5 X1")
        x2 = open_request(h.args.port, "/s5/l1/x.txt", rcvbuf=SLOW_RCVBUF)
        await_first_byte(x2, "S5 X2")
        # Reading earlier would resume the frame; the timeout handler frees
        # the request before the worker can accept the follow-up.
        h.wait_for_log(TIMED_OUT, 2, always=True)

        # Each is cut short, or the timeout never ended it mid-frame.
        for label, x in (("X1", x1), ("X2", x2)):
            body = read_response(x, f"S5 {label}")
            try:
                test_encoding.decompress_payload(h.args.zstd_bin, body)
            except RuntimeError:
                continue
            raise RuntimeError(f"S5 {label}: the frame completed before the timeout")

        body_y = read_response(open_request(h.args.port, "/s1/y.txt"), "/s1/y.txt")
        lines = h.all_closed(len(S3_FILLERS) + 3)

    return lines, [*results, ("follow-up", body_y, files["y.txt"])]


def s5_verdict(lines: list[str]) -> str | None:
    """S5 schedule guard, then the slot reuse; None when both hold."""
    acq, verdict = acquisitions(lines, "S5", len(S3_FILLERS) + 3)
    if verdict is not None:
        return verdict
    x1, x2, y = acq[-3:]
    if not x1.reused or x2.cached:
        return (
            f"FAIL[order]: S5: expected X1 on loan and X2 uncached; got: "
            f"{zstd_part(x1.line)} / {zstd_part(x2.line)}"
        )
    returned = {w.conn for w in witnesses(lines, RETURNED_RE)}
    for label, x in (("X1", x1), ("X2", x2)):
        # Ended by the timer, mid-frame, with no body-filter return.
        timed_out = first_index(lines, rf"\*{x.conn} {TIMED_OUT}", x.index)
        if timed_out is None or frame_done(lines, x) is not None:
            return f"FAIL[order]: S5: {label} was not ended mid-frame by send_timeout"
        if x.conn in returned:
            return (
                f"FAIL[order]: S5: {label} returned its context from the body "
                f"filter, so the pool cleanup was not what returned it"
            )

    order = (x1.index, closed(lines, x1), y.index)
    if None in order or sorted(order) != list(order):
        return (
            f"FAIL[order]: S5: expected X1 acquire < X1 close request < "
            f"follow-up acquire, got lines {order}"
        )
    if not (y.reused and y.ptr == x1.ptr):
        return (
            f"FAIL[cleanup]: S5: X1's request ended mid-frame, but the "
            f"follow-up did not reuse its context {x1.ptr} -- the pool cleanup "
            f"did not return it; the follow-up logged: {zstd_part(y.line)}"
        )
    return None


SCENARIOS = (
    ("S1", scenario_s1, s1_verdict),
    ("S2", scenario_s2, s2_verdict),
    ("S3", scenario_s3, s3_verdict),
    ("S4", scenario_s4, s4_verdict),
    ("S4u", scenario_s4u, s4u_verdict),
    ("S5", scenario_s5, s5_verdict),
)


def main() -> int:
    """Run every scenario; exit 1 if any assertion failed, 2 on errors only."""
    args = parse_args()
    nginx = pathlib.Path(args.nginx_binary)
    if not nginx.exists():
        raise FileNotFoundError(nginx)
    h = Harness(args, nginx)

    v = subprocess.run([str(nginx), "-V"], capture_output=True, text=True, check=False)
    if "zstd" not in v.stderr:
        raise RuntimeError("nginx -V shows no zstd module")
    if h.debug and "--with-debug" not in v.stderr:
        raise RuntimeError(
            "the witnesses are ngx_log_debug lines: this tool needs an nginx "
            "built --with-debug, or --log-level warn to skip them"
        )
    if not h.debug:
        print("witnesses skipped (--log-level warn): only the decodes gate this run")

    os.umask(0o022)
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
            "OK: a request returns its CCtx when its frame ends or fails, not "
            "when its request ends, and never twice; one ended mid-frame "
            "returns it from its pool cleanup; every stream decoded byte-exact"
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
