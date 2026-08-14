"""Run a real multi-process PySearch cluster locally, without containers.

    uv run python scripts/run_cluster.py

Starts one uvicorn process per shard plus a coordinator, each with its own
database, waits for the cluster to report ready, and prints the coordinator's
address. Ctrl-C stops everything.

The point is that the distributed system does not depend on Docker: these are
ordinary OS processes talking real HTTP over real sockets. Compose packages the
same topology reproducibly, but nothing here needs it.
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

READY_TIMEOUT_SECONDS = 30.0


def start_node(port: int, environment: dict[str, str]) -> subprocess.Popen[bytes]:
    """Launch one uvicorn process with the given configuration."""
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env={**os.environ, **environment},
    )


def wait_until_ready(url: str, timeout: float = READY_TIMEOUT_SECONDS) -> bool:
    """Poll a node's readiness endpoint until it answers 200 or time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=3, help="number of shard nodes")
    parser.add_argument("--base-port", type=int, default=9000, help="first shard port")
    parser.add_argument("--coordinator-port", type=int, default=8000)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="where shard databases live; a temporary directory by default",
    )
    arguments = parser.parse_args()

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if arguments.data_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="pysearch-cluster-")
        data_dir = Path(temporary.name)
    else:
        data_dir = arguments.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

    processes: list[subprocess.Popen[bytes]] = []
    shard_urls = []

    try:
        for shard_id in range(arguments.shards):
            port = arguments.base_port + shard_id
            shard_urls.append(f"http://127.0.0.1:{port}")
            processes.append(
                start_node(
                    port,
                    {
                        "PYSEARCH_APP_NAME": f"pysearch-shard-{shard_id}",
                        "PYSEARCH_NODE_ROLE": "shard",
                        "PYSEARCH_SHARD_ID": str(shard_id),
                        "PYSEARCH_SHARD_COUNT": str(arguments.shards),
                        # One database per shard, never shared.
                        "PYSEARCH_STORAGE_PATH": str(data_dir / f"shard-{shard_id}.db"),
                    },
                )
            )

        for shard_id, url in enumerate(shard_urls):
            if not wait_until_ready(f"{url}/ready"):
                print(f"shard {shard_id} did not become ready", file=sys.stderr)
                return 1
            print(f"shard {shard_id} ready at {url}")

        processes.append(
            start_node(
                arguments.coordinator_port,
                {
                    "PYSEARCH_APP_NAME": "pysearch-coordinator",
                    "PYSEARCH_NODE_ROLE": "coordinator",
                    "PYSEARCH_SHARD_COUNT": str(arguments.shards),
                    "PYSEARCH_SHARD_URLS": ",".join(shard_urls),
                },
            )
        )

        coordinator_url = f"http://127.0.0.1:{arguments.coordinator_port}"
        if not wait_until_ready(f"{coordinator_url}/ready"):
            print("coordinator did not become ready", file=sys.stderr)
            return 1

        print(f"\ncluster ready: {coordinator_url}")
        print(f"data directory: {data_dir}")
        print("press Ctrl-C to stop\n")

        signal.pause()
    except KeyboardInterrupt:
        print("\nstopping cluster")
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if temporary is not None:
            temporary.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
