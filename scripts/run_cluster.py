"""Run a real multi-process PySearch cluster locally, without containers.

    uv run python scripts/run_cluster.py

Starts one uvicorn process per physical node plus a coordinator, each with its
own database, waits for the cluster to report ready, and prints the
coordinator's address. Ctrl-C stops everything.

With ``--replication-factor 2`` every logical shard gets a primary and a
replica, which is the topology that survives losing a node: kill a primary and
search keeps working from its replica, while writes to that shard start failing.

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
    parser.add_argument("--shards", type=int, default=3, help="number of logical shards")
    parser.add_argument(
        "--replication-factor",
        type=int,
        default=1,
        choices=(1, 2),
        help="physical copies per logical shard",
    )
    parser.add_argument("--base-port", type=int, default=9000, help="first primary port")
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
    replica_urls = []
    replicated = arguments.replication_factor > 1

    try:
        for shard_id in range(arguments.shards):
            primary_port = arguments.base_port + shard_id
            replica_port = arguments.base_port + 100 + shard_id
            primary_url = f"http://127.0.0.1:{primary_port}"
            replica_url = f"http://127.0.0.1:{replica_port}"
            shard_urls.append(primary_url)

            processes.append(
                start_node(
                    primary_port,
                    {
                        "PYSEARCH_APP_NAME": f"pysearch-shard-{shard_id}-primary",
                        "PYSEARCH_NODE_ID": f"shard-{shard_id}-primary",
                        "PYSEARCH_NODE_ROLE": "shard",
                        "PYSEARCH_SHARD_ID": str(shard_id),
                        "PYSEARCH_SHARD_COUNT": str(arguments.shards),
                        "PYSEARCH_REPLICA_ROLE": "primary",
                        "PYSEARCH_REPLICA_URLS": replica_url if replicated else "",
                        # One database per physical node, never shared.
                        "PYSEARCH_STORAGE_PATH": str(data_dir / f"shard-{shard_id}-primary.db"),
                    },
                )
            )
            if replicated:
                replica_urls.append(replica_url)

        # Primaries first: a replica verifies itself against its primary while
        # starting up, and will refuse to serve if it cannot reach one.
        for shard_id, url in enumerate(shard_urls):
            if not wait_until_ready(f"{url}/ready"):
                print(f"shard {shard_id} primary did not become ready", file=sys.stderr)
                return 1
            print(f"shard {shard_id} primary ready at {url}")

        for shard_id, _ in enumerate(replica_urls):
            processes.append(
                start_node(
                    arguments.base_port + 100 + shard_id,
                    {
                        "PYSEARCH_APP_NAME": f"pysearch-shard-{shard_id}-replica",
                        "PYSEARCH_NODE_ID": f"shard-{shard_id}-replica",
                        "PYSEARCH_NODE_ROLE": "shard",
                        "PYSEARCH_SHARD_ID": str(shard_id),
                        "PYSEARCH_SHARD_COUNT": str(arguments.shards),
                        "PYSEARCH_REPLICA_ROLE": "replica",
                        "PYSEARCH_PRIMARY_URL": shard_urls[shard_id],
                        "PYSEARCH_STORAGE_PATH": str(data_dir / f"shard-{shard_id}-replica.db"),
                    },
                )
            )

        for shard_id, url in enumerate(replica_urls):
            if not wait_until_ready(f"{url}/ready"):
                print(f"shard {shard_id} replica did not become ready", file=sys.stderr)
                return 1
            print(f"shard {shard_id} replica ready at {url}")

        processes.append(
            start_node(
                arguments.coordinator_port,
                {
                    "PYSEARCH_APP_NAME": "pysearch-coordinator",
                    "PYSEARCH_NODE_ROLE": "coordinator",
                    "PYSEARCH_SHARD_COUNT": str(arguments.shards),
                    "PYSEARCH_SHARD_URLS": ",".join(shard_urls),
                    "PYSEARCH_REPLICA_URLS": ";".join(replica_urls),
                },
            )
        )

        coordinator_url = f"http://127.0.0.1:{arguments.coordinator_port}"
        if not wait_until_ready(f"{coordinator_url}/ready"):
            print("coordinator did not become ready", file=sys.stderr)
            return 1

        print(f"\ncluster ready: {coordinator_url}")
        print(f"logical shards: {arguments.shards}, copies each: {arguments.replication_factor}")
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
