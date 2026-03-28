"""

Run cassandra docker


docker run --name cassandra-spider -d --memory="16g" --memory-swap="8g"  -v /mnt/disk2/tampm/spider20-cassandra-data/:/var/lib/cassandra  -p 7000:7000 -p 9042:9042 -e HEAP_NEWSIZE=128M -e MAX_HEAP_SIZE=2048M cassandra:5.0-jammy
docker exec -it cassandra-spider bash -c 'echo $MAX_HEAP_SIZE'

docker stop cassandra-spider
docker rm cassandra-spider
rm -rf spider20-cassandra-data/
docker start cassandra-spider

cassandra still take more than 32GB RAM ??? WHY


"""

from cassandra.cluster import Cluster, Session
from cassandra.query import PreparedStatement, ConsistencyLevel
from datetime import datetime
from typing import Optional, Iterable

class CassandraKV:
    def __init__(
        self,
        contact_points: list[str],
        port: int=7000,
        keyspace: str = "kv",
        table: str = "store",
        consistency: ConsistencyLevel = ConsistencyLevel.LOCAL_QUORUM,
    ):
        self._cluster = Cluster(contact_points, port=port,
                                protocol_version=3
                                )
        """

        https://stackoverflow.com/questions/78770533/how-can-i-avoid-connectionshutdowncrc-mismatch-on-header
        """
        self._session: Session = self._cluster.connect()
        self._ensure_keyspace(keyspace)
        self._session.set_keyspace(keyspace)
        self._table = table
        self._consistency = consistency

        self._session.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
          k text PRIMARY KEY,
          v blob,
        )
        """)
        # Prepared statements
        self._put_ps: PreparedStatement = self._session.prepare(
            # f"INSERT INTO {table} (k, v, updated_at) VALUES (?, ?, ?) USING TTL ?"
            f"INSERT INTO {table} (k, v) VALUES (?, ?) IF NOT EXISTS;"
        )
        # BUG14122025 : timestamp : updated_at: use timestamp causes issues
        self._put_ps.consistency_level = consistency

        self._get_ps: PreparedStatement = self._session.prepare(
            f"SELECT v FROM {table} WHERE k = ?"
        )
        self._get_ps.consistency_level = consistency

        self._del_ps: PreparedStatement = self._session.prepare(
            f"DELETE FROM {table} WHERE k = ?"
        )
        self._del_ps.consistency_level = consistency

        self._exists_ps: PreparedStatement = self._session.prepare(
            f"SELECT k FROM {table} WHERE k = ?"
        )
        self._exists_ps.consistency_level = consistency

    def _ensure_keyspace(self, keyspace: str) -> None:
        # repl_str = "{ " + ", ".join(parts) + " }"
        self._session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication  = {{ 'class': 'SimpleStrategy', 'replication_factor': '1' }} ;"
        )

    # ----- bytes API -----
    def put(self, key: str, value: bytes, ttl_seconds: int = 0) -> None:
        ttl = int(ttl_seconds) if ttl_seconds and ttl_seconds > 0 else None
        # self._session.execute(self._put_ps, (key, bytearray(value), datetime.utcnow(), ttl))
        self._session.execute(self._put_ps, (key, bytearray(value)))

    def get(self, key: str) -> Optional[bytes]:
        row = self._session.execute(self._get_ps, (key,)).one()
        return bytes(row.v) if row and row.v is not None else None

    def count(self, table: str) -> int:
        """
        Count number of rows in the table.
        :param table:
        :return:
        """
        row = self._session.execute(f"SELECT COUNT(*) FROM {table}").one()
        # import pdb; pdb.set_trace()
        return row.count if row else 0

    def delete(self, key: str) -> None:
        self._session.execute(self._del_ps, (key,))

    def exists(self, key: str) -> bool:
        return self._session.execute(self._exists_ps, (key,)).one() is not None

    # ----- convenience: text/JSON helpers -----
    def put_text(self, key: str, text: str, ttl_seconds: int = 0) -> None:
        self.put(key, text.encode("utf-8"), ttl_seconds)

    def get_text(self, key: str) -> Optional[str]:
        b = self.get(key)
        return b.decode("utf-8") if b is not None else None

    def put_json(self, key: str, obj, ttl_seconds: int = 0) -> None:
        import json
        self.put_text(key, json.dumps(obj, separators=(",", ":")), ttl_seconds)

    def get_json(self, key: str):
        import json
        s = self.get_text(key)
        return json.loads(s) if s is not None else None

    # ----- batch-ish helpers (concurrent single-row ops recommended) -----
    def put_many(self, items: Iterable[tuple[str, bytes]], ttl_seconds: int = 0) -> None:
        # Use execute_concurrent for parallelism without large logged batches
        from cassandra.concurrent import execute_concurrent_with_args
        ttl = int(ttl_seconds) if ttl_seconds and ttl_seconds > 0 else None
        # args = [(k, bytearray(v), datetime.utcnow(), ttl) for k, v in items]
        args = [(k, bytearray(v)) for k, v in items]
        execute_concurrent_with_args(self._session, self._put_ps, args, raise_on_first_error=True)

    def get_many(self, keys: Iterable[str]) -> dict[str, Optional[bytes]]:
        from cassandra.concurrent import execute_concurrent_with_args
        results: dict[str, Optional[bytes]] = {}
        fut = execute_concurrent_with_args(self._session, self._get_ps, [(k,) for k in keys])
        for (success, res), key in zip(fut, keys):
            if success:
                row = res.one()
                results[key] = bytes(row.v) if row and row.v is not None else None
            else:
                results[key] = None
        return results

    def close(self):
        self._session.shutdown()
        self._cluster.shutdown()
        del self._session
        del self._cluster
    @staticmethod
    def check_existence(contact_points: list[str], port: int, keyspace: str, table: str) -> bool:
        cluster = Cluster(contact_points, port=port)
        session: Session = cluster.connect()
        try:
            session.set_keyspace(keyspace)
            row = session.execute(f"SELECT COUNT(*) FROM {table}").one()
            return row is not None
        except Exception:
            return False
        finally:
            session.shutdown()
            cluster.shutdown()
            del session
            del cluster
if __name__ == "__main__":
    kv = CassandraKV(["127.0.0.1"], port=9042)

    kv.put_text("user:1", "Alice", ttl_seconds=0)
    print(kv.get_text("user:1"))  # "Alice"
    print(kv.exists("user:1"))  # True
    kv.put_json("cfg:app", {"v": 1})
    print(kv.get_json("cfg:app"))  # {"v": 1}
    kv.delete("user:1")

    # Bulk
    pairs = [(f"k:{i}", f"val{i}".encode()) for i in range(1000)]
    kv.put_many(pairs)
    vals = kv.get_many([f"k:{i}" for i in range(10)])

    kv.close()