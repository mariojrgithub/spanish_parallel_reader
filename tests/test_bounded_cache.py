"""
Tests: _BoundedCache LRU session cache helper.
Reproduced without importing app to avoid Streamlit side effects.
"""
from collections import OrderedDict


class _BoundedCache:
    def __init__(self, max_entries: int) -> None:
        self._max = max(1, max_entries)
        self._data: OrderedDict = OrderedDict()

    def get(self, key: object) -> object:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: object, value: object) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


# ── basic behaviour ───────────────────────────────────────────────────────────

def test_get_missing_returns_none():
    c = _BoundedCache(10)
    assert c.get("x") is None


def test_put_and_get_roundtrip():
    c = _BoundedCache(10)
    c.put("a", 42)
    assert c.get("a") == 42


def test_len_reflects_entries():
    c = _BoundedCache(10)
    c.put("a", 1)
    c.put("b", 2)
    assert len(c) == 2


def test_clear_empties_cache():
    c = _BoundedCache(10)
    c.put("a", 1)
    c.put("b", 2)
    c.clear()
    assert len(c) == 0
    assert c.get("a") is None


# ── capacity enforcement ──────────────────────────────────────────────────────

def test_cap_not_exceeded():
    c = _BoundedCache(3)
    for i in range(10):
        c.put(i, i)
    assert len(c) <= 3


def test_oldest_evicted_first():
    c = _BoundedCache(3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.put("d", 4)  # evicts "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("d") == 4


def test_get_promotes_to_mru():
    c = _BoundedCache(3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.get("a")       # promotes "a" — "b" is now oldest
    c.put("d", 4)    # evicts "b"
    assert c.get("b") is None
    assert c.get("a") == 1


def test_update_existing_key_no_growth():
    c = _BoundedCache(3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.put("a", 99)   # update, not insert
    assert len(c) == 3
    assert c.get("a") == 99


# ── edge cases ────────────────────────────────────────────────────────────────

def test_max_entries_one():
    c = _BoundedCache(1)
    c.put("a", 1)
    c.put("b", 2)
    assert len(c) == 1
    assert c.get("b") == 2
    assert c.get("a") is None


def test_tuple_keys_work():
    """Cache keys in app.py are tuples."""
    c = _BoundedCache(5)
    key = ("chunk text", "qwen2.5:7b", "B1", "Natural", "Spain", "Closest meaning",
           False, False, False, 0.1)
    c.put(key, "result")
    assert c.get(key) == "result"
