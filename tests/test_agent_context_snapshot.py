"""Tests for agent_context_snapshot."""

from __future__ import annotations

import json
import time

import pytest

from agent_context_snapshot import ContextSnapshot, SnapshotDiff, SnapshotStore

# ---------------------------------------------------------------------------
# ContextSnapshot
# ---------------------------------------------------------------------------


def test_snapshot_basic_fields():
    msgs = [{"role": "user", "content": "hi"}]
    snap = ContextSnapshot(index=0, messages=msgs)
    assert snap.index == 0
    assert snap.messages == msgs
    assert snap.label == ""
    assert snap.metadata == {}
    assert snap.turn is None
    assert isinstance(snap.timestamp, float)


def test_snapshot_with_label_and_turn():
    snap = ContextSnapshot(index=2, messages=[], label="checkpoint", turn=5)
    assert snap.label == "checkpoint"
    assert snap.turn == 5


def test_snapshot_message_count():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    snap = ContextSnapshot(index=0, messages=msgs)
    assert snap.message_count() == 2


def test_snapshot_message_count_empty():
    snap = ContextSnapshot(index=0, messages=[])
    assert snap.message_count() == 0


def test_snapshot_roles():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    snap = ContextSnapshot(index=0, messages=msgs)
    assert snap.roles() == ["system", "user", "assistant"]


def test_snapshot_roles_missing_role_key():
    msgs = [{"content": "no role here"}, {"role": "user", "content": "hi"}]
    snap = ContextSnapshot(index=0, messages=msgs)
    assert snap.roles() == ["", "user"]


def test_snapshot_to_dict():
    msgs = [{"role": "user", "content": "hello"}]
    snap = ContextSnapshot(
        index=1,
        messages=msgs,
        label="test",
        metadata={"key": "val"},
        timestamp=1000.0,
        turn=3,
    )
    d = snap.to_dict()
    assert d["index"] == 1
    assert d["label"] == "test"
    assert d["message_count"] == 1
    assert d["messages"] == msgs
    assert d["metadata"] == {"key": "val"}
    assert d["timestamp"] == 1000.0
    assert d["turn"] == 3


def test_snapshot_to_json():
    snap = ContextSnapshot(index=0, messages=[], timestamp=1234.0)
    result = json.loads(snap.to_json())
    assert result["index"] == 0
    assert result["messages"] == []
    assert result["timestamp"] == 1234.0


def test_snapshot_repr_no_label():
    snap = ContextSnapshot(index=0, messages=[{"role": "user", "content": "x"}])
    r = repr(snap)
    assert "ContextSnapshot(index=0" in r
    assert "messages=1" in r
    assert "label" not in r


def test_snapshot_repr_with_label():
    snap = ContextSnapshot(index=2, messages=[], label="start")
    r = repr(snap)
    assert "label='start'" in r


def test_snapshot_custom_timestamp():
    snap = ContextSnapshot(index=0, messages=[], timestamp=9999.5)
    assert snap.timestamp == 9999.5


# ---------------------------------------------------------------------------
# SnapshotDiff
# ---------------------------------------------------------------------------


def test_diff_is_empty_true():
    diff = SnapshotDiff(from_index=0, to_index=1)
    assert diff.is_empty is True


def test_diff_is_empty_false_messages_added():
    diff = SnapshotDiff(
        from_index=0, to_index=1, messages_added=[{"role": "user", "content": "hi"}]
    )
    assert diff.is_empty is False


def test_diff_is_empty_false_messages_removed():
    diff = SnapshotDiff(
        from_index=0,
        to_index=1,
        messages_removed=[{"role": "user", "content": "hi"}],
    )
    assert diff.is_empty is False


def test_diff_is_empty_false_metadata_added():
    diff = SnapshotDiff(from_index=0, to_index=1, metadata_added={"k": "v"})
    assert diff.is_empty is False


def test_diff_is_empty_false_metadata_removed():
    diff = SnapshotDiff(from_index=0, to_index=1, metadata_removed={"k": "v"})
    assert diff.is_empty is False


def test_diff_is_empty_false_metadata_changed():
    diff = SnapshotDiff(
        from_index=0, to_index=1, metadata_changed={"k": ("old", "new")}
    )
    assert diff.is_empty is False


def test_diff_repr():
    diff = SnapshotDiff(
        from_index=0,
        to_index=1,
        messages_added=[{"role": "user", "content": "x"}],
    )
    r = repr(diff)
    assert "from=0" in r
    assert "to=1" in r
    assert "+1 msgs" in r
    assert "-0 msgs" in r


# ---------------------------------------------------------------------------
# SnapshotStore — basic operations
# ---------------------------------------------------------------------------


def test_store_empty_at_start():
    store = SnapshotStore()
    assert store.count() == 0
    assert len(store) == 0
    assert store.latest() is None
    assert store.first() is None
    assert store.all() == []
    assert store.labels() == []


def test_store_take_returns_snapshot():
    store = SnapshotStore()
    msgs = [{"role": "user", "content": "hello"}]
    snap = store.take(msgs, label="first")
    assert isinstance(snap, ContextSnapshot)
    assert snap.index == 0
    assert snap.label == "first"
    assert snap.messages == msgs


def test_store_take_increments_index():
    store = SnapshotStore()
    s0 = store.take([], label="a")
    s1 = store.take([], label="b")
    s2 = store.take([], label="c")
    assert s0.index == 0
    assert s1.index == 1
    assert s2.index == 2


def test_store_count():
    store = SnapshotStore()
    store.take([])
    store.take([])
    store.take([])
    assert store.count() == 3
    assert len(store) == 3


def test_store_latest():
    store = SnapshotStore()
    store.take([], label="first")
    store.take([], label="second")
    assert store.latest().label == "second"


def test_store_first():
    store = SnapshotStore()
    store.take([], label="first")
    store.take([], label="second")
    assert store.first().label == "first"


def test_store_all_returns_copy():
    store = SnapshotStore()
    store.take([])
    store.take([])
    result = store.all()
    assert len(result) == 2
    result.append("extra")
    assert store.count() == 2  # original unchanged


def test_store_labels():
    store = SnapshotStore()
    store.take([], label="alpha")
    store.take([])
    store.take([], label="gamma")
    assert store.labels() == ["alpha", "", "gamma"]


def test_store_get_by_index():
    store = SnapshotStore()
    store.take([], label="zero")
    store.take([], label="one")
    assert store.get(0).label == "zero"
    assert store.get(1).label == "one"


def test_store_get_index_error():
    store = SnapshotStore()
    with pytest.raises(IndexError):
        store.get(0)


def test_store_get_negative_index_error():
    store = SnapshotStore()
    store.take([])
    with pytest.raises(IndexError):
        store.get(-1)


def test_store_get_out_of_range():
    store = SnapshotStore()
    store.take([])
    with pytest.raises(IndexError):
        store.get(1)


def test_store_get_by_label():
    store = SnapshotStore()
    store.take([], label="start")
    store.take([], label="middle")
    store.take([], label="end")
    assert store.get_by_label("middle").index == 1


def test_store_get_by_label_first_match():
    store = SnapshotStore()
    store.take([], label="dup")
    store.take([], label="dup")
    assert store.get_by_label("dup").index == 0


def test_store_get_by_label_key_error():
    store = SnapshotStore()
    store.take([], label="exists")
    with pytest.raises(KeyError):
        store.get_by_label("missing")


def test_store_take_with_metadata():
    store = SnapshotStore()
    snap = store.take([], metadata={"model": "claude-3", "temp": 0.7})
    assert snap.metadata == {"model": "claude-3", "temp": 0.7}


def test_store_take_with_turn():
    store = SnapshotStore()
    snap = store.take([], turn=3)
    assert snap.turn == 3


def test_store_take_with_timestamp():
    store = SnapshotStore()
    snap = store.take([], timestamp=42.0)
    assert snap.timestamp == 42.0


# ---------------------------------------------------------------------------
# Deep copy verification
# ---------------------------------------------------------------------------


def test_take_deep_copies_messages():
    store = SnapshotStore()
    msgs = [{"role": "user", "content": "original"}]
    snap = store.take(msgs)
    msgs[0]["content"] = "modified"
    assert snap.messages[0]["content"] == "original"


def test_take_deep_copies_metadata():
    store = SnapshotStore()
    meta = {"key": "original"}
    snap = store.take([], metadata=meta)
    meta["key"] = "modified"
    assert snap.metadata["key"] == "original"


def test_take_appending_to_original_list_does_not_affect_snapshot():
    store = SnapshotStore()
    msgs = [{"role": "user", "content": "hi"}]
    snap = store.take(msgs)
    msgs.append({"role": "assistant", "content": "hello"})
    assert snap.message_count() == 1


def test_take_deep_copies_nested_message_content():
    store = SnapshotStore()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "original"}]}]
    snap = store.take(msgs)
    msgs[0]["content"][0]["text"] = "modified"
    assert snap.messages[0]["content"][0]["text"] == "original"


def test_take_deep_copies_nested_metadata():
    store = SnapshotStore()
    meta = {"config": {"temp": 0.7}}
    snap = store.take([], metadata=meta)
    meta["config"]["temp"] = 9.9
    assert snap.metadata["config"]["temp"] == 0.7


def test_to_dict_deep_copies_nested_messages():
    snap = ContextSnapshot(
        index=0,
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    )
    d = snap.to_dict()
    d["messages"][0]["content"][0]["text"] = "mutated"
    assert snap.messages[0]["content"][0]["text"] == "hi"


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def test_diff_messages_added():
    store = SnapshotStore()
    store.take([{"role": "user", "content": "hi"}], label="before")
    store.take(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        label="after",
    )
    diff = store.diff(0, 1)
    assert len(diff.messages_added) == 1
    assert diff.messages_added[0]["role"] == "assistant"
    assert diff.messages_removed == []


def test_diff_messages_removed():
    store = SnapshotStore()
    store.take(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    store.take([{"role": "user", "content": "hi"}])
    diff = store.diff(0, 1)
    assert len(diff.messages_removed) == 1
    assert diff.messages_removed[0]["role"] == "assistant"
    assert diff.messages_added == []


def test_diff_no_changes():
    store = SnapshotStore()
    msgs = [{"role": "user", "content": "same"}]
    store.take(msgs)
    store.take(msgs)
    diff = store.diff(0, 1)
    assert diff.is_empty


def test_diff_metadata_added():
    store = SnapshotStore()
    store.take([], metadata={"a": 1})
    store.take([], metadata={"a": 1, "b": 2})
    diff = store.diff(0, 1)
    assert diff.metadata_added == {"b": 2}
    assert diff.metadata_removed == {}
    assert diff.metadata_changed == {}


def test_diff_metadata_removed():
    store = SnapshotStore()
    store.take([], metadata={"a": 1, "b": 2})
    store.take([], metadata={"a": 1})
    diff = store.diff(0, 1)
    assert diff.metadata_removed == {"b": 2}
    assert diff.metadata_added == {}


def test_diff_metadata_changed():
    store = SnapshotStore()
    store.take([], metadata={"a": "old"})
    store.take([], metadata={"a": "new"})
    diff = store.diff(0, 1)
    assert diff.metadata_changed == {"a": ("old", "new")}


def test_diff_combined():
    store = SnapshotStore()
    store.take(
        [{"role": "user", "content": "hi"}],
        metadata={"model": "fast"},
    )
    store.take(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        metadata={"model": "smart", "temp": 0.5},
    )
    diff = store.diff(0, 1)
    assert len(diff.messages_added) == 1
    assert diff.metadata_changed == {"model": ("fast", "smart")}
    assert diff.metadata_added == {"temp": 0.5}
    assert not diff.is_empty


def test_diff_from_to_indexes():
    store = SnapshotStore()
    store.take([])
    store.take([])
    diff = store.diff(0, 1)
    assert diff.from_index == 0
    assert diff.to_index == 1


def test_diff_invalid_index_raises():
    store = SnapshotStore()
    store.take([])
    with pytest.raises(IndexError):
        store.diff(0, 5)


# ---------------------------------------------------------------------------
# Clear and prune
# ---------------------------------------------------------------------------


def test_store_clear():
    store = SnapshotStore()
    store.take([])
    store.take([])
    store.clear()
    assert store.count() == 0
    assert store.latest() is None


def test_store_prune_before():
    store = SnapshotStore()
    store.take([], label="a")
    store.take([], label="b")
    store.take([], label="c")
    store.take([], label="d")
    removed = store.prune_before(2)
    assert removed == 2
    assert store.count() == 2
    assert store.get(0).label == "c"
    assert store.get(1).label == "d"


def test_store_prune_before_reindexes():
    store = SnapshotStore()
    store.take([], label="x")
    store.take([], label="y")
    store.take([], label="z")
    store.prune_before(1)
    snaps = store.all()
    for i, s in enumerate(snaps):
        assert s.index == i


def test_store_prune_before_zero_removes_none():
    store = SnapshotStore()
    store.take([])
    store.take([])
    removed = store.prune_before(0)
    assert removed == 0
    assert store.count() == 2


def test_store_prune_beyond_end():
    store = SnapshotStore()
    store.take([])
    store.take([])
    removed = store.prune_before(10)
    assert removed == 2
    assert store.count() == 0


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_store_repr():
    store = SnapshotStore()
    assert repr(store) == "SnapshotStore(count=0)"
    store.take([])
    store.take([])
    assert repr(store) == "SnapshotStore(count=2)"


# ---------------------------------------------------------------------------
# Timestamp auto-set
# ---------------------------------------------------------------------------


def test_snapshot_timestamp_auto_set():
    before = time.time()
    store = SnapshotStore()
    snap = store.take([])
    after = time.time()
    assert before <= snap.timestamp <= after
