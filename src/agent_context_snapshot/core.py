"""Point-in-time snapshots of agent conversation state.

:class:`ContextSnapshot` captures messages, metadata, and a label at a
specific moment.  :class:`SnapshotStore` accumulates snapshots and supports
retrieval, diffing, and rollback.

Example::

    from agent_context_snapshot import SnapshotStore

    store = SnapshotStore()

    messages = [{"role": "system", "content": "You are helpful."}]
    store.take(messages, label="start")

    messages.append({"role": "user", "content": "Hello"})
    messages.append({"role": "assistant", "content": "Hi!"})
    store.take(messages, label="after_turn_1")

    # Roll back to an earlier state
    previous = store.get(0).messages
    # or
    previous = store.get_by_label("start").messages

    # Diff two snapshots
    diff = store.diff(0, 1)
    print(diff.messages_added)   # [user msg, assistant msg]
    print(diff.messages_removed) # []
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]


@dataclass
class ContextSnapshot:
    """An immutable snapshot of agent context at a point in time.

    Attributes:
        index:     0-based position within the :class:`SnapshotStore`.
        label:     Optional human-readable label.
        messages:  Deep copy of the conversation messages at snapshot time.
        metadata:  Deep copy of user-supplied metadata.
        timestamp: Unix timestamp (seconds) when the snapshot was taken.
        turn:      Optional turn number for the conversation.
    """

    index: int
    messages: list[Message]
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    turn: int | None = None

    def message_count(self) -> int:
        """Number of messages in this snapshot."""
        return len(self.messages)

    def roles(self) -> list[str]:
        """Ordered list of roles for all messages."""
        return [m.get("role", "") for m in self.messages]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "index": self.index,
            "label": self.label,
            "message_count": self.message_count(),
            "messages": copy.deepcopy(self.messages),
            "metadata": copy.deepcopy(self.metadata),
            "timestamp": self.timestamp,
            "turn": self.turn,
        }

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        label_part = f", label={self.label!r}" if self.label else ""
        return (
            f"ContextSnapshot(index={self.index}, "
            f"messages={self.message_count()}{label_part})"
        )


@dataclass
class SnapshotDiff:
    """Difference between two :class:`ContextSnapshot` objects.

    Attributes:
        from_index:        Index of the *earlier* snapshot.
        to_index:          Index of the *later* snapshot.
        messages_added:    Messages present in *to* but not in *from*.
        messages_removed:  Messages present in *from* but not in *to*.
        metadata_added:    Keys added in *to*.
        metadata_removed:  Keys removed from *from*.
        metadata_changed:  Keys whose values changed.
    """

    from_index: int
    to_index: int
    messages_added: list[Message] = field(default_factory=list)
    messages_removed: list[Message] = field(default_factory=list)
    metadata_added: dict[str, Any] = field(default_factory=dict)
    metadata_removed: dict[str, Any] = field(default_factory=dict)
    metadata_changed: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """``True`` if no differences were detected."""
        return (
            not self.messages_added
            and not self.messages_removed
            and not self.metadata_added
            and not self.metadata_removed
            and not self.metadata_changed
        )

    def __repr__(self) -> str:
        return (
            f"SnapshotDiff(from={self.from_index}, to={self.to_index}, "
            f"+{len(self.messages_added)} msgs, "
            f"-{len(self.messages_removed)} msgs)"
        )


def _diff_messages(
    from_msgs: list[Message], to_msgs: list[Message]
) -> tuple[list[Message], list[Message]]:
    """Simple sequential diff: count how many messages were added/removed."""
    n_from = len(from_msgs)
    n_to = len(to_msgs)
    # Find longest common prefix
    prefix = 0
    for i in range(min(n_from, n_to)):
        if from_msgs[i] == to_msgs[i]:
            prefix += 1
        else:
            break
    added = list(to_msgs[prefix:])
    removed = list(from_msgs[prefix:])
    return added, removed


def _diff_metadata(
    from_meta: dict[str, Any], to_meta: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, tuple[Any, Any]]]:
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    changed: dict[str, tuple[Any, Any]] = {}
    all_keys = set(from_meta) | set(to_meta)
    for k in all_keys:
        if k in to_meta and k not in from_meta:
            added[k] = to_meta[k]
        elif k in from_meta and k not in to_meta:
            removed[k] = from_meta[k]
        elif from_meta[k] != to_meta[k]:
            changed[k] = (from_meta[k], to_meta[k])
    return added, removed, changed


class SnapshotStore:
    """Accumulate and query context snapshots.

    Example::

        store = SnapshotStore()
        store.take(messages, label="initial")
        # ... after more turns ...
        store.take(messages, label="after_turn_3")

        snap = store.latest()
        snap = store.get(0)
        snap = store.get_by_label("initial")
        diff = store.diff(0, 1)
    """

    def __init__(self) -> None:
        self._snapshots: list[ContextSnapshot] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def take(
        self,
        messages: list[Message],
        *,
        label: str = "",
        metadata: dict[str, Any] | None = None,
        turn: int | None = None,
        timestamp: float | None = None,
    ) -> ContextSnapshot:
        """Take a snapshot of the current *messages*.

        Args:
            messages:  Current conversation messages (deep-copied).
            label:     Human-readable label for this snapshot.
            metadata:  Optional key/value data to capture alongside messages.
            turn:      Optional conversation turn number.
            timestamp: Optional Unix timestamp (default: ``time.time()``).

        Returns:
            The new :class:`ContextSnapshot`.
        """
        snap = ContextSnapshot(
            index=len(self._snapshots),
            messages=copy.deepcopy(messages),
            label=label,
            metadata=copy.deepcopy(metadata) if metadata is not None else {},
            timestamp=timestamp if timestamp is not None else time.time(),
            turn=turn,
        )
        self._snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, index: int) -> ContextSnapshot:
        """Return snapshot at *index*.

        Raises:
            IndexError: if out of range.
        """
        if index < 0 or index >= len(self._snapshots):
            raise IndexError(f"No snapshot at index {index}")
        return self._snapshots[index]

    def get_by_label(self, label: str) -> ContextSnapshot:
        """Return the first snapshot with the given *label*.

        Raises:
            KeyError: if no snapshot has that label.
        """
        for s in self._snapshots:
            if s.label == label:
                return s
        raise KeyError(f"No snapshot with label {label!r}")

    def latest(self) -> ContextSnapshot | None:
        """Return the most recent snapshot, or ``None`` if empty."""
        return self._snapshots[-1] if self._snapshots else None

    def first(self) -> ContextSnapshot | None:
        """Return the oldest snapshot, or ``None`` if empty."""
        return self._snapshots[0] if self._snapshots else None

    def all(self) -> list[ContextSnapshot]:
        """Return all snapshots in order."""
        return list(self._snapshots)

    def count(self) -> int:
        """Number of snapshots stored."""
        return len(self._snapshots)

    def labels(self) -> list[str]:
        """Return labels for all snapshots (empty string for unlabelled)."""
        return [s.label for s in self._snapshots]

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(self, from_index: int, to_index: int) -> SnapshotDiff:
        """Compute the diff between two snapshots.

        Args:
            from_index: Index of the earlier snapshot.
            to_index:   Index of the later snapshot.

        Returns:
            :class:`SnapshotDiff` describing what changed.
        """
        snap_from = self.get(from_index)
        snap_to = self.get(to_index)
        added, removed = _diff_messages(snap_from.messages, snap_to.messages)
        meta_added, meta_removed, meta_changed = _diff_metadata(
            snap_from.metadata, snap_to.metadata
        )
        return SnapshotDiff(
            from_index=from_index,
            to_index=to_index,
            messages_added=added,
            messages_removed=removed,
            metadata_added=meta_added,
            metadata_removed=meta_removed,
            metadata_changed=meta_changed,
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all snapshots."""
        self._snapshots.clear()

    def prune_before(self, index: int) -> int:
        """Remove all snapshots with index < *index*.

        Returns the number of snapshots removed.
        """
        to_keep = [s for s in self._snapshots if s.index >= index]
        removed = len(self._snapshots) - len(to_keep)
        # Re-index
        for i, s in enumerate(to_keep):
            object.__setattr__(s, "index", i)  # type: ignore[arg-type]
        self._snapshots = to_keep
        return removed

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._snapshots)

    def __repr__(self) -> str:
        return f"SnapshotStore(count={self.count()})"
