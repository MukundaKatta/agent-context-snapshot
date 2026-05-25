# agent-context-snapshot

Point-in-time snapshots of agent conversation state for debugging and checkpointing.

Capture messages and metadata at any moment. Roll back, diff two moments, or prune old history.

## Install

```bash
pip install agent-context-snapshot
```

## Usage

```python
from agent_context_snapshot import SnapshotStore

store = SnapshotStore()

messages = [{"role": "system", "content": "You are helpful."}]
store.take(messages, label="start")

messages.append({"role": "user", "content": "Hello"})
messages.append({"role": "assistant", "content": "Hi!"})
store.take(messages, label="after_turn_1", turn=1)

# Roll back to an earlier state
previous = store.get(0).messages
# or by label
previous = store.get_by_label("start").messages

# Diff two snapshots
diff = store.diff(0, 1)
print(diff.messages_added)   # [user msg, assistant msg]
print(diff.messages_removed) # []
print(diff.is_empty)         # False
```

## SnapshotStore API

```python
store = SnapshotStore()

# Record
snap = store.take(messages, label="...", metadata={}, turn=1, timestamp=None)

# Retrieve
store.get(index)            # by index, raises IndexError if missing
store.get_by_label("name")  # first match, raises KeyError if missing
store.latest()              # most recent, or None
store.first()               # oldest, or None
store.all()                 # list of all snapshots
store.count()               # number of snapshots
store.labels()              # list of labels (empty string for unlabelled)

# Diff
diff = store.diff(0, 1)     # SnapshotDiff between two snapshots

# Mutation
store.clear()               # remove all snapshots
store.prune_before(n)       # remove snapshots with index < n, returns count removed
```

## ContextSnapshot fields

| Field | Description |
|-------|-------------|
| `index` | 0-based position in the store |
| `label` | Optional human-readable label |
| `messages` | Deep copy of conversation messages at snapshot time |
| `metadata` | Deep copy of user-supplied metadata |
| `timestamp` | Unix timestamp when the snapshot was taken |
| `turn` | Optional turn number |

Methods: `message_count()`, `roles()`, `to_dict()`, `to_json()`

## SnapshotDiff fields

| Field | Description |
|-------|-------------|
| `messages_added` | Messages in *to* but not in *from* |
| `messages_removed` | Messages in *from* but not in *to* |
| `metadata_added` | Keys added in *to* |
| `metadata_removed` | Keys removed from *from* |
| `metadata_changed` | Keys whose values changed: `{key: (old, new)}` |
| `is_empty` | `True` if no differences were detected |

## License

MIT
