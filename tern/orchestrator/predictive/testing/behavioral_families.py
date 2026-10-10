"""Controles estruturais independentes de nomes de fixtures externos."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BehavioralFamily:
    name: str
    files: dict[str, str]
    problem: str
    api: str
    implementation: str | None
    effect: str | None


def behavioral_families():
    return (
        BehavioralFamily("direct", {"api.py": "def read():\n    return helper()\ndef helper():\n    return 1\n"},
            "api.py read() produces unexpected output", "read", "helper", "RETURNS_VALUE"),
        BehavioralFamily("imported", {"api.py": "from impl import helper as fetch\ndef read():\n    return fetch()\n",
            "impl.py": "def helper():\n    return 1\n"}, "api.py read() unexpected output", "read", "helper", "RETURNS_VALUE"),
        BehavioralFamily("relative", {"pkg/api.py": "from .impl import helper\ndef read():\n    return helper()\n",
            "pkg/impl.py": "def helper():\n    return 1\n"}, "pkg/api.py read() unexpected output", "read", "helper", "RETURNS_VALUE"),
        BehavioralFamily("bound", {"api.py": "class Api:\n    def read(self):\n        return self.helper()\n    def helper(self):\n        return 1\n"},
            "api.py Api.read() unexpected output", "Api.read", "Api.helper", "RETURNS_VALUE"),
        BehavioralFamily("dynamic_negative", {"api.py": "def read(receiver):\n    return receiver.helper()\ndef helper():\n    return 1\n"},
            "api.py read() unexpected output", "read", None, "RETURNS_VALUE"),
        BehavioralFamily("state", {"api.py": "class Api:\n    def update(self, item):\n        self.current = item\n"},
            "api.py Api.update() stores unexpected state", "Api.update", "Api.update", "WRITES_ATTRIBUTE"),
        BehavioralFamily("generator", {"api.py": "def stream():\n    yield 1\n"},
            "api.py stream() produces unexpected iteration", "stream", "stream", "YIELDS_VALUE"),
        BehavioralFamily("handler", {"api.py": "def read():\n    try:\n        raise ValueError()\n    except ValueError:\n        return None\n"},
            "api.py read() unexpected output", "read", "read", "HANDLES_EXCEPTION"),
        BehavioralFamily("subscript", {"api.py": "def update(items):\n    items[0] = 1\n"},
            "api.py update() stores unexpected state", "update", "update", "MUTATES_CONTAINER"),
        BehavioralFamily("distractor_negative", {"api.py": "def read():\n    # helper return None validation\n    return 1\ndef helper():\n    return None\n"},
            "api.py read() unexpected output", "read", "read", "RETURNS_VALUE"),
        BehavioralFamily("void_negative", {"api.py": "def read():\n    pass\n"},
            "api.py read() unexpected output", "read", "read", None),
        BehavioralFamily("decorated_negative", {"api.py": "def read():\n    return helper()\n@decorate\ndef helper():\n    return 1\n"},
            "api.py read() unexpected output", "read", None, "RETURNS_VALUE"),
    )
