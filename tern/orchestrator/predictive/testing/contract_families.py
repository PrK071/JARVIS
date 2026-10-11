"""Familias controladas: oracle nao participa do runtime."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ContractFamily:
    name: str
    source: str
    problem: str
    expected: str
    investigation_symbol: str | None
    violation: bool = False


def contract_families():
    return (
        ContractFamily('literal_mismatch', 'def read() -> int:\n    return None\n',
            'read() wrong output\nExpected: an integer\nObserved: None',
            'EXPECTED_CONFIRMED_ROOT', 'read', True),
        ContractFamily('valid_literal', 'def read() -> int:\n    return 1\n',
            'read() reported wrong output\nExpected: an integer\nObserved: incorrect result',
            'EXPECTED_HYPOTHESIS', 'read'),
        ContractFamily('downstream', 'def read() -> int:\n    return 1\ndef consume():\n    return read() + 2\n',
            'consume() wrong output\nExpected: 1\nObserved: 3',
            'EXPECTED_HYPOTHESIS', 'consume'),
        ContractFamily('state', 'class Api:\n    def update(self, item):\n        self.current = item\n    def read(self):\n        return self.current\n',
            'Api.update() wrong state\nExpected: stored item\nObserved: different state',
            'EXPECTED_HYPOTHESIS', 'Api.update'),
        ContractFamily('state_guard', 'class Api:\n    def update(self, item):\n        if item:\n            self.current = item\n',
            'Api.update() wrong state\nExpected: assigned item\nObserved: unchanged state',
            'EXPECTED_HYPOTHESIS', 'Api.update'),
        ContractFamily('yield', 'def read():\n    yield 1\n',
            'read() iteration wrong\nExpected: two items\nObserved: one item',
            'EXPECTED_HYPOTHESIS', 'read'),
        ContractFamily('forward', 'def read():\n    return helper()\ndef helper():\n    return None\n',
            'read() wrong output\nExpected: data\nObserved: None',
            'EXPECTED_HYPOTHESIS', 'read'),
        ContractFamily('conditional', 'def read(flag) -> int:\n    if flag:\n        return None\n    return 1\n',
            'read() wrong output\nExpected: integer\nObserved: incorrect result',
            'EXPECTED_HYPOTHESIS', 'read'),
        ContractFamily('reported_contract', 'def read(item):\n    return item\n',
            'read() wrong output\nExpected: integer\nObserved: None',
            'EXPECTED_HYPOTHESIS', 'read'),
        ContractFamily('unrelated', 'def other():\n    return None\n',
            'Unknown operation produces incorrect results', 'EXPECTED_ABSTENTION', None),
        ContractFamily('ambiguous', 'def read(receiver):\n    return receiver.helper()\n',
            'read() wrong output\nExpected: data\nObserved: missing data',
            'EXPECTED_HYPOTHESIS', 'read'),
        ContractFamily('no_effect', 'def read():\n    pass\n', 'read() wrong output',
            'EXPECTED_ABSTENTION', None),
    )
