"""Quatro inputs separados; nenhuma informacao do oracle em inputs."""
import re


def input_variants(original, additions):
    expected = re.search(r'(?:Expected (?:Result|behavior)|What you expected)[^\n]*\n+([^#<]+)', original, re.I)
    blocks = re.findall(r'```[^\n]*\n(.*?)```', original, re.S)
    structured = ('Observed report (unverified):\n'+original+'\nExpected: '+
        (expected.group(1).strip() if expected else 'UNKNOWN')+
        '\nReported input/examples (unverified):\n'+'\n'.join(blocks)+'\n')
    return {'A': {'text': original, 'available': True, 'source': 'ORIGINAL_REPORT'},
        'B': {'text': structured, 'available': True, 'source': 'SAME_REPORT_REORGANIZED', 'new_information': False},
        'C': {'text': structured+'\n'.join('Untrusted pre-fix test source (not executed):\n'+a['text'] for a in additions)
            if additions else None, 'available': bool(additions), 'source': 'BASE_COMMIT_TEST_CITED_IN_REPORT'},
        'D': {'text': None, 'available': False, 'source': 'NO_ADDITIONAL_PREEXISTING_RUNTIME_EVIDENCE',
            'reason': 'Logs in original report already belong to A/B; none newly available.'}}
