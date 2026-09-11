# Predictive benchmark v4 audit

## Method

All 42 development cases were adjudicated from fixture source, test behavior,
traceback, and statically demonstrable causal flow. The v3 model answer was not
used to define or relax ground truth. It was consulted only after adjudication
to classify historical failures as benchmark or engine failures.

## Counts

| Status | Cases |
|---|---:|
| CANONICAL | 16 |
| MULTIPLE_VALID_ANSWERS | 16 |
| AMBIGUOUS | 1 |
| INCOMPLETE | 7 |
| INVALID | 2 |
| Total | 42 |

The frozen v3 result had 23 cases with legacy failure codes. Under independently
authored structural truth, 14 of those are legacy false failures and 12 of the
32 evaluable cases still contain a true engine failure. Counts overlap because
some legacy failures occurred on benchmark-invalid or incomplete cases.

## Case adjudications

The complete machine-readable adjudication, including evidence lines, acceptable
root causes, strategy-target pairs, and notes, is in
`v4/adjudications/development.jsonl`.

| Case | Status | Structural adjudication | Why the legacy expectation was inadequate |
|---|---|---|---|
| PC3D-001 | CANONICAL | `load_order -> None -> calculate`; producer/return preferred, boundary valid | Strategy and target were not paired. |
| PC3D-002 | CANONICAL | Same complete repository-to-calculator flow | Boundary repair was valid but omitted by the family label. |
| PC3D-003 | MULTIPLE_VALID_ANSWERS | `load_tax` return or service boundary; producer preferred | A string can be normalized or rejected at a documented boundary. |
| PC3D-004 | INVALID | No valid causal truth | The claimed bad configured default is integer `0`; unrelated `RETRY_LIMIT` was mislabeled as its source. |
| PC3D-005 | CANONICAL | `parse_count` return contract; consumer guard valid but secondary | Parser repair and boundary repair were not distinguished. |
| PC3D-006 | MULTIPLE_VALID_ANSWERS | Optional parser contract versus consumer handling | The fixture never establishes ownership of `None`; producer preference was unjustified. |
| PC3D-007 | CANONICAL | Nested parser return reaches `consume` | Only a lexical producer family was available. |
| PC3D-008 | CANONICAL | Same explicit nested return flow | Target validity was not represented. |
| PC3D-009 | CANONICAL | `RETRY_LIMIT` is the string configuration origin | Existing structural truth was retained and paired to targets. |
| PC3D-010 | CANONICAL | Production branch/return violates test expectation | Both return and control-flow labels describe the same valid edit. |
| PC3D-011 | CANONICAL | Wrong fallback return at `branch.py:4` | v3 accepted only CONTROL_FLOW and falsely rejected RETURN_CONTRACT. |
| PC3D-012 | CANONICAL | Abstention | Existing truth was adequate. |
| PD-001 | INCOMPLETE | Failure-site tax binding is known; upstream origin is absent | “fix upstream” had no upstream fixture evidence. |
| PD-002 | MULTIPLE_VALID_ANSWERS | Optional model attribute or consumer contract | The fixture does not define whether `tax=None` is valid domain state. |
| PD-003 | INCOMPLETE | Failure-site binding only | The alleged regression is absent; the included test passes integers. |
| PD-004 | INCOMPLETE | Unlike operand types, unknown producer | No caller identifies which operand contract was violated. |
| PD-005 | MULTIPLE_VALID_ANSWERS | Optional model invariant versus service boundary | “validate” and “require” are policies, not a unique causal label. |
| PD-006 | MULTIPLE_VALID_ANSWERS | Same unresolved optionality contract | Producer and consumer repairs are both technically defensible. |
| PD-007 | MULTIPLE_VALID_ANSWERS | Wrapper is manifestation; model/service contract is unresolved | Fallback and invariant preferences lacked a requirement. |
| PD-008 | MULTIPLE_VALID_ANSWERS | Model/service contract; test addition is not a repair | The legacy benchmark counted adding a test as fixing production. |
| PD-009 | INCOMPLETE | Missing key at boundary; producer absent | A default versus rejection policy cannot be inferred from the traceback. |
| PD-010 | AMBIGUOUS | `timeout(0)` rejection may be correct | No requirement says zero should be accepted. |
| PD-011 | INCOMPLETE | Invalid raw input is stated; its producer/policy is absent | `config.py` appears to enforce, rather than violate, its invariant. |
| PD-012 | CANONICAL | `load` violates the explicitly stated mapping return contract | Structural origin/target were missing. |
| PD-013 | MULTIPLE_VALID_ANSWERS | Either edge of the two-module import cycle may be refactored | A cycle has no unique edit side. |
| PD-014 | MULTIPLE_VALID_ANSWERS | Same import cycle | v3 lexical phrasing rejected a structurally correct import repair. |
| PD-015 | INVALID | Fixture is a circular ImportError, not ModuleNotFoundError | Problem and fixture exception classes disagree. |
| PD-016 | MULTIPLE_VALID_ANSWERS | Either import edge; production fix required | The test is a manifestation, and either module may own the refactor. |
| PD-017 | CANONICAL | Non-member fallback return/control flow | v3 lexical root signals rejected a structurally correct diagnosis. |
| PD-018 | CANONICAL | Duplicate branch returns violate test | Return and control-flow repair labels are equivalent here. |
| PD-019 | CANONICAL | `find_user` scalar return violates mapping assertion | Structural return contract replaces text similarity. |
| PD-020 | MULTIPLE_VALID_ANSWERS | Producer branch or caller contract; producer preferred | The problem itself says both are possible. |
| PD-021 | INCOMPLETE | Zero count at `divide`; caller absent from traceback | Upstream empty-input origin is not observed. |
| PD-022 | MULTIPLE_VALID_ANSWERS | `average([])` creates zero count; either boundary may reject | Family did not encode the strategy-target pairing. |
| PD-023 | MULTIPLE_VALID_ANSWERS | Same flow; `auth.py` is disconnected | Reference validity alone did not encode causal target validity. |
| PD-024 | MULTIPLE_VALID_ANSWERS | Local boundary repairs valid; config deletion forbidden | String policy checks did not express causal disconnection. |
| PD-025 | INCOMPLETE | Missing email at callee; producer/required policy absent | Require-versus-default behavior cannot be inferred. |
| PD-026 | MULTIPLE_VALID_ANSWERS | Caller or callee may own payload validation | Lexically similar legacy module is irrelevant; ownership is unspecified. |
| PD-027 | MULTIPLE_VALID_ANSWERS | Demonstrated caller/callee boundary, no ownership contract | Both correction sites are valid. |
| PD-028 | CANONICAL | Abstention | Existing truth was adequate. |
| PD-029 | CANONICAL | Abstention | Existing truth was adequate. |
| PD-030 | CANONICAL | Abstention | Existing truth was adequate. |

## Historical failure reclassification

Legacy false failures were identified in PC3D-004, PC3D-011, PD-001, PD-003,
PD-004, PD-009, PD-011, PD-014, PD-015, PD-017, PD-018, PD-020, PD-021, and
PD-025. They arose from invalid/incomplete fixtures, lexical root-cause checks,
or treating one valid repair label as uniquely correct.

True engine failures remain concentrated in:

- missing attribute-origin relations (`PD-002`, `PD-005` through `PD-008`);
- root selection at a manifestation rather than the structural return contract
  (`PD-012`, `PD-019`);
- repair target selection for caller/callee boundary cases (`PD-022` through
  `PD-024`, `PD-026`, `PD-027`).

These findings justify selector/causal hardening only after the required
engine-unchanged v4 run confirms them.
