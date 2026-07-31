from __future__ import annotations

import re
from typing import Any, Mapping


class SchemaError(ValueError):
    pass


def validate(value: Any, schema: Mapping[str, Any], location: str = "$") -> None:
    if "anyOf" in schema:
        for candidate in schema["anyOf"]:
            try:
                validate(value, candidate, location)
                break
            except SchemaError:
                pass
        else:
            raise SchemaError(f"{location}: nenhuma alternativa valida")

    expected = schema.get("type")
    checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected and (expected not in checks or not checks[expected](value)):
        raise SchemaError(f"{location}: esperado {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{location}: valor fora do enum")

    if expected == "object":
        required = set(schema.get("required", ()))
        missing = sorted(required.difference(value))
        if missing:
            raise SchemaError(f"{location}: campos obrigatorios ausentes: {', '.join(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value).difference(properties))
            if extra:
                raise SchemaError(f"{location}: campos desconhecidos: {', '.join(extra)}")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{location}.{key}")
    elif expected == "array":
        if len(value) > schema.get("maxItems", len(value)):
            raise SchemaError(f"{location}: itens demais")
        if len(value) < schema.get("minItems", 0):
            raise SchemaError(f"{location}: itens insuficientes")
        for index, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{location}[{index}]")
    elif expected == "string":
        if len(value) < schema.get("minLength", 0):
            raise SchemaError(f"{location}: texto curto demais")
        if len(value) > schema.get("maxLength", len(value)):
            raise SchemaError(f"{location}: texto longo demais")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise SchemaError(f"{location}: formato invalido")
    elif expected in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaError(f"{location}: valor abaixo do minimo")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaError(f"{location}: valor acima do maximo")
