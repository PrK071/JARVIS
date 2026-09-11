from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import RuntimeService


class Contract:
    service: "RuntimeService"
