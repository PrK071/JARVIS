from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import Service


class Contract:
    service: "Service"
