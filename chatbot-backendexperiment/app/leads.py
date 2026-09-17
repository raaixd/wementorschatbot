"""
Lead collection and storage have been decommissioned.

The WeMentors chatbot routes demo inquiries and enrollment questions directly
to official contact channels (email and phone) without requesting, storing,
or transmitting personal user data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class DemoLead:
    """Backwards compatibility stub. No lead data is collected or persisted."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.status = "disabled"

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"status": "disabled"}

    @classmethod
    def from_dict(cls, data: Dict[str, Optional[str]]) -> DemoLead:
        return cls()

    def has_any_field(self) -> bool:
        return False

    def get_missing_field_labels(self) -> List[str]:
        return []

    def is_complete(self) -> bool:
        return False


def extract_lead_fields(text: str, current_lead: Optional[DemoLead] = None) -> Tuple[DemoLead, List[str]]:
    return DemoLead(), []


def format_lead_summary(lead: DemoLead) -> str:
    return ""


def format_partial_prompt(lead: DemoLead, newly_found: List[str]) -> str:
    return ""
