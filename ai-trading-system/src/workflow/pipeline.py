"""High-level workflow orchestrator used by the CLI entry point."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings


@dataclass
class WorkflowResult:
    """Simple container describing workflow execution summary."""

    executed: bool
    demo_mode: bool
    dry_run: bool
    today_only: bool


@dataclass
class DailyWorkflow:
    """Co-ordinates the end-to-end trading workflow (placeholder implementation)."""

    settings: Settings

    def run(self, *, demo: bool, dry_run: bool, today: bool) -> WorkflowResult:
        """Execute the configured workflow and return a summary of actions taken."""

        # In the scaffold we simply acknowledge the intended flags. Concrete
        # implementations will wire API ingestion, feature engineering, AI
        # scoring, risk checks, and execution as described in the system plan.
        return WorkflowResult(
            executed=True,
            demo_mode=demo or self.settings.demo_mode,
            dry_run=dry_run,
            today_only=today,
        )


__all__ = ["DailyWorkflow", "WorkflowResult"]
