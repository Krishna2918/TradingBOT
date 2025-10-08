"""Dashboard package exports."""

from .state_manager import (
    trading_state,
    STATE_STORE,
    DEMO_STATE_PATH,
    reset_in_memory_state,
    load_trading_state,
    save_trading_state,
)
from .ui_components import (
    create_navbar,
    create_enhanced_navbar,
    create_status_pill,
    create_regime_badge,
)

__all__ = [
    "trading_state",
    "STATE_STORE",
    "DEMO_STATE_PATH",
    "reset_in_memory_state",
    "load_trading_state",
    "save_trading_state",
    "create_navbar",
    "create_enhanced_navbar",
    "create_status_pill",
    "create_regime_badge",
]
from .sections import (
    create_hybrid_control_status,
    create_risk_panel,
    create_learning_panel,
    create_alerts_feed,
    create_ai_activity_monitor,
)

__all__.extend([
    'create_hybrid_control_status',
    'create_risk_panel',
    'create_learning_panel',
    'create_alerts_feed',
    'create_ai_activity_monitor',
    'create_performance_tabs',
])
