"""
SIP (Systematic Investment Plan) Package

Automated ETF investing with profit allocation
"""

from .sip_simulator import (
    SIPSimulator,
    SIPTransaction,
    get_sip_simulator
)

__all__ = [
    'SIPSimulator',
    'SIPTransaction',
    'get_sip_simulator'
]

