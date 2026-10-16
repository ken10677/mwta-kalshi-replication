"""
Kalshi Fee Calculator

Kalshi charges fees on trades using the formula:
  Taker: round_up(0.07 × contracts × price × (1 - price))
  Maker: round_up(0.0175 × contracts × price × (1 - price))

Where price is in dollars (e.g., 50¢ = 0.50).

Key points:
- Fees are charged on BOTH entry and exit trades
- Max fee is at 50¢ price (~1.75¢/contract for takers)
- Fees decrease toward extreme prices (near 0¢ or 100¢)

Source: https://kalshi.com/fee-schedule
"""

import math
from typing import Tuple


# Fee coefficients
TAKER_FEE_COEFFICIENT = 0.07
MAKER_FEE_COEFFICIENT = 0.0175


def calculate_fee_cents(price_cents: int, quantity: int = 1, is_taker: bool = True) -> int:
    """
    Calculate the fee in cents for a trade using Kalshi's formula.

    Formula: round_up(coefficient × contracts × price × (1 - price))

    Args:
        price_cents: Contract price in cents (1-99)
        quantity: Number of contracts
        is_taker: True for taker orders (immediate fill), False for maker (resting)

    Returns:
        Fee in cents (rounded up to nearest cent)
    """
    # Convert cents to dollars for the formula
    price_dollars = price_cents / 100.0

    # Choose coefficient based on order type
    coefficient = TAKER_FEE_COEFFICIENT if is_taker else MAKER_FEE_COEFFICIENT

    # Kalshi formula: coefficient × contracts × price × (1 - price)
    fee_dollars = coefficient * quantity * price_dollars * (1 - price_dollars)

    # Convert to cents and round UP (Kalshi always rounds up)
    fee_cents = math.ceil(fee_dollars * 100)

    return fee_cents


def calculate_round_trip_fees(
    entry_price_cents: int,
    exit_price_cents: int,
    quantity: int = 1,
    is_taker: bool = True
) -> int:
    """
    Calculate total fees for a round-trip trade (entry + exit).

    Args:
        entry_price_cents: Price paid on entry
        exit_price_cents: Price received on exit
        quantity: Number of contracts
        is_taker: Whether orders are taker (immediate) or maker (resting)

    Returns:
        Total fees in cents for both entry and exit
    """
    entry_fee = calculate_fee_cents(entry_price_cents, quantity, is_taker)
    exit_fee = calculate_fee_cents(exit_price_cents, quantity, is_taker)
    return entry_fee + exit_fee


def calculate_net_pnl(
    entry_price_cents: int,
    exit_price_cents: int,
    side: str,
    quantity: int = 1,
    is_taker: bool = True
) -> Tuple[int, int, int]:
    """
    Calculate net P&L after fees for a closed position.

    Args:
        entry_price_cents: Price paid on entry
        exit_price_cents: Price received on exit (current bid for sells)
        side: 'yes' or 'no'
        quantity: Number of contracts
        is_taker: Whether orders are taker or maker

    Returns:
        Tuple of (gross_pnl_cents, total_fees_cents, net_pnl_cents)
    """
    # Calculate gross P&L (before fees)
    # For BOTH sides: profit = exit_price - entry_price
    # (entry and exit prices are always in the same denomination as the side traded)
    # YES: bought YES at 30c, sell YES at 50c → profit = 50 - 30 = +20c
    # NO:  bought NO at 18c, sell NO at 60c  → profit = 60 - 18 = +42c
    gross_pnl_per_contract = exit_price_cents - entry_price_cents

    gross_pnl = gross_pnl_per_contract * quantity

    # Calculate fees for entry and exit
    total_fees = calculate_round_trip_fees(
        entry_price_cents, exit_price_cents, quantity, is_taker
    )

    # Net P&L = Gross - Fees
    net_pnl = gross_pnl - total_fees

    return (gross_pnl, total_fees, net_pnl)


def calculate_expected_value(
    entry_price_cents: int,
    true_probability: float,
    quantity: int = 1,
    side: str = "yes",
    is_taker: bool = True
) -> float:
    """
    Calculate expected value of a trade after fees.

    Args:
        entry_price_cents: Price you're paying in cents
        true_probability: Your estimate of P(YES wins) - always the YES probability!
                          Do NOT invert this for NO side trades - this function handles it.
        quantity: Number of contracts
        side: 'yes' or 'no'
        is_taker: Whether this is a taker order

    Returns:
        Expected value in cents (total, not per contract)
        Positive = profitable bet, negative = unprofitable

    Example:
        If consensus says YES=25%, and you're buying NO at 69c:
        - Call with true_probability=0.25 (not 0.75!)
        - Function computes: NO wins with prob 1-0.25=0.75
        - EV = 0.75 * (100-69) - 0.25 * 69 - fees = +6.0c (approx)
    """
    # Entry fee is always paid
    entry_fee = calculate_fee_cents(entry_price_cents, quantity, is_taker)

    if side == "yes":
        # Buying YES at price P:
        # Win (prob = true_probability): receive 100¢, pay exit fee at ~100¢
        # Lose (prob = 1 - true_probability): lose entry price

        # If we win, we get 100¢ per contract but pay exit fee
        # Exit fee at 100¢ is essentially 0 (price × (1-price) = 0)
        exit_fee_if_win = calculate_fee_cents(99, quantity, is_taker)  # ~0
        profit_if_win = (100 - entry_price_cents) * quantity - entry_fee - exit_fee_if_win

        loss_if_lose = entry_price_cents * quantity + entry_fee  # Lose entry + entry fee

        ev = (true_probability * profit_if_win) - ((1 - true_probability) * loss_if_lose)

    else:  # side == "no"
        # Buying NO (equivalent to selling YES at entry_price_cents)
        # Win when event doesn't happen

        exit_fee_if_win = calculate_fee_cents(1, quantity, is_taker)  # ~0
        profit_if_win = (100 - entry_price_cents) * quantity - entry_fee - exit_fee_if_win

        loss_if_lose = entry_price_cents * quantity + entry_fee

        # For NO, we win when the event doesn't happen
        ev = ((1 - true_probability) * profit_if_win) - (true_probability * loss_if_lose)

    return ev


def minimum_gross_profit_for_net_target(
    entry_price_cents: int,
    target_net_profit_cents: int,
    quantity: int = 1,
    is_taker: bool = True
) -> int:
    """
    Calculate the minimum gross profit needed to achieve a target net profit.

    Useful for setting take-profit thresholds that account for fees.

    Args:
        entry_price_cents: Entry price
        target_net_profit_cents: Desired net profit per contract after fees
        quantity: Number of contracts
        is_taker: Whether orders are taker

    Returns:
        Required gross profit per contract in cents
    """
    # We need: gross_profit - fees >= target_net_profit
    # Fees depend on exit price, so we iterate to find the answer

    for gross_profit in range(target_net_profit_cents, 100):
        exit_price = entry_price_cents + gross_profit
        if exit_price > 99:
            exit_price = 99

        _, total_fees, net_pnl = calculate_net_pnl(
            entry_price_cents, exit_price, "yes", quantity, is_taker
        )

        if net_pnl >= target_net_profit_cents * quantity:
            return gross_profit

    # Fallback: need significant gross profit
    return target_net_profit_cents + 4  # Rough estimate


def should_take_trade(
    market_price_cents: int,
    posterior_probability: float,
    min_ev_cents: float = 2.0,
    side: str = "yes"
) -> Tuple[bool, float, str]:
    """
    Decide whether a trade meets profitability criteria.

    Args:
        market_price_cents: Current market price in cents
        posterior_probability: Your blended probability estimate (0-1)
        min_ev_cents: Minimum expected value to accept
        side: 'yes' or 'no'

    Returns:
        Tuple of (should_trade, expected_value, reason)
    """
    ev = calculate_expected_value(
        entry_price_cents=market_price_cents,
        true_probability=posterior_probability,
        quantity=1,
        side=side,
        is_taker=True
    )

    if ev >= min_ev_cents:
        return (True, ev, f"EV of {ev:.1f}¢ exceeds minimum {min_ev_cents}¢")
    else:
        return (False, ev, f"EV of {ev:.1f}¢ below minimum {min_ev_cents}¢")


def estimate_net_ev_total(
    entry_price_cents: int,
    gross_ev_per_contract: float,
    quantity: int,
    is_taker: bool = True,
    slippage_cents: int = 0
) -> dict:
    """
    Estimate the total net EV for an order after fees and slippage.

    This is the key function for fee-aware trade gating. It computes:
      net_ev_total = (gross_ev_per_contract * qty) - entry_fee - exit_fee_estimate - slippage

    Args:
        entry_price_cents: Entry price in cents
        gross_ev_per_contract: Expected gross EV per contract (from calculate_expected_value)
        quantity: Number of contracts
        is_taker: Whether this is a taker order (default True)
        slippage_cents: Additional slippage estimate (default 0)

    Returns:
        Dict with:
            - gross_ev_total: Gross EV for all contracts
            - entry_fee_cents: Entry fee estimate
            - exit_fee_cents: Exit fee estimate (at entry price, conservative)
            - total_fee_cents: Total fees
            - slippage_cents: Slippage amount
            - net_ev_total: Net EV after fees and slippage
            - profitable: Whether net_ev_total > 0

    Example:
        >>> result = estimate_net_ev_total(5, 3.0, 50)
        >>> result['net_ev_total']  # May be negative due to fees on penny trades
    """
    gross_ev_total = gross_ev_per_contract * quantity

    # Entry fee
    entry_fee = calculate_fee_cents(entry_price_cents, quantity, is_taker)

    # Exit fee estimate - use entry price as conservative estimate
    # (actual exit price unknown, but entry price gives worst-case at mid-range)
    exit_fee = calculate_fee_cents(entry_price_cents, quantity, is_taker)

    total_fee = entry_fee + exit_fee
    net_ev_total = gross_ev_total - total_fee - slippage_cents

    return {
        "gross_ev_total": round(gross_ev_total, 2),
        "entry_fee_cents": entry_fee,
        "exit_fee_cents": exit_fee,
        "total_fee_cents": total_fee,
        "slippage_cents": slippage_cents,
        "net_ev_total": round(net_ev_total, 2),
        "profitable": net_ev_total > 0
    }


# For backwards compatibility
def calculate_fee_per_contract(price_cents: int, is_taker: bool = True) -> float:
    """Legacy function - returns fee per single contract."""
    return float(calculate_fee_cents(price_cents, 1, is_taker))


def calculate_trade_fees(
    price_cents: int,
    quantity: int,
    side: str,
    is_taker: bool = True
) -> Tuple[float, float]:
    """Legacy function - returns (fee_if_win, fee_if_lose)."""
    fee = calculate_fee_cents(price_cents, quantity, is_taker)
    return (float(fee), 0.0)
