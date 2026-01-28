"""
Online Signal Temporal Logic (STL) monitoring library.

This library provides efficient online monitoring of STL formulas with multiple semantics:
- Boolean: classic true/false evaluation
- Quantitative: robustness as a single float value
- Robustness (RoSI): robustness as an interval (min, max)

Example:
    >>> import ostl_python
    >>> # Create formula: Always[0,5](x > 0.5)
    >>> phi = ostl_python.Formula.always(0, 5, ostl_python.Formula.gt('x', 0.5))
    >>> # Create monitor with robustness semantics
    >>> monitor = ostl_python.Monitor(phi, semantics='robustness')
    >>> # Feed data
    >>> result = monitor.update('x', 1.0, 0.0)
    >>> for eval in result['evaluations']:
    ...     print(eval['outputs'])
"""

from typing import Literal, TypedDict, Union, List, Tuple

__version__: str

class OutputDict(TypedDict):
    """A single output verdict."""

    timestamp: float
    """The timestamp this verdict is for."""
    value: Union[bool, float, Tuple[float, float]]
    """The verdict value. Type depends on monitor semantics:
    - bool for boolean semantics
    - float for quantitative semantics
    - tuple[float, float] for robustness semantics (min, max)
    """

class EvaluationDict(TypedDict):
    """Result of evaluating a synchronized step."""

    sync_step_signal: str
    """Signal name of the synchronized step (may be interpolated)."""
    sync_step_timestamp: float
    """Timestamp of the synchronized step."""
    sync_step_value: float
    """Value of the synchronized step."""
    outputs: List[OutputDict]
    """List of output verdicts produced by evaluating this sync step."""

class MonitorOutputDict(TypedDict):
    """Output from a monitor update operation."""

    input_signal: str
    """The signal name that was updated."""
    input_timestamp: float
    """The timestamp of the input that triggered this update."""
    evaluations: List[EvaluationDict]
    """List of evaluations, one for each synchronized step. May be empty if data is buffered."""

class Formula:
    """
    Signal Temporal Logic (STL) formula.

    Use static methods to construct formulas. Formulas can be composed using
    boolean and temporal operators.

    Example:
        >>> # G[0,5](x > 0.5)
        >>> phi = Formula.always(0, 5, Formula.gt('x', 0.5))
        >>> # (x > 0.3) AND F[0,3](y < 0.8)
        >>> phi2 = Formula.and_(
        ...     Formula.gt('x', 0.3),
        ...     Formula.eventually(0, 3, Formula.lt('y', 0.8))
        ... )
    """

    @staticmethod
    def gt(signal: str, value: float) -> "Formula":
        """
        Create a greater-than atomic predicate.

        Args:
            signal: Name of the signal to compare
            value: Threshold value to compare against

        Returns:
            Formula representing: signal > value

        Example:
            >>> Formula.gt('x', 0.5)  # x > 0.5
        """
        ...

    @staticmethod
    def lt(signal: str, value: float) -> "Formula":
        """
        Create a less-than atomic predicate.

        Args:
            signal: Name of the signal to compare
            value: Threshold value to compare against

        Returns:
            Formula representing: signal < value

        Example:
            >>> Formula.lt('x', 0.5)  # x < 0.5
        """
        ...

    @staticmethod
    def true_() -> "Formula":
        """
        Create a constant true formula.

        Returns:
            Formula that is always satisfied (⊤)

        Example:
            >>> Formula.true_()
        """
        ...

    @staticmethod
    def false_() -> "Formula":
        """
        Create a constant false formula.

        Returns:
            Formula that is never satisfied (⊥)

        Example:
            >>> Formula.false_()
        """
        ...

    @staticmethod
    def and_(left: "Formula", right: "Formula") -> "Formula":
        """
        Create a conjunction (AND) of two formulas.

        Args:
            left: First formula
            right: Second formula

        Returns:
            Formula representing: left ∧ right

        Example:
            >>> Formula.and_(Formula.gt('x', 0.5), Formula.lt('x', 1.0))
        """
        ...

    @staticmethod
    def or_(left: "Formula", right: "Formula") -> "Formula":
        """
        Create a disjunction (OR) of two formulas.

        Args:
            left: First formula
            right: Second formula

        Returns:
            Formula representing: left ∨ right

        Example:
            >>> Formula.or_(Formula.gt('x', 1.0), Formula.lt('x', 0.0))
        """
        ...

    @staticmethod
    def not_(child: "Formula") -> "Formula":
        """
        Create a negation (NOT) of a formula.

        Args:
            child: Formula to negate

        Returns:
            Formula representing: ¬child

        Example:
            >>> Formula.not_(Formula.gt('x', 0.5))  # x <= 0.5
        """
        ...

    @staticmethod
    def implies(left: "Formula", right: "Formula") -> "Formula":
        """
        Create an implication between two formulas.

        Args:
            left: Antecedent (condition)
            right: Consequent (result)

        Returns:
            Formula representing: left → right (if left then right)

        Example:
            >>> # If x > 2.0, then y < 1.0
            >>> Formula.implies(Formula.gt('x', 2.0), Formula.lt('y', 1.0))
        """
        ...

    @staticmethod
    def always(start: float, end: float, child: "Formula") -> "Formula":
        """
        Create a globally (always) temporal formula.

        The formula must hold at all time points in the interval [start, end]
        relative to the current time.

        Args:
            start: Start of time interval (seconds)
            end: End of time interval (seconds)
            child: Formula that must hold throughout the interval

        Returns:
            Formula representing: G[start,end](child)

        Example:
            >>> # For the next 5 seconds, x must be > 0.5
            >>> Formula.always(0, 5, Formula.gt('x', 0.5))
        """
        ...

    @staticmethod
    def eventually(start: float, end: float, child: "Formula") -> "Formula":
        """
        Create an eventually (finally) temporal formula.

        The formula must hold at some time point in the interval [start, end]
        relative to the current time.

        Args:
            start: Start of time interval (seconds)
            end: End of time interval (seconds)
            child: Formula that must hold at some point in the interval

        Returns:
            Formula representing: F[start,end](child)

        Example:
            >>> # Within 3 seconds, y must drop below 0.8
            >>> Formula.eventually(0, 3, Formula.lt('y', 0.8))
        """
        ...

    @staticmethod
    def until(start: float, end: float, left: "Formula", right: "Formula") -> "Formula":
        """
        Create an until temporal formula.

        The left formula must hold until the right formula becomes true,
        and the right formula must become true within the interval [start, end].

        Args:
            start: Start of time interval (seconds)
            end: End of time interval (seconds)
            left: Formula that must hold until right becomes true
            right: Formula that must eventually become true

        Returns:
            Formula representing: left U[start,end] right

        Example:
            >>> # x > 0 must hold until y < 0.5 (within 5 seconds)
            >>> Formula.until(0, 5, Formula.gt('x', 0), Formula.lt('y', 0.5))
        """
        ...

    def __str__(self) -> str:
        """Return string representation of the formula."""
        ...

    def __repr__(self) -> str:
        """Return detailed representation of the formula."""
        ...

SemanticsType = Literal["boolean", "quantitative", "robustness"]
StrategyType = Literal["incremental", "naive"]
ModeType = Literal["eager", "strict"]

class Monitor:
    """
    Online STL monitor.

    Monitors signal traces against an STL formula and produces verdicts.
    Supports multiple semantics (boolean, quantitative, robustness), strategies
    (incremental, naive), and evaluation modes (eager, strict).

    The monitor processes signals incrementally and produces verdicts when
    sufficient information is available.

    Example:
        >>> phi = Formula.always(0, 5, Formula.gt('x', 0.5))
        >>> monitor = Monitor(phi, semantics='robustness')
        >>> for t in range(10):
        ...     result = monitor.update('x', 0.8, float(t))
        ...     for eval in result['evaluations']:
        ...         for output in eval['outputs']:
        ...             print(f"t={t}: {output}")
    """

    def __init__(
        self,
        formula: Formula,
        semantics: SemanticsType = "boolean",
        strategy: StrategyType = "incremental",
        mode: ModeType | None = None,
    ) -> None:
        """
        Create a new STL monitor.

        Args:
            formula: The STL formula to monitor
            semantics: Output semantics. Options:
                - "boolean": Returns True/False (classic propositional evaluation)
                - "quantitative": Returns float robustness value (+ = satisfied, - = violated)
                - "robustness": Returns (min, max) robustness interval (RoSI semantics)
            strategy: Monitoring strategy. Options:
                - "incremental": Efficient online monitoring with sliding windows (recommended)
                - "naive": Simple baseline implementation (slower but easier to understand)
            mode: Evaluation mode. Options:
                - "eager": Produce verdicts as soon as possible (default for robustness)
                - "strict": Wait for complete information (default for boolean/quantitative)
                - None: Auto-select based on semantics

        Raises:
            ValueError: If invalid semantics, strategy, or mode is specified
            ValueError: If eager mode is used with quantitative semantics (not supported)

        Note:
            - Quantitative semantics does not support eager evaluation mode
            - For single-signal formulas, signal synchronization is automatically disabled
              for better performance

        Example:
            >>> # Boolean monitoring with strict mode
            >>> m1 = Monitor(phi, semantics="boolean", strategy="incremental")
            >>>
            >>> # Quantitative robustness
            >>> m2 = Monitor(phi, semantics="quantitative")
            >>>
            >>> # Robustness intervals with eager evaluation
            >>> m3 = Monitor(phi, semantics="robustness", mode="eager")
        """
        ...

    def update(self, signal: str, value: float, timestamp: float) -> MonitorOutputDict:
        """
        Update the monitor with a new signal value.

        Processes the new data point and produces verdicts when sufficient
        information is available. Each update may trigger multiple evaluations,
        one for each synchronized step (which may include interpolated values).

        Args:
            signal: Name of the signal being updated
            value: Signal value at this timestamp
            timestamp: Timestamp in seconds (must be monotonically increasing)

        Returns:
            Dictionary containing:
                - 'input_signal': The signal name that was updated
                - 'input_timestamp': The timestamp of this update
                - 'evaluations': List of evaluation dictionaries, each containing:
                    - 'sync_step_signal': Signal name of the synchronized step
                    - 'sync_step_timestamp': Timestamp of the synchronized step
                    - 'sync_step_value': Value of the synchronized step
                    - 'outputs': List of output dictionaries with:
                        - 'timestamp': The time this verdict is for
                        - 'value': The verdict value (type depends on semantics)

        Note:
            - Evaluations list may be empty if data is being buffered
            - For multi-signal formulas, the synchronizer may produce interpolated steps
            - Each evaluation corresponds to a specific synchronized step
            - Multiple outputs may be produced for each synchronized step

        Example:
            >>> result = monitor.update('x', 0.8, 1.0)
            >>> print(f"Input: {result['input_signal']} at t={result['input_timestamp']}")
            >>> for eval in result['evaluations']:
            ...     print(f"Sync step: {eval['sync_step_signal']} = {eval['sync_step_value']} at t={eval['sync_step_timestamp']}")
            ...     for output in eval['outputs']:
            ...         print(f"  Verdict at t={output['timestamp']}: {output['value']}")
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of the monitor."""
        ...
