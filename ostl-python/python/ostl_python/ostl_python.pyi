"""
Online Signal Temporal Logic (STL) monitoring library.

This library provides efficient online monitoring of STL formulas with multiple semantics:
- StrictSatisfaction/EagerSatisfaction: classic true/false evaluation
- Robustness: robustness as a single float value
- Rosi: robustness as an interval (min, max)

Example using parse_formula (recommended):
    >>> import ostl_python
    >>> # Parse formula using the same DSL syntax as Rust's stl! macro
    >>> phi = ostl_python.parse_formula('G[0, 5](x > 0.5)')
    >>> # Create monitor with Robustness semantics
    >>> monitor = ostl_python.Monitor(phi, semantics='Robustness')
    >>> # Feed data
    >>> output = monitor.update('x', 1.0, 0.0)
    >>> # Print using Rust's Display formatting
    >>> print(output)
    >>> # Access structured data
    >>> print(output.to_dict())

Example using Formula builder methods:
    >>> import ostl_python
    >>> # Create formula: Always[0,5](x > 0.5)
    >>> phi = ostl_python.Formula.always(0, 5, ostl_python.Formula.gt('x', 0.5))
    >>> # Create monitor with Rosi semantics
    >>> monitor = ostl_python.Monitor(phi, semantics='Rosi')
    >>> # Feed data
    >>> output = monitor.update('x', 1.0, 0.0)
    >>> # Use __str__ and __repr__ for Rust-style formatting
    >>> print(str(output))   # Display format
    >>> print(repr(output))  # Debug format
"""

from typing import Literal, TypedDict, Union, List, Tuple

__version__: str


def parse_formula(formula_str: str) -> "Formula":
    """
    Parse an STL formula from a string using the same DSL syntax as Rust's `stl!` macro.

    This allows you to write formulas using the same syntax as Rust, making it easy
    to port formulas between Python and Rust code.

    Syntax:
        Predicates:
            - ``signal > value`` - Signal greater than value
            - ``signal < value`` - Signal less than value
            - ``signal >= value`` - Signal greater than or equal
            - ``signal <= value`` - Signal less than or equal

        Boolean Constants:
            - ``true`` - Always true
            - ``false`` - Always false

        Unary Operators:
            - ``!(sub)`` or ``not(sub)`` - Negation
            - ``G[start, end](sub)`` or ``globally[start, end](sub)`` - Globally (always)
            - ``F[start, end](sub)`` or ``eventually[start, end](sub)`` - Eventually (finally)

        Binary Operators:
            - ``left && right`` or ``left and right`` - Conjunction
            - ``left || right`` or ``left or right`` - Disjunction
            - ``left -> right`` or ``left implies right`` - Implication
            - ``left U[start, end] right`` or ``left until[start, end] right`` - Until

    Args:
        formula_str: A string containing an STL formula

    Returns:
        The parsed Formula object

    Raises:
        ValueError: If the formula string cannot be parsed

    Examples:
        >>> # Simple predicate
        >>> f = parse_formula("x > 5")
        >>> # Globally operator
        >>> f = parse_formula("G[0, 10](x > 5)")
        >>> # Complex formula
        >>> f = parse_formula("G[0, 10](x > 5) && F[0, 5](y < 3)")
        >>> # Using keyword syntax
        >>> f = parse_formula("globally[0, 10](x > 5) and eventually[0, 5](y < 3)")
    """
    ...


class OutputDict(TypedDict):
    """A single output verdict."""

    timestamp: float
    """The timestamp this verdict is for."""
    value: Union[bool, float, Tuple[float, float]]
    """The verdict value. Type depends on monitor semantics:
    - bool for qualitative semantics
    - float for quantitative semantics
    - tuple[float, float] for RoSI semantics (min, max)
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

SemanticsType = Literal["StrictSatisfaction", "EagerSatisfaction", "Robustness", "Rosi"]
AlgorithmType = Literal["Incremental", "Naive"]
SynchronizationType = Literal["ZeroOrderHold", "Linear", "None"]


class MonitorOutput:
    """
    Output from a monitor update operation.

    This class wraps the Rust MonitorOutput structure and provides:
    - Rust-style Display formatting via __str__()
    - Rust-style Debug formatting via __repr__()
    - Structured data access via to_dict()
    - Convenient properties and methods

    The string representation shows verdicts in the format:
        ``t={timestamp}: {value}``

    For multiple verdicts, they are shown on separate lines.
    If no verdicts are available, it shows "No verdicts available".

    Example:
        >>> output = monitor.update('x', 1.0, 0.0)
        >>> # Rust-style Display
        >>> print(output)
        t=5s: 0.5
        >>> # Access properties
        >>> print(output.input_signal, output.input_timestamp)
        >>> # Get finalized verdicts
        >>> for ts, val in output.finalize():
        ...     print(f"Verdict at {ts}: {val}")
    """

    @property
    def input_signal(self) -> str:
        """The name of the signal that was updated."""
        ...

    @property
    def input_timestamp(self) -> float:
        """The timestamp (in seconds) of the input that triggered this update."""
        ...

    @property
    def input_value(self) -> float:
        """The value of the input signal."""
        ...

    def has_outputs(self) -> bool:
        """
        Check if there are any output verdicts.

        Returns:
            True if there is at least one output verdict, False otherwise.
        """
        ...

    def total_outputs(self) -> int:
        """
        Get the total number of output verdicts.

        Returns:
            The total count of output verdicts across all evaluations.
        """
        ...

    def is_empty(self) -> bool:
        """
        Check if the evaluations list is empty.

        Returns:
            True if no evaluations were triggered, False otherwise.
        """
        ...

    def finalize(self) -> List[Tuple[float, Union[bool, float, Tuple[float, float]]]]:
        """
        Get the finalized verdicts as a list of (timestamp, value) tuples.

        This returns the latest verdict for each unique timestamp,
        matching the behavior of Rust's ``finalize()`` method.

        Returns:
            List of (timestamp, value) tuples. The value type depends on the
            monitor semantics:
            - bool for qualitative semantics
            - float for robustness semantics
            - tuple[float, float] for RoSI semantics
        """
        ...

    def to_dict(self) -> MonitorOutputDict:
        """
        Convert the monitor output to a dictionary.

        Returns:
            Dictionary containing:
            - 'input_signal': the signal name
            - 'input_timestamp': the input timestamp
            - 'input_value': the input value
            - 'evaluations': list of evaluation dictionaries

        Example:
            >>> output = monitor.update('x', 1.0, 0.0)
            >>> d = output.to_dict()
            >>> print(d['input_signal'], d['input_timestamp'])
        """
        ...

    def __str__(self) -> str:
        """
        Return Rust-style Display representation of the output.

        Shows verdicts in the format: ``t={timestamp}: {value}``
        For multiple verdicts, they are shown on separate lines.
        If no verdicts are available, returns "No verdicts available".
        """
        ...

    def __repr__(self) -> str:
        """
        Return Rust-style Debug representation of the output.

        Shows the full internal structure of the MonitorOutput,
        matching Rust's Debug formatting.
        """
        ...

class Monitor:
    """
    Online STL monitor.

    Monitors signal traces against an STL formula and produces verdicts.
    Supports multiple semantics (StrictSatisfaction, EagerSatisfaction, Robustness, Rosi),
    algorithms (Incremental, Naive), and synchronization strategies (ZeroOrderHold, Linear, None).

    The monitor processes signals incrementally and produces verdicts when
    sufficient information is available.

    Example:
        >>> phi = Formula.always(0, 5, Formula.gt('x', 0.5))
        >>> monitor = Monitor(phi, semantics='Rosi')
        >>> for t in range(10):
        ...     result = monitor.update('x', 0.8, float(t))
        ...     for eval in result['evaluations']:
        ...         for output in eval['outputs']:
        ...             print(f"t={t}: {output}")
    """

    def __init__(
        self,
        formula: Formula,
        semantics: SemanticsType = "Robustness",
        algorithm: AlgorithmType = "Incremental",
        synchronization: SynchronizationType = "ZeroOrderHold",
    ) -> None:
        """
        Create a new STL monitor.

        Args:
            formula: The STL formula to monitor
            semantics: Output semantics. Options:
                - "StrictSatisfaction": Returns True/False with strict evaluation
                - "EagerSatisfaction": Returns True/False with eager evaluation
                - "Robustness": Returns float robustness value (+ = satisfied, - = violated, default)
                - "Rosi": Returns (min, max) robustness interval
            algorithm: Monitoring algorithm. Options:
                - "Incremental": Efficient online monitoring with sliding windows (default)
                - "Naive": Simple baseline implementation
            synchronization: Signal synchronization method. Options:
                - "ZeroOrderHold": Zero-order hold (default)
                - "Linear": Linear interpolation
                - "None": No interpolation

        Raises:
            ValueError: If invalid semantics, algorithm, or synchronization is specified
            ValueError: If Naive algorithm is used with EagerSatisfaction (not supported)

        Note:
            - For single-signal formulas, signal synchronization is automatically disabled
              for better performance

        Example:
            >>> # StrictSatisfaction monitoring
            >>> m1 = Monitor(phi, semantics="StrictSatisfaction", algorithm="Incremental")
            >>>
            >>> # Robustness (quantitative)
            >>> m2 = Monitor(phi, semantics="Robustness")
            >>>
            >>> # Rosi intervals with eager evaluation
            >>> m3 = Monitor(phi, semantics="Rosi")
        """
        ...

    def update(self, signal: str, value: float, timestamp: float) -> MonitorOutput:
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
            MonitorOutput object containing:
                - Display/Debug formatting via __str__() and __repr__()
                - Properties: input_signal, input_timestamp, input_value
                - Methods: has_outputs(), total_outputs(), is_empty(), finalize()
                - Structured data access via to_dict()

        Note:
            - Evaluations list may be empty if data is being buffered
            - For multi-signal formulas, the synchronizer may produce interpolated steps
            - Each evaluation corresponds to a specific synchronized step
            - Multiple outputs may be produced for each synchronized step

        Example:
            >>> output = monitor.update('x', 0.8, 1.0)
            >>> # Use Rust-style Display formatting
            >>> print(output)
            >>> # Access properties
            >>> print(f"Input: {output.input_signal} at t={output.input_timestamp}")
            >>> # Get finalized verdicts
            >>> for ts, val in output.finalize():
            ...     print(f"Verdict at {ts}: {val}")
            >>> # Access as dictionary
            >>> d = output.to_dict()
            >>> for eval in d['evaluations']:
            ...     for out in eval['outputs']:
            ...         print(f"  t={out['timestamp']}: {out['value']}")
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of the monitor."""
        ...
