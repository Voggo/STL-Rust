"""
Comprehensive test suite for ostl_python.

Tests all formula types, monitor types, and error handling.
"""

import pytest
import ostl_python.ostl_python as ostl


class TestParseFormula:
    """Test the parse_formula function that uses the same DSL as Rust's stl! macro."""

    def test_simple_greater_than(self):
        """Test parsing a simple greater-than predicate."""
        formula = ostl.parse_formula("x > 5")
        assert formula is not None
        assert "x" in str(formula)

    def test_simple_less_than(self):
        """Test parsing a simple less-than predicate."""
        formula = ostl.parse_formula("y < 3.14")
        assert formula is not None
        assert "y" in str(formula)

    def test_greater_equal(self):
        """Test parsing greater-than-or-equal predicate."""
        formula = ostl.parse_formula("x >= 5")
        assert formula is not None

    def test_less_equal(self):
        """Test parsing less-than-or-equal predicate."""
        formula = ostl.parse_formula("x <= 5")
        assert formula is not None

    def test_boolean_true(self):
        """Test parsing true constant."""
        formula = ostl.parse_formula("true")
        assert formula is not None

    def test_boolean_false(self):
        """Test parsing false constant."""
        formula = ostl.parse_formula("false")
        assert formula is not None

    def test_globally(self):
        """Test parsing globally operator."""
        formula = ostl.parse_formula("G[0, 10](x > 5)")
        assert formula is not None
        assert "G" in str(formula)

    def test_globally_keyword(self):
        """Test parsing globally keyword syntax."""
        formula = ostl.parse_formula("globally[0, 10](x > 5)")
        assert formula is not None

    def test_eventually(self):
        """Test parsing eventually operator."""
        formula = ostl.parse_formula("F[0, 5](y < 3)")
        assert formula is not None
        assert "F" in str(formula)

    def test_eventually_keyword(self):
        """Test parsing eventually keyword syntax."""
        formula = ostl.parse_formula("eventually[0, 5](y < 3)")
        assert formula is not None

    def test_and_symbols(self):
        """Test parsing conjunction with && symbols."""
        formula = ostl.parse_formula("x > 5 && y < 3")
        assert formula is not None

    def test_and_keyword(self):
        """Test parsing conjunction with 'and' keyword."""
        formula = ostl.parse_formula("x > 5 and y < 3")
        assert formula is not None

    def test_or_symbols(self):
        """Test parsing disjunction with || symbols."""
        formula = ostl.parse_formula("x > 5 || y < 3")
        assert formula is not None

    def test_or_keyword(self):
        """Test parsing disjunction with 'or' keyword."""
        formula = ostl.parse_formula("x > 5 or y < 3")
        assert formula is not None

    def test_not_symbol(self):
        """Test parsing negation with ! symbol."""
        formula = ostl.parse_formula("!(x > 5)")
        assert formula is not None

    def test_not_keyword(self):
        """Test parsing negation with 'not' keyword."""
        formula = ostl.parse_formula("not(x > 5)")
        assert formula is not None

    def test_implies_symbol(self):
        """Test parsing implication with -> symbol."""
        formula = ostl.parse_formula("x > 5 -> y < 3")
        assert formula is not None

    def test_implies_keyword(self):
        """Test parsing implication with 'implies' keyword."""
        formula = ostl.parse_formula("x > 5 implies y < 3")
        assert formula is not None

    def test_until_symbol(self):
        """Test parsing until operator with U symbol."""
        formula = ostl.parse_formula("x > 5 U[0, 10] y < 3")
        assert formula is not None

    def test_until_keyword(self):
        """Test parsing until operator with 'until' keyword."""
        formula = ostl.parse_formula("x > 5 until[0, 10] y < 3")
        assert formula is not None

    def test_nested_temporal(self):
        """Test parsing nested temporal operators."""
        formula = ostl.parse_formula("G[0, 10](F[0, 5](x > 0))")
        assert formula is not None

    def test_complex_formula(self):
        """Test parsing complex formula."""
        formula = ostl.parse_formula("G[0, 10](x > 5) && F[0, 5](y < 3)")
        assert formula is not None

    def test_whitespace_tolerance(self):
        """Test that parser handles extra whitespace."""
        formula = ostl.parse_formula("  G  [  0  ,  10  ]  (  x  >  5  )  ")
        assert formula is not None

    def test_decimal_interval(self):
        """Test parsing intervals with decimal values."""
        formula = ostl.parse_formula("G[0.5, 10.5](x > 5)")
        assert formula is not None

    def test_signal_with_underscore(self):
        """Test parsing signal names with underscores."""
        formula = ostl.parse_formula("my_signal > 5")
        assert formula is not None
        assert "my_signal" in str(formula)

    def test_negative_threshold(self):
        """Test parsing predicates with negative thresholds."""
        formula = ostl.parse_formula("x > -5")
        assert formula is not None

    def test_error_empty_input(self):
        """Test that empty input raises an error."""
        with pytest.raises(ValueError):
            ostl.parse_formula("")

    def test_error_invalid_syntax(self):
        """Test that invalid syntax raises an error."""
        with pytest.raises(ValueError):
            ostl.parse_formula("not a valid formula !!!")

    def test_error_missing_interval(self):
        """Test that missing interval raises an error."""
        with pytest.raises(ValueError):
            ostl.parse_formula("G(x > 5)")

    def test_with_monitor(self):
        """Test using parse_formula with a monitor."""
        formula = ostl.parse_formula("G[0, 5](x > 0.5)")
        monitor = ostl.Monitor(formula, semantics="Robustness")
        result = monitor.update("x", 1.0, 0.0)
        assert "evaluations" in result

    def test_complex_with_monitor(self):
        """Test complex parsed formula with a monitor."""
        formula = ostl.parse_formula("G[0, 5](x > 0) && F[0, 3](y < 10)")
        monitor = ostl.Monitor(formula, semantics="Rosi")
        result = monitor.update("x", 1.0, 0.0)
        assert "evaluations" in result


class TestFormulaCreation:
    """Test that all formula types can be created."""

    def test_gt_formula(self):
        """Test greater-than atomic predicate."""
        formula = ostl.Formula.gt("x", 5.0)
        assert formula is not None
        assert "x" in str(formula)

    def test_lt_formula(self):
        """Test less-than atomic predicate."""
        formula = ostl.Formula.lt("y", 10.0)
        assert formula is not None
        assert "y" in str(formula)

    def test_true_formula(self):
        """Test constant true formula."""
        formula = ostl.Formula.true_()
        assert formula is not None

    def test_false_formula(self):
        """Test constant false formula."""
        formula = ostl.Formula.false_()
        assert formula is not None

    def test_and_formula(self):
        """Test conjunction of formulas."""
        f1 = ostl.Formula.gt("x", 1.0)
        f2 = ostl.Formula.lt("x", 10.0)
        formula = ostl.Formula.and_(f1, f2)
        assert formula is not None

    def test_or_formula(self):
        """Test disjunction of formulas."""
        f1 = ostl.Formula.gt("x", 5.0)
        f2 = ostl.Formula.lt("y", 3.0)
        formula = ostl.Formula.or_(f1, f2)
        assert formula is not None

    def test_not_formula(self):
        """Test negation of formula."""
        f = ostl.Formula.gt("x", 5.0)
        formula = ostl.Formula.not_(f)
        assert formula is not None

    def test_implies_formula(self):
        """Test implication."""
        f1 = ostl.Formula.gt("x", 0.0)
        f2 = ostl.Formula.lt("y", 100.0)
        formula = ostl.Formula.implies(f1, f2)
        assert formula is not None

    def test_always_formula(self):
        """Test globally (always) temporal operator."""
        f = ostl.Formula.gt("x", 5.0)
        formula = ostl.Formula.always(0.0, 10.0, f)
        assert formula is not None

    def test_eventually_formula(self):
        """Test eventually (finally) temporal operator."""
        f = ostl.Formula.lt("x", 5.0)
        formula = ostl.Formula.eventually(0.0, 5.0, f)
        assert formula is not None

    def test_until_formula(self):
        """Test until temporal operator."""
        f1 = ostl.Formula.gt("x", 0.0)
        f2 = ostl.Formula.gt("x", 10.0)
        formula = ostl.Formula.until(0.0, 5.0, f1, f2)
        assert formula is not None

    def test_complex_formula(self):
        """Test complex nested formula."""
        # Always[0,5](x > 0.5 AND Eventually[0,2](y < 0.8))
        f1 = ostl.Formula.gt("x", 0.5)
        f2 = ostl.Formula.lt("y", 0.8)
        f3 = ostl.Formula.eventually(0.0, 2.0, f2)
        f4 = ostl.Formula.and_(f1, f3)
        formula = ostl.Formula.always(0.0, 5.0, f4)
        assert formula is not None


class TestMonitorCreation:
    """Test that all monitor types can be created."""

    def test_monitor_default(self):
        """Test monitor with default parameters."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula)
        assert monitor is not None
        assert "Robustness" in repr(monitor)

    def test_boolean_monitor_incremental_eager(self):
        """Test boolean monitor with explicit parameters."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor_eager = ostl.Monitor(
            formula, semantics="EagerSatisfaction", algorithm="Incremental"
        )
        assert monitor_eager is not None

    def test_boolean_monitor_incremental_strict(self):
        """Test boolean monitor with incremental algorithm and strict semantics."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor_strict = ostl.Monitor(
            formula, semantics="StrictSatisfaction", algorithm="Incremental"
        )
        assert monitor_strict is not None

    def test_boolean_monitor_naive(self):
        """Test boolean monitor with naive algorithm."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction", algorithm="Naive")
        assert monitor is not None

    def test_quantitative_monitor(self):
        """Test quantitative monitor."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="Robustness")
        assert monitor is not None
        assert "Robustness" in repr(monitor)

    def test_robustness_monitor(self):
        """Test robustness interval monitor."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="Rosi")
        assert monitor is not None
        assert "Rosi" in repr(monitor)

    def test_robustness_monitor_eager(self):
        """Test robustness monitor (rosi semantics)."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="Rosi")
        assert monitor is not None


class TestMonitorUpdate:
    """Test that monitor update works for all semantics."""

    def test_boolean_update(self):
        """Test boolean monitor update."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        result = monitor.update("x", 10.0, 0.0)

        assert "input_signal" in result
        assert result["input_signal"] == "x"
        assert "input_timestamp" in result
        assert result["input_timestamp"] == 0.0
        assert "evaluations" in result
        assert isinstance(result["evaluations"], list)

        # Check structure of evaluations
        if len(result["evaluations"]) > 0:
            eval_dict = result["evaluations"][0]
            assert "sync_step_signal" in eval_dict
            assert "sync_step_timestamp" in eval_dict
            assert "sync_step_value" in eval_dict
            assert "outputs" in eval_dict

            if len(eval_dict["outputs"]) > 0:
                output = eval_dict["outputs"][0]
                assert "timestamp" in output
                assert "value" in output
                assert isinstance(output["value"], bool)

    def test_quantitative_update(self):
        """Test quantitative monitor update."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="Robustness")

        result = monitor.update("x", 10.0, 0.0)

        assert "input_signal" in result
        assert "evaluations" in result

        if len(result["evaluations"]) > 0:
            eval_dict = result["evaluations"][0]
            if len(eval_dict["outputs"]) > 0:
                output = eval_dict["outputs"][0]
                assert isinstance(output["value"], float)

    def test_robustness_update(self):
        """Test robustness monitor update."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="Rosi")

        result = monitor.update("x", 10.0, 0.0)

        assert "input_signal" in result
        assert "evaluations" in result

        if len(result["evaluations"]) > 0:
            eval_dict = result["evaluations"][0]
            if len(eval_dict["outputs"]) > 0:
                output = eval_dict["outputs"][0]
                assert isinstance(output["value"], tuple)
                assert len(output["value"]) == 2
                assert isinstance(output["value"][0], float)
                assert isinstance(output["value"][1], float)

    def test_multiple_updates(self):
        """Test multiple sequential updates."""
        formula = ostl.Formula.always(0.0, 2.0, ostl.Formula.gt("x", 5.0))
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        for t in range(5):
            result = monitor.update("x", 10.0, float(t))
            assert result is not None
            assert result["input_timestamp"] == float(t)

    def test_multi_signal_update(self):
        """Test updates with multiple signals."""
        f1 = ostl.Formula.gt("x", 5.0)
        f2 = ostl.Formula.lt("y", 10.0)
        formula = ostl.Formula.and_(f1, f2)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        result1 = monitor.update("x", 10.0, 0.0)
        assert result1["input_signal"] == "x"

        result2 = monitor.update("y", 5.0, 0.0)
        assert result2["input_signal"] == "y"


class TestErrorHandling:
    """Test that invalid inputs are properly caught."""

    def test_invalid_semantics(self):
        """Test that invalid semantics raises ValueError."""
        formula = ostl.Formula.gt("x", 5.0)

        with pytest.raises(ValueError, match="Invalid semantics"):
            ostl.Monitor(formula, semantics="invalid")

    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises ValueError."""
        formula = ostl.Formula.gt("x", 5.0)

        with pytest.raises(ValueError, match="Invalid algorithm"):
            ostl.Monitor(formula, semantics="strict", algorithm="invalid")

    def test_naive_algorithm_with_eager(self):
        """Test that naive algorithm with eager semantics raises error."""
        formula = ostl.Formula.gt("x", 5.0)

        with pytest.raises(ValueError, match="Naive algorithm does not support"):
            ostl.Monitor(formula, semantics="EagerSatisfaction", algorithm="Naive")

    def test_invalid_synchronization(self):
        """Test that invalid synchronization raises ValueError."""
        formula = ostl.Formula.gt("x", 5.0)

        with pytest.raises(ValueError, match="Invalid synchronization"):
            ostl.Monitor(formula, semantics="strict", synchronization="invalid")


class TestsynchronizationStrategies:
    """Test that all synchronization strategies can be created."""

    def test_zoh_synchronization(self):
        """Test zero-order hold synchronization."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction", synchronization="ZeroOrderHold")
        assert monitor is not None
        assert "ZeroOrderHold" in repr(monitor)

    def test_linear_synchronization(self):
        """Test linear synchronization."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction", synchronization="Linear")
        assert monitor is not None
        assert "Linear" in repr(monitor)

    def test_none_synchronization(self):
        """Test no synchronization."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction", synchronization="None")
        assert monitor is not None
        assert "None" in repr(monitor)

    def test_synchronization_with_quantitative(self):
        """Test synchronization strategies work with quantitative semantics."""
        formula = ostl.Formula.gt("x", 5.0)

        monitor_zoh = ostl.Monitor(
            formula, semantics="Robustness", synchronization="ZeroOrderHold"
        )
        assert monitor_zoh is not None

        monitor_linear = ostl.Monitor(
            formula, semantics="Robustness", synchronization="Linear"
        )
        assert monitor_linear is not None

        monitor_none = ostl.Monitor(
            formula, semantics="Robustness", synchronization="None"
        )
        assert monitor_none is not None

    def test_synchronization_with_rosi(self):
        """Test synchronization strategies work with rosi semantics."""
        formula = ostl.Formula.gt("x", 5.0)

        monitor_zoh = ostl.Monitor(formula, semantics="Rosi", synchronization="ZeroOrderHold")
        assert monitor_zoh is not None

        monitor_linear = ostl.Monitor(formula, semantics="Rosi", synchronization="Linear")
        assert monitor_linear is not None

        monitor_none = ostl.Monitor(formula, semantics="Rosi", synchronization="None")
        assert monitor_none is not None


class TestTemporalFormulas:
    """Test temporal formulas produce correct structure."""

    def test_always_produces_verdicts(self):
        """Test that Always formula eventually produces verdicts."""
        formula = ostl.Formula.always(0.0, 2.0, ostl.Formula.gt("x", 5.0))
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        # Feed data points
        results = []
        for t in range(5):
            result = monitor.update("x", 10.0, float(t))
            results.append(result)

        # At least some results should have evaluations
        total_evals = sum(len(r["evaluations"]) for r in results)
        assert total_evals > 0

    def test_eventually_produces_verdicts(self):
        """Test that Eventually formula produces verdicts."""
        formula = ostl.Formula.eventually(0.0, 2.0, ostl.Formula.gt("x", 5.0))
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        results = []
        for t in range(5):
            result = monitor.update("x", 10.0, float(t))
            results.append(result)

        total_evals = sum(len(r["evaluations"]) for r in results)
        assert total_evals > 0

    def test_until_produces_verdicts(self):
        """Test that Until formula produces verdicts."""
        f1 = ostl.Formula.gt("x", 0.0)
        f2 = ostl.Formula.gt("x", 10.0)
        formula = ostl.Formula.until(0.0, 3.0, f1, f2)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        results = []
        # Start with low value, then high
        for t in range(2):
            result = monitor.update("x", 5.0, float(t))
            results.append(result)
        for t in range(2, 5):
            result = monitor.update("x", 15.0, float(t))
            results.append(result)

        total_evals = sum(len(r["evaluations"]) for r in results)
        assert total_evals > 0


class TestBooleanOperators:
    """Test boolean operators produce correct behavior."""

    def test_and_operator(self):
        """Test AND operator."""
        f1 = ostl.Formula.gt("x", 5.0)
        f2 = ostl.Formula.lt("x", 15.0)
        formula = ostl.Formula.and_(f1, f2)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        # Value in range should satisfy
        result = monitor.update("x", 10.0, 0.0)
        assert result is not None

    def test_or_operator(self):
        """Test OR operator."""
        f1 = ostl.Formula.gt("x", 100.0)
        f2 = ostl.Formula.lt("x", 5.0)
        formula = ostl.Formula.or_(f1, f2)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        result = monitor.update("x", 2.0, 0.0)
        assert result is not None

    def test_not_operator(self):
        """Test NOT operator."""
        f = ostl.Formula.gt("x", 5.0)
        formula = ostl.Formula.not_(f)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        result = monitor.update("x", 2.0, 0.0)
        assert result is not None

    def test_implies_operator(self):
        """Test IMPLIES operator."""
        f1 = ostl.Formula.gt("x", 5.0)
        f2 = ostl.Formula.lt("y", 10.0)
        formula = ostl.Formula.implies(f1, f2)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        result = monitor.update("x", 10.0, 0.0)
        assert result is not None


class TestEvaluationStructure:
    """Test the structure of evaluation results."""

    def test_evaluation_has_sync_step_info(self):
        """Test that evaluations include sync step information."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="StrictSatisfaction")

        result = monitor.update("x", 10.0, 0.0)

        if len(result["evaluations"]) > 0:
            eval_dict = result["evaluations"][0]

            # Check all required fields are present
            assert "sync_step_signal" in eval_dict
            assert "sync_step_timestamp" in eval_dict
            assert "sync_step_value" in eval_dict
            assert "outputs" in eval_dict

            # Check types
            assert isinstance(eval_dict["sync_step_signal"], str)
            assert isinstance(eval_dict["sync_step_timestamp"], float)
            assert isinstance(eval_dict["sync_step_value"], float)
            assert isinstance(eval_dict["outputs"], list)

    def test_output_structure(self):
        """Test that outputs have correct structure."""
        formula = ostl.Formula.gt("x", 5.0)
        monitor = ostl.Monitor(formula, semantics="Rosi")

        result = monitor.update("x", 10.0, 0.0)

        if len(result["evaluations"]) > 0:
            eval_dict = result["evaluations"][0]
            if len(eval_dict["outputs"]) > 0:
                output = eval_dict["outputs"][0]

                assert "timestamp" in output
                assert "value" in output
                assert isinstance(output["timestamp"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
