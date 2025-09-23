#[cfg(test)]
mod tests {
    use stl_types_traits::{stl::operators::STLFormula, stl::operators::TimeInterval};
    use std::time::Duration;

    fn get_formula() -> STLFormula {
        STLFormula::And(
            Box::new(STLFormula::Always(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(10),
                },
                Box::new(STLFormula::LessThan(5.0)),
            )),
            Box::new(STLFormula::Until(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(20),
                },
                Box::new(STLFormula::False),
                Box::new(STLFormula::True),
            )),
        )
    }
    #[test]
    fn print_stl_formula() {
        println!("{}", get_formula().to_string());
        assert_eq!(
            get_formula().to_string(),
            "(G[0, 10](x < 5)) /\\ ((False) U[0, 20] (True))"
        );
    }

    #[test]
    fn to_tree_string() {
        println!("{}", get_formula().to_tree_string(10));
    }

    #[test]
    fn test_max_lookahead() {
        let formula = get_formula();
        assert_eq!(formula.get_max_lookahead(), Duration::from_secs(20));
    }
}
