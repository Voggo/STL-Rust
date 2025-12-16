#[macro_export]
macro_rules! stl {
        // --- ATOMICS ---
        (true) => {
            $crate::stl::formula_definition::FormulaDefinition::True
        };
        (false) => {
            $crate::stl::formula_definition::FormulaDefinition::False
        };

        // --- PREDICATES (Atomics) ---
        ($signal:ident > $val:expr) => {
            $crate::stl::formula_definition::FormulaDefinition::GreaterThan(stringify!($signal), $val as f64)
        };
        ($signal:ident < $val:expr) => {
            $crate::stl::formula_definition::FormulaDefinition::LessThan(stringify!($signal), $val as f64)
        };
        // Syntactic sugar for >=, <=, ==
        ($signal:ident >= $val:expr) => {
            // Not(signal < val)
            $crate::stl!(!($signal < $val))
        };
        ($signal:ident <= $val:expr) => {
            // Not(signal > val)
            $crate::stl!(!($signal > $val))
        };
        ($signal:ident == $val:expr) => {
            // (signal >= val) && (signal <= val)
            $crate::stl!(($signal >= $val) && ($signal <= $val))
        };

        // --- UNARY OPERATORS ---
        (! ($($sub:tt)+) ) => {
            $crate::stl::formula_definition::FormulaDefinition::Not(
                Box::new($crate::stl!($($sub)+))
            )
        };
        // Alias for !
        (not ($($sub:tt)+) ) => {
            $crate::stl!(!($($sub)+))
        };

        (G [$start:expr, $end:expr] ($($sub:tt)+) ) => {
            $crate::stl::formula_definition::FormulaDefinition::Globally(
                $crate::stl::core::TimeInterval {
                    start: std::time::Duration::from_secs($start as u64),
                    end: std::time::Duration::from_secs($end as u64),
                },
                Box::new($crate::stl!($($sub)+))
            )
        };
        // Alias for G
        (globally [$start:expr, $end:expr] ($($sub:tt)+) ) => {
            $crate::stl!(G [$start, $end] ($($sub)+))
        };

        (F [$start:expr, $end:expr] ($($sub:tt)+) ) => {
            $crate::stl::formula_definition::FormulaDefinition::Eventually(
                $crate::stl::core::TimeInterval {
                    start: std::time::Duration::from_secs($start as u64),
                    end: std::time::Duration::from_secs($end as u64),
                },
                Box::new($crate::stl!($($sub)+))
            )
        };
        // Alias for F
        (eventually [$start:expr, $end:expr] ($($sub:tt)+) ) => {
            $crate::stl!(F [$start, $end] ($($sub)+))
        };

        // --- BINARY OPERATORS (Infix) ---
        ( ($($left:tt)+) && ($($right:tt)+) ) => {
            $crate::stl::formula_definition::FormulaDefinition::And(
                Box::new($crate::stl!($($left)+)),
                Box::new($crate::stl!($($right)+))
            )
        };
        // Alias for &&
        ( ($($left:tt)+) and ($($right:tt)+) ) => {
            $crate::stl!(($($left)+) && ($($right)+))
        };

        ( ($($left:tt)+) || ($($right:tt)+) ) => {
            $crate::stl::formula_definition::FormulaDefinition::Or(
                Box::new($crate::stl!($($left)+)),
                Box::new($crate::stl!($($right)+))
            )
        };
        // Alias for ||
        ( ($($left:tt)+) or ($($right:tt)+) ) => {
            $crate::stl!(($($left)+) || ($($right)+))
        };

        ( ($($left:tt)+) -> ($($right:tt)+) ) => {
            $crate::stl::formula_definition::FormulaDefinition::Implies(
                Box::new($crate::stl!($($left)+)),
                Box::new($crate::stl!($($right)+))
            )
        };
        // Alias for ->
        ( ($($left:tt)+) implies ($($right:tt)+) ) => {
            $crate::stl!(($($left)+) -> ($($right)+))
        };

        ( ($($left:tt)+) U [$start:expr, $end:expr] ($($right:tt)+) ) => {
            $crate::stl::formula_definition::FormulaDefinition::Until(
                $crate::stl::core::TimeInterval {
                    start: std::time::Duration::from_secs($start as u64),
                    end: std::time::Duration::from_secs($end as u64),
                },
                Box::new($crate::stl!($($left)+)),
                Box::new($crate::stl!($($right)+))
            )
        };
        // Alias for U
        ( ($($left:tt)+) until [$start:expr, $end:expr] ($($right:tt)+) ) => {
            $crate::stl!(($($left)+) U [$start, $end] ($($right)+))
        };

        // --- PARENTHESES ---
        ( ( $($sub:tt)+ ) ) => {
            $crate::stl!($($sub)+)
        };

        // --- INTERPOLATION / FALLBACK ---
        // This catches variables, function calls, or blocks that return a FormulaDefinition.
        // IMPORTANT: This must be the LAST rule to avoid shadowing the DSL syntax.
        ($e:expr) => {
            $e
        };
    }

#[cfg(test)]
mod tests {
    use crate::stl::formula_definition::FormulaDefinition;
    #[test]
    fn test_stl_macro() {
        let formula: FormulaDefinition = stl!(
            G[0,5]((signal>5) and ((x>0)U[0,2](true)))
        );
        println!("{}", formula.to_tree_string(2))
    }
    #[test]
    fn test_stl_macro_aliases() {
        let formula: FormulaDefinition = stl! {
            globally[0,5]((signal>5) and ((x>0)until[0,2](true)))
        };
        println!("{}", formula.to_tree_string(2))
    }
    #[test]
    fn test_stl_macro_2() {
        let formula: FormulaDefinition = stl! {
            (eventually [0, 2] (x > 5)) and (globally [0, 2] (x > 0))
        };
        println!("{}", formula.to_tree_string(2))
    }
    #[test]
    fn test_stl_macro_2_aliases() {
        let formula: FormulaDefinition = stl! {
           (F [0, 2] (x > 5)) && (G[0, 2] (x > 0))
        };
        println!("{}", formula.to_tree_string(2))
    }
    #[test]
    fn test_stl_object_in_stl_macro() {
        let formula_1: FormulaDefinition = stl! {
            (eventually [0, 2] (x > 5)) and (globally [0, 2] (x > 0))
        };
        let formula_2: FormulaDefinition = stl! {
            (formula_1) or (false)
        };
        println!("{}", formula_2.to_tree_string(2))
    }
}
