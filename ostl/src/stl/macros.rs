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
        (alw [$start:expr, $end:expr] ($($sub:tt)+) ) => {
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
        (ev [$start:expr, $end:expr] ($($sub:tt)+) ) => {
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
    }

#[cfg(test)]
mod tests {
    use crate::stl::formula_definition::FormulaDefinition;
    #[test]
    fn test_stl_macro() {
        let _: FormulaDefinition = stl! {
            G[0,5]((signal>5) and ((x>0)U[0,2](true)))
        };
    }
}