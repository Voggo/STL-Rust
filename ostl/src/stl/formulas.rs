use crate::stl::formula_definition::FormulaDefinition;
use crate::stl::core::TimeInterval;
use std::time::Duration;
use crate::stl;

/// Returns the vector of Signal Temporal Logic formulas.
/// If `ids` is not empty, returns only the formulas with the specified IDs.
pub fn get_formulas(ids: &[usize]) -> Vec<(usize, FormulaDefinition)> {
    let mut formulas: Vec<(usize, FormulaDefinition)> = Vec::new();

    // // --- Basic Formulas (Lines 1-12) ---
    formulas.push((1, stl!((x < 0.5) and (x > -0.5))));
    formulas.push((2, stl!((x < 0.5) or (x > -0.5))));
    formulas.push((3, stl!(not (x < 0.5))));

    // // 4-6. Globally (Always) 
    formulas.push((4, stl!(alw[0, 10] (x < 0.5))));
    formulas.push((5, stl!(alw[0, 100] (x < 0.5))));
    formulas.push((6, stl!(alw[0, 1000] (x < 0.5))));

    // 7-9. Eventually 
    formulas.push((7, stl!(ev[0, 10] (x < 0.5))));
    formulas.push((8, stl!(ev[0, 100] (x < 0.5))));
    formulas.push((9, stl!(ev[0, 1000] (x < 0.5))));

    // 10-12. Until 
    formulas.push((10, stl!((x < 0.5) until[0, 10] (x > -0.5))));
    formulas.push((11, stl!((x < 0.5) until[0, 100] (x > -0.5))));
    formulas.push((12, stl!((x < 0.5) until[0, 1000] (x > -0.5))));

    // --- Complex Nested Formulas (Lines 13-21) ---
    let zero_ten = TimeInterval { start: Duration::from_secs(0), end: Duration::from_secs(10) };

    // Pattern A
    let make_and_ev_chain = |depth: usize| -> FormulaDefinition {
        let mut curr = stl!(x > 0.0); // Base case
        for _ in 0..depth {
            curr = FormulaDefinition::And(
                Box::new(FormulaDefinition::Eventually(
                    zero_ten.clone(),
                    Box::new(curr.clone()),
                )),
                Box::new(FormulaDefinition::Eventually(
                    zero_ten.clone(),
                    Box::new(curr.clone()),
                )),
            );
        }
        curr
    };

    let next_id_start = 13usize;
    formulas.push((next_id_start, make_and_ev_chain(2))); 
    formulas.push((next_id_start + 1, make_and_ev_chain(3))); 
    formulas.push((next_id_start + 2, make_and_ev_chain(5))); 

    // Pattern B
    let make_ev_alw_chain = |depth: usize| -> FormulaDefinition {
        let mut curr = stl!(x > 0.0); // Base case
        for _ in 0..depth {
            let alw_layer = FormulaDefinition::Globally(zero_ten.clone(), Box::new(curr));
            curr = FormulaDefinition::Eventually(zero_ten.clone(), Box::new(alw_layer));
        }
        curr
    };

    formulas.push((next_id_start + 3, make_ev_alw_chain(2))); 
    formulas.push((next_id_start + 4, make_ev_alw_chain(3))); 
    formulas.push((next_id_start + 5, make_ev_alw_chain(5)));  

    // Pattern C
    let make_until_chain = |depth: usize| -> FormulaDefinition {
        let mut curr = stl!(x > 0.0); // Base case
        for _ in 0..depth {
            curr = FormulaDefinition::Until(
                zero_ten.clone(),
                Box::new(curr.clone()),
                Box::new(curr),
            );
        }
        curr
    };

    formulas.push((next_id_start + 6, make_until_chain(2)));
    formulas.push((next_id_start + 7, make_until_chain(3))); 
    formulas.push((next_id_start + 8, make_until_chain(5))); 

    if ids.is_empty() {
        formulas
    } else {
        formulas.into_iter().filter(|(id, _)| ids.contains(id)).collect()
    }
}
