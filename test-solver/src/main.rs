use std::ffi::c_void;

use cvode_wrap::*;

fn main() {
    let y0 = [0., 1.];
    //define the right-hand-side
    fn f(_t: F, y: &[F; 2], ydot: &mut [F; 2], k: &F) -> RhsResult {
        *ydot = [y[1], -y[0] * k];
        RhsResult::Ok
    }
    wrap!(wrapped_f, f, F);
    //initialize the solver
    let mut solver = Solver::new(
        LinearMultistepMethod::ADAMS,
        wrapped_f,
        0.,
        &y0,
        1e-4,
        AbsTolerance::scalar(1e-4),
        1e-2
    )
    .unwrap();
    //and solve
    let ts: Vec<_> = (1..100).collect();
    println!("0,{},{}", y0[0], y0[1]);
    for &t in &ts {
        let (_tret, &[x, xdot]) = solver.step(t as _, StepKind::Normal).unwrap();
        println!("{},{},{}", t, x, xdot);
    }
}
