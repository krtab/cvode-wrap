use std::ffi::c_void;

use cvode_wrap::*;

fn main() {
    let y0 = [0., 1.];
    //define the right-hand-side
    fn f(_t: F, y: &[F; 2], ydot: &mut [F; 2], _data: *mut c_void) -> RhsResult {
        *ydot = [y[1], -y[0] / 10.];
        RhsResult::Ok
    }
    wrap!(wrapped_f, f);
    //initialize the solver
    let mut solver =
        Solver::new(LinearMultistepMethod::ADAMS, wrapped_f, 0., &y0, 1e-4, 1e-4).unwrap();
    //and solve
    let ts: Vec<_> = (1..100).collect();
    println!("0,{},{}", y0[0], y0[1]);
    for &t in &ts {
        let (_tret, &[x, xdot]) = solver.step(t as _, StepKind::Normal).unwrap();
        println!("{},{},{}", t, x, xdot);
    }
}
