A wrapper around the sundials ODE solver.

# Example

An oscillatory 2-D system.

```rust
use cvode_wrap::*;

let y0 = [0., 1.];
// define the right-hand-side as a rust function of type RhsF<Realtype, 2>
fn f(
  _t: Realtype,
   y: &[Realtype; 2],
   ydot: &mut [Realtype; 2],
   k: &Realtype,
) -> RhsResult {
    *ydot = [y[1], -y[0] * k];
    RhsResult::Ok
}
// Use the `wrap!` macro to define a `wrapped_f` function callable
// from C (of type `c_wrapping::RhsFCtype`) that wraps `f`.
wrap!(wrapped_f, f, Realtype, 2);
//initialize the solver
let mut solver = Solver::new(
    LinearMultistepMethod::Adams,
    wrapped_f,
    0.,
    &y0,
    1e-4,
    AbsTolerance::scalar(1e-4),
    1e-2,
)
.unwrap();
//and solve
let ts: Vec<_> = (1..100).collect();
println!("0,{},{}", y0[0], y0[1]);
for &t in &ts {
    let (_tret, &[x, xdot]) = solver.step(t as _, StepKind::Normal).unwrap();
    println!("{},{},{}", t, x, xdot);
}
```