A wrapper around the cvode(S) ODE solver from sundials.

[ [documentation](https://docs.rs/cvode-wrap) ] [ [lib.rs](https://lib.rs/crates/cvode-wrap) ] [ [git repository](https://gitlab.inria.fr/InBio/Public/cvode-rust-wrap) ]

# Examples

## Oscillator

An oscillatory system defined by `x'' = -k * x`.

### Without sensitivities

```rust
let y0 = [0., 1.];
//define the right-hand-side
fn f(_t: Realtype, y: &[Realtype; 2], ydot: &mut [Realtype; 2], k: &Realtype) -> RhsResult {
    *ydot = [y[1], -y[0] * k];
    RhsResult::Ok
}
//initialize the solver
let mut solver = SolverNoSensi::new(
    LinearMultistepMethod::Adams,
    f,
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

### With sensitivities

The sensitivities are computed with respect to `x(0)`, `x'(0)` and `k`.

```rust
let y0 = [0., 1.];
//define the right-hand-side
fn f(_t: Realtype, y: &[Realtype; 2], ydot: &mut [Realtype; 2], k: &Realtype) -> RhsResult {
    *ydot = [y[1], -y[0] * k];
    RhsResult::Ok
}
//define the sensitivity function for the right hand side
fn fs(
    _t: Realtype,
    y: &[Realtype; 2],
    _ydot: &[Realtype; 2],
    ys: [&[Realtype; 2]; N_SENSI],
    ysdot: [&mut [Realtype; 2]; N_SENSI],
    k: &Realtype,
) -> RhsResult {
    // Mind that when indexing sensitivities, the first index
    // is the parameter index, and the second the state variable
    // index
    *ysdot[0] = [ys[0][1], -ys[0][0] * k];
    *ysdot[1] = [ys[1][1], -ys[1][0] * k];
    *ysdot[2] = [ys[2][1], -ys[2][0] * k - y[0]];
    RhsResult::Ok
}

const N_SENSI: usize = 3;

// the sensitivities in order are d/dy0[0], d/dy0[1] and d/dk
let ys0 = [[1., 0.], [0., 1.], [0., 0.]];

//initialize the solver
let mut solver = SolverSensi::new(
    LinearMultistepMethod::Adams,
    f,
    fs,
    0.,
    &y0,
    &ys0,
    1e-4,
    AbsTolerance::scalar(1e-4),
    SensiAbsTolerance::scalar([1e-4; N_SENSI]),
    1e-2,
)
.unwrap();
//and solve
let ts: Vec<_> = (1..100).collect();
println!("0,{},{}", y0[0], y0[1]);
for &t in &ts {
    let (_tret, &[x, xdot], [&[dy0_dy00, dy1_dy00], &[dy0_dy01, dy1_dy01], &[dy0_dk, dy1_dk]]) =
        solver.step(t as _, StepKind::Normal).unwrap();
    println!(
        "{},{},{},{},{},{},{},{},{}",
        t, x, xdot, dy0_dy00, dy1_dy00, dy0_dy01, dy1_dy01, dy0_dk, dy1_dk
    );
}
```