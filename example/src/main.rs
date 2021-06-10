use std::env::args;

use cvode_wrap::*;

fn main() {
    let y0 = [0., 1.];
    //define the right-hand-side
    fn f(_t: Realtype, y: &[Realtype; 2], ydot: &mut [Realtype; 2], k: &Realtype) -> RhsResult {
        *ydot = [y[1], -y[0] * k];
        RhsResult::Ok
    }
    // If there is any command line argument compute the sensitivities, else don't.
    if args().nth(1).is_none() {
        //initialize the solver
        let mut solver = cvode::Solver::new(
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
    } else {
        const N_SENSI: usize = 3;
        // the sensitivities in order are d/dy0[0], d/dy0[1] and d/dk
        let ys0 = [[1., 0.], [0., 1.], [0., 0.]];

        fn fs(
            _t: Realtype,
            y: &[Realtype; 2],
            _ydot: &[Realtype; 2],
            ys: [&[Realtype; 2]; N_SENSI],
            ysdot: [&mut [Realtype; 2]; N_SENSI],
            k: &Realtype,
        ) -> RhsResult {
            *ysdot[0] = [ys[0][1], -ys[0][0] * k];
            *ysdot[1] = [ys[1][1], -ys[1][0] * k];
            *ysdot[2] = [ys[2][1], -ys[2][0] * k - y[0]];
            RhsResult::Ok
        }

        //initialize the solver
        let mut solver = cvode_sens::Solver::new(
            LinearMultistepMethod::Adams,
            f,
            fs,
            0.,
            &y0,
            &ys0,
            1e-4,
            AbsTolerance::scalar(1e-4),
            cvode_sens::SensiAbsTolerance::scalar([1e-4; N_SENSI]),
            1e-2,
        )
        .unwrap();
        //and solve
        let ts: Vec<_> = (1..100).collect();
        println!("0,{},{}", y0[0], y0[1]);
        for &t in &ts {
            let (
                _tret,
                &[x, xdot],
                [&[dy0_dy00, dy1_dy00], &[dy0_dy01, dy1_dy01], &[dy0_dk, dy1_dk]],
            ) = solver.step(t as _, StepKind::Normal).unwrap();
            println!(
                "{},{},{},{},{},{},{},{},{}",
                t, x, xdot, dy0_dy00, dy1_dy00, dy0_dy01, dy1_dy01, dy0_dk, dy1_dk
            );
        }
    }
}
