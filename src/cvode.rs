//! Wrapper around cvode, without sensitivities

use std::{convert::TryInto, os::raw::c_int, pin::Pin};

use sundials_sys::{SUNComm, SUNContext, SUNLinearSolver, SUNMatrix};

use crate::{
    check_flag_is_succes, check_non_null, sundials_create_context, sundials_free_context,
    AbsTolerance, CvodeMemoryBlock, CvodeMemoryBlockNonNullPtr, LinearMultistepMethod,
    NVectorSerial, NVectorSerialHeapAllocated, Realtype, Result, RhsResult, StepKind,
};

struct WrappingUserData<UserData, F> {
    actual_user_data: UserData,
    f: F,
}

/// The ODE solver without sensitivities.
///
/// # Type Arguments
///
/// - `F` is the type of the right-hand side function
///
///  - `UserData` is the type of the supplementary arguments for the
/// right-hand-side. If unused, should be `()`.
///
/// - `N` is the "problem size", that is the dimension of the state space.
pub struct Solver<UserData, F, const N: usize> {
    mem: CvodeMemoryBlockNonNullPtr,
    y0: NVectorSerialHeapAllocated<N>,
    sunmatrix: SUNMatrix,
    linsolver: SUNLinearSolver,
    atol: AbsTolerance<N>,
    user_data: Pin<Box<WrappingUserData<UserData, F>>>,
    context: SUNContext,
}

extern "C" fn wrap_f<UserData, F, const N: usize>(
    t: Realtype,
    y: *const NVectorSerial<N>,
    ydot: *mut NVectorSerial<N>,
    data: *const WrappingUserData<UserData, F>,
) -> c_int
where
    F: Fn(Realtype, &[Realtype; N], &mut [Realtype; N], &UserData) -> RhsResult,
{
    let y = unsafe { &*y }.as_slice();
    let ydot = unsafe { &mut *ydot }.as_slice_mut();
    let WrappingUserData {
        actual_user_data: data,
        f,
    } = unsafe { &*data };
    let res = f(t, y, ydot, data);
    match res {
        RhsResult::Ok => 0,
        RhsResult::RecoverableError(e) => e as c_int,
        RhsResult::NonRecoverableError(e) => -(e as c_int),
    }
}

impl<UserData, F, const N: usize> Solver<UserData, F, N>
where
    F: Fn(Realtype, &[Realtype; N], &mut [Realtype; N], &UserData) -> RhsResult,
{
    /// Create a new solver.
    pub fn new(
        method: LinearMultistepMethod,
        f: F,
        t0: Realtype,
        y0: &[Realtype; N],
        rtol: Realtype,
        atol: AbsTolerance<N>,
        user_data: UserData,
    ) -> Result<Self> {
        // Create context, required from version 6 and above on
        let context = sundials_create_context()?;

        assert_eq!(y0.len(), N);
        let mem: CvodeMemoryBlockNonNullPtr = {
            let mem_maybenull = unsafe { sundials_sys::CVodeCreate(method as c_int, context) };
            check_non_null(mem_maybenull as *mut CvodeMemoryBlock, "CVodeCreate")?.into()
        };
        let y0 = NVectorSerialHeapAllocated::new_from(y0, context);
        let matrix = {
            let matrix = unsafe {
                sundials_sys::SUNDenseMatrix(N.try_into().unwrap(), N.try_into().unwrap(), context)
            };
            check_non_null(matrix, "SUNDenseMatrix")?
        };
        let linsolver = {
            let linsolver =
                unsafe { sundials_sys::SUNLinSol_Dense(y0.as_raw(), matrix.as_ptr(), context) };
            check_non_null(linsolver, "SUNDenseLinearSolver")?
        };
        let user_data = Box::pin(WrappingUserData {
            actual_user_data: user_data,
            f,
        });
        let res = Solver {
            mem,
            y0,
            sunmatrix: matrix.as_ptr(),
            linsolver: linsolver.as_ptr(),
            atol,
            user_data,
            context,
        };
        {
            let fn_ptr = wrap_f::<UserData, F, N> as extern "C" fn(_, _, _, _) -> _;
            let flag = unsafe {
                sundials_sys::CVodeInit(
                    mem.as_raw(),
                    Some(std::mem::transmute(fn_ptr)),
                    t0,
                    res.y0.as_raw(),
                )
            };
            check_flag_is_succes(flag, "CVodeInit")?;
        }
        match &res.atol {
            &AbsTolerance::Scalar(atol) => {
                let flag = unsafe { sundials_sys::CVodeSStolerances(mem.as_raw(), rtol, atol) };
                check_flag_is_succes(flag, "CVodeSStolerances")?;
            }
            AbsTolerance::Vector(atol) => {
                let flag =
                    unsafe { sundials_sys::CVodeSVtolerances(mem.as_raw(), rtol, atol.as_raw()) };
                check_flag_is_succes(flag, "CVodeSVtolerances")?;
            }
        }
        {
            let flag = unsafe {
                sundials_sys::CVodeSetLinearSolver(
                    mem.as_raw(),
                    linsolver.as_ptr(),
                    matrix.as_ptr(),
                )
            };
            check_flag_is_succes(flag, "CVodeSetLinearSolver")?;
        }
        {
            let flag = unsafe {
                sundials_sys::CVodeSetUserData(
                    mem.as_raw(),
                    std::mem::transmute(res.user_data.as_ref().get_ref()),
                )
            };
            check_flag_is_succes(flag, "CVodeSetUserData")?;
        }
        Ok(res)
    }

    /// Takes a step according to `step_kind` (see [`StepKind`]).
    ///
    /// Returns a tuple `(t_out,&y(t_out))` where `t_out` is the time
    /// reached by the solver as dictated by `step_kind`, and `y(t_out)` is an
    /// array of the state variables at that time.
    pub fn step(
        &mut self,
        tout: Realtype,
        step_kind: StepKind,
    ) -> Result<(Realtype, &[Realtype; N])> {
        let mut tret = 0.;
        let flag = unsafe {
            sundials_sys::CVode(
                self.mem.as_raw(),
                tout,
                self.y0.as_raw(),
                &mut tret,
                step_kind as c_int,
            )
        };
        check_flag_is_succes(flag, "CVode")?;
        Ok((tret, self.y0.as_slice()))
    }
}

impl<UserData, F, const N: usize> Drop for Solver<UserData, F, N> {
    fn drop(&mut self) {
        unsafe { sundials_sys::CVodeFree(&mut self.mem.as_raw()) }
        unsafe { sundials_sys::SUNLinSolFree(self.linsolver) };
        unsafe { sundials_sys::SUNMatDestroy(self.sunmatrix) };
        // We cannot do anything if this fails. So we ignore the output
        let _ = sundials_free_context(self.context);
    }
}

#[cfg(test)]
mod tests {
    use crate::RhsResult;

    use super::*;

    fn f(
        _t: super::Realtype,
        y: &[Realtype; 2],
        ydot: &mut [Realtype; 2],
        _data: &(),
    ) -> RhsResult {
        *ydot = [y[1], -y[0]];
        RhsResult::Ok
    }

    #[test]
    fn create() {
        let y0 = [0., 1.];
        let _solver = Solver::new(
            LinearMultistepMethod::Adams,
            f,
            0.,
            &y0,
            1e-4,
            AbsTolerance::Scalar(1e-4),
            (),
        )
        .unwrap();
    }
}
