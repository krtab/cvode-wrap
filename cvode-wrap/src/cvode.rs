use std::{convert::TryInto, ffi::c_void, os::raw::c_int, pin::Pin, ptr::NonNull};

use cvode_5_sys::{SUNLinearSolver, SUNMatrix};

use crate::{
    check_flag_is_succes, check_non_null, LinearMultistepMethod, NVectorSerial,
    NVectorSerialHeapAllocated, Realtype, Result, RhsResult, StepKind, WrappingUserData,
};

#[repr(C)]
struct CvodeMemoryBlock {
    _private: [u8; 0],
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
struct CvodeMemoryBlockNonNullPtr {
    ptr: NonNull<CvodeMemoryBlock>,
}

impl CvodeMemoryBlockNonNullPtr {
    fn new(ptr: NonNull<CvodeMemoryBlock>) -> Self {
        Self { ptr }
    }

    fn as_raw(self) -> *mut c_void {
        self.ptr.as_ptr() as *mut c_void
    }
}

impl From<NonNull<CvodeMemoryBlock>> for CvodeMemoryBlockNonNullPtr {
    fn from(x: NonNull<CvodeMemoryBlock>) -> Self {
        Self::new(x)
    }
}

/// An enum representing the choice between a scalar or vector absolute tolerance
pub enum AbsTolerance<const SIZE: usize> {
    Scalar(Realtype),
    Vector(NVectorSerialHeapAllocated<SIZE>),
}

impl<const SIZE: usize> AbsTolerance<SIZE> {
    pub fn scalar(atol: Realtype) -> Self {
        AbsTolerance::Scalar(atol)
    }

    pub fn vector(atol: &[Realtype; SIZE]) -> Self {
        let atol = NVectorSerialHeapAllocated::new_from(atol);
        AbsTolerance::Vector(atol)
    }
}

/// The main struct of the crate. Wraps a sundials solver.
///
/// Args
/// ----
/// `UserData` is the type of the supplementary arguments for the
/// right-hand-side. If unused, should be `()`.
///
/// `N` is the "problem size", that is the dimension of the state space.
///
/// See [crate-level](`crate`) documentation for more.
pub struct Solver<UserData, F, const N: usize> {
    mem: CvodeMemoryBlockNonNullPtr,
    y0: NVectorSerialHeapAllocated<N>,
    sunmatrix: SUNMatrix,
    linsolver: SUNLinearSolver,
    atol: AbsTolerance<N>,
    user_data: Pin<Box<WrappingUserData<UserData, F>>>,
}

/// The wrapping function.
///
/// Internally used in [`wrap`].
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
    pub fn new(
        method: LinearMultistepMethod,
        f: F,
        t0: Realtype,
        y0: &[Realtype; N],
        rtol: Realtype,
        atol: AbsTolerance<N>,
        user_data: UserData,
    ) -> Result<Self> {
        assert_eq!(y0.len(), N);
        let mem: CvodeMemoryBlockNonNullPtr = {
            let mem_maybenull = unsafe { cvode_5_sys::CVodeCreate(method as c_int) };
            check_non_null(mem_maybenull as *mut CvodeMemoryBlock, "CVodeCreate")?.into()
        };
        let y0 = NVectorSerialHeapAllocated::new_from(y0);
        let matrix = {
            let matrix = unsafe {
                cvode_5_sys::SUNDenseMatrix(N.try_into().unwrap(), N.try_into().unwrap())
            };
            check_non_null(matrix, "SUNDenseMatrix")?
        };
        let linsolver = {
            let linsolver = unsafe { cvode_5_sys::SUNLinSol_Dense(y0.as_raw(), matrix.as_ptr()) };
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
        };
        {
            let fn_ptr = wrap_f::<UserData, F, N> as extern "C" fn(_, _, _, _) -> _;
            let flag = unsafe {
                cvode_5_sys::CVodeInit(
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
                let flag = unsafe { cvode_5_sys::CVodeSStolerances(mem.as_raw(), rtol, atol) };
                check_flag_is_succes(flag, "CVodeSStolerances")?;
            }
            AbsTolerance::Vector(atol) => {
                let flag =
                    unsafe { cvode_5_sys::CVodeSVtolerances(mem.as_raw(), rtol, atol.as_raw()) };
                check_flag_is_succes(flag, "CVodeSVtolerances")?;
            }
        }
        {
            let flag = unsafe {
                cvode_5_sys::CVodeSetLinearSolver(mem.as_raw(), linsolver.as_ptr(), matrix.as_ptr())
            };
            check_flag_is_succes(flag, "CVodeSetLinearSolver")?;
        }
        {
            let flag = unsafe {
                cvode_5_sys::CVodeSetUserData(
                    mem.as_raw(),
                    std::mem::transmute(res.user_data.as_ref().get_ref()),
                )
            };
            check_flag_is_succes(flag, "CVodeSetUserData")?;
        }
        Ok(res)
    }

    pub fn step(
        &mut self,
        tout: Realtype,
        step_kind: StepKind,
    ) -> Result<(Realtype, &[Realtype; N])> {
        let mut tret = 0.;
        let flag = unsafe {
            cvode_5_sys::CVode(
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
        unsafe { cvode_5_sys::CVodeFree(&mut self.mem.as_raw()) }
        unsafe { cvode_5_sys::SUNLinSolFree(self.linsolver) };
        unsafe { cvode_5_sys::SUNMatDestroy(self.sunmatrix) };
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
        );
    }
}
