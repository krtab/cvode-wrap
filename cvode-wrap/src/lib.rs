use std::convert::TryInto;
use std::{ffi::c_void, intrinsics::transmute, os::raw::c_int, ptr::NonNull};

use cvode::SUNMatrix;
use cvode_5_sys::{
    cvode::{self, realtype, SUNLinearSolver},
    nvector_serial::N_VGetArrayPointer,
};

mod nvector;
pub use nvector::NVectorSerial;

pub type F = realtype;
pub type CVector = cvode::N_Vector;

#[repr(u32)]
#[derive(Debug)]
pub enum LinearMultistepMethod {
    ADAMS = cvode::CV_ADAMS,
    BDF = cvode::CV_BDF,
}

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

pub struct Solver<const N: usize> {
    mem: CvodeMemoryBlockNonNullPtr,
    _y0: NVectorSerial<N>,
    _sunmatrix: SUNMatrix,
    _linsolver: SUNLinearSolver,
}

pub enum RhsResult {
    Ok,
    RecoverableError(u8),
    NonRecoverableError(u8),
}

type RhsF<const N: usize> = fn(F, &[F; N], &mut [F; N], *mut c_void) -> RhsResult;

pub fn wrap_f<const N: usize>(
    f: RhsF<N>,
    t: F,
    y: CVector,
    ydot: CVector,
    data: *mut c_void,
) -> c_int {
    let y = unsafe { transmute(N_VGetArrayPointer(y as _)) };
    let ydot = unsafe { transmute(N_VGetArrayPointer(ydot as _)) };
    let res = f(t, y, ydot, data);
    match res {
        RhsResult::Ok => 0,
        RhsResult::RecoverableError(e) => e as c_int,
        RhsResult::NonRecoverableError(e) => -(e as c_int),
    }
}

#[macro_export]
macro_rules! wrap {
    ($wrapped_f_name: ident, $f_name: ident) => {
        extern "C" fn $wrapped_f_name(
            t: F,
            y: CVector,
            ydot: CVector,
            data: *mut std::ffi::c_void,
        ) -> std::os::raw::c_int {
            wrap_f($f_name, t, y, ydot, data)
        }
    };
}

type RhsFCtype = extern "C" fn(F, CVector, CVector, *mut c_void) -> c_int;

#[repr(u32)]
pub enum StepKind {
    Normal = cvode::CV_NORMAL,
    OneStep = cvode::CV_ONE_STEP,
}

#[derive(Debug)]
pub enum Error {
    NullPointerError { func_id: &'static str },
    ErrorCode { func_id: &'static str, flag: c_int },
}

pub type Result<T> = std::result::Result<T, Error>;

fn check_non_null<T>(ptr: *mut T, func_id: &'static str) -> Result<NonNull<T>> {
    NonNull::new(ptr).ok_or_else(|| Error::NullPointerError { func_id })
}

fn check_flag_is_succes(flag: c_int, func_id: &'static str) -> Result<()> {
    if flag == cvode::CV_SUCCESS as i32 {
        Ok(())
    } else {
        Err(Error::ErrorCode { flag, func_id })
    }
}

impl<const N: usize> Solver<N> {
    pub fn new(
        method: LinearMultistepMethod,
        f: RhsFCtype,
        t0: F,
        y0: &[F; N],
        atol: F,
        rtol: F,
    ) -> Result<Self> {
        assert_eq!(y0.len(), N);
        let mem_maybenull = unsafe { cvode::CVodeCreate(method as c_int) };
        let mem: CvodeMemoryBlockNonNullPtr =
            check_non_null(mem_maybenull as *mut CvodeMemoryBlock, "CVodeCreate")?.into();
        let y0 = NVectorSerial::new_from(y0);
        let flag = unsafe { cvode::CVodeInit(mem.as_raw(), Some(f), t0, y0.as_raw() as _) };
        check_flag_is_succes(flag, "CVodeInit")?;
        let flag = unsafe { cvode::CVodeSStolerances(mem.as_raw(), atol, rtol) };
        check_flag_is_succes(flag, "CVodeSStolerances")?;
        let matrix = unsafe {
            cvode_5_sys::sunmatrix_dense::SUNDenseMatrix(
                N.try_into().unwrap(),
                N.try_into().unwrap(),
            )
        };
        check_non_null(matrix, "SUNDenseMatrix")?;
        let linsolver = unsafe {
            cvode_5_sys::sunlinsol_dense::SUNDenseLinearSolver(y0.as_raw() as _, matrix as _)
        };
        check_non_null(linsolver, "SUNDenseLinearSolver")?;
        let flag =
            unsafe { cvode::CVodeSetLinearSolver(mem.as_raw(), linsolver as _, matrix as _) };
        check_flag_is_succes(flag, "CVodeSetLinearSolver")?;
        Ok(Solver {
            mem,
            _y0: y0,
            _sunmatrix: matrix as _,
            _linsolver: linsolver as _,
        })
    }

    pub fn step(&mut self, tout: F, step_kind: StepKind) -> Result<(F, &[F; N])> {
        let mut tret = 0.;
        let flag = unsafe {
            cvode::CVode(
                self.mem.as_raw(),
                tout,
                self._y0.as_raw() as _,
                &mut tret,
                step_kind as c_int,
            )
        };
        check_flag_is_succes(flag, "CVode")?;
        Ok((tret, self._y0.as_ref()))
    }
}

impl<const N: usize> Drop for Solver<N> {
    fn drop(&mut self) {
        unsafe { cvode::CVodeFree(&mut self.mem.as_raw()) }
        unsafe { cvode::SUNLinSolFree(self._linsolver) };
        unsafe { cvode::SUNMatDestroy(self._sunmatrix) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(_t: super::F, y: &[F; 2], ydot: &mut [F; 2], _data: *mut c_void) -> RhsResult {
        *ydot = [y[1], -y[0]];
        RhsResult::Ok
    }

    wrap!(wrapped_f, f);

    #[test]
    fn create() {
        let y0 = [0., 1.];
        let _solver = Solver::new(LinearMultistepMethod::ADAMS, wrapped_f, 0., &y0, 1e-4, 1e-4);
    }
}
