use std::{convert::TryInto, ffi::c_void, os::raw::c_int, pin::Pin, ptr::NonNull};

use sundials_sys::{SUNLinearSolver, SUNMatrix, CV_STAGGERED};

use crate::{
    check_flag_is_succes, check_non_null, AbsTolerance, LinearMultistepMethod, NVectorSerial,
    NVectorSerialHeapAllocated, Realtype, Result, RhsResult, StepKind,
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

pub enum SensiAbsTolerance<const SIZE: usize, const N_SENSI: usize> {
    Scalar([Realtype; N_SENSI]),
    Vector([NVectorSerialHeapAllocated<SIZE>; N_SENSI]),
}

impl<const SIZE: usize, const N_SENSI: usize> SensiAbsTolerance<SIZE, N_SENSI> {
    pub fn scalar(atol: [Realtype; N_SENSI]) -> Self {
        SensiAbsTolerance::Scalar(atol)
    }

    pub fn vector(atol: &[[Realtype; SIZE]; N_SENSI]) -> Self {
        SensiAbsTolerance::Vector(
            array_init::from_iter(
                atol.iter()
                    .map(|arr| NVectorSerialHeapAllocated::new_from(arr)),
            )
            .unwrap(),
        )
    }
}

impl From<NonNull<CvodeMemoryBlock>> for CvodeMemoryBlockNonNullPtr {
    fn from(x: NonNull<CvodeMemoryBlock>) -> Self {
        Self::new(x)
    }
}

struct WrappingUserData<UserData, F, FS> {
    actual_user_data: UserData,
    f: F,
    fs: FS,
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
pub struct Solver<UserData, F, FS, const N: usize, const N_SENSI: usize> {
    mem: CvodeMemoryBlockNonNullPtr,
    y0: NVectorSerialHeapAllocated<N>,
    y_s0: Box<[NVectorSerialHeapAllocated<N>; N_SENSI]>,
    sunmatrix: SUNMatrix,
    linsolver: SUNLinearSolver,
    atol: AbsTolerance<N>,
    atol_sens: SensiAbsTolerance<N, N_SENSI>,
    user_data: Pin<Box<WrappingUserData<UserData, F, FS>>>,
    sensi_out_buffer: [NVectorSerialHeapAllocated<N>; N_SENSI],
}

/// The wrapping function.
///
/// Internally used in [`wrap`].
extern "C" fn wrap_f<UserData, F, FS, const N: usize>(
    t: Realtype,
    y: *const NVectorSerial<N>,
    ydot: *mut NVectorSerial<N>,
    data: *const WrappingUserData<UserData, F, FS>,
) -> c_int
where
    F: Fn(Realtype, &[Realtype; N], &mut [Realtype; N], &UserData) -> RhsResult,
{
    let y = unsafe { &*y }.as_slice();
    let ydot = unsafe { &mut *ydot }.as_slice_mut();
    let WrappingUserData {
        actual_user_data: data,
        f,
        ..
    } = unsafe { &*data };
    let res = f(t, y, ydot, data);
    match res {
        RhsResult::Ok => 0,
        RhsResult::RecoverableError(e) => e as c_int,
        RhsResult::NonRecoverableError(e) => -(e as c_int),
    }
}

extern "C" fn wrap_f_sens<UserData, F, FS, const N: usize, const N_SENSI: usize>(
    _n_s: c_int,
    t: Realtype,
    y: *const NVectorSerial<N>,
    ydot: *const NVectorSerial<N>,
    y_s: *const [*const NVectorSerial<N>; N_SENSI],
    y_sdot: *mut [*mut NVectorSerial<N>; N_SENSI],
    data: *const WrappingUserData<UserData, F, FS>,
    _tmp1: *const NVectorSerial<N>,
    _tmp2: *const NVectorSerial<N>,
) -> c_int
where
    FS: Fn(
        Realtype,
        &[Realtype; N],
        &[Realtype; N],
        [&[Realtype; N]; N_SENSI],
        [&mut [Realtype; N]; N_SENSI],
        &UserData,
    ) -> RhsResult,
{
    let y = unsafe { &*y }.as_slice();
    let ydot = unsafe { &*ydot }.as_slice();
    let y_s = unsafe { &*y_s };
    let y_s: [&[Realtype; N]; N_SENSI] =
        array_init::from_iter(y_s.iter().map(|&v| unsafe { &*v }.as_slice())).unwrap();
    let y_sdot = unsafe { &mut *y_sdot };
    let y_sdot: [&mut [Realtype; N]; N_SENSI] = array_init::from_iter(
        y_sdot
            .iter_mut()
            .map(|&mut v| unsafe { &mut *v }.as_slice_mut()),
    )
    .unwrap();
    let WrappingUserData {
        actual_user_data: data,
        fs,
        ..
    } = unsafe { &*data };
    let res = fs(t, y, ydot, y_s, y_sdot, data);
    match res {
        RhsResult::Ok => 0,
        RhsResult::RecoverableError(e) => e as c_int,
        RhsResult::NonRecoverableError(e) => -(e as c_int),
    }
}

impl<UserData, F, FS, const N: usize, const N_SENSI: usize> Solver<UserData, F, FS, N, N_SENSI>
where
    F: Fn(Realtype, &[Realtype; N], &mut [Realtype; N], &UserData) -> RhsResult,
    FS: Fn(
        Realtype,
        &[Realtype; N],
        &[Realtype; N],
        [&[Realtype; N]; N_SENSI],
        [&mut [Realtype; N]; N_SENSI],
        &UserData,
    ) -> RhsResult,
{
    #[allow(clippy::clippy::too_many_arguments)]
    pub fn new(
        method: LinearMultistepMethod,
        f: F,
        f_sens: FS,
        t0: Realtype,
        y0: &[Realtype; N],
        y_s0: &[[Realtype; N]; N_SENSI],
        rtol: Realtype,
        atol: AbsTolerance<N>,
        atol_sens: SensiAbsTolerance<N, N_SENSI>,
        user_data: UserData,
    ) -> Result<Self> {
        assert_eq!(y0.len(), N);
        let mem: CvodeMemoryBlockNonNullPtr = {
            let mem_maybenull = unsafe { sundials_sys::CVodeCreate(method as c_int) };
            check_non_null(mem_maybenull as *mut CvodeMemoryBlock, "CVodeCreate")?.into()
        };
        let y0 = NVectorSerialHeapAllocated::new_from(y0);
        let y_s0 = Box::new(
            array_init::from_iter(
                y_s0.iter()
                    .map(|arr| NVectorSerialHeapAllocated::new_from(arr)),
            )
            .unwrap(),
        );
        let matrix = {
            let matrix = unsafe {
                sundials_sys::SUNDenseMatrix(N.try_into().unwrap(), N.try_into().unwrap())
            };
            check_non_null(matrix, "SUNDenseMatrix")?
        };
        let linsolver = {
            let linsolver = unsafe { sundials_sys::SUNLinSol_Dense(y0.as_raw(), matrix.as_ptr()) };
            check_non_null(linsolver, "SUNDenseLinearSolver")?
        };
        let user_data = Box::pin(WrappingUserData {
            actual_user_data: user_data,
            f,
            fs: f_sens,
        });
        let res = Solver {
            mem,
            y0,
            y_s0,
            sunmatrix: matrix.as_ptr(),
            linsolver: linsolver.as_ptr(),
            atol,
            atol_sens,
            user_data,
            sensi_out_buffer: array_init::array_init(|_| NVectorSerialHeapAllocated::new()),
        };
        {
            let flag = unsafe {
                sundials_sys::CVodeSetUserData(
                    mem.as_raw(),
                    res.user_data.as_ref().get_ref() as *const _ as _,
                )
            };
            check_flag_is_succes(flag, "CVodeSetUserData")?;
        }
        {
            let fn_ptr = wrap_f::<UserData, F, FS, N> as extern "C" fn(_, _, _, _) -> _;
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
        {
            let fn_ptr = wrap_f_sens::<UserData, F, FS, N, N_SENSI>
                as extern "C" fn(_, _, _, _, _, _, _, _, _) -> _;
            let flag = unsafe {
                sundials_sys::CVodeSensInit(
                    mem.as_raw(),
                    N_SENSI as c_int,
                    CV_STAGGERED as _,
                    Some(std::mem::transmute(fn_ptr)),
                    res.y_s0.as_ptr() as _,
                )
            };
            check_flag_is_succes(flag, "CVodeSensInit")?;
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
        match &res.atol_sens {
            SensiAbsTolerance::Scalar(atol) => {
                let flag = unsafe {
                    sundials_sys::CVodeSensSStolerances(mem.as_raw(), rtol, atol.as_ptr() as _)
                };
                check_flag_is_succes(flag, "CVodeSensSStolerances")?;
            }
            SensiAbsTolerance::Vector(atol) => {
                let flag = unsafe {
                    sundials_sys::CVodeSensSVtolerances(mem.as_raw(), rtol, atol.as_ptr() as _)
                };
                check_flag_is_succes(flag, "CVodeSensSVtolerances")?;
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
        Ok(res)
    }

    #[allow(clippy::clippy::type_complexity)]
    pub fn step(
        &mut self,
        tout: Realtype,
        step_kind: StepKind,
    ) -> Result<(Realtype, &[Realtype; N], [&[Realtype; N]; N_SENSI])> {
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
        let flag = unsafe {
            sundials_sys::CVodeGetSens(
                self.mem.as_raw(),
                &mut tret,
                self.sensi_out_buffer.as_mut_ptr() as _,
            )
        };
        check_flag_is_succes(flag, "CVodeGetSens")?;
        let sensi_ptr_array =
            array_init::from_iter(self.sensi_out_buffer.iter().map(|v| v.as_slice())).unwrap();
        Ok((tret, self.y0.as_slice(), sensi_ptr_array))
    }
}

impl<UserData, F, FS, const N: usize, const N_SENSI: usize> Drop
    for Solver<UserData, F, FS, N, N_SENSI>
{
    fn drop(&mut self) {
        unsafe { sundials_sys::CVodeFree(&mut self.mem.as_raw()) }
        unsafe { sundials_sys::SUNLinSolFree(self.linsolver) };
        unsafe { sundials_sys::SUNMatDestroy(self.sunmatrix) };
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

    fn fs<const N_SENSI: usize>(
        _t: super::Realtype,
        _y: &[Realtype; 2],
        _ydot: &[Realtype; 2],
        _ys: [&[Realtype; 2]; N_SENSI],
        ysdot: [&mut [Realtype; 2]; N_SENSI],
        _data: &(),
    ) -> RhsResult {
        for ysdot_i in std::array::IntoIter::new(ysdot) {
            *ysdot_i = [0., 0.];
        }
        RhsResult::Ok
    }

    #[test]
    fn create() {
        let y0 = [0., 1.];
        let y_s0 = [[0.; 2]; 4];
        let _solver = Solver::new(
            LinearMultistepMethod::Adams,
            f,
            fs,
            0.,
            &y0,
            &y_s0,
            1e-4,
            AbsTolerance::scalar(1e-4),
            SensiAbsTolerance::scalar([1e-4; 4]),
            (),
        ).unwrap();
    }
}
