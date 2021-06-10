//! A wrapper around cvode and cvodes from the sundials tool suite.
//!
//! Users should be mostly interested in [`SolverSensi`] and [`SolverNoSensi`].

use std::{ffi::c_void, os::raw::c_int, ptr::NonNull};

use sundials_sys::realtype;

mod nvector;
pub use nvector::{NVectorSerial, NVectorSerialHeapAllocated};

mod cvode;
mod cvode_sens;

pub use cvode::Solver as SolverNoSensi;
pub use cvode_sens::Solver as SolverSensi;

/// The floatting-point type sundials was compiled with
pub type Realtype = realtype;

#[repr(i32)]
#[derive(Debug)]
/// An integration method.
pub enum LinearMultistepMethod {
    /// Recomended for non-stiff problems.
    Adams = sundials_sys::CV_ADAMS,
    /// Recommended for stiff problems.
    Bdf = sundials_sys::CV_BDF,
}

/// A return type for the right-hand-side rust function.
///
/// Adapted from Sundials cv-ode guide version 5.7 (BSD Licensed), setcion 4.6.1 :
///
/// > If a recoverable error occurred, `cvode` will attempt to correct,
/// > if the error is unrecoverable, the integration is halted.
/// >
/// > A recoverable failure error return is typically used to flag a value of
/// > the dependent variableythat is “illegal” in some way (e.g., negative where
/// > only a non-negative value is physically meaningful).  If such a return is
/// > made, `cvode` will attempt to recover (possibly repeating the nonlinear solve,
/// > or reducing the step size) in order to avoid this recoverable error return.
pub enum RhsResult {
    /// Indicates that there was no error
    Ok,
    /// Indicate that there was a recoverable error and its code
    RecoverableError(u8),
    /// Indicatest hat there was a non recoverable error
    NonRecoverableError(u8),
}

/// Type of integration step
#[repr(i32)]
pub enum StepKind {
    /// The `NORMAL`option causes the solver to take internal steps
    /// until it has reached or just passed the user-specified time.
    /// The solver then interpolates in order to return an approximate
    /// value of y at the desired time.
    Normal = sundials_sys::CV_NORMAL,
    /// The `CV_ONE_STEP` option tells the solver to take just one
    /// internal step and then return thesolution at the point reached
    /// by that step.
    OneStep = sundials_sys::CV_ONE_STEP,
}

/// The error type for this crate
#[derive(Debug)]
pub enum Error {
    NullPointerError { func_id: &'static str },
    ErrorCode { func_id: &'static str, flag: c_int },
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

/// An enum representing the choice between scalars or vectors absolute tolerances
/// for sensitivities.
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

/// A short-hand for `std::result::Result<T, crate::Error>`
pub type Result<T> = std::result::Result<T, Error>;

fn check_non_null<T>(ptr: *mut T, func_id: &'static str) -> Result<NonNull<T>> {
    NonNull::new(ptr).ok_or(Error::NullPointerError { func_id })
}

fn check_flag_is_succes(flag: c_int, func_id: &'static str) -> Result<()> {
    if flag == sundials_sys::CV_SUCCESS {
        Ok(())
    } else {
        Err(Error::ErrorCode { flag, func_id })
    }
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
