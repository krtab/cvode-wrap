use std::{os::raw::c_int, ptr::NonNull};

use cvode_5_sys::realtype;

mod nvector;
pub use nvector::{NVectorSerial, NVectorSerialHeapAllocated};

pub mod cvode;
pub mod cvode_sens;

/// The floatting-point type sundials was compiled with
pub type Realtype = realtype;

#[repr(u32)]
#[derive(Debug)]
/// An integration method.
pub enum LinearMultistepMethod {
    /// Recomended for non-stiff problems.
    Adams = cvode_5_sys::CV_ADAMS,
    /// Recommended for stiff problems.
    Bdf = cvode_5_sys::CV_BDF,
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
#[repr(u32)]
pub enum StepKind {
    /// The `NORMAL`option causes the solver to take internal steps
    /// until it has reached or just passed the user-specified time.
    /// The solver then interpolates in order to return an approximate
    /// value of y at the desired time.
    Normal = cvode_5_sys::CV_NORMAL,
    /// The `CV_ONE_STEP` option tells the solver to take just one
    /// internal step and then return thesolution at the point reached
    /// by that step.
    OneStep = cvode_5_sys::CV_ONE_STEP,
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

/// A short-hand for `std::result::Result<T, crate::Error>`
pub type Result<T> = std::result::Result<T, Error>;

fn check_non_null<T>(ptr: *mut T, func_id: &'static str) -> Result<NonNull<T>> {
    NonNull::new(ptr).ok_or(Error::NullPointerError { func_id })
}

fn check_flag_is_succes(flag: c_int, func_id: &'static str) -> Result<()> {
    if flag == cvode_5_sys::CV_SUCCESS as i32 {
        Ok(())
    } else {
        Err(Error::ErrorCode { flag, func_id })
    }
}
