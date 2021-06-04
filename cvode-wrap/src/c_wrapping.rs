//! Code used to wrap the RHS function for C.
//!
//! The most user friendly way to interact with this module is through the
//! [`wrap`] macro.

use std::os::raw::c_int;

use cvode_5_sys::N_VGetArrayPointer;

use crate::{NVectorSerial, Realtype, RhsF, RhsResult};

/// The type of the function pointer for the right hand side that is passed to C.
///
/// The advised method to declare such a function is to use the [`wrap`] macro.
pub type RhsFCtype<UserData, const N: usize> = extern "C" fn(
    t: Realtype,
    y: *const NVectorSerial<N>,
    ydot: *mut NVectorSerial<N>,
    user_data: *const UserData,
) -> c_int;

/// The wrapping function.
///
/// Internally used in [`wrap`].
pub fn wrap_f<UserData, const N: usize>(
    f: RhsF<UserData, N>,
    t: Realtype,
    y: *const NVectorSerial<N>,
    ydot: *mut NVectorSerial<N>,
    data: &UserData,
) -> c_int {
    let y = unsafe { &*(N_VGetArrayPointer(y as _) as *const [f64; N]) };
    let ydot = unsafe { &mut *(N_VGetArrayPointer(ydot as _) as *mut [f64; N]) };
    let res = f(t, y, ydot, data);
    match res {
        RhsResult::Ok => 0,
        RhsResult::RecoverableError(e) => e as c_int,
        RhsResult::NonRecoverableError(e) => -(e as c_int),
    }
}

/// Declares an `extern "C"` function of type [`RhsFCtype`] that wraps a
/// normal Rust `fn` of type [`RhsF`]
#[macro_export]
macro_rules! wrap {
    ($wrapped_f_name: ident, $f_name: ident, $user_data_type: ty, $problem_size: expr) => {
        extern "C" fn $wrapped_f_name(
            t: Realtype,
            y: *const NVectorSerial<$problem_size>,
            ydot: *mut NVectorSerial<$problem_size>,
            data: *const $user_data_type,
        ) -> std::os::raw::c_int {
            let data = unsafe { &*data };
            c_wrapping::wrap_f($f_name, t, y, ydot, data)
        }
    };
}
