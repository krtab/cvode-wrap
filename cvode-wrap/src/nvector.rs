use std::{convert::TryInto, intrinsics::transmute, ops::Deref, ptr::NonNull};

use cvode_5_sys::{cvode::realtype, nvector_serial};


#[repr(transparent)]
#[derive(Debug)]
pub struct NVectorSerialHeapAlloced<const SIZE: usize> {
    inner: NonNull<nvector_serial::_generic_N_Vector>,
}

impl<const SIZE: usize> Deref for NVectorSerialHeapAlloced<SIZE> {
    type Target = nvector_serial::_generic_N_Vector;

    fn deref(&self) -> &Self::Target {
        unsafe {self.inner.as_ref()}
    }
 }

impl<const SIZE: usize> NVectorSerialHeapAlloced<SIZE> {
    pub fn as_slice(&self) -> &[realtype; SIZE] {
        unsafe { transmute(nvector_serial::N_VGetArrayPointer_Serial(self.as_raw())) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [realtype; SIZE] {
        unsafe { transmute(nvector_serial::N_VGetArrayPointer_Serial(self.as_raw())) }
    }

    pub fn new() -> Self {
        Self {
            inner: NonNull::new(unsafe { nvector_serial::N_VNew_Serial(SIZE.try_into().unwrap()) })
                .unwrap(),
        }
    }

    pub fn new_from(data: &[realtype; SIZE]) -> Self {
        let mut res = Self::new();
        res.as_slice_mut().copy_from_slice(data);
        res
    }

    pub fn as_raw(&self) -> nvector_serial::N_Vector {
        self.inner.as_ptr()
    }
}

impl<const SIZE: usize> Drop for NVectorSerialHeapAlloced<SIZE> {
    fn drop(&mut self) {
        unsafe { nvector_serial::N_VDestroy(self.as_raw()) }
    }
}
