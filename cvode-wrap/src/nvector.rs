use std::{
    convert::TryInto,
    intrinsics::transmute,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use cvode_5_sys::{cvode::realtype, nvector_serial};

#[repr(transparent)]
#[derive(Debug)]
pub struct NVectorSerial<const SIZE: usize> {
    inner: nvector_serial::_generic_N_Vector,
}

impl<const SIZE: usize> NVectorSerial<SIZE> {
    pub unsafe fn as_raw(&self) -> nvector_serial::N_Vector {
        std::mem::transmute(&self.inner)
    }

    pub fn as_slice(&self) -> &[realtype; SIZE] {
        unsafe { transmute(nvector_serial::N_VGetArrayPointer_Serial(self.as_raw())) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [realtype; SIZE] {
        unsafe { transmute(nvector_serial::N_VGetArrayPointer_Serial(self.as_raw())) }
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct NVectorSerialHeapAlloced<const SIZE: usize> {
    inner: NonNull<NVectorSerial<SIZE>>,
}

impl<const SIZE: usize> Deref for NVectorSerialHeapAlloced<SIZE> {
    type Target = NVectorSerial<SIZE>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.inner.as_ref() }
    }
}

impl<const SIZE: usize> DerefMut for NVectorSerialHeapAlloced<SIZE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.inner.as_mut() }
    }
}

impl<const SIZE: usize> NVectorSerialHeapAlloced<SIZE> {
    pub fn new() -> Self {
        let raw_c = unsafe { nvector_serial::N_VNew_Serial(SIZE.try_into().unwrap()) };
        Self {
            inner: NonNull::new(raw_c as *mut NVectorSerial<SIZE>).unwrap(),
        }
    }

    pub fn new_from(data: &[realtype; SIZE]) -> Self {
        let mut res = Self::new();
        res.as_slice_mut().copy_from_slice(data);
        res
    }
}

impl<const SIZE: usize> Drop for NVectorSerialHeapAlloced<SIZE> {
    fn drop(&mut self) {
        unsafe { nvector_serial::N_VDestroy(self.as_raw()) }
    }
}
