use std::{
    convert::TryInto,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};
use sundials_sys::realtype;
use crate::SunContext;

/// A sundials `N_Vector_Serial`.
#[repr(transparent)]
#[derive(Debug)]
pub struct NVectorSerial<const SIZE: usize> {
    inner: sundials_sys::_generic_N_Vector,
}

impl<const SIZE: usize> NVectorSerial<SIZE> {
    pub(crate) unsafe fn as_raw(&self) -> sundials_sys::N_Vector {
        std::mem::transmute(&self.inner)
    }

    /// Returns a reference to the inner slice of the vector.
    pub fn as_slice(&self) -> &[realtype; SIZE] {
        unsafe { &*(sundials_sys::N_VGetArrayPointer_Serial(self.as_raw()) as *const [f64; SIZE]) }
    }

    /// Returns a mutable reference to the inner slice of the vector.
    pub fn as_slice_mut(&mut self) -> &mut [realtype; SIZE] {
        unsafe {
            &mut *(sundials_sys::N_VGetArrayPointer_Serial(self.as_raw()) as *mut [f64; SIZE])
        }
    }
}

#[repr(transparent)]
#[derive(Debug)]
/// An owning pointer to a sundials [`NVectorSerial`] on the heap.
pub struct NVectorSerialHeapAllocated<const SIZE: usize> {
    inner: NonNull<NVectorSerial<SIZE>>,
}

impl<const SIZE: usize> Deref for NVectorSerialHeapAllocated<SIZE> {
    type Target = NVectorSerial<SIZE>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.inner.as_ref() }
    }
}

impl<const SIZE: usize> DerefMut for NVectorSerialHeapAllocated<SIZE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.inner.as_mut() }
    }
}

impl<const SIZE: usize> NVectorSerialHeapAllocated<SIZE> {
    unsafe fn new_inner_uninitialized(context: SunContext) -> NonNull<NVectorSerial<SIZE>> {
        let raw_c = sundials_sys::N_VNew_Serial(SIZE.try_into().unwrap(), context.as_ptr());
        NonNull::new(raw_c as *mut NVectorSerial<SIZE>).unwrap()
    }

    /// Creates a new vector, filled with 0.
    pub fn new(context: SunContext) -> Self {
        let inner = unsafe {
            let x = Self::new_inner_uninitialized(context);
            let ptr = sundials_sys::N_VGetArrayPointer_Serial(x.as_ref().as_raw());
            for off in 0..SIZE {
                *ptr.add(off) = 0.;
            }
            x
        };
        Self { inner }
    }

    /// Creates a new vector, filled with data from `data`.
    pub fn new_from(data: &[realtype; SIZE], context: SunContext) -> Self {
        let inner = unsafe {
            let x = Self::new_inner_uninitialized(context);
            let ptr = sundials_sys::N_VGetArrayPointer_Serial(x.as_ref().as_raw());
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, SIZE);
            x
        };
        Self { inner }
    }
}

impl<const SIZE: usize> Drop for NVectorSerialHeapAllocated<SIZE> {
    fn drop(&mut self) {
        unsafe { sundials_sys::N_VDestroy(self.as_raw()) }
    }
}
