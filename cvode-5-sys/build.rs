fn main() -> () {
    for lib in &["sundials_cvodes", "sundials_nvecserial"] {
        println!("cargo:rustc-link-lib={}", lib);
    }
}
