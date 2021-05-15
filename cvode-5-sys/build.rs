use bindgen::builder;
use std::env;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    for lib in &["sundials_cvodes", "sundials_nvecserial"] {
        println!("cargo:rustc-link-lib={}", lib);
    }

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("generated.rs");

    let bindings = builder()
        .header("src/include.h")
        .generate()
        .map_err(|()| anyhow::anyhow!("Couldn't generate bindings."))?;

    // Write the generated bindings to an output file.
    bindings.write_to_file(dest_path)?;
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/include.h");
    Ok(())
}
