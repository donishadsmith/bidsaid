use pyo3::prelude::*;

#[pymodule]
mod _rust {
    use gzp::{
        Compression, GzpError, ZWriter,
        deflate::Mgzip,
        par::compress::{ParCompress, ParCompressBuilder},
    };
    use pyo3::prelude::*;
    use std::{
        fs::{File, read, remove_file},
        io::Write,
    };

    #[pyfunction]
    // https://github.com/nipy/nibabel/issues/1477
    fn compress_nifti_image(
        src_file: &str,
        dst_file: &str,
        compress_level: u32,
        delete_src_file: bool,
    ) -> PyResult<()> {
        let data_bytes: Vec<u8> = read(src_file)?;
        let file_out: File = File::create(dst_file)?;
        let mut writer: ParCompress<Mgzip, _> = ParCompressBuilder::new()
            .compression_level(Compression::new(compress_level))
            .from_writer(file_out);
        writer.write_all(&data_bytes)?;
        writer.finish().map_err(|e: GzpError| {
            pyo3::exceptions::PyIOError::new_err(format!("NIfTI compression failed: {e}"))
        })?;

        if delete_src_file {
            remove_file(src_file)?;
        }

        Ok(())
    }
}
