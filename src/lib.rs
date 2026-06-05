use pyo3::prelude::*;

/// Rust implementations for the BIDSAid.
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

    // https://github.com/nipy/nibabel/issues/1477
    /// Parallel file compression for any file but used for NIfTI images
    /// in BIDSAid.
    ///
    /// # Arguments
    /// * src_file - String reference of path to source file location.
    /// * dst_file - String reference of path to destination file location.
    /// * compress_level - The desired compression level.
    /// * delete_src_file - Whether or not to delete the source file.
    ///
    /// # Errors
    /// Returns `PyIOError` if the source file cannot be read, the destination cannot be
    /// created or written, compression fails, or the source file deletion fails.
    #[pyfunction]
    pub fn compress_file(
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
            pyo3::exceptions::PyIOError::new_err(format!("File compression failed: {e}"))
        })?;

        if delete_src_file {
            remove_file(src_file)?;
        }

        Ok(())
    }
}
