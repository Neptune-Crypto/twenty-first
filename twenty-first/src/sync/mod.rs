//! Provides simplified lock types for sharing data between threads

mod atomic_mutex;
mod atomic_rw;
pub mod traits;

pub use atomic_mutex::AtomicMutex;
pub use atomic_rw::AtomicRw;
