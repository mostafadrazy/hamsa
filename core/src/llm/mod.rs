//! Local language-model helper: manages an official prebuilt llama.cpp (Windows, Vulkan) and runs `llama-server` as a
//! hidden background process. Shared by every feature that needs an LLM (smart cleanup now, audio models later).
//!
//! Files (all inside the data folder, never in the shared `models` folder):
//! - `llama\<version>\`: the unpacked llama.cpp release (downloaded on first need, SHA-256 pinned).
//! - `llama\server.log`: the helper's output, for troubleshooting.
//!
//! The server is started lazily, kept alive while it is used, killed after `IDLE_UNLOAD` of silence, and lives in a
//! Windows job object that kills it when Hamsa exits (even by crashing).

pub mod asr;
pub mod cleanup;

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

/// llama.cpp release in use. Must stay at b9199 or newer (audio models need it).
pub const LLAMA_VERSION: &str = "b12651";
pub const LLAMA_URL: &str =
    "https://github.com/ggml-org/llama.cpp/releases/download/b12651/llama-b12651-bin-win-vulkan-x64.zip";
pub const LLAMA_SHA256: &str = "e58fa9a137d8046c69334005ad6fb3e904ddc7e56f2a61ad1b844335826ef3f4";
pub const LLAMA_BYTES: u64 = 38_939_211;

/// The server is unloaded after this long without a request.
pub const IDLE_UNLOAD: Duration = Duration::from_secs(5 * 60);
/// How long a model may take to load before the start counts as failed.
const LOAD_TIMEOUT: Duration = Duration::from_secs(90);
/// After a failed start, the next attempt waits this long (so a broken setup doesn't slow every dictation).
const RETRY_AFTER: Duration = Duration::from_secs(60);

/// Where the model runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Gpu,
    Cpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Gpu => "GPU",
            Backend::Cpu => "CPU",
        }
    }
}

// ---------- Download ----------

/// Download `url` to `path`, checking size progress through `on_bytes` (total bytes received so far) and the pinned
/// SHA-256. The file is written as `<path>.part` and renamed only when the checksum matches.
pub fn download_verified(
    url: &str,
    path: &Path,
    sha256: &str,
    expected_bytes: u64,
    on_bytes: &mut dyn FnMut(u64),
) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let partial = path.with_file_name(format!("{}.part", path.file_name().and_then(|n| n.to_str()).unwrap_or("download")));
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Download failed: {e}. Check your internet connection."))?;
    let _ = expected_bytes;
    let mut reader = response.into_reader();
    let mut file = fs::File::create(&partial).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut received = 0u64;
    loop {
        let n = match reader.read(&mut buf) {
            Ok(n) => n,
            Err(e) => {
                let _ = fs::remove_file(&partial);
                return Err(format!("Download interrupted: {e}"));
            }
        };
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        hasher.update(&buf[..n]);
        received += n as u64;
        on_bytes(received);
    }
    file.flush().map_err(|e| e.to_string())?;
    drop(file);
    let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if digest != sha256 {
        let _ = fs::remove_file(&partial);
        return Err("The download was corrupted (checksum mismatch). Please try again.".into());
    }
    fs::rename(&partial, path).map_err(|e| e.to_string())
}

pub fn llama_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("llama")
}

pub fn server_exe(data_dir: &Path) -> PathBuf {
    llama_dir(data_dir).join(LLAMA_VERSION).join("llama-server.exe")
}

pub fn llama_installed(data_dir: &Path) -> bool {
    server_exe(data_dir).is_file()
}

/// Make sure the pinned llama.cpp release is unpacked in `llama\<version>\`; download it if not.
/// `on_bytes` gets the bytes received of the zip (`LLAMA_BYTES` in total).
pub fn install_llama(data_dir: &Path, on_bytes: &mut dyn FnMut(u64)) -> Result<PathBuf, String> {
    let exe = server_exe(data_dir);
    if exe.is_file() {
        return Ok(exe);
    }
    let root = llama_dir(data_dir);
    fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let archive = root.join(format!("llama-{LLAMA_VERSION}.zip"));
    download_verified(LLAMA_URL, &archive, LLAMA_SHA256, LLAMA_BYTES, on_bytes)?;
    let staging = root.join(format!("{LLAMA_VERSION}.tmp"));
    let _ = fs::remove_dir_all(&staging);
    let unpacked = unzip(&archive, &staging);
    let _ = fs::remove_file(&archive);
    if let Err(e) = unpacked {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("Couldn't unpack llama.cpp: {e}"));
    }
    let target = root.join(LLAMA_VERSION);
    let _ = fs::remove_dir_all(&target);
    fs::rename(&staging, &target).map_err(|e| e.to_string())?;
    // Older releases are no longer needed.
    if let Ok(entries) = fs::read_dir(&root) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if entry.path().is_dir() && name != LLAMA_VERSION {
                let _ = fs::remove_dir_all(entry.path());
            }
        }
    }
    if exe.is_file() {
        Ok(exe)
    } else {
        Err("llama.cpp was unpacked but llama-server.exe is missing.".into())
    }
}

fn unzip(archive: &Path, into: &Path) -> Result<(), String> {
    let file = fs::File::open(archive).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    fs::create_dir_all(into).map_err(|e| e.to_string())?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        // Entries that would land outside `into` are skipped.
        let Some(relative) = entry.enclosed_name() else { continue };
        let out = into.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            continue;
        }
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut dest = fs::File::create(&out).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut dest).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ---------- The server process ----------

struct Running {
    child: Child,
    port: u16,
    backend: Backend,
}

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

enum Phase {
    Stopped,
    Starting,
    Ready(Running),
    Failed(String),
}

struct Inner {
    phase: Phase,
    /// The model being loaded or served, and its multimodal projector (audio models).
    model: Option<PathBuf>,
    mmproj: Option<PathBuf>,
    /// GPU offload failed once this session: use the CPU from now on.
    cpu_only: bool,
    last_used: Instant,
    failed_at: Option<Instant>,
}

/// What the helper is doing, for the UI.
#[derive(Clone, Debug, PartialEq)]
pub enum HostState {
    Stopped,
    Starting,
    Ready(Backend),
    Failed(String),
}

pub struct LlamaHost {
    data_dir: PathBuf,
    inner: Mutex<Inner>,
    cv: Condvar,
    notify: Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>,
    idle_thread: AtomicBool,
}

impl LlamaHost {
    pub fn new(data_dir: &Path) -> Arc<Self> {
        Arc::new(Self {
            data_dir: data_dir.to_path_buf(),
            inner: Mutex::new(Inner { phase: Phase::Stopped, model: None, mmproj: None, cpu_only: false, last_used: Instant::now(), failed_at: None }),
            cv: Condvar::new(),
            notify: Mutex::new(Vec::new()),
            idle_thread: AtomicBool::new(false),
        })
    }

    /// `f` is called (from any thread) whenever the state changes. Several listeners may be added.
    pub fn on_change(&self, f: Arc<dyn Fn() + Send + Sync>) {
        if let Ok(mut n) = self.notify.lock() {
            n.push(f);
        }
    }

    fn changed(&self) {
        let listeners = self.notify.lock().map(|n| n.clone()).unwrap_or_default();
        for f in listeners {
            f();
        }
    }

    pub fn state(&self) -> HostState {
        let Ok(g) = self.inner.lock() else { return HostState::Stopped };
        match &g.phase {
            Phase::Stopped => HostState::Stopped,
            Phase::Starting => HostState::Starting,
            Phase::Ready(r) => HostState::Ready(r.backend),
            Phase::Failed(e) => HostState::Failed(e.clone()),
        }
    }

    /// The model the helper is loading or serving.
    pub fn model(&self) -> Option<PathBuf> {
        self.inner.lock().ok().and_then(|g| g.model.clone())
    }

    /// Stop the server (if running) and free its memory.
    pub fn stop(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.phase = Phase::Stopped; // dropping the old phase kills the process
            g.model = None;
            g.mmproj = None;
        }
        self.cv.notify_all();
        self.changed();
    }

    /// Begin loading `model` in the background (no-op when it is already loading or loaded).
    /// `warmup` is an optional chat-completions body sent once after loading, so the first real request is fast
    /// (the first one pays for building the compute graphs, a couple of seconds on a GPU).
    pub fn start_async(self: &Arc<Self>, model: &Path, warmup: Option<String>) {
        self.start_with(model, None, warmup);
    }

    /// As `start_async`, for an audio model: `mmproj` is its projector file (`--mmproj`).
    pub fn start_with(self: &Arc<Self>, model: &Path, mmproj: Option<&Path>, warmup: Option<String>) {
        {
            let Ok(mut g) = self.inner.lock() else { return };
            let same_model = g.model.as_deref() == Some(model) && g.mmproj.as_deref() == mmproj;
            let failed_recently = g.failed_at.is_some_and(|t| t.elapsed() < RETRY_AFTER);
            let keep = match &mut g.phase {
                Phase::Starting => true,
                Phase::Ready(r) => same_model && r.child.try_wait().ok().flatten().is_none(),
                Phase::Failed(_) => same_model && failed_recently,
                Phase::Stopped => false,
            };
            if keep {
                return;
            }
            g.phase = Phase::Starting; // kills a previous server
            g.model = Some(model.to_path_buf());
            g.mmproj = mmproj.map(Path::to_path_buf);
            g.failed_at = None;
        }
        self.ensure_idle_thread();
        self.changed();
        let me = self.clone();
        let model = model.to_path_buf();
        let mmproj = mmproj.map(Path::to_path_buf);
        std::thread::Builder::new()
            .name("hamsa-llama-start".into())
            .spawn(move || me.launch(&model, mmproj.as_deref(), warmup.as_deref()))
            .ok();
    }

    /// Start `model` if needed and wait up to `timeout` for it to answer. Returns the server's port and backend.
    pub fn wait_ready(self: &Arc<Self>, model: &Path, warmup: Option<String>, timeout: Duration) -> Result<(u16, Backend), String> {
        self.wait_ready_with(model, None, warmup, timeout)
    }

    /// As `wait_ready`, for an audio model with a projector file.
    pub fn wait_ready_with(
        self: &Arc<Self>,
        model: &Path,
        mmproj: Option<&Path>,
        warmup: Option<String>,
        timeout: Duration,
    ) -> Result<(u16, Backend), String> {
        self.start_with(model, mmproj, warmup);
        let deadline = Instant::now() + timeout;
        let mut g = self.inner.lock().map_err(|_| "internal error".to_string())?;
        loop {
            let same_model = g.model.as_deref() == Some(model);
            let outcome = match &g.phase {
                Phase::Ready(r) if same_model => Some(Ok((r.port, r.backend))),
                Phase::Failed(e) => Some(Err(e.clone())),
                Phase::Stopped => Some(Err("The helper is not running.".to_string())),
                _ => None,
            };
            if let Some(outcome) = outcome {
                if outcome.is_ok() {
                    g.last_used = Instant::now();
                }
                return outcome;
            }
            let now = Instant::now();
            if now >= deadline {
                return Err("The helper is still loading.".into());
            }
            g = self.cv.wait_timeout(g, deadline - now).map_err(|_| "internal error".to_string())?.0;
        }
    }

    fn launch(self: &Arc<Self>, model: &Path, mmproj: Option<&Path>, warmup: Option<&str>) {
        let result = self.start_server(model, mmproj, warmup);
        if let Ok(mut g) = self.inner.lock() {
            // Superseded (stopped or switched to another model) while loading: drop the new server.
            if matches!(g.phase, Phase::Starting) && g.model.as_deref() == Some(model) {
                match result {
                    Ok(running) => {
                        g.phase = Phase::Ready(running);
                        g.last_used = Instant::now();
                    }
                    Err(e) => {
                        eprintln!("hamsa: llama helper: {e}");
                        g.phase = Phase::Failed(e);
                        g.failed_at = Some(Instant::now());
                    }
                }
            }
        }
        self.cv.notify_all();
        self.changed();
    }

    /// GPU first, then the CPU once if that fails (and the CPU stays the choice for the rest of the session).
    fn start_server(&self, model: &Path, mmproj: Option<&Path>, warmup: Option<&str>) -> Result<Running, String> {
        let exe = server_exe(&self.data_dir);
        if !exe.is_file() {
            return Err("llama.cpp is not downloaded.".into());
        }
        if !model.is_file() {
            return Err("The model file is missing.".into());
        }
        if mmproj.is_some_and(|p| !p.is_file()) {
            return Err("The model's audio file is missing.".into());
        }
        let cpu_only = self.inner.lock().map(|g| g.cpu_only).unwrap_or(false);
        let order: &[Backend] = if cpu_only { &[Backend::Cpu] } else { &[Backend::Gpu, Backend::Cpu] };
        let mut last = String::new();
        for &backend in order {
            match spawn_server(&exe, model, mmproj, backend, warmup, &llama_dir(&self.data_dir).join("server.log")) {
                Ok(r) => {
                    if backend == Backend::Cpu && !cpu_only {
                        eprintln!("hamsa: llama helper: GPU start failed ({last}); using the CPU");
                        if let Ok(mut g) = self.inner.lock() {
                            g.cpu_only = true;
                        }
                    }
                    return Ok(r);
                }
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn ensure_idle_thread(self: &Arc<Self>) {
        if self.idle_thread.swap(true, Ordering::SeqCst) {
            return;
        }
        let me = self.clone();
        std::thread::Builder::new()
            .name("hamsa-llama-idle".into())
            .spawn(move || loop {
                std::thread::sleep(Duration::from_secs(10));
                let unloaded = me
                    .inner
                    .lock()
                    .map(|mut g| {
                        let idle = matches!(g.phase, Phase::Ready(_)) && g.last_used.elapsed() >= IDLE_UNLOAD;
                        if idle {
                            g.phase = Phase::Stopped;
                            g.model = None;
                            g.mmproj = None;
                        }
                        idle
                    })
                    .unwrap_or(false);
                if unloaded {
                    eprintln!("hamsa: llama helper unloaded after being idle");
                    me.changed();
                }
            })
            .ok();
    }

    /// POST a JSON body to the running server. Counts as use (keeps the server loaded).
    pub fn post_json(&self, port: u16, path: &str, body: &str, timeout: Duration) -> Result<String, String> {
        if let Ok(mut g) = self.inner.lock() {
            g.last_used = Instant::now();
        }
        let agent = ureq::AgentBuilder::new().timeout(timeout).build();
        match agent.post(&format!("http://127.0.0.1:{port}{path}")).set("Content-Type", "application/json").send_string(body) {
            Ok(r) => r.into_string().map_err(|e| format!("bad response: {e}")),
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                Err(format!("server answered {code}: {}", text.chars().take(160).collect::<String>()))
            }
            Err(e) => Err(format!("request failed: {e}")),
        }
    }
}

fn free_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| format!("no free port: {e}"))?;
    listener.local_addr().map(|a| a.port()).map_err(|e| e.to_string())
}

/// Launch `llama-server` hidden on a free port and wait until it reports healthy.
fn spawn_server(
    exe: &Path,
    model: &Path,
    mmproj: Option<&Path>,
    backend: Backend,
    warmup: Option<&str>,
    log_path: &Path,
) -> Result<Running, String> {
    let port = free_port()?;
    let log = fs::File::create(log_path).map_err(|e| e.to_string())?;
    let log2 = log.try_clone().map_err(|e| e.to_string())?;
    let mut cmd = Command::new(exe);
    cmd.current_dir(exe.parent().unwrap_or(Path::new(".")))
        .args(["--host", "127.0.0.1", "--port", &port.to_string(), "-m"])
        .arg(model)
        // Audio models need room for the sound as well as the words.
        .args(["-ngl", if backend == Backend::Gpu { "99" } else { "0" }, "-c", if mmproj.is_some() { "8192" } else { "4096" }])
        .args(["--jinja", "-np", "1", "--no-webui"]);
    if let Some(mmproj) = mmproj {
        cmd.arg("--mmproj").arg(mmproj);
    }
    if backend == Backend::Cpu {
        cmd.args(["--device", "none"]);
        if mmproj.is_some() {
            cmd.arg("--no-mmproj-offload");
        }
    }
    cmd.stdin(Stdio::null()).stdout(log).stderr(log2);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let child = cmd.spawn().map_err(|e| format!("couldn't start llama-server: {e}"))?;
    #[cfg(windows)]
    job::assign(&child);
    let mut running = Running { child, port, backend };

    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(1)).build();
    let url = format!("http://127.0.0.1:{port}/health");
    let started = Instant::now();
    loop {
        if let Ok(Some(status)) = running.child.try_wait() {
            return Err(format!("llama-server stopped ({status}): {}", last_log_line(log_path)));
        }
        if agent.get(&url).call().is_ok() {
            if let Some(body) = warmup {
                // Failing to warm up isn't fatal; the first real request will just be slower.
                let t = Instant::now();
                let again = ureq::AgentBuilder::new().timeout(Duration::from_secs(60)).build();
                let sent = again
                    .post(&format!("http://127.0.0.1:{port}/v1/chat/completions"))
                    .set("Content-Type", "application/json")
                    .send_string(body);
                eprintln!("hamsa: llama helper warm-up {} in {} ms", if sent.is_ok() { "done" } else { "failed" }, t.elapsed().as_millis());
            }
            return Ok(running);
        }
        if started.elapsed() > LOAD_TIMEOUT {
            return Err("llama-server took too long to load the model".into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn last_log_line(path: &Path) -> String {
    fs::read_to_string(path)
        .ok()
        .and_then(|t| t.lines().rev().find(|l| !l.trim().is_empty()).map(|l| l.trim().chars().take(160).collect()))
        .unwrap_or_default()
}

/// A Windows job object that kills everything assigned to it when Hamsa's process ends, however it ends.
#[cfg(windows)]
mod job {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;
    use std::sync::OnceLock;

    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    static JOB: OnceLock<usize> = OnceLock::new();

    pub fn assign(child: &Child) {
        let job = *JOB.get_or_init(|| unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() {
                return 0;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            handle as usize
        });
        if job != 0 {
            unsafe {
                AssignProcessToJobObject(job as _, child.as_raw_handle() as _);
            }
        }
    }
}
