//! cargo run --release --example transcribe -- <model archive (.bps.tar.zst) or unpacked dir> <audio.wav>
//!
//! The WAV must be 16 kHz mono (16-bit PCM or 32-bit float).

use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: transcribe <model.bps.tar.zst | model dir> <audio.wav>");
        std::process::exit(2);
    }
    let model = Path::new(&args[1]);
    let dir = if model.is_file() {
        let dest = model.with_extension("").with_extension("unpacked");
        match phonon::find_model_dir(&dest) {
            Some(d) => d,
            None => {
                let t = Instant::now();
                let d = phonon::unpack(model, &dest).unwrap_or_else(|e| fail(&e));
                eprintln!("unpacked in {:.1} s", t.elapsed().as_secs_f32());
                d
            }
        }
    } else {
        model.to_path_buf()
    };

    eprintln!("memory before load: {:.0} MB", private_mb());
    let t = Instant::now();
    let engine = phonon::Engine::load(&dir).unwrap_or_else(|e| fail(&e));
    eprintln!(
        "loaded in {:.2} s, {:.0} MB of weights in memory, encoder on {}{}",
        t.elapsed().as_secs_f32(),
        engine.memory_bytes() as f32 / 1e6,
        engine.device(),
        engine.gpu_unavailable_reason().map(|r| format!(" ({r})")).unwrap_or_default()
    );

    eprintln!("memory after load: {:.0} MB", private_mb());
    let mut reader = hound::WavReader::open(&args[2]).unwrap_or_else(|e| fail(&e.to_string()));
    let spec = reader.spec();
    if spec.sample_rate != phonon::SAMPLE_RATE || spec.channels != 1 {
        fail(&format!("need 16 kHz mono, got {} Hz × {} channels", spec.sample_rate, spec.channels));
    }
    let pcm: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };
    let seconds = pcm.len() as f32 / phonon::SAMPLE_RATE as f32;

    // Run a few times: the first pass also warms up the thread pool and caches.
    let terms: Vec<String> = std::env::var("BOOST").map(|v| v.split(',').map(String::from).collect()).unwrap_or_default();
    let boost = engine.boost(&terms);
    let runs: usize = std::env::var("RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    for run in 0..runs {
        let t = Instant::now();
        let script = std::env::var("LANG_LOCK").ok().and_then(|l| phonon::script_for_language(&l));
        let result = engine.transcribe_opts(&pcm, &phonon::Options { script, boost: Some(&boost) });
        let took = t.elapsed().as_secs_f32();
        if run == 0 {
            println!("{}", result.text);
        }
        let tm = result.timings;
        eprintln!(
            "run {}: {seconds:.1} s of audio in {took:.3} s ({:.0}× realtime) | mel {:.0} ms, encoder {:.0} ms, decoder {:.0} ms",
            run + 1,
            seconds / took,
            tm.mel_ms,
            tm.encoder_ms,
            tm.decoder_ms
        );
        eprintln!("        memory: {:.0} MB", private_mb());
        let stages: Vec<String> = phonon::profile_report().iter().map(|(n, ms)| format!("{n} {ms:.0}")).collect();
        eprintln!("        {}", stages.join(", "));
    }
}

/// Private memory of this process, in MB (Windows).
fn private_mb() -> f64 {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        GetProcessMemoryInfo(GetCurrentProcess(), &mut c as *mut _ as *mut PROCESS_MEMORY_COUNTERS, c.cb);
        return c.PrivateUsage as f64 / 1e6;
    }
    #[allow(unreachable_code)]
    0.0
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}
