use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use tiktoken_rs::{cl100k_base, o200k_base, CoreBPE};
use tokenizers::Tokenizer;

// ============================================================
// OpenAI — tiktoken-rs (vocab bundled in the crate, no network)
// ============================================================
#[derive(Deserialize)]
struct EncodingKwargs {
    encoding: String,
}

/// Lazily build and cache each tiktoken encoder. Leaked to `'static` so the
/// cache holds a stable reference across calls.
fn encoder(name: &str) -> PolarsResult<&'static CoreBPE> {
    static CACHE: OnceLock<RwLock<HashMap<String, &'static CoreBPE>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(bpe) = cache.read().unwrap().get(name) {
        return Ok(*bpe);
    }
    let bpe = match name {
        "o200k_base" => o200k_base(),
        "cl100k_base" => cl100k_base(),
        other => {
            return Err(PolarsError::ComputeError(
                format!("polars-llm-accel: unknown encoding {other:?}").into(),
            ))
        }
    }
    .map_err(|e| PolarsError::ComputeError(format!("polars-llm-accel: {e}").into()))?;
    let leaked: &'static CoreBPE = Box::leak(Box::new(bpe));
    cache.write().unwrap().insert(name.to_string(), leaked);
    Ok(leaked)
}

#[polars_expr(output_type = Int64)]
fn count_openai(inputs: &[Series], kwargs: EncodingKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let bpe = encoder(&kwargs.encoding)?;
    let name = ca.name().clone();
    let mut out: Int64Chunked = ca
        .into_iter()
        .map(|opt| opt.map(|s| bpe.encode_ordinary(s).len() as i64))
        .collect();
    out.rename(name);
    Ok(out.into_series())
}

// ============================================================
// Gemini — the shared Gemma SentencePiece tokenizer (tokenizers crate)
// ============================================================
#[derive(Deserialize)]
struct GemmaKwargs {
    path: String,
}

/// Load and cache the Gemma `tokenizer.json` by path (it is large; load once).
fn gemma_tokenizer(path: &str) -> PolarsResult<&'static Tokenizer> {
    static CACHE: OnceLock<RwLock<HashMap<String, &'static Tokenizer>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(tok) = cache.read().unwrap().get(path) {
        return Ok(*tok);
    }
    let tok = Tokenizer::from_file(path).map_err(|e| {
        PolarsError::ComputeError(
            format!("polars-llm-accel: load gemma tokenizer {path:?}: {e}").into(),
        )
    })?;
    let leaked: &'static Tokenizer = Box::leak(Box::new(tok));
    cache.write().unwrap().insert(path.to_string(), leaked);
    Ok(leaked)
}

#[polars_expr(output_type = Int64)]
fn count_gemma(inputs: &[Series], kwargs: GemmaKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let tok = gemma_tokenizer(&kwargs.path)?;
    let name = ca.name().clone();

    // Collect non-null rows, batch-encode (parallel inside tokenizers), scatter back.
    let mut idx: Vec<usize> = Vec::new();
    let mut texts: Vec<&str> = Vec::new();
    for (i, opt) in ca.into_iter().enumerate() {
        if let Some(s) = opt {
            idx.push(i);
            texts.push(s);
        }
    }
    let mut vals: Vec<Option<i64>> = vec![None; ca.len()];
    if !texts.is_empty() {
        let encs = tok.encode_batch(texts, true).map_err(|e| {
            PolarsError::ComputeError(format!("polars-llm-accel: gemma encode: {e}").into())
        })?;
        for (i, e) in idx.into_iter().zip(encs) {
            vals[i] = Some(e.len() as i64);
        }
    }
    let mut out: Int64Chunked = vals.into_iter().collect();
    out.rename(name);
    Ok(out.into_series())
}
