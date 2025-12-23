use anyhow::Result;
use burn::prelude::*;
use burn::tensor::backend::Backend as BurnBackend;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

pub mod downloader;
pub mod llm_singleton;
pub mod model_manager;
pub mod model_utils;

use downloader::ModelDownloader;
pub use llm_singleton::LLMSingleton;
pub use model_manager::{FileInfo, ModelInfo, ModelManager, ModelRegistry, ModelStatus};

/// Simple placeholder model for Burn migration
#[derive(Clone)]
pub struct SimpleModel<B: BurnBackend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: BurnBackend> SimpleModel<B> {
    pub fn new(_device: &B::Device) -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
    
    pub fn forward(&self, _input: Tensor<B, 2>, _start_pos: usize) -> Result<Tensor<B, 3>> {
        // Placeholder implementation
        Ok(Tensor::zeros([1, 1, 32000], &Default::default()))
    }
}

/// Backend type - using ndarray as default
#[cfg(feature = "ndarray")]
type Backend = burn_ndarray::NdArray<f32>;

#[cfg(feature = "wgpu")]
type Backend = burn_wgpu::Wgpu<f32, i32>;

#[cfg(feature = "candle")]
type Backend = burn_candle::Candle<f32>;

#[cfg(feature = "cuda")]
type Backend = burn_cuda::Cuda<f32>;

/// Mini LLM wrapper for running lightweight models (Qwen, Phi, etc.)
pub struct MiniLLM {
    model: SimpleModel<Backend>,
    tokenizer: Tokenizer,
    device: <Backend as burn::tensor::backend::Backend>::Device,
    config: LLMConfig,
}

impl MiniLLM {
    /// Clear the model's KV cache
    pub fn clear_cache(&mut self) {
        // Placeholder for now
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub repeat_penalty: f32,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            model_path: None, // Will auto-download if None
            tokenizer_path: None,
            max_tokens: 1024,
            // Temperature 0.7 provides good balance between creativity and coherence
            // (avoids greedy sampling that can cause repetition with small models)
            temperature: 0.7,
            top_p: 0.95,
            // Higher repeat_penalty discourages the model from repeating tokens
            repeat_penalty: 1.3,
        }
    }
}

impl MiniLLM {
    /// Create a new MiniLLM instance
    pub fn new(config: LLMConfig) -> Result<Self> {
        Self::new_with_progress(config, |_| {})
    }

    /// Create a new MiniLLM instance with progress callback
    pub fn new_with_progress<F>(config: LLMConfig, mut progress_fn: F) -> Result<Self>
    where
        F: FnMut(&str),
    {
        // Initialize device
        progress_fn("Initializing device...");
        let device = Self::get_device()?;

        // Get or download model files
        let (model_paths, tokenizer_path, config_path) =
            if config.model_path.is_none() || config.tokenizer_path.is_none() {
                tracing::info!("No model path provided, downloading from Hugging Face...");
                let downloader = ModelDownloader::new()?;
                let files = downloader.download_qwen_model_with_progress(&mut progress_fn)?;
                (files.model_paths, files.tokenizer_path, files.config_path)
            } else {
                // If paths are provided, assume config is in same directory
                let model_path = config.model_path.clone().unwrap();
                let config_path = model_path
                    .parent()
                    .ok_or_else(|| anyhow::anyhow!("Invalid model path"))?
                    .join("config.json");
                (
                    vec![model_path],
                    config.tokenizer_path.clone().unwrap(),
                    config_path,
                )
            };

        tracing::info!("Loading model from {} files", model_paths.len());
        progress_fn("Loading model into memory...");
        
        // Create simple placeholder model for now
        let model = SimpleModel::new(&device);

        tracing::info!("Loading tokenizer from: {:?}", tokenizer_path);
        progress_fn("Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        progress_fn("Model ready!");

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    /// Get the best available device
    fn get_device() -> Result<<Backend as burn::tensor::backend::Backend>::Device> {
        // For now, use default device
        // In a real implementation, we would check for available backends
        tracing::info!("Using default device");
        Ok(Default::default())
    }



    /// Generate text from a prompt
    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_with_system("", prompt, max_tokens).await
    }

    /// Generate text with a system prompt
    pub async fn generate_with_system(
        &mut self,
        system: &str,
        user: &str,
        max_tokens: usize,
    ) -> Result<String> {
        // Try fast generation first
        match self.generate_fast(system, user, max_tokens).await {
            Ok(text) => Ok(text),
            Err(e) => {
                tracing::warn!(
                    "Fast generation failed: {}. Falling back to robust (slow) generation.",
                    e
                );
                self.generate_robust(system, user, max_tokens).await
            }
        }
    }

    /// Generate text with streaming callback
    ///
    /// This is similar to `generate_fast`, but it calls `callback` with each new token string.
    /// The callback should return `true` to continue generating, or `false` to stop.
    pub async fn generate_stream<F>(
        &mut self,
        system: &str,
        user: &str,
        max_tokens: usize,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(String) -> bool,
    {
        // Simple placeholder for now
        let response = "This is a placeholder streaming response from the Burn backend migration.";
        for word in response.split_whitespace() {
            if !callback(format!("{} ", word)) {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        
        Ok(())
    }

    /// Fast generation using KV caching (O(N))
    async fn generate_fast(
        &mut self,
        system: &str,
        user: &str,
        max_tokens: usize,
    ) -> Result<String> {
        self.clear_cache();

        // Format prompt
        let formatted_prompt = if system.is_empty() {
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user
            )
        } else {
            format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                system, user
            )
        };

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(formatted_prompt.clone(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let prompt_tokens = encoding.get_ids().to_vec();
        let eos_token = self.tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);

        let mut all_tokens = prompt_tokens.clone();

        // Simple placeholder generation for now
        // Just return a fixed response to test the pipeline
        let text = "This is a placeholder response from the Burn backend migration.".to_string();

        tracing::debug!("Fast generation finished. Length: {}", text.len());
        Ok(text)
    }

    /// Robust (but slow) generation that re-processes context every step (O(N^2))
    /// Used as fallback if fast generation fails due to shape/cache errors.
    async fn generate_robust(
        &mut self,
        system: &str,
        user: &str,
        max_tokens: usize,
    ) -> Result<String> {
        // Simple placeholder for now
        let text = "This is a placeholder response from the Burn backend migration (robust fallback).".to_string();
        
        tracing::debug!(
            "LLM Generated Output length: {} chars",
            text.chars().count()
        );

        Ok(text)
    }
}

/// Get the shared LLM instance (lazy initialized)
pub fn get_llm() -> Result<Arc<MiniLLM>> {
    use std::sync::Mutex;

    static INSTANCE: Mutex<Option<Arc<MiniLLM>>> = Mutex::new(None);

    let mut guard = INSTANCE.lock().unwrap();
    if guard.is_none() {
        let config = LLMConfig::default();
        *guard = Some(Arc::new(MiniLLM::new(config)?));
    }

    Ok(guard.as_ref().unwrap().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_selection() {
        let device = MiniLLM::get_device();
        assert!(device.is_ok());
    }

    #[test]
    fn test_default_config() {
        let config = LLMConfig::default();
        assert_eq!(config.max_tokens, 1024);
        assert_eq!(config.temperature, 0.7);
    }
}
