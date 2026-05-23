use crate::agent_manager::models::StreamChunk;
use crate::message::ChatMessage;
use anyhow::Result;
use futures_util::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::OnceCell;

#[derive(Clone)]
pub struct BurnClient {
    model_id: String,
    state: Arc<OnceCell<BurnModelState>>,
}

struct BurnModelState {
    tokenizer_path: std::path::PathBuf,
    config_path: std::path::PathBuf,
    backend: &'static str,
}

impl BurnClient {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            state: Arc::new(OnceCell::new()),
        }
    }

    pub async fn chat_text_stream(
        &self,
        _messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let state = self.prepare_model().await?;
        let model_id = self.model_id.clone();
        let backend = state.backend;
        let tokenizer_path = state.tokenizer_path.display().to_string();
        let config_path = state.config_path.display().to_string();

        let stream = futures_util::stream::once(async move {
            anyhow::bail!(
                "burn provider loaded metadata for '{model_id}' on {backend}, but transformer weight mapping is not implemented yet (tokenizer: {tokenizer_path}, config: {config_path})"
            )
        });
        Ok(Box::pin(stream))
    }

    pub async fn context_window(&self) -> Result<Option<usize>> {
        Ok(None)
    }

    async fn prepare_model(&self) -> Result<&BurnModelState> {
        self.state
            .get_or_try_init(|| {
                let model_id = self.model_id.clone();
                async move {
                    tokio::task::spawn_blocking(move || load_model_state(&model_id))
                        .await
                        .map_err(|e| anyhow::anyhow!("burn model loader panicked: {e}"))?
                }
            })
            .await
    }
}

fn load_model_state(model_id: &str) -> Result<BurnModelState> {
    let client = hf_hub::HFClientSync::new()?;
    let (owner, name) = split_hf_model_id(model_id);
    let repo = client.model(owner, name);
    let tokenizer_path = repo.download_file().filename("tokenizer.json").send()?;
    let config_path = repo.download_file().filename("config.json").send()?;

    let _tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer for {model_id}: {e}"))?;

    Ok(BurnModelState {
        tokenizer_path,
        config_path,
        backend: selected_backend(),
    })
}

fn split_hf_model_id(model_id: &str) -> (&str, &str) {
    model_id.split_once('/').unwrap_or(("", model_id))
}

fn selected_backend() -> &'static str {
    if cfg!(feature = "burn-rocm") {
        return "rocm";
    }
    if cfg!(feature = "burn-cuda") {
        return "cuda";
    }
    if cfg!(feature = "burn-wgpu") {
        return "wgpu";
    }
    "ndarray"
}
