use anyhow::Result;
use burn::prelude::*;

/// Sample next token from logits with repetition penalty (Burn version)
pub fn sample_token_burn<B: burn::tensor::backend::Backend>(
    logits: &Tensor<B, 1>,
    temperature: f64,
    top_p: f64,
    repeat_penalty: f32,
    previous_tokens: &[u32],
) -> Result<u32> {
    // Convert tensor to Vec<f32>
    let logits_data = logits.to_data();
    let logits_slice = logits_data.as_slice::<f32>().unwrap_or(&[]);
    let mut logits_vec = logits_slice.to_vec();
    
    // Apply repetition penalty to previously generated tokens
    if repeat_penalty != 1.0 {
        for &token_id in previous_tokens {
            let idx = token_id as usize;
            if idx < logits_vec.len() {
                // Penalize by dividing if logit is positive, multiplying if negative
                if logits_vec[idx] > 0.0 {
                    logits_vec[idx] /= repeat_penalty;
                } else {
                    logits_vec[idx] *= repeat_penalty;
                }
            }
        }
    }

    if temperature <= 0.0 {
        // Greedy sampling
        let mut best_idx = 0;
        let mut best_logit = logits_vec[0];
        for (idx, &logit) in logits_vec.iter().enumerate().skip(1) {
            if logit > best_logit {
                best_logit = logit;
                best_idx = idx;
            }
        }
        return Ok(best_idx as u32);
    }

    // Apply temperature
    let mut probs: Vec<f32> = logits_vec
        .iter()
        .map(|&l| (l / temperature as f32).exp())
        .collect();

    // Normalize to probabilities
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p (nucleus) sampling
    if top_p < 1.0 {
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        // Sort by probability (descending), handle NaN
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find nucleus
        let mut cumsum = 0.0;
        let mut nucleus_size = 0;
        for (_, p) in indexed_probs.iter() {
            cumsum += p;
            nucleus_size += 1;
            if cumsum >= top_p as f32 {
                break;
            }
        }

        // Zero out probabilities outside nucleus
        let nucleus_indices: std::collections::HashSet<usize> = indexed_probs
            .iter()
            .take(nucleus_size)
            .map(|(i, _)| *i)
            .collect();

        for (i, p) in probs.iter_mut().enumerate() {
            if !nucleus_indices.contains(&i) {
                *p = 0.0;
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    // Sample from distribution
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let sample: f32 = rng.gen();

    let mut cumsum = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumsum += p;
        if sample <= cumsum {
            return Ok(idx as u32);
        }
    }

    // Fallback
    Ok((probs.len() - 1) as u32)
}


