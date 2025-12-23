use anyhow::Result;
use linggen_llm::{LLMConfig, MiniLLM};

#[tokio::main]
async fn main() -> Result<()> {
    println!("Testing Burn backend migration...");
    
    // Create config
    let config = LLMConfig::default();
    
    // Try to create MiniLLM instance
    println!("Creating MiniLLM instance with Burn backend...");
    let mut llm = MiniLLM::new(config)?;
    
    println!("MiniLLM created successfully with Burn backend!");
    println!("This confirms the basic migration is working.");
    
    // Test generation
    println!("\nTesting generation...");
    let response = llm.generate("Hello, how are you?", 50).await?;
    println!("Response: {}", response);
    
    Ok(())
}