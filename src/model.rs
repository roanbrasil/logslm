// Tiny decoder-only GPT in candle. One block is enough for the log grammar.
use candle_core::{Device, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, ops::softmax, Embedding, LayerNorm, Linear,
    Module, VarBuilder,
};

#[derive(Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
}

// causal mask: 0 on and below the diagonal, -inf above (forbids looking ahead)
fn causal_mask(t: usize, dev: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (t, t), dev)
}

struct CausalSelfAttention {
    qkv: Linear,
    proj: Linear,
    n_head: usize,
}

impl CausalSelfAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let qkv = linear_no_bias(cfg.n_embd, 3 * cfg.n_embd, vb.pp("qkv"))?;
        let proj = linear_no_bias(cfg.n_embd, cfg.n_embd, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            n_head: cfg.n_head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let hs = c / self.n_head;

        let qkv = self.qkv.forward(x)?; // [b, t, 3c]
        let q = qkv.narrow(2, 0, c)?;
        let k = qkv.narrow(2, c, c)?;
        let v = qkv.narrow(2, 2 * c, c)?;

        let split = |z: &Tensor| -> Result<Tensor> {
            z.reshape((b, t, self.n_head, hs))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(&q)?;
        let k = split(&k)?;
        let v = split(&v)?;

        let att = (q.matmul(&k.transpose(2, 3)?)? * (1.0 / (hs as f64).sqrt()))?;
        let mask = causal_mask(t, x.device())?;
        let att = att.broadcast_add(&mask)?;
        let att = softmax(&att, D::Minus1)?;

        let y = att.matmul(&v)?; // [b, n_head, t, hs]
        let y = y.transpose(1, 2)?.reshape((b, t, c))?;
        self.proj.forward(&y)
    }
}

struct Block {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

impl Block {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(cfg.n_embd, 1e-5, vb.pp("ln1"))?,
            attn: CausalSelfAttention::new(cfg, vb.pp("attn"))?,
            ln2: layer_norm(cfg.n_embd, 1e-5, vb.pp("ln2"))?,
            ff1: linear(cfg.n_embd, 4 * cfg.n_embd, vb.pp("ff1"))?,
            ff2: linear(4 * cfg.n_embd, cfg.n_embd, vb.pp("ff2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln1.forward(x)?)?)?;
        let h = self.ff1.forward(&self.ln2.forward(&x)?)?.gelu()?;
        let h = self.ff2.forward(&h)?;
        x + h
    }
}

pub struct GptLog {
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head: Linear,
    pub block_size: usize,
}

impl GptLog {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            blocks.push(Block::new(cfg, vb.pp(format!("block{i}")))?);
        }
        Ok(Self {
            tok_emb: embedding(cfg.vocab_size, cfg.n_embd, vb.pp("tok_emb"))?,
            pos_emb: embedding(cfg.block_size, cfg.n_embd, vb.pp("pos_emb"))?,
            blocks,
            ln_f: layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_f"))?,
            head: linear(cfg.n_embd, cfg.vocab_size, vb.pp("head"))?,
            block_size: cfg.block_size,
        })
    }

    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (_b, t) = idx.dims2()?;
        let pos = Tensor::arange(0u32, t as u32, idx.device())?;
        let tok = self.tok_emb.forward(idx)?;
        let pe = self.pos_emb.forward(&pos)?;
        let mut x = tok.broadcast_add(&pe)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.head.forward(&x)
    }
}
