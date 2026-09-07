// Character-level tokenizer: the "language" of logs has a tiny alphabet, so
// every distinct character becomes an id. Deterministic across runs.
use std::collections::{BTreeMap, BTreeSet};

pub struct CharTokenizer {
    stoi: BTreeMap<char, u32>,
    itos: Vec<char>,
}

impl CharTokenizer {
    pub fn from_corpus(text: &str) -> Self {
        let set: BTreeSet<char> = text.chars().collect();
        let itos: Vec<char> = set.into_iter().collect(); // already sorted (BTreeSet)
        let stoi = itos
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        Self { stoi, itos }
    }

    pub fn encode(&self, s: &str) -> Vec<u32> {
        s.chars().map(|c| *self.stoi.get(&c).unwrap_or(&0)).collect()
    }

    #[allow(dead_code)]
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.itos[i as usize]).collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }
}
