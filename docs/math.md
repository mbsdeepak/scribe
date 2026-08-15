# The mathematics of scribe

A from-scratch decoder-only language model. This document derives every moving
part and points at the code that implements it. Companion to the diffusion
project's `docs/math.tex` — same spirit, language side.

Notation: a sequence of tokens $x = (x_1, \dots, x_T)$, each $x_t \in \{1,\dots,V\}$
with vocabulary size $V = 50257$ (GPT-2 BPE). The model has parameters $\theta$,
embedding dimension $d$ ($=384$), $L$ layers ($=6$), $H$ heads ($=6$).

---

## 1. The objective: autoregressive language modelling

A language model factorizes the joint probability of a sequence by the chain
rule, left to right:

$$
p_\theta(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}).
$$

Training is maximum likelihood: choose $\theta$ to maximize the log-probability
of the data, equivalently to **minimize the negative log-likelihood**, which for
a categorical distribution is exactly the **cross-entropy** between the model's
predicted next-token distribution and the true next token:

$$
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_{t} \mid x_{<t})
= -\frac{1}{T}\sum_{t=1}^{T} \log \mathrm{softmax}\big(z_t\big)_{x_t},
$$

where $z_t \in \mathbb{R}^V$ are the logits the network produces at position
$t$. "Teacher forcing" means at every position the true prefix $x_{<t}$ is fed
in; the target at position $t$ is simply the next token $x_{t+1}$.

**Code:** `src/model.py` — `F.cross_entropy(logits.view(-1, V), targets.view(-1))`,
with `targets` the inputs shifted by one (`src/data.py`, `TokenWindows`).

**Perplexity** is just the exponentiated loss, $\mathrm{PPL} = e^{\mathcal{L}}$ —
the effective number of equally-likely choices the model is torn between. A
loss of $1.3$ nats $\Rightarrow$ perplexity $\approx 3.7$.

---

## 2. Tokenization

Raw text is turned into token ids by **byte-pair encoding** (BPE): start from
bytes, greedily merge the most frequent adjacent pair repeatedly until the vocab
reaches $V$. We reuse GPT-2's exact merges (via `tiktoken`), so the model is
tokenizer-compatible with `ember`. Stories are separated by the special
end-of-text token $\text{EOT}=50256$, which teaches the model where documents
begin and end. **Code:** `src/tokenizer.py`.

---

## 3. The model: a decoder-only transformer

### 3.1 Embeddings

The input ids are mapped to vectors and given a learned position code:

$$
h^0_t = W_e[x_t] + W_p[t], \qquad W_e \in \mathbb{R}^{V\times d}, \; W_p \in \mathbb{R}^{T_{\max}\times d}.
$$

**Code:** `wte`, `wpe` in `src/model.py`.

### 3.2 A transformer block (pre-LayerNorm)

Each of the $L$ blocks refines the hidden states with two residual sub-layers —
attention then MLP — each preceded by LayerNorm:

$$
\begin{aligned}
h' &= h + \mathrm{Attn}(\mathrm{LN}(h)), \\
h'' &= h' + \mathrm{MLP}(\mathrm{LN}(h')).
\end{aligned}
$$

Residual connections let gradients flow directly; pre-LN (normalizing the
*input* of each sub-layer) is what makes deep transformers train stably.
**Code:** `Block.forward`.

**LayerNorm** normalizes each token vector to zero mean / unit variance, then
applies a learned affine map:

$$
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
\quad \mu = \tfrac{1}{d}\sum_i x_i, \; \sigma^2 = \tfrac{1}{d}\sum_i (x_i-\mu)^2.
$$

### 3.3 Causal self-attention

Project the (normalized) hidden states to queries, keys, values, split into $H$
heads of size $d_k = d/H$, and for each head compute scaled dot-product
attention with a **causal mask** $M$ ($M_{ij}=0$ if $j\le i$, else $-\infty$):

$$
Q = XW_Q,\; K = XW_K,\; V = XW_V, \qquad
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V.
$$

- The $QK^\top$ term scores how much each position should attend to each other
  position; the $1/\sqrt{d_k}$ scaling keeps the dot products from growing with
  dimension and saturating the softmax.
- The **causal mask** enforces autoregression: position $t$ may only attend to
  positions $\le t$, so the prediction of $x_{t+1}$ never sees the future. This
  is the single constraint that makes the model a valid factorization of §1.
- Heads are computed in parallel and concatenated, then projected back by $W_O$.

**Code:** `CausalSelfAttention` — one fused `c_attn` matmul produces $Q,K,V$;
`F.scaled_dot_product_attention(..., is_causal=True)` applies the mask and
softmax in one (flash) kernel; `c_proj` is $W_O$.

### 3.4 MLP

A position-wise two-layer network with a $4d$ hidden size and a GELU
nonlinearity:

$$
\mathrm{MLP}(x) = W_2\,\mathrm{GELU}(W_1 x), \quad W_1\in\mathbb{R}^{4d\times d},\; W_2\in\mathbb{R}^{d\times 4d}.
$$

$\mathrm{GELU}(u) = u\,\Phi(u)$ (Gaussian CDF $\Phi$); we use the $\tanh$
approximation, matching GPT-2. **Code:** `MLP`.

### 3.5 Output head (weight tying)

After the final LayerNorm, logits are produced by projecting with the **transpose
of the token-embedding matrix** — the head *shares* $W_e$ (weight tying), which
saves $V\times d \approx 19\text{M}$ parameters and regularizes:

$$
z_t = W_e\, \mathrm{LN}_f(h^L_t) \in \mathbb{R}^V, \qquad
p_\theta(\cdot \mid x_{\le t}) = \mathrm{softmax}(z_t).
$$

**Code:** `F.linear(x, self.wte.weight)` in `GPT.forward`.

### 3.6 Why init loss $\approx \ln V$

At initialization the logits are ~uniform, so $p \approx 1/V$ for every token and
the cross-entropy is $-\ln(1/V) = \ln V$. For $V=50257$ that's $\approx 10.82$;
for the smoke test's $V=512$, $\ln 512 \approx 6.24$. Seeing this exact value on
step 0 is a quick proof the logits and loss are wired correctly.
**Code:** `scripts/smoke_test.py` check [2].

---

## 4. Optimization

### 4.1 AdamW

Per-parameter adaptive step with **decoupled** weight decay (Loshchilov &
Hutter). With gradients $g_t$:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, &\hat m_t &= m_t/(1-\beta_1^t),\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2, &\hat v_t &= v_t/(1-\beta_2^t),\\
\theta_t &= \theta_{t-1} - \eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\Big).
\end{aligned}
$$

"Decoupled" means the decay $\lambda\theta$ is applied directly to the weights,
not folded into the gradient. We decay only the 2-D matmul weights; biases,
LayerNorm gains, and embeddings are left alone. $\beta=(0.9, 0.95)$ (the LM
convention — a slightly lower $\beta_2$ than vision's $0.999$).
**Code:** `GPT.configure_optimizers`.

### 4.2 Learning-rate schedule

Linear **warmup** for $W$ steps (avoids destabilizing the fresh, high-variance
network), then **cosine decay** to a floor $\eta_{\min}$ over the planned
horizon $S$ steps:

$$
\eta(s) =
\begin{cases}
\eta_{\max}\,\dfrac{s+1}{W}, & s < W,\\[2ex]
\eta_{\min} + \tfrac{1}{2}\big(\eta_{\max}-\eta_{\min}\big)\Big(1 + \cos\pi\,\dfrac{s-W}{S-W}\Big), & s \ge W.
\end{cases}
$$

The schedule is a pure function of the **global step** $s$, which is exactly why
training resumes bit-for-bit after a pause: restore $s$ and the LR continues on
its curve. **Code:** `lr_at` in `src/train.py`.

### 4.3 Gradient accumulation & clipping

To get an effective batch of $B_\text{eff} = B \cdot A$ without the memory of a
$B_\text{eff}$-sized batch, we average gradients over $A$ micro-batches before
each optimizer step (each micro-loss scaled by $1/A$). Global-norm clipping
$g \leftarrow g\cdot\min(1, c/\lVert g\rVert)$ caps rare loss spikes.
**Code:** the `grad_accum` loop + `clip_grad_norm_` in `src/train.py`.

---

## 5. Parameter count (default config)

$d=384,\ L=6,\ H=6,\ V=50257,\ T_{\max}=256$:

| Component | Formula | Params |
| :-- | :-- | --: |
| Token embedding $W_e$ (tied to head) | $V\,d$ | 19.30M |
| Position embedding $W_p$ | $T_{\max}\,d$ | 0.10M |
| Attention / block | $4d^2 + 4d$ | 0.59M |
| MLP / block | $8d^2 + 5d$ | 1.18M |
| LayerNorms / block | $4d$ | ~0.002M |
| $\times\,L=6$ blocks | | 10.65M |
| Final LN | $2d$ | ~0.001M |
| **Total** | | **≈ 30.0M** |

The embedding table is $\sim\!64\%$ of the parameters — typical for a small model
with a large BPE vocabulary, and the reason weight tying matters so much here.

---

## 6. Generation

Given a prompt, sample one token at a time, appending each to the context. The
next-token distribution can be shaped:

- **Temperature** $\tau$: divide logits before softmax, $p \propto \exp(z/\tau)$.
  $\tau\to 0$ is greedy (argmax); $\tau>1$ flattens (more random).
- **Top-$k$**: keep only the $k$ highest-logit tokens, renormalize.
- **Top-$p$ (nucleus)**: keep the smallest set whose cumulative probability
  exceeds $p$.

**Code:** `GPT.generate` (local sampler); `ember` does the production version.

At inference the causal mask means the keys/values of past tokens don't change,
so they can be **cached** rather than recomputed — this is the KV cache, and
`ember` pages it. The math of a single attention step is identical to §3.3; only
the bookkeeping differs. That shared math is why a model trained here serves
there unchanged.

---

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2019)
- Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* (2023)
- Karpathy, *nanoGPT* — the reference minimal GPT training loop
