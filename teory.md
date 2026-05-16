## Theory — sequence-model background (slides pp. 102–156, "13. Sequence models - part I.pdf")

The choices baked into `CFG` and the pipeline below map 1-to-1 onto the lecture deck. Cross-references are kept for the report.

### Problem framing (slides 102–105, 109, 111)

A discrete-time dynamical system in state-space form `x[k+1]=f(x[k],u[k])`, `y[k]=g(x[k])` is equivalent (under mild assumptions) to the **regression form** `y[k] = f(u[k..k-nu+1], y[k-1..k-ny])` — no hidden state, only past I/O samples. This is exactly the SISO task here.

- **NNARX** (slide 109) realises `f` directly with a static MLP over a lagged regressor — what `build_narx_windows` + `NNARX` (cell 14) do.
- **RNN/LSTM/GRU** (slide 111) realise the full state-space form: the recurrent layers play the role of `f(·)` (state-to-state), the linear head plays `g(·)` (state-to-output) — what `SeqModel` (cell 16) does.

### Two error definitions (slides 106–108, 110)

The deck distinguishes:

- **1-step-ahead prediction error (OSA)** — output at `k` computed from *true* past samples of the test set.
- **Simulation error** — output at `k` computed from the *model's own* past predictions (free-run).

Slide 110 makes the key claim: NNARX trained on 1-step loss carries an implicit equation-error noise structure → strong OSA, weak simulation. RNNs trained against the simulation error mitigate this. Our reporting (`vaf_osa`, `vaf_3step`, `vaf_sim` from cells 22/29) covers both regimes so this asymmetry is visible in Phase 3. The k-step curriculum (`narx_kstep_schedule`, `rec_kstep_schedule`) is the practical bridge between the two losses.

### Vanishing / exploding gradients (slides 118–121, Pascanu et al. 2013)

BPTT gradient through `t-k` time steps factors as

```
∂a<t>/∂a<k> = ∏_{k<i≤t} W_aaᵀ · σ'(a<i-1>)
```

so `‖W_aa‖<1` → vanishing, `‖W_aa‖>1` → exploding. This is the textbook motivation for:

- preferring **LSTM/GRU over vanilla RNN** for long horizons (additive cell-state path bypasses the product);
- **gradient clipping** (`CFG["rec_grad_clip"]=1.0` in `train_seq`).

### Cell equations (slides 122–147)

Vanilla RNN unit (slides 122–126):

```
a<t> = tanh(W_a [a<t-1>, x<t>] + b_a)
y<t> = W_y a<t> + b_y
```

GRU — full form, slide 137 (what `nn.GRU` implements):

```
g_r = σ(W_r [c<t-1>, x<t>] + b_r)        # relevance / reset gate
g_u = σ(W_u [c<t-1>, x<t>] + b_u)        # update gate
c̃<t> = tanh(W_c [g_r · c<t-1>, x<t>] + b_c)
c<t>  = g_u · c̃<t> + (1 - g_u) · c<t-1>
a<t>  = c<t>
```

LSTM — slides 144–147 (matches `nn.LSTM`):

```
g_u = σ(W_u [a<t-1>, x<t>] + b_u)        # input / update
g_f = σ(W_f [a<t-1>, x<t>] + b_f)        # forget
g_o = σ(W_o [a<t-1>, x<t>] + b_o)        # output
c̃<t> = tanh(W_c [a<t-1>, x<t>] + b_c)
c<t>  = g_u · c̃<t> + g_f · c<t-1>
a<t>  = g_o · tanh(c<t>)
```

The additive `c<t> = g_u · c̃<t> + g_f · c<t-1>` recurrence is what makes the Jacobian product on slide 121 well-behaved — when `g_f ≈ 1`, gradients pass through unattenuated.

### Bidirectional RNN (slides 148–151) — not used here

`y^<t> = g(W_y [a→<t>, a←<t>] + b_y)` uses both future and past hidden states. Closed-loop simulation requires causal access only, so a bidirectional pass would leak future samples into the prediction and inflate test metrics unrealistically. We keep the recurrent stack strictly causal.

### Deep RNN (slides 152–156)

Stacking recurrent layers — `a^{[ℓ]<t>} = g(W^{[ℓ]} [a^{[ℓ]<t-1>}, a^{[ℓ-1]<t>}] + b^{[ℓ]})` — adds depth in both time and feature axes. This is `CFG["rec_layers"]`; the Phase 2 sweep over layer count exists for exactly this reason.