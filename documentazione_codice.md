# Documentazione del codice — `project-si-mlicv.ipynb`

Spiegazione dettagliata, **cella per cella e funzione per funzione**, del notebook che realizza il progetto §2.5 (System Identification del dataset EEG-wrist di Vlaar et al., 2018). Il filo conduttore è duplice:

1. **Cosa fa il codice** (input, output, struttura interna).
2. **Perché è scritto così** (vincoli del problema, scelte di modellazione, ottimizzazioni).

Riferimenti incrociati: il file [specs.md](specs.md) fissa la consegna; [teory.md](teory.md) mappa cellule ↔ slide di teoria; [benchmark_eeg_literature.md](benchmark_eeg_literature.md) fissa i numeri della letteratura usati come target.

---

## 0. Architettura generale

Il notebook è organizzato come una **pipeline monolitica guidata da un unico dizionario di configurazione `CFG`** ([cella 4](project-si-mlicv.ipynb)). Tutte le fasi (caricamento, split, training, valutazione, sweep) leggono parametri da `CFG` (o da copie di `CFG` modificate via `dict(CFG)`). Questa convenzione è ribadita anche in [CLAUDE.md](CLAUDE.md): non c'è un secondo oggetto di configurazione, e qualunque variazione passa per la mutazione di una copia.

Pipeline a stadi:

| Stadio | Celle | Output |
|---|---|---|
| Imports + device | 2 | `DEVICE`, seed globale |
| Configurazione | 4 | `CFG` |
| Loaders | 6 | `u_all, y_all` di shape `[S, M, N]` |
| Split | 10 | `(u_tr, y_tr), (u_va, y_va), (u_te, y_te)` |
| Metriche | 12 | `nrmse`, `vaf`, `count_params` |
| NNARX | 14 | `NNARX`, `build_narx_windows`, `build_narx_rollout` |
| Ricorrenti | 16–18 | `RNNModel`, `LSTMModel`, `GRUModel` (cell hand-rolled) |
| Training | 20 | `train_narx`, `train_seq` |
| Eval simulazione | 22 | `eval_narx`, `simulate_seq`, `kstep_seq` |
| FLOPs | 24–27 | `flops_per_sample` |
| Runner Phase 1 | 29–33 | `run_nnarx`, `run_rnn`, `run_lstm`, `run_gru`, `run_arx_linear` |
| Sysidentpy add-on | 37 | `run_nnarx_frols`, `run_poly_narmax` |
| Phase 2 | 45–53 | sweep architetturali |
| Phase 3 | 55 | tabella comparativa |
| LOSO esterno | 56 | cross-validation per soggetto |

---

## 1. Imports e device — [cella 2](project-si-mlicv.ipynb)

```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

* **Seed unico** per `random`, `numpy`, `torch`. Ricomparirà ad ogni runner (es. `_train_one_nnarx`, `repeat_seeds`) per garantire **riproducibilità** anche con il multi-seed di Phase 2.
* `DEVICE` auto-detect: il notebook è pensato per girare sia in locale su CPU che su Kaggle con GPU senza modifiche.

---

## 2. Configurazione — [cella 4](project-si-mlicv.ipynb)

Il dizionario `CFG` raccoglie **tutti gli iper-parametri** delle tre fasi. Punti di rilievo:

### Dataset e split
* `dataset: "medium"` — i dati grezzi `[M, P, N_raw]` per soggetto (il `"small"` è la versione già pre-processata).
* `split: "loso"` — leave-one-subject-out, il setup raccomandato dalla letteratura (Vlaar et al.) per evitare leakage tra realizzazioni dello stesso soggetto.

### NNARX
* `narx_u_taps = range(0, 20)` e `narx_y_taps = range(1, 6)` — **lag consecutivi densi** allineati a Gu 2021 (NARMAX), coprono ~78 ms di input e ~20 ms di feedback a 256 Hz. La versione precedente usava lag **dilatati** (0,1,2,4,8,16,32,64) ma saltava l'intervallo 4–16 dove cade la dinamica più informativa.
* `narx_act = "gelu"`, `narx_skip = True`, `narx_residual = True`, `narx_u_feats = ("du",)` — irrobustimenti dell'MLP:
  * skip-connection lineare in parallelo all'MLP (ARX residuo);
  * predizione di `Δy[k] = y[k] - y[k-1]` invece di `y[k]` (residuo);
  * feature derivata `du[k] = u[k] - u[k-1]` per fornire esplicitamente la velocità dell'angolo di stimolo.
* `narx_y_noise_std = 0.05` — **noise injection** sulla feedback in input al regressor (in teacher forcing), riduce il gap fra TF e simulazione (Bengio 2015).
* `narx_kstep_schedule = (1, 5, 10, 20, 40, 80, 160, 320)` con `narx_kstep_advance = "plateau"` — curriculum k-step: si inizia in teacher forcing puro (k=1) e si avanza a un orizzonte più lungo **solo quando** la val-NRMSE è in plateau (vedi `_resolve_kstep_schedule` e `train_narx`). Allunga fino a 320 (= ~1.25 s a 256 Hz) per coprire l'orizzonte di simulazione realistico.
* `narx_curriculum_lr_drop = 0.5` + `narx_curriculum_restore = True` — ad ogni avanzamento del curriculum si ricarica il best-so-far e si dimezza la LR (warm restart): evita di partire da un punto già peggiorato dal nuovo loss.
* `narx_sched_sampling_p_max = 0.5` — scheduled sampling (Bengio 2015): nei rollout k>1 si miscela ground-truth e predizione del modello come input al passo successivo, con probabilità che cresce linearmente.
* `narx_ensemble = 3` — media di 3 reti addestrate con seed diversi, riduce la varianza di simulazione.
* `narx_spec_lambda = 0.1` con `narx_spec_kstep_min = 16` — termine **spettrale** nella loss: `|FFT(y_pred) - FFT(y_true)|^2`. Si attiva solo quando il rollout copre abbastanza campioni perché la FFT sia informativa. Aiuta a evitare drift in frequenza.

### Ricorrenti
* `rec_hidden = 64`, `rec_layers = 1` — taglia di partenza tipica.
* `rec_kstep_schedule = (1, 5, 20, 64, 256, 512, 1024)` — analogo curriculum k-step, fino a 1024 (metà del segmento di test, `N = 2048`). BPTT su 2048 timestep esploderebbe la memoria, quindi è abilitato `rec_tbptt_len = 256` che **disaccoppia la finestra di gradiente dall'orizzonte di rollout** (vedi `train_seq`).
* `rec_use_layernorm = True`, `rec_var_dropout = 0.1` — gli ingredienti di stabilità per LSTM/GRU su rollout lunghi (`_HandLSTM`, `_HandGRU` implementano LayerNorm sulle pre-attivazioni dei gate e dropout **variazionale** con maschera locked-in-time).

### Training
* `epochs_narx = epochs_rec = 200`, `batch_narx = 512`, `lr = 1e-3`.
* `phase2_seeds = 3` — il multi-seed reporting in Phase 2 produce `<metric>_std` accanto a `<metric>`.

---

## 3. Caricamento dataset — [cella 6](project-si-mlicv.ipynb)

### `find_mat_file(filename)`
Risolve il path del `.mat` cercando in (1) sottocartella locale, (2) cartella corrente, (3) `/kaggle/input/...` (per supportare Kaggle out-of-the-box). Lancia `FileNotFoundError` se il file non esiste.

### `load_small(mat_path)`
Carica il `.mat` già preprocessato e ritorna direttamente `u, y` di shape `[S, M, N]` in `float32`.

### `load_medium(mat_path, ds=8)`
È la **traduzione fedele in Python** dello script MATLAB `Load_Plot_EEG_2.m`:

1. Per ogni soggetto estrae `angle [M, P, N_raw]` (input) e `comp [M, P, N_raw]` (output ICA).
2. **Downsampling** per indice `[::ds]` con `ds=8` → da 2048 Hz a 256 Hz (allineato a `CFG["fs"]`).
3. **Media sui P periodi** (axis=1) → shape `[M, N]` per soggetto.
4. **Zero-mean per realizzazione** (`axis=-1`) — rimuove offset DC.
5. **Scaling per soggetto** dividendo per la media (sulle M realizzazioni) della std lungo il tempo: `u / mean_m(std_t(u))`. Non si normalizza per realizzazione perché distruggerebbe le differenze di ampiezza tra realizzazioni dello stesso soggetto, che sono informazione fisiologica.
6. **`np.roll(u, 5, axis=-1)`** — equivalente al MATLAB `circshift(u, 5, 3)`: introduce il delay di stimolo coerente con il delay della risposta corticale.

Tutte queste scelte (downsampling, mean-of-stds, delay) **devono restare in sync con la reference MATLAB**: cambiarle inquinerebbe il confronto con la letteratura — cfr. nota in [CLAUDE.md](CLAUDE.md).

### Selettore finale
```python
if CFG["dataset"] == "small":
    u_all, y_all = load_small(...)
elif CFG["dataset"] == "medium":
    u_all, y_all = load_medium(...)
```

`S, M, N` (`participants, realizations, samples`) sono variabili globali usate nei loop di LOSO.

---

## 4. Plot ispettivo — [cella 8](project-si-mlicv.ipynb)

Plot di una singola realizzazione (`u_all[0,0]` e `y_all[0,0]`) come check sanity sul caricamento. Non altera lo stato.

---

## 5. Split train/val/test — [cella 10](project-si-mlicv.ipynb)

### `make_split(u_all, y_all, cfg)`
Split **within-participant**: usa gli indici di realizzazione `train_real_idx`, `val_real_idx`, `test_real_idx` per partizionare lungo l'axis `M`. Il `reshape(-1, N)` appiattisce `[S, M, N]` → `[S*M, N]` in modo che tutto il downstream sia split-agnostic.

### `make_loso_split(u_all, y_all, cfg)`
Split **leave-one-subject-out**:
* Tutte le realizzazioni del soggetto `loso_test_subject` finiscono nel test set.
* Sui restanti `S-1` soggetti, le realizzazioni `loso_val_real_idx` (default `[5, 6]`) vanno in validation; il resto in training.

Entrambe le funzioni ritornano triple `[num_seqs, N]`. È una scelta consapevole: il downstream (`build_narx_windows`, `simulate_seq`, …) opera su **liste/array di sequenze**, mai su sottocubi `[S, M, N]`.

---

## 6. Metriche — [cella 12](project-si-mlicv.ipynb)

* `nrmse(y_true, y_pred) = RMSE(y_true - y_pred) / std(y_true)` — il `1e-12` evita divisioni per zero (segmenti costanti).
* `vaf(y_true, y_pred) = 100 * (1 - var(err) / var(y_true))` — *Variance Accounted For*, in **%**, la metrica standard nei paper EEG.
* `count_params(model)` — somma dei `numel()` per i parametri `requires_grad`. Esclude buffer (es. le mask `_u_taps_t` registrate come buffer non-persistent in `NNARX`).

---

## 7. NNARX — [cella 14](project-si-mlicv.ipynb)

Questa è la cella più densa, contiene il **cuore** del NNARX: regressor, classi, e due funzioni di costruzione dei batch.

### `make_u_feats(u, u_feats=())`
Costruisce un tensore `[T, F]` con colonne `[u, du, u**2]` (sottoinsieme selezionato da `u_feats`). Pre-calcola `du` come differenze prime con `du[0] = 0`. `usq` (u quadrato) è disponibile ma non attivo di default.

### `_normalize_taps(u_taps, y_taps, nu, ny)`
* Se `u_taps`/`y_taps` sono `None`, sostituisce con tap consecutivi `0..nu-1` e `1..ny`.
* **Asserzioni**: `u_taps >= 0` (può includere 0, cioè `u[k]` corrente), `y_taps >= 1` (esclude `y[k]` corrente per evitare leakage — il modello deve **predire** `y[k]`, non leggerlo).

### `build_narx_windows(u_seqs, y_seqs, u_taps, y_taps, residual, u_feats)`
Costruzione **vettorizzata** dei dataset 1-step teacher-forced.

* Crea `uf = make_u_feats(u, u_feats)` di shape `[T, F]`.
* Calcola `L = max(Lu, Ly)` (lag warm-up) e gli indici di campionamento `ks = arange(L, T)`.
* Costruisce per fancy-indexing tutti i lag in un colpo: `uf[ks[:,None] - u_taps[None,:]]` produce `[K, nu, F]` → reshape a `[K, nu*F]`.
* Concatenazione finale `[u_lags | y_lags]` (ordine colonne canonico, importante perché `train_narx` slica `xb[:, y_lo:y_hi]` per iniettare rumore solo sulla y feedback).
* Se `residual=True`, il target è `Δy = y[L:] - y[L-1:-1]`.

La motivazione della vettorizzazione: con `K ≈ N - L ≈ 2000` campioni × `S*M ≈ 70` realizzazioni si supera facilmente `1e5` righe; un loop Python sarebbe collo di bottiglia.

### `build_narx_rollout(u_seqs, y_seqs, u_taps, y_taps, kstep, u_feats)`
Costruisce le finestre per il **rollout differenziabile** `k`-step:

* `U_win` di shape `[K, Lu + kstep - 1, F]` — copre `u[k_start - Lu + 1 .. k_start + kstep - 1]`, cioè tutti gli `u` che serviranno al rollout.
* `Y_warm` di shape `[K, Ly]` — gli ultimi `Ly` valori veri di `y` per inizializzare il buffer di feedback.
* `Y_tgt` di shape `[K, kstep]` — i target per il rollout.

Nota la formula `win_idx = (ks - Lu + 1)[:, None] + arange(win_len)[None, :]` — è una broadcasting matrix `[K, win_len]` che evita loop. Lo stesso pattern è usato in `eval_3step_narx`.

### Classe `NNARX(nn.Module)`

Costruttore (`__init__`):
* Memorizza `u_taps`, `y_taps`, deriva `nu`, `ny`, `Lu`, `Ly`, `L = max(Lu, Ly)`.
* `n_u_feats = 1 + len(u_feats)` (i.e. `u` da solo, o `[u, du]`, ecc.).
* `in_dim = nu * n_u_feats + ny`: prima parte input (con eventuale `du`), seconda parte y feedback.
* MLP `nn.Sequential` con larghezze `[in_dim] + hidden + [1]` e attivazione configurabile (`tanh`/`relu`/`gelu`).
* `self.skip = nn.Linear(in_dim, 1)` se `skip=True` — è una **skip lineare globale** (ARX linearizzato in parallelo all'MLP).
* `register_buffer(..., persistent=False)` per i tap come tensori GPU — accelerano l'indicizzazione in `rollout` evitando conversioni numpy/torch ad ogni passo.

`_head(x)`: applica MLP + skip (somma additiva).

`forward(x)`: alias di `_head` (la classe è usata sia in TF puro che in rollout, ma il forward "esterno" è solo il TF batch).

`simulate(u_seq, y_warmup)` — **simulazione free-run** in puro NumPy:
* Estrae i parametri dell'MLP in tensori NumPy una volta sola.
* Loop sequenziale da `L` a `N`, ricostruendo il regressor `[u_lags | y_hat_lags]` con il `y_hat` parzialmente costruito.
* Su CPU, la rete è abbastanza piccola da fare overhead di kernel-launch CUDA peggiore di un puro NumPy → da qui la scelta di tenere la simulazione in NumPy (commentato esplicitamente nel codice).
* Se `residual=True`, somma `y_hat[k-1]` all'output (interpretato come `Δy`).

`simulate_osa(u_seq, y_seq)` — **one-step-ahead**: come `simulate` ma usa `y_seq` (vero) come feedback, mai `y_hat`. Resta su GPU perché qui non c'è dipendenza ricorrente nei dati (è teacher forced). Serve per la colonna **OSA** della tabella Phase 1/3.

`rollout(U_win, Y_warm, kstep, Y_true=None, p_sample=0.0)` — **rollout differenziabile** k-step, usato in training:
* Per ogni step `j` in `range(kstep)`:
  * Seleziona `u_slice = U_win[:, Lu-1+j - u_taps, :]` (è un `index_select` su una dimensione fissa, quindi differenziabile e veloce).
  * Costruisce il regressor con il buffer corrente `y_buf` e fa forward.
  * Aggiorna `y_buf` *roll-and-append*: rimuove il più vecchio, appende `y_fb` (feedback).
* **Scheduled sampling**: se `p_sample > 0` e `Y_true` è fornito, `y_fb = mask*Y_true[:, j] + (1-mask)*y_pred.detach()`. Il `.detach()` è cruciale: il gradiente del passo j+1 non risale fino al passo j attraverso il feedback predetto (altrimenti il grafo computazionale esplode con `kstep` lunghi).
* Loss però è sempre calcolata su `preds` (output del modello), mai sul feedback misto.

---

## 8. Modelli ricorrenti — [celle 16, 17, 18](project-si-mlicv.ipynb)

### Convenzione comune
Tutti i modelli (`RNNModel`, `LSTMModel`, `GRUModel`) espongono lo stesso contratto:
* Input: `u [B, T, 2]` con due canali `(u, y_prev)`.
* Output: `[B, T, 1]`.
* Espongono `.rnn` (cella) e `.head` (lineare 1) come attributi, perché `train_seq` e `simulate_seq` chiamano `model.rnn(x_t, h)` un passo alla volta nel rollout k-step.

### `RNNModel`
Wrapper su `nn.RNN(nonlinearity="tanh", batch_first=True)` + linear head. È un Elman RNN classico, scelto come baseline più semplice tra i ricorrenti.

### `_HandLSTM` e `_HandGRU` — celle 17, 18
**Perché reimplementare a mano LSTM/GRU?** Tre ragioni:

1. **LayerNorm sui gate**: `nn.LSTM` PyTorch non supporta LayerNorm fra i gate. Lo stile Cooijmans 2016 (LSTM) e Ba 2016 (GRU) richiede di normalizzare il vettore pre-attivazione di shape `[4H]` (o `[3H]`). Solo una cella manuale lo permette.
2. **Dropout variazionale**: maschera di dropout su `h_list[l]` **bloccata nel tempo** (Gal & Ghahramani 2016). `nn.LSTM` applica dropout solo *tra layer*, non sul hidden state — quindi non utile per stabilizzare rollout di 1024 step.
3. **Param-count parity**: la layout di `W_ih [in→4H]` e `W_hh [H→4H]` (con bias entrambi) replica esattamente PyTorch — quindi il `count_params(model)` è identico a un `nn.LSTM` della stessa taglia, e i confronti di FLOPs/parametri restano onesti.

Forward `_HandLSTM`:
* Inizializza `h, c` a zero se non passati.
* Pre-campiona le mask dropout (una per layer, mantenute identiche su tutti i T step).
* Loop `for t in range(T)`:
  * Per ogni layer: `gates = W_ih(layer_in) + W_hh(h * mask)`, eventuale `LayerNorm`, split `i, f, g, o`, gate math standard.
  * `layer_in = h_new` propaga verso il layer successivo.
* Output `[B, T, H]` + stato finale `(h_n, c_n)`.

`_HandGRU` analogo: 3 gate (reset, update, new); convenzione PyTorch `n = tanh(x_n + r * h_n)` (il reset gate moltiplica la parte hidden-to-new, non il tutto). LayerNorm separato su `W_ih(x)` e `W_hh(h)`.

I due wrapper finali `LSTMModel`/`GRUModel` pesano un linear head sopra la cella.

---

## 9. Training loops — [cella 20](project-si-mlicv.ipynb)

### `_resolve_kstep_schedule(kstep_schedule, epochs)`
Normalizza la schedule del curriculum k-step in due rappresentazioni parallele:
* `sched_fixed = [(start_epoch, k), ...]` per la modalità "fixed-epoch" (legacy).
* `sched_k = [k0, k1, ...]` per la modalità "plateau" (usata di default).

Accetta sia tuple `(threshold, k)` che interi bare; sopporta soglie frazionarie in `(0, 1]` (che vengono mappate su `epochs`).

### `_stage_min_epochs(min_epochs, stage_idx)`
Helper per leggere il minimo numero di epoche allo stage corrente da una sequenza/scalar.

### `train_narx(model, u_tr, y_tr, u_va, y_va, ...)`

Funzione ampia. La spiego per blocchi logici.

**Setup**:
* Adam optimizer + opzionale CosineAnnealingLR.
* `loss_fn = MSELoss`.
* `DataLoader` per il branch 1-step (`dl1` da `build_narx_windows`).
* `rollout_cache = {}` — cache lazy dei `DataLoader` k-step, perché ricostruire `build_narx_rollout` ad ogni epoca sarebbe sprecato.

**Loop di training**:
```python
for ep in range(epochs):
    # 1) determina kstep corrente (plateau-gated o fixed)
    # 2) se cambiato rispetto a prev_kstep -> curriculum advance:
    #    - reset patience, plateau counter
    #    - se curriculum_restore: ricarica best_state
    #    - se curriculum_lr_drop: pg["lr"] *= drop
    # 3) ramp p_sample (scheduled sampling) linearmente 0->p_max
    # 4) train one epoch:
    #    - se kstep <= 1: 1-step TF + noise injection su y feedback
    #    - se kstep > 1: rollout differenziabile con loss MSE
    #      + opzionale FFT-magnitude loss
    #      + opzionale geometric step weighting w_j = step_decay**j
    #    - grad clip + step
    # 5) val periodica con eval_narx (simulazione **vera**, non TF)
    # 6) patience + best-state restore
    # 7) plateau-advance: se stage_bad_checks >= kstep_plateau_checks
    #    AND elapsed_in_stage >= kstep_min_epochs[stage] -> advance
```

**Dettagli importanti**:
* `y_lo, y_hi` localizzano le colonne y nel regressor per applicare rumore solo a quelle (le colonne u vanno lasciate pulite).
* La FFT-magnitude loss attiva solo per `kstep >= spec_kstep_min` (FFT su un orizzonte troppo corto è priva di senso spettrale).
* Lo `step_decay` pesa i passi di rollout in modo geometrico: `step_decay = 0.95` darebbe più peso ai primi passi (predizioni più certe). Default `1.0` (off).
* `bad_checks` (early stopping su val sim NRMSE) si **resetta ad ogni curriculum advance**: cambia il problema, ricomincia il conteggio.
* `stage_bad_checks` invece traccia il **plateau locale**: serve a decidere quando avanzare al prossimo k.

**Why plateau-gated curriculum?** Schedule fissi (es. avanza a k=5 a epoch 30) sono fragili: se il modello a k=1 non ha ancora convergito, lo si butta in k=5 prematuramente e diverge. Il gate "elapsed ≥ min_epochs AND val NRMSE in plateau" è più robusto.

### `train_seq(model, u_tr, y_tr, ...)`

Analogo per i ricorrenti. Differenze chiave:

**Tensorizzazione**: i dati di training sono **tutti** su GPU come `U_tr [B, T, 1]`, `Y_tr [B, T, 1]`, `Y_prev_tr` (versione shiftata di 1 di Y_tr). Possibile perché un'intera realizzazione sta in memoria.

**Branch 1-step**: vettorializzato:
```python
X = cat([U_tr, Y_prev_tr + noise], dim=-1)
loss = MSE(model(X), Y_tr)
```
Tutte le `[B, T]` predizioni in un singolo forward.

**Branch k-step (TBPTT chunked rollout)**:
```python
chunk = tbptt_len or kstep
h = None
for t in range(T):
    if t == 0: y_prev = 0
    elif t % kstep == 0: y_prev = Y_tr[:, t-1]  # full TF reset
    else:
        # scheduled sampling on non-reset steps
        y_prev = mask*Y_tr[:, t-1] + (1-mask)*last_pred
    out, h = model.rnn(cat([u_t, y_prev]), h)
    yp = model.head(out)
    preds.append(yp); last_pred = yp
    # close chunk every `chunk` steps
    if ((t+1) - chunk_start) >= chunk or t == T-1:
        loss_chunk.backward()  # backprop only on this chunk
        opt.step(); opt.zero_grad()
        h = h.detach()   # cut graph
        chunk_start = t+1
```

**Perché disaccoppiare `chunk` da `kstep`?** Se `kstep=1024` e si fa backward su tutto il chunk, l'OOM è garantito su qualsiasi GPU consumer. Ma vogliamo comunque che `y_prev` sia il **predetto** del modello per quasi tutti i passi (non il vero ogni `chunk` step). Quindi: il **reset feedback** avviene ogni `kstep` step (definisce la difficoltà del task), ma il **detach del grafo** avviene ogni `chunk` step (definisce la memoria). Questa separazione è esplicitata nei commenti del codice e nel CFG come `rec_tbptt_len = 256`.

**Validation**: usa `simulate_seq` (cella 22) — è una **vera simulazione free-run** con warm-up, non TF. La metrica di selezione è quindi `nrmse(y_sim, y_true)`, che è ciò che conta in test.

---

## 10. Valutazione in simulazione — [cella 22](project-si-mlicv.ipynb)

### `eval_narx(model, u_seqs, y_seqs)`
Per ogni sequenza, chiama `model.simulate(u, y[:L])` (warm-up = primi `L` campioni veri di y, poi free-run). Restituisce predizioni e target concatenati.

### `eval_seq(model, u_seqs, y_seqs)`
Valutazione **teacher-forced** vettorializzata per i ricorrenti. Veloce, usata come "OSA" (one-step-ahead) per la tabella, perché in TF il modello vede sempre `y[t-1]` vero.

### `simulate_seq(model, u_seqs, y_seqs, warmup=10)`
**Simulazione free-run** dei ricorrenti, una sequenza alla volta:
* Per `k=0`: `y_prev = 0`.
* Per `1 <= k <= warmup`: `y_prev = y[k-1]` (warm-up TF: stabilizza lo stato hidden).
* Per `k > warmup`: `y_prev = y_hat[k-1]` (free-run vero).

Il warm-up TF è necessario perché lo stato nascosto iniziale (`h=None` → zeros) richiede qualche step per "agganciarsi" al segnale. Senza warm-up il primo secondo di simulazione è inutilizzabile e gonfia la NRMSE.

### `_detach_state(h)`
Helper: se `h` è una tupla (LSTM: `(h, c)`), detacha entrambi; altrimenti detacha singolarmente. Usato in `kstep_seq`.

### `eval_3step_narx(model, u_seqs, y_seqs, k=3)`
Predizione **k-step-ahead** per NNARX (default `k=3`, allineato alle convenzioni Gu 2021 / Santos 2023):
* Usa `build_narx_rollout` per costruire le finestre.
* Chiama `model.rollout(U, Yw, k)[:, -1]` — prende **solo l'ultimo step** di ogni rollout.
* I target sono `Y_tgt[:, -1]`.

Il senso: ogni predizione è "tra 3 step da ora, condizionato su y vero fino al passo corrente e u vero fino a 3 step avanti". È un'estensione naturale di OSA verso la simulazione.

### `kstep_seq(model, u_seqs, y_seqs, k=3)`
Analogo per i ricorrenti. **Più sottile** perché lo stato nascosto è continuo nel tempo:
* Prima fa un trunk teacher-forced lungo tutta la sequenza, salvando `h_trace[t]` ad ogni step.
* Poi per ogni `t` da `k` a `T`: prende `h_branch = h_trace[t-k]`, parte da `y[t-k]` come `y_prev`, fa `k` step free-run dentro un branch isolato (`_detach_state` evita di sporcare il trunk), e tiene l'ultima predizione come `y_hat_k[t]`.
* Drop dei primi `k` campioni (non c'è abbastanza storia).

---

## 11. Conteggio FLOPs — [celle 24–27](project-si-mlicv.ipynb)

Il MAC (multiply-accumulate) è contato come **2 FLOP** (1 mul + 1 add), seguendo la convenzione standard.

### `flops_rnn(model)`
Elman RNN a singolo gate (`tanh`):
* Layer 1 vede `[u, y_prev] (in=2)`: `2 * 1 * (2*H + H*H) + H` FLOPs.
* Layer `>= 2` vede `H`: `2 * 1 * (H*H + H*H) + H`.
* Head lineare: `2*H + 1`.

### `flops_lstm(model)`
4 gate (input/forget/cell/output): identico ma `gates = 4`, più `5H` per le nonlinearità (sigm i/f/o, tanh g, tanh c, hadamards collapsati a ~5H). Se LayerNorm: `+ L * 6 * 4H` (≈ 6 op per feature, su `[4H]`).

### `flops_gru(model)`
3 gate; `3H` per le nonlinearità (sigm r/z, tanh n + hadamards in `h = (1-z)*n + z*h`). LayerNorm aggiunge `L * 2 * 6 * 3H` (LN separato su `W_ih(x)` e `W_hh(h)`).

### `flops_per_sample(model)`
Dispatcher: ispeziona il tipo di modello e chiama il helper giusto. Per NNARX, **cammina su `model.net`** e somma `2 * prev_dim * out_dim + bias` per ogni `nn.Linear`, più la skip lineare se presente. Questa scelta evita di hard-codare la geometria del MLP — funziona per qualsiasi `hidden` tuple.

**Perché analytic e non `ptflops`?** Le celle ricorrenti `ptflops` non le gestisce correttamente (gates non standard, LayerNorm custom, dropout variazionale). Affidarsi a un conteggio analitico esplicito è più robusto, ma **occhio**: cambiare la matematica delle cell senza aggiornare anche il conteggio FLOPs porta a numeri sbagliati in tabella — vedi nota in [CLAUDE.md](CLAUDE.md).

---

## 12. Runner Phase 1 — [celle 29–33](project-si-mlicv.ipynb)

### `plot_sim`, `plot_three_sims`, `plot_error_spectrum` — cella 29
Helper di plotting condivisi (NRMSE/VAF nelle didascalie). `plot_three_sims` genera tre pannelli separati (OSA / 3-step / sim) per evidenziare il **divario** tra teacher-forced e free-run — è il tipico failure mode di un NNARX overfittato.

`plot_error_spectrum` fa FFT del residuo per diagnostica: se l'errore concentra in una banda specifica, c'è una dinamica non catturata.

### `RESULTS = {}`
Dizionario globale che accumula i risultati di ogni modello. Letto da Phase 3 (cella 55) per la tabella finale.

### `_make_nnarx(cfg)`
Factory che istanzia un `NNARX` leggendo i parametri da `cfg`. Centralizzato per essere riutilizzato sia da `run_nnarx` che da `_train_one_nnarx` ed evitare drift di parametri.

### `_train_one_nnarx(cfg, seed_offset)`
Addestra **una** rete NNARX:
* Legge `narx_seed_base` da cfg (passato da `repeat_seeds`); il seed effettivo è `base + seed_offset`. Questo doppio livello permette ad un ensemble (intra-runner) di variare il seed e a Phase 2 multi-seed di variare il seed *del base* indipendentemente.
* Chiama `train_narx` passando tutti i parametri NNARX dal cfg con `.get(...)` (i default coprono i casi in cui un campo manca).

### `run_nnarx(cfg, tag="nnarx")` — cella 29
Runner completo:
1. Cicla `n_ens = narx_ensemble` volte chiamando `_train_one_nnarx(cfg, seed_offset=i)`.
2. Per ogni modello calcola la simulazione su test (`model.simulate`), OSA (`model.simulate_osa`), 3-step (`eval_3step_narx`).
3. **Media le predizioni** dei membri dell'ensemble (non i pesi). Le metriche finali sono calcolate sulla media.
4. `flops_per_sample` viene **moltiplicato per `n_ens`** — questo è onesto: per ottenere la predizione finale serve far girare tutti gli ensemble member.
5. Plot e ritorno.

### `run_rnn`, `run_lstm`, `run_gru` — celle 30–32
Stessa struttura:
1. Costruisci il modello.
2. `train_seq(...)` con tutti i parametri `rec_*` (incluso `tbptt_len`, `sched_sampling_p_max`, `use_layernorm`/`var_dropout` per LSTM/GRU).
3. Simulazione free-run (`simulate_seq`), OSA (`eval_seq`), 3-step (`kstep_seq`).
4. Plot e ritorno.

### `run_arx_linear(cfg, ny=5, nu=20, u_delay=0)` — cella 33
Baseline **ARX lineare** chiusa-loop fitted con OLS:
* Costruisce regressor `[y_lags | u_lags]` con `y_taps = 1..ny`, `u_taps = 0..nu-1`.
* `np.linalg.lstsq` per fit dei coefficienti `theta`.
* Simulazione/OSA con cicli Python espliciti.
* **Guard di divergenza**: se la simulazione free-run supera `50 * std(y_true)` (polo instabile), clippa i valori a quella banda e setta `free_run_diverged = True`. Senza questo, una sola sequenza divergente esplode tutta la NRMSE e rende il confronto inutile.
* Le metriche `nrmse_3step`/`vaf_3step` sono `NaN` perché un ARX lineare a 3-step non aggiunge informazione (è linearmente estrapolabile da OSA).
* `params = len(theta)`, `flops_per_sample = 2 * len(theta)`.

**Perché un ARX baseline?** È il "ne vale la pena" check di ogni paper di system ID: se la NN non batte un ARX, il NN non sta imparando nulla di non lineare.

---

## 13. NNARX + sysidentpy — [cella 37](project-si-mlicv.ipynb)

Add-on con due varianti FROLS-based:

### `frols_select_lags(u_seqs, y_seqs, ylag_max=64, xlag_max=64, n_terms=20, degree=1)`
Usa `sysidentpy.FROLS` (Forward Regression with Orthogonal Least Squares) con basi polinomiali di grado 1 (lineari) per **selezionare automaticamente i lag** rilevanti:
* Fit FROLS sui dati train flattened.
* Estrae i codici dei termini selezionati (`var_id`, `lag`) — convenzione sysidentpy: `var_id=1` → y, `var_id=2` → u.
* Restituisce `u_taps`, `y_taps` come tuple di interi unici ordinati.

### `run_nnarx_frols(cfg)`
1. Esegue `frols_select_lags` per derivare `(u_taps, y_taps)` data-driven.
2. Crea `cfg_f = dict(cfg)` con i lag selezionati.
3. Chiama `run_nnarx(cfg_f)` come al solito.
4. Etichetta il modello come `"NNARX-FROLS"`.

Razionale: il NNARX hand-tuned usa lag scelti da Gu 2021; il FROLS controlla se uno **selezionatore automatico** trova lag che vanno meglio sul dataset specifico. Spesso converge su un subset più piccolo (più sparso) di lag.

### `run_poly_narmax(cfg, ylag=15, xlag=15, degree=2, n_terms=15)`
Modello **polinomiale NARMAX** completo via sysidentpy:
* FROLS sceglie i monomi polinomiali (di grado 2), OLS fitta i coefficienti — il tutto in pochi secondi.
* Riporta **tre paia di metriche**:
  * OSA (`steps_ahead=1`): per parità con Gu 2021 / Santos 2023.
  * 3-step (`steps_ahead=3`): guard di divergenza per-sequenza (clipping a `50*std`).
  * Free-run (`steps_ahead=None`): stessa guard.
* `params = len(poly.theta)`, `flops_per_sample = 2 * params`.

Questo è un **secondo baseline** non lineare (ma chiuso, non NN) — confronto onesto per capire se l'NNARX guadagna davvero sui modelli classici.

---

## 14. Phase 2 — sweep architetturale

### `repeat_seeds(runner_fn, cfg, n_seeds, base_seed)` — cella 45
Wrapper che esegue un runner Phase 1 `n_seeds` volte con seed diversi e aggrega i risultati:
* Lo stub `n_seeds <= 1` ritorna direttamente `runner_fn(cfg)` (legacy behaviour).
* Altrimenti per ogni seed: re-seed globale, **stamp `narx_seed_base` e `rec_seed_base` nel cfg** (così `_train_one_nnarx` legge il base e l'ensemble lo varia internamente), esegue il runner.
* Aggrega `mean` e `std` per le metriche numeriche; aggiunge `<metric>_std` accanto a ogni `<metric>`.
* Droppa `history` (le loss curve gonfierebbero il JSON di output).

### Sweep RNN/LSTM/GRU — celle 46–48
Per ogni `(hidden, layers)` in `{32, 64, 128} × {1, 2}`:
* `cfg = dict(CFG)` con override delle taglie.
* `epochs_rec = 100` (sweep usa metà delle epoch della baseline Phase 1).
* `repeat_seeds(run_rnn, cfg, n_seeds=phase2_seeds)`.
* Appende al `PHASE2` con metadata `kind`, `hidden`, `layers`.

### Riepilogo + dump JSON — cella 49
Print ordinato per `nrmse_sim`, scrive `results/phase2_rec.json` (esclude `history`).

### Sweep NNARX `(u_taps, y_taps, hidden)` — cella 51
5 configurazioni di taps (dilated short/medium/long + consec-8 + consec-16) × 3 hidden = 15 run. Curriculum fixed-epoch (più semplice per uno sweep). Dump in `results/phase2_narx.json`.

### Phase 2c — sweep spettrale + dataset opzionale — cella 53
* `RUN_SPECTRAL_SWEEP = True` → sweep di `spec_lambda ∈ {0.0, 0.05, 0.1, 0.2}` su NNARX-FROLS. Abbassa `spec_kstep_min` a 5 (altrimenti con 100 epoch il curriculum non arriva mai a kstep=16 e il loss spettrale non si attiva mai, dando risultati identici).
* `RUN_MEDIUM_DATASET_TRIAL = False` (default) — se attivato e siamo su small, ricarica i dati medium, sostituisce le globali, e gira Phase 1 sui dati medium.

---

## 15. Phase 3 — tabella comparativa — [cella 55](project-si-mlicv.ipynb)

Itera `RESULTS` nell'ordine canonico `(arx_linear, nnarx, nnarx_frols, poly_narmax, rnn, lstm, gru)` e stampa una tabella formattata: `params`, `flops/sample`, `t_train`, NRMSE/VAF in **tre orizzonti** (OSA, 3-step, sim).

* I modelli con `free_run_diverged` o `three_step_diverged` ricevono un asterisco nel nome — la guard nei runner non maschera la diagnosi, la rende esplicita.
* Dump in `results/phase1_metrics.json`.
* Scatter plot `params (log) vs sim NRMSE` — il classico Pareto front di system ID (vuoi i modelli **in basso a sinistra**: pochi parametri, bassa NRMSE).

---

## 16. LOSO CV esterna — [cella 56](project-si-mlicv.ipynb)

Quando `RUN_LOSO_CV = True` e `CFG["split"] == "loso"`:

* Loop `for s in range(S)`: per ogni soggetto come test fold:
  * Override `cfg["loso_test_subject"] = s`.
  * `make_loso_split` per ricostruire i set.
  * Esegui NNARX, RNN, LSTM, GRU.
  * Append `{"fold": s, ...}` a `fold_rows`.
* Aggregazione: `by_model = {model_name: [rows...]}` raccoglie i fold per modello.
* Stampa:
  * Una tabella verbosa `mean ± pstdev` per OSA/3-step/sim NRMSE/VAF.
  * Una tabella compatta in layout Phase 3, con un asterisco se almeno un fold è divergito.
* Dump `results/loso_folds.json`.

Questo loop è il punto **decisivo** del progetto secondo la convenzione del paper Vlaar: la metrica di interesse non è "modello su un test set fisso" ma "modello mediamente su tutti i soggetti held-out", per misurare la **generalizzazione cross-subject**.

---

## 17. Note trasversali su scelte progettuali

### Perché un solo notebook?
La consegna del corso (vedi [feedback_notebook_kaggle.md] nella memoria) richiede deliverable singolo per Kaggle. `CLAUDE.md` lo ribadisce: non factorizzare in moduli.

### Convenzione `[S, M, N]` invariata fino al split
I loader producono cubi `[S, M, N]`. **Solo gli splitter li appiattiscono** a `[num_seqs, N]`. Da lì in poi tutto downstream lavora su sequenze 1D, mai su cubi. È un'invariante esplicitamente protetta — molte celle indicizzano con `axis=-1` assumendo `N` ultimo.

### Vettorizzazione vs loop
Il principio è: **vettorizza dove il prefetch è batched (build_narx_*)**, ma **resta sequenziale dove c'è dipendenza ricorrente vera (simulate, simulate_seq, kstep_seq)**. Un altro caso interessante: `NNARX.simulate` è in **numpy puro**, perché su una rete piccola il kernel-launch CUDA per step domina il forward stesso.

### Ensemble + multi-seed: due livelli ortogonali
* **Ensemble** (`narx_ensemble`): N reti con seed diversi addestrate nello stesso run, **predizioni mediate**. Riduce la varianza della predizione finale.
* **Multi-seed Phase 2** (`phase2_seeds`): N run indipendenti, **metriche aggregate (mean + std)**. Riduce la varianza della stima della performance.

Sono complementari: il primo migliora la prestazione, il secondo migliora l'affidabilità della stima della prestazione.

### Tre orizzonti di valutazione
* **OSA** (k=1): TF puro, mostra quanto bene il modello cattura la dinamica locale.
* **3-step**: la convenzione di Gu 2021 e Santos 2023, allineata alla letteratura.
* **Free-run** (sim): il vero compito di system ID secondo Vlaar. È quello che conta in deploy.

Reportarli tutti e tre **insieme** rende immediatamente visibile il gap OSA→sim — che è il sintomo della discrepanza teacher-forced/free-run, target dell'intero apparato curriculum + scheduled sampling + noise injection.

### Guard di divergenza
ARX lineare e Poly-NARMAX possono divergere in free-run su segmenti specifici (poli instabili o monomi che amplificano errori). Il pattern è uniforme: rilevazione (NaN o `|y_hat| > 50 * std`), clipping, flag `_diverged` propagato fino alla tabella di Phase 3 dove appare come asterisco. Mai mascherare il problema, sempre renderlo visibile.

---

## 18. Mappa cellula → file di output

| Cella | Output su disco |
|---|---|
| 55 (Phase 3) | `results/phase1_metrics.json` |
| 49 (Phase 2 rec) | `results/phase2_rec.json` |
| 51 (Phase 2 NNARX) | `results/phase2_narx.json` |
| 53 (Phase 2 aux) | `results/phase2_aux.json` |
| 56 (LOSO CV) | `results/loso_folds.json` |

Tutti i JSON escludono `history` (i loss curve esplodono in dimensione e non sono usati dal report).

---

## Riferimenti
* [specs.md](specs.md) — consegna del progetto.
* [CLAUDE.md](CLAUDE.md) — convenzioni e vincoli del codebase.
* [teory.md](teory.md) — mapping celle ↔ slide di teoria.
* [benchmark_eeg_literature.md](benchmark_eeg_literature.md) — numeri target dalla letteratura.
* [review_miglioramenti_progetto_system_identification.md](review_miglioramenti_progetto_system_identification.md) — analisi gap e priorità di miglioramento.
