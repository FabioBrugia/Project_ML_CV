# Revisione del progetto — System Identification of Nonlinear Dynamical Systems

## Valutazione generale

Il progetto è strutturato molto bene e copre correttamente le tre fasi richieste dalle specifiche:

1. Addestramento di modelli differenti (NNARX, RNN, LSTM, GRU)
2. Esplorazione architetturale
3. Confronto in termini di accuratezza, training time e complessità

Aspetti particolarmente positivi:

- notebook molto ordinato e leggibile
- separazione chiara tra training, evaluation e benchmarking
- presenza di metriche corrette (NRMSE, VAF)
- implementazione sia custom che con `sysidentpy`
- introduzione di FLOPs e parameter counting
- presenza di split LOSO (molto importante scientificamente)
- buon livello di documentazione markdown

Il progetto è già sopra la media per completezza e struttura.

---

# Miglioramenti principali

## 1. Correggere alcuni errori sintattici nel notebook

Nel notebook sono presenti alcune celle con errori sintattici che probabilmente derivano da editing o conversioni.

### Problemi trovati

#### Cell GRU sweep

```python
 do # Phase 2 -- GRU sweep over (hidden, layers)
```

`do` genera errore Python.

Va sostituito con:

```python
# Phase 2 -- GRU sweep over (hidden, layers)
```

---

#### Cell NNARX Phase2

```python
1PHASE2_NARX = []
```

Nome variabile non valido.

Corretto:

```python
PHASE2_NARX = []
```

---

#### Cell ptflops

```python
1 !pip install ptflops
```

Il prefisso `1` causa errore.

Corretto:

```python
!pip install ptflops
```

---

# Miglioramenti metodologici

## 2. Aggiungere baseline lineari

Le specifiche parlano di “challenging nonlinear systems”, quindi mostrare il confronto con modelli lineari rafforzerebbe molto il lavoro.

Consigliato aggiungere:

- ARX lineare
- Linear State Space
- Ridge regression autoregressiva

Questo permetterebbe di dimostrare quantitativamente:

- quanto la non linearità sia realmente necessaria
- il guadagno ottenuto con RNN/LSTM

Molto utile nella discussione finale.

---

## 3. Aggiungere teacher forcing decay

Attualmente il training ricorrente sembra usare teacher forcing standard.

Miglioramento importante:

### Scheduled sampling

Ridurre progressivamente il teacher forcing ratio:

```python
teacher_forcing_ratio = max(0.1, 1 - epoch / epochs)
```

Benefici:

- maggiore stabilità in closed-loop simulation
- riduzione dell’error accumulation
- simulazioni più realistiche

Questo è molto rilevante nei problemi di system identification.

---

## 4. Early stopping più robusto

Sembra già presente una forma di validazione, ma si può migliorare con:

- patience configurabile
- restore best weights
- smoothing della validation loss

Esempio:

```python
best_state = deepcopy(model.state_dict())
```

alla minima validation loss.

---

## 5. Seed e riproducibilità completa

Hai impostato i seed, ma per completa reproducibility scientifica puoi aggiungere:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Questo migliora la qualità sperimentale del progetto.

---

# Miglioramenti sulle metriche

## 6. Aggiungere MAE e fit percentage

Hai già NRMSE e VAF.

Puoi aggiungere:

### MAE

Più interpretabile dell’MSE.

### Fit percentage

Molto usata in system identification:

```python
fit = 100 * (1 - norm(y - yhat)/norm(y - mean(y)))
```

Aumenta il rigore sperimentale.

---

## 7. Confidence intervals sui risultati

Nella Phase 2 puoi eseguire più run con seed differenti:

- media
- deviazione standard

Esempio:

| Model | NRMSE mean ± std |
|---|---|
| LSTM | 0.12 ± 0.01 |

Questo rende il confronto molto più scientifico.

---

# Miglioramenti architetturali

## 8. Bidirectional RNN — NON consigliata

Per system identification causale non è appropriata.

Meglio evitare.

---

## 9. Residual connections nelle stacked LSTM

Molto interessante se usi >2 layers.

Esempio:

```python
h2 = h2 + h1
```

Può migliorare:

- stabilità
- gradient flow
- training profondo

---

## 10. Layer normalization

Molto utile nelle RNN.

Specialmente con:

- sequenze lunghe
- hidden size elevato

Può stabilizzare significativamente il training.

---

## 11. Sequence-to-sequence training

Attualmente sembra esserci training one-step con simulazione separata.

Miglioramento avanzato:

Allenare direttamente su rollout multipli:

```python
loss = mse(y_rollout, y_true)
```

Questo ottimizza direttamente la simulazione closed-loop.

Molto forte come upgrade scientifico.

---

# Miglioramenti Phase 3

## 12. Grafici più informativi

Molto consigliati:

### Accuracy vs Parameters

Scatter plot:

- x = #params
- y = NRMSE

### Accuracy vs FLOPs

Mostra il tradeoff accuratezza/complessità.

### Training time vs Accuracy

Molto utile per la discussione finale.

---

## 13. Tabelle comparative finali

Aggiungere una tabella finale tipo:

| Model | Params | FLOPs | NRMSE | VAF | Train Time |
|---|---|---|---|---|---|

Questo è quasi obbligatorio in un progetto di benchmarking.

---

# Miglioramenti scientifici avanzati

## 14. Ablation study

Molto apprezzata.

Esempio:

- con/senza dropout
- hidden size piccolo/grande
- 1 vs 2 layers
- con/senza derivative features

Serve a mostrare comprensione del sistema.

---

## 15. Analisi errore nel dominio del tempo e frequenza

Attualmente fai soprattutto time-domain evaluation.

Molto interessante aggiungere:

- PSD comparison
- FFT comparison
- errore spettrale

Specialmente perché il dataset è neurophysiological.

---

## 16. Discussione fisica del sistema

Attualmente il progetto è molto ML-oriented.

Puoi migliorarlo aggiungendo:

- interpretazione dinamica
- memoria del sistema
- tempi caratteristici
- perché LSTM funziona meglio
- correlazione con dinamiche corticali

Questo aumenta molto il livello accademico.

---

# Miglioramento più importante

Se dovessi scegliere il miglior upgrade complessivo:

## Priorità alta

1. Correggere gli errori sintattici
2. Aggiungere baseline lineare
3. Aggiungere rollout training / scheduled sampling
4. Tabelle comparative finali
5. Confidence intervals su più run

---

# Valutazione finale

## Livello del progetto

Il progetto è:

- corretto metodologicamente
- ben strutturato
- abbastanza avanzato
- superiore a un classico assignment universitario

Punti forti principali:

- completezza
- benchmarking serio
- implementazioni multiple
- FLOPs/complexity analysis
- LOSO split

Con i miglioramenti suggeriti può diventare molto vicino a un piccolo lavoro di ricerca applicata.

