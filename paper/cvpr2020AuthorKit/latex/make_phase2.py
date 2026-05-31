import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open("../../../results/results/phase2_rec.json"))

fams = {"rnn": ("RNN", "tab:red", "o"),
        "lstm": ("LSTM", "tab:purple", "s"),
        "gru": ("GRU", "tab:green", "^")}

fig, ax = plt.subplots(figsize=(7.0, 4.2))

for kind, (label, color, marker) in fams.items():
    rows = [r for r in data if r["kind"] == kind]
    rows.sort(key=lambda r: r["params"])
    for ls, lw, fill, dash in [(1, 1.8, color, "-"), (2, 1.4, "white", "--")]:
        sub = [r for r in rows if r["layers"] == ls]
        if not sub:
            continue
        x = [r["params"] for r in sub]
        y = [r["vaf_sim"] for r in sub]
        e = [r.get("vaf_sim_std", 0.0) for r in sub]
        ax.errorbar(x, y, yerr=e, color=color, lw=lw, ls=dash,
                    marker=marker, ms=8, mfc=fill, mec=color, mew=1.5,
                    capsize=3, label=f"{label} (L={ls})")

# NNARX FROLS-K20 plateau band (from phase1/phase2 NNARX sweep: ~25-27% sim)
ax.axhspan(25.0, 27.5, color="tab:blue", alpha=0.12, zorder=0)
ax.text(2.2e5, 26.2, "NNARX-FROLS plateau", color="tab:blue",
        fontsize=9, ha="right", va="center")

ax.set_xscale("log")
ax.set_xlabel("trainable parameters")
ax.set_ylabel(r"free-run VAF$_\mathrm{sim}$ [\%]" if False else "free-run VAF (sim) [%]")
ax.set_title("Phase 2: recurrent architectural sweep (3 seeds)")
ax.set_ylim(0, 30)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(ncol=3, fontsize=8, loc="upper center", framealpha=0.9)
fig.tight_layout()
fig.savefig("figures/phase2_sweep.png", dpi=150)
print("saved figures/phase2_sweep.png")
