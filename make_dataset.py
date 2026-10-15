#!/usr/bin/env python3
"""
make_dataset.py  --  Sprint 1 dataset generator
Physics-informed state estimation for distribution grids under PMU dropout.

WHAT THIS SCRIPT PRODUCES (per feeder, saved in  <out>/<system>/):
    train.npz            complete measurements (no PMU missing), labels = full state
    val.npz              PMU dropout drawn at random from {10,20,30,40,50} % per snapshot
    test_drop10..50.npz  SAME underlying snapshots, five different dropout levels (paired test)
    meta.npz             Ybus (complex admittance matrix), slack index, PMU bus indices, ...

PIPELINE (one loop iteration = one "load-flow snapshot"):
    1. draw a time of day + random load scaling      -> new load vector (P, Q)
    2. solve the AC power flow (pandapower)          -> TRUE state: |V|, theta at every bus
    3. corrupt the true values with sensor noise     -> what a PMU would report
    4. corrupt the true injections with forecast err -> "pseudo-measurements" (used by physics loss)
    5. apply a dropout mask                          -> some PMU channels become unavailable

Run:   python make_dataset.py --out data
Quick: python make_dataset.py --n_train 300 --n_val 100 --n_test 100 --out data_small
"""
import argparse
import os
import urllib.request

import numpy as np
import pandapower as pp
import pandapower.networks as pn




# ----------------------------------------------------------------------------------
# 1. BUILD THE NETWORKS
# ----------------------------------------------------------------------------------
def _read_matrix(text, name):
    """Extract the numeric block  `mpc.<name> = [ ... ];`  from a MATPOWER .m file."""
    start = text.index(f"mpc.{name} = [")
    end = text.index("];", start)
    rows = []
    for line in text[start:end].splitlines()[1:]:       # skip the 'mpc.xxx = [' line
        line = line.split("%")[0].replace(";", " ").strip()   # strip MATLAB comments / ';'
        if line:
            rows.append([float(v) for v in line.split()])
    return np.array(rows)





def build_net(system, case69_path=None):
    if system == "ieee33":
        return pn.case33bw()               # Baran-Wu 33-bus, built into pandapower


# ----------------------------------------------------------------------------------
# 2. LOAD MODEL  (a day's load profile + randomness)
# ----------------------------------------------------------------------------------
def daily_profile(hour):
    """Smooth residential-style daily curve: base load + morning bump + evening peak.
    Output is a multiplier on the nominal feeder load (~0.55 at night, ~1.0 at 19:00)."""
    base = 0.55
    morning = 0.25 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
    evening = 0.45 * np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)
    return base + morning + evening


# ----------------------------------------------------------------------------------
# 3. SNAPSHOT SIMULATION
# ----------------------------------------------------------------------------------
def get_ybus(net):
    """Return the bus admittance matrix Y (complex, per-unit) in pandapower bus order.
    Physics: I_injection = Y @ V.  Needed later for the power-flow residual loss."""
    pp.runpp(net, calculate_voltage_angles=True)
    Yppc = net._ppc["internal"]["Ybus"].toarray()
    lut = net._pd2ppc_lookups["bus"][net.bus.index.values]   # pandapower idx -> internal idx
    return Yppc[np.ix_(lut, lut)]


def simulate_snapshots(net, n, rng, p0, q0):
    """Solve n random AC load flows. Returns TRUE states and TRUE injections (per-unit).
    Snapshots where the solver diverges are discarded and redrawn."""
    nb = len(net.bus)
    vm = np.zeros((n, nb)); va = np.zeros((n, nb))          # true |V| [pu], theta [rad]
    pinj = np.zeros((n, nb)); qinj = np.zeros((n, nb))      # true injections [pu]
    hour = np.zeros(n)
    nl = len(net.load)
    k = 0
    while k < n:
        h = rng.uniform(0, 24)                                       # time of day
        day = np.clip(rng.normal(1.0, 0.05), 0.85, 1.15)             # day-to-day variation
        per_load = np.clip(rng.normal(1.0, 0.10, nl), 0.5, 1.5)      # each load deviates a bit
        scale = daily_profile(h) * day * per_load
        net.load["p_mw"] = p0 * scale
        net.load["q_mvar"] = q0 * scale * np.clip(rng.normal(1.0, 0.02, nl), 0.9, 1.1)
        try:
            pp.runpp(net, calculate_voltage_angles=True)             # <-- AC power flow
        except pp.LoadflowNotConverged:
            continue
        r = net.res_bus.loc[net.bus.index]
        vm[k] = r["vm_pu"].values
        va[k] = np.deg2rad(r["va_degree"].values)
        # pandapower reports p_mw as CONSUMPTION (+); injection convention is generation (+) -> minus sign
        pinj[k] = -r["p_mw"].values / net.sn_mva
        qinj[k] = -r["q_mvar"].values / net.sn_mva
        hour[k] = h
        k += 1
    return dict(vm=vm, va=va, pinj=pinj, qinj=qinj, hour=hour)


# ----------------------------------------------------------------------------------
# 4. MEASUREMENT MODEL
# ----------------------------------------------------------------------------------
def make_measurements(sim, pmu_idx, slack, rng, sig_vm, sig_va_deg, sig_pseudo):
    """Build noisy PMU readings (|V|, theta at PMU buses) and noisy pseudo-measurements
    of P/Q injections at every bus.

    PMU noise: additive Gaussian (sig_vm in pu, sig_va in degrees).
    Pseudo-measurements: multiplicative Gaussian error (load FORECAST error, e.g. 10 %).
        - multiplicative => zero-injection buses stay exactly 0 (physically correct)
        - slack bus injection is unknown (it supplies the whole feeder) -> set to 0 and
          excluded from the physics residual later.
    """
    n = sim["vm"].shape[0]
    pm_vm = sim["vm"][:, pmu_idx] + rng.normal(0, sig_vm, (n, len(pmu_idx)))
    pm_va = sim["va"][:, pmu_idx] + rng.normal(0, np.deg2rad(sig_va_deg), (n, len(pmu_idx)))
    p_ps = sim["pinj"] * (1 + rng.normal(0, sig_pseudo, sim["pinj"].shape))
    q_ps = sim["qinj"] * (1 + rng.normal(0, sig_pseudo, sim["qinj"].shape))
    p_ps[:, slack] = 0.0
    q_ps[:, slack] = 0.0
    return pm_vm.astype(np.float32), pm_va.astype(np.float32), \
        p_ps.astype(np.float32), q_ps.astype(np.float32)


def make_mask(rng, n, k, drop_frac):
    """Random PMU dropout. In EVERY snapshot exactly round(drop_frac*k) of the k PMUs are
    lost, chosen uniformly at random and independently per snapshot.
    mask = 1 -> PMU available, 0 -> PMU missing.  `drop_frac` can be a scalar or an array."""
    drop_frac = np.broadcast_to(np.asarray(drop_frac, dtype=float), (n,))
    n_drop = np.rint(drop_frac * k).astype(int)
    ranks = rng.random((n, k)).argsort(1).argsort(1)    # random permutation rank per row
    return (ranks >= n_drop[:, None]).astype(np.float32)  # lowest-ranked n_drop entries dropped


def pack(sim, meas, mask, drop_frac):
    """Assemble the arrays stored in a .npz file. Missing channels are ZEROED in
    pmu_vm / pmu_va (the model must use `mask` to know they are missing, not 'zero volts')."""
    pm_vm, pm_va, p_ps, q_ps = meas
    return dict(
        pmu_vm=pm_vm * mask, pmu_va=pm_va * mask, mask=mask,        # model inputs (masked)
        pmu_vm_full=pm_vm, pmu_va_full=pm_va,                       # unmasked copy (for re-masking)
        p_pseudo=p_ps, q_pseudo=q_ps,                               # model inputs + physics target
        vm_true=sim["vm"].astype(np.float32), va_true=sim["va"].astype(np.float32),  # LABELS
        p_true=sim["pinj"].astype(np.float32), q_true=sim["qinj"].astype(np.float32),
        drop_frac=np.broadcast_to(drop_frac, (len(mask),)).astype(np.float32),
        hour=sim["hour"].astype(np.float32))


# ----------------------------------------------------------------------------------
# 5. SANITY CHECK  (does the AC power-flow equation hold for our stored data?)
# ----------------------------------------------------------------------------------
def check_physics(Y, sim, slack):
    """S_i = V_i * conj( sum_k Y_ik V_k )  must reproduce the injections pandapower reports.
    If this max error is ~1e-6 the Ybus / sign / per-unit conventions are all correct."""
    V = sim["vm"] * np.exp(1j * sim["va"])                 # polar -> complex, shape (n, nb)
    S = V * np.conj(V @ Y.T)                               # row-wise  V_i * conj((Y V)_i)
    err = np.abs(S - (sim["pinj"] + 1j * sim["qinj"]))
    err[:, slack] = 0                                      # slack row is not constrained
    return err.max()


# ----------------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--systems", nargs="+", default=["ieee33"])
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--n_val", type=int, default=1)
    ap.add_argument("--n_test", type=int, default=1)
    ap.add_argument("--pmu_frac", type=float, default=0.5, help="fraction of non-slack buses that own a PMU")
    ap.add_argument("--sig_vm", type=float, default=0.001, help="PMU |V| noise std [pu]")
    ap.add_argument("--sig_va", type=float, default=0.05, help="PMU angle noise std [deg]")
    ap.add_argument("--sig_pseudo", type=float, default=0.10, help="load-forecast relative error std")
    ap.add_argument("--levels", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--case69_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    for system in a.systems:
        print(f"\n=== {system} ===")
        net = build_net(system, a.case69_path)
        nb = len(net.bus)
        slack = int(np.where(net.bus.index == net.ext_grid.bus.iloc[0])[0][0])
        p0, q0 = net.load["p_mw"].values.copy(), net.load["q_mvar"].values.copy()  # nominal loads
        Y = get_ybus(net)

        # Fixed PMU placement (same for all splits). Slack excluded: its V=1.0, theta=0 is known.
        place_rng = np.random.default_rng(a.seed)
        cand = np.array([i for i in range(nb) if i != slack])
        k = int(round(a.pmu_frac * len(cand)))
        pmu_idx = np.sort(place_rng.choice(cand, k, replace=False))
        print(f"buses={nb}  slack={slack}  PMUs={k} at buses {pmu_idx.tolist()}")

        outdir = os.path.join(a.out, system)
        os.makedirs(outdir, exist_ok=True)
        # Independent RNG streams per split -> no snapshot is shared between train/val/test.
        rng = {s: np.random.default_rng(a.seed + 1000 * (i + 1)) for i, s in enumerate(["train", "val", "test"])}

        def sim_and_meas(split, n):
            sim = simulate_snapshots(net, n, rng[split], p0, q0)
            meas = make_measurements(sim, pmu_idx, slack, rng[split], a.sig_vm, a.sig_va, a.sig_pseudo)
            return sim, meas

        # ---- TRAIN: no missing PMU channels (mask = all ones) ----
        sim, meas = sim_and_meas("train", a.n_train)
        print("physics check (max |S_calc - S_true|, pu):", f"{check_physics(Y, sim, slack):.2e}")
        np.savez_compressed(os.path.join(outdir, "train.npz"),
                            **pack(sim, meas, np.ones((a.n_train, k), np.float32), 0.0))

        # ---- VAL: dropout level drawn per snapshot from the list of levels ----
        sim, meas = sim_and_meas("val", a.n_val)
        lv = rng["val"].choice(a.levels, a.n_val)
        np.savez_compressed(os.path.join(outdir, "val.npz"),
                            **pack(sim, meas, make_mask(rng["val"], a.n_val, k, lv), lv))

        # ---- TEST: identical snapshots, one file per dropout level (paired comparison) ----
        sim, meas = sim_and_meas("test", a.n_test)
        for lvl in a.levels:
            m = make_mask(rng["test"], a.n_test, k, lvl)
            np.savez_compressed(os.path.join(outdir, f"test_drop{int(round(lvl * 100))}.npz"),
                                **pack(sim, meas, m, lvl))

        # ---- META: everything the physics loss / training code needs ----
        np.savez(os.path.join(outdir, "meta.npz"), Ybus_real=Y.real, Ybus_imag=Y.imag,
                 slack=slack, pmu_idx=pmu_idx, n_bus=nb, base_mva=net.sn_mva,
                 sig_vm=a.sig_vm, sig_va_deg=a.sig_va, sig_pseudo=a.sig_pseudo)
        print(f"saved to {outdir}/  (min |V| in train = {sim['vm'].min():.3f} pu)")


if __name__ == "__main__":
    main()
