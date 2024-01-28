"""Function to sample from the set of UDGs corresponding to non-empty UECs"""
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
from scripts.grues.sample import sample_causal_dag
from scripts.thesis.mcmc_algorithm import InputData


def gen_udgs_mcmc(d=10,
                  init="empty",
                  desired_ed=None,
                  n_models=1,
                  explore=True,
                  rng=np.random.default_rng(0),
                  plot=True):
    """
    Randomly generate a set of UDGs among non-empty UECs
    
    Args
    ----
    d: int
        Number of nodes
    init: str or array-like
        Initialization of MCMC algorithm. If init="udg", initialize at a UDG with edge density = desired_ed. 
        Else, initialize with whatever is passed to init.
    desired_ed: float
        See above
    n_models: int
        Number of UDGs to generate
    explore: bool
        Whether to discard the proper MCMC set-up in favor of a trajectory with maximal space exploration.
        If False, a uniform target distribution is forced. Recommendation is to set to True.
    plot: bool
        Whether to plot sample MCMC trajctory and evidence of a uniform sample
    """
    directory = f"{os.path.expanduser('~')}/mcmc-estimation-causal-vae-architectures/storage/"
    pattern = fr"{'explore' if explore else 'uniform'}_{init}{desired_ed or ''}_buckets_{d}_{n_models}_"
    # List all files in the directory
    files = os.listdir(directory)
    # Check if any of the file names match the pattern
    loaded = False
    for file in files:
        if re.match(f"^{pattern}", file):
            full_path = os.path.join(directory, file)
            if os.path.isfile(full_path):
                loaded = True
                break    
    prng = [p / 10 for p in range(1,10)]
    # If already stored, load and return.
    if loaded:
        buckets = {}
        for key in prng:
            buckets[key] = np.load(f"{directory}/{pattern}{key}.npy")
        return buckets
    
    full = np.triu(np.ones((d,d)), k=1).sum()        
    X = rng.normal(size=(10**3, d))
    max_moves=10**3
    burn_in = 500
    if explore:
        alpha_exit = 1
        alpha_stay = 0
        uniform = False
        p = {"add": 1 / 3,
         "delete": 1 / 3,
         "reverse": 1 / 3,
         "stay": 0}
    else:
        # Force uniform target distribution
        alpha_exit = 1
        alpha_stay = 1
        uniform = True
        p = {"add": 1 / 4,
         "delete": 1 / 4,
         "reverse": 1 / 4,
         "stay": 1 / 4}
    buckets = {ed:[] for ed in prng}
    buckets_counts = {ed:0 for ed in prng}
    while min(buckets_counts.values()) < n_models:
        if init == "udg":
            # Randomly generate the DAG initialization until we have UDG edge density = desired_ed if specified
            retry_condition = True
            while retry_condition:
                ep = min(abs(rng.normal(scale=0.05)), 1) # truncated normal
                uec, dag, sample = sample_causal_dag(num_nodes=d,
                                                      edge_prob=ep,
                                                      samp_size=0,
                                                      return_uec=True,
                                                      rng=rng)
                ed = np.triu(uec, k=1).sum() / full
                if desired_ed is None:
                    retry_condition = False
                else:
                    retry_condition = np.round(ed, 1) != desired_ed
            _init = dag.astype(int)
        else:
            _init = init 
        idt = InputData(samples=X, factor_model=True)
        idt.mcmc(init=_init,
                 max_moves=max_moves, 
                 burn_in=burn_in,
                 explore=explore,
                 alpha_exit=alpha_exit,
                 alpha_stay=alpha_stay,
                 uniform=uniform,
                 p=p)

        # Frequency
        burned_chain = idt.markov_chain_udg[idt.burn_in::]
        udg_counts = Counter(hash(udg.tobytes()) for udg in burned_chain)
        total_counts = sum(udg_counts.values())

        # Frequency of UDGs per edge density
        density = pd.Series({hash(udg.tobytes()):np.triu(udg, k=1).sum() / full for udg in burned_chain}).rename("density")
        freq = pd.DataFrame(udg_counts.most_common()).rename(columns={0:"hash", 1:"freq"})

        df = freq.join(density, on="hash", how='left')
        grpd = df.groupby("density").agg({'freq': ['sum', 'count']})
        grpd.columns = grpd.columns.droplevel()

        hist = grpd.reset_index()
        hist["density"] = np.round(hist["density"],1)
        hist["density"] = hist["density"].replace({0:0.1, 1:0.9})
        hist = hist.groupby("density").sum().reset_index()
        print(buckets_counts)
        need_additions = [i for i, j in buckets_counts.items() if j < n_models]
        if len(set(need_additions) & set(list(hist.loc[hist["count"]>0, "density"]))) == 0:
            continue
        # Choose UDG in this chain
        if explore:
            # Explore mode so randomly choose UDGs in chain
            hist["choice"] = hist["count"].map(lambda x: rng.choice(np.arange(x)))
        else:
            # Uniform so choose last in chain
            hist["choice"] = hist["count"].map(lambda x: x-1)
        visited = []
        counts = {ed:0 for ed in prng}
        chosen = {ed:None for ed in prng}
        for udg in burned_chain:
            hashed = hash(udg.tobytes())
            if hashed in visited:
                continue
            else:
                ed = np.triu(udg, k=1).sum() / full
                ed = np.round(ed, 1)
                if ed == 0:
                    ed = 0.1
                elif ed == 1:
                    ed = 0.9
                choice = hist.loc[hist["density"]==ed, "choice"].item()
                if counts[ed] < choice:
                    counts[ed] += 1
                elif counts[ed] == choice:
                    chosen[ed] = udg
                    counts[ed] += 1
                visited.append(hashed)

        for key in need_additions:
            if chosen[key] is not None:
                buckets[key].append(chosen[key])
                buckets_counts[key] += 1
    # Save
    for key, bucket in buckets.items():
        np.save(f"{directory}/{pattern}{key}.npy", np.array(bucket))
    if plot:
        # BIC
        fig, ax = plt.subplots()
        ax.plot(idt.visited)
        ax.set_xlabel("MCMC iteration")
        ax.set_ylabel("BIC using N(0,1) sample")
        fig.suptitle(f"Search trajectory, initialization: {init} {(desired_ed or '')}")
        fig.tight_layout()
        fig.savefig(f"udg_chain_bic.png", bbox_inches='tight')
        # Freq
        freq = np.array(udg_counts.most_common())
        fig, ax = plt.subplots()
        ax.plot(np.arange(freq.shape[0]), freq[:,1] / total_counts, "-o")
        fig.suptitle("Frequency of UDGs visited (highest-lowest)")
        ax.set_ylabel("Freq of UDG")
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_xticks([])
        fig.tight_layout()
        fig.savefig(f"udg_chain_freq.png", bbox_inches='tight')

        plot = grpd.plot()
        plot.set(title="UDG frequency per edge density")
        fig = plot.get_figure()
        fig.tight_layout()
        fig.savefig("udg_freq_ed_cont.png", bbox_inches='tight')

        plot = hist.plot.bar(x="density", y=["sum", "count"])
        plot.set(title="UDG frequency per edge density")
        fig = plot.get_figure()
        fig.tight_layout()
        fig.savefig("udg_freq_ed_disc.png", bbox_inches='tight')

        # Evidence of uniform sample
        prng = [i / 10 for i in range(1,10)]

        fig, axs = plt.subplots(3,3)
        for i in range(9):
            bucket = np.load(f"{directory}/{pattern}{prng[i]}.npy")
            avg = np.mean(bucket, axis=0)
            ax =  axs[i//3, i%3]
            sns.heatmap(np.log(avg), linewidth=0.5, ax=ax)
        for ax in fig.get_axes():
            ax.label_outer()
        fig.suptitle("Log elementwise average per edge density")
        fig.tight_layout()
        fig.savefig("log_elementwise_avg_ed.png", bbox_inches='tight')

        fig, ax = plt.subplots()
        buckets = np.empty((n_models*9,d,d))
        for i in range(9):
            bucket = np.load(f"{directory}/{pattern}{prng[i]}.npy")
            buckets[n_models*i:n_models*(i+1), :, :] = bucket
        avg = np.mean(buckets, axis=0)
        sns.heatmap(np.log(avg), linewidth=0.5, ax=ax)
        fig.suptitle("Log elementwise average across edge densities")
        fig.tight_layout()
        fig.savefig("log_elementwise_avg.png", bbox_inches='tight')        
    return buckets