"""MCMC estimation of UEC representative"""
import numpy as np
from scipy.stats import beta
from itertools import product
import pandas as pd
from collections import Counter

from scripts.medil.gauss_obs_l0_pen import GaussObsL0Pen
from scripts.thesis.mcmc_helpers import check_dag, udg_from_dag
from scripts.medil.independence_testing import estimate_UDG

class InputData(object):
    r"""Metropolis-Hastings algorithm with DAG posterior as target distribution
    geared to find BIC and MAP estimate of the UDG. The proposal kernel
    leverages the transformational characterization of unconditional
    equivalence of DAGs derived in Markham, A. et al. (2022). The algorithm
    draws from GrUES (Deligeorgaki, D. et al., 2022) with the key difference
    that it operates on the level of DAGs with a different proposal kernel.
    """
    def __init__(self, samples, factor_model=False, rng=np.random.default_rng(0)):
        self.samples = np.array(samples, dtype=float)
        self.factor_model = factor_model
        self.rng = rng

        self.num_samps, self.num_feats = self.samples.shape
        self.debug = False

    def mcmc(self, **kwargs):
        """Metropolis-Hastings implementation saving the DAG and UDG Markov chains,
        BIC-optimal DAG and UDG, and BIC trace of visited models.
        """
        self.init_mcmc(**kwargs)
        while self.active() == True:
            self.old_dag = np.copy(self.dag)
            self.old_udg = np.copy(self.udg)
            self.old_an_graph = np.copy(self.an_graph)

            moves_dict = {
                "add": self.add,
                "delete": self.delete,
                "reverse": self.reverse,
                "stay": self.stay
            }
            
            poss_moves, considered, p = self.compute_transition_kernel(moves_dict, action=None)
            move = self.rng.choice(poss_moves, p=p)

            q = float(self.q[move] * p[poss_moves == move]) # self.q[move] is the probability that we chose considered given move

            moves_dict[move](considered[move])
            
            if self.debug:
                try:
                    self.run_checks(move)
                except AssertionError:
                    import pdb, traceback

                    exc = traceback.format_exc()
                    pdb.set_trace()

            inv_moves = {
                "add": "delete",
                "delete": "add",
                "reverse": "reverse",
                "stay": "stay"
            }
            inv_move = inv_moves[move]
            cons = (considered[move][::-1] if move=="reverse" else considered[move]) # reversed edge has different address
            action = {"move":inv_move, "considered":cons}
            poss_moves, _, p = self.compute_transition_kernel(moves_dict, action=action)
            q_inv = float(self.q[inv_move] * p[poss_moves == inv_move])
            
            likelihood_ratio, new_rss, new_bic = self.get_likelihood_ratio()
            if new_bic < self.optimal_bic:
                self.optimal_bic = new_bic
                self.optimal_dag = self.dag
                self.optimal_udg = self.udg

            likelihood_and_transition_ratio = likelihood_ratio * (q_inv / q)
            if self.prior:
                prior_ratio = self.prior(self)
                likelihood_and_transition_ratio *= prior_ratio
            else:
                # Uniform prior
                prior_ratio = 1
                likelihood_and_transition_ratio *= prior_ratio
            h = min(1, likelihood_and_transition_ratio)
            if self.explore:
                h = 1
            make_move = self.rng.choice((True, False), p=(h, 1 - h))
            self.moves += 1
            if make_move:
                self.old_rss = new_rss
                self.old_bic = new_bic
                self.accepted += 1
                if self.tuning_mode: # sanity check uec exits
                    if self.stayed[move]:
                        assert (self.udg == self.old_udg).all()
                    else:
                        assert not (self.udg == self.old_udg).all()
            else:
                self.dag = self.old_dag
                self.udg = self.old_udg
                self.an_graph = self.old_an_graph
                
            if self.conv_crit is not None:
                if make_move:
                    hashed_udg = hash(self.udg.tobytes())
                    b = hashed_udg in self.unique_udg_hashed
                    if b:
                        self.counter += 1
                    else:
                        self.counter = 0
                        self.unique_udg_hashed.append(hashed_udg)
                else:
                    self.counter += 1
            self.visited[self.moves] = self.old_bic
            self.markov_chain_udg[self.moves] = self.udg
            self.markov_chain_dag[self.moves] = self.dag
    
    def init_mcmc(self, init="empty", p="uniform", max_moves=10000, conv_crit=None, prior=None, alpha_stay=1, alpha_exit=1, burn_in=0, explore=False, lmbda=1, tuning_mode=False, uniform=False):
        """Initialize the MCMC chain.
        """
        self.explore = explore # force h = 1
        self.uniform = uniform
        if p == "uniform":
            self.p = {
                "add": 1 / 4,
                "delete": 1 / 4,
                "reverse": 1 / 4,
                "stay": 1 / 4}
        else:
            self.p = p
        self.alpha_stay = alpha_stay
        self.alpha_exit = alpha_exit
        self.prior = prior
        self.lmbda = lmbda
        
        self.init_dag_udg(init)
        self.moves = 0
        self.max_moves = max_moves
        self.conv_crit = conv_crit
        self.counter = 0
        self.unique_udg_hashed = []
        self.burn_in = (burn_in or 0)
        self.accepted = 1 # starting point is accepted
        self.visited = np.empty(self.max_moves, float) # stores BIC scores

        if self.factor_model:
            graph = np.copy(self.udg).astype(int)
            np.fill_diagonal(graph, 0)
        else:
            graph = self.dag
        
        self.old_rss = self.compute_mle_rss(graph=graph)
        self.visited[0] = self.optimal_bic = self.old_bic = self.num_samps * self.old_rss + graph.sum() * self.lmbda * np.log(self.num_samps)
        
        self.markov_chain_dag = np.empty((self.max_moves, self.num_feats, self.num_feats), int)
        self.markov_chain_udg = np.empty((self.max_moves, self.num_feats, self.num_feats), bool)
        self.markov_chain_dag[0] = self.dag
        self.markov_chain_udg[0] = self.udg
        
        self.tuning_mode = max(tuning_mode, self.alpha_stay != self.alpha_exit)
        if self.tuning_mode:
            self.stayed = {}
            self.balance = {move:[] for move in self.p.keys()}
    
    def active(self):
        if self.moves % 1000 == 0:
            print(f"Iteration {self.moves} complete")
        # Always abort at latest at max_moves
        if self.moves >= self.max_moves - 1:
            return False
        if self.conv_crit is not None:
            # If no new UDG for the last self.conv_crit steps, stop the simulation
            if self.counter < self.conv_crit:
                return True
            else:
                self.trim()
                return False
        else:
            return True
    
    def trim(self):
        self.visited = self.visited[:self.moves + 1]
        self.markov_chain_dag = self.markov_chain_dag[:self.moves + 1]
        self.markov_chain_udg = self.markov_chain_udg[:self.moves + 1]
        return
    
    def run_checks(self):
        check_dag(self.dag)
        assert np.diag(self.udg).sum == self.num_feats

    def init_dag_udg(self, init):
        if type(init) == str:
            if init == "empty":
                self.dag = np.zeros((self.num_feats, self.num_feats))
            elif init == "complete":
                self.dag = np.triu(np.ones((self.num_feats, self.num_feats)), k=1)
        else:
            assert type(init) == np.ndarray
            self.dag = np.array(init)
        self.udg = udg_from_dag(self.dag)

        self.an_graph = self.an(self.dag)
        self.optimal_udg = self.udg
        self.optimal_dag = self.dag

    def get_likelihood_ratio(self):
        if self.factor_model:
            graph = np.copy(self.udg).astype(int)
            np.fill_diagonal(graph, 0)
        else:
            graph = self.dag
        new_rss = self.compute_mle_rss(graph=graph)
        new_bic = self.num_samps * new_rss # -2*ll
        new_bic += graph.sum() * self.lmbda * np.log(self.num_samps)
        if self.uniform:
            ll_ratio = 1
        else:
            # Replace ll ratio with ratio of (-BIC)
            ll_ratio = np.exp(self.old_bic - new_bic)
        return ll_ratio, new_rss, new_bic

    def compute_mle_rss(self, graph):
        omegas = GaussObsL0Pen(data=self.samples, method="raw")._mle_full(graph)[1]
        return np.log(omegas).sum()

    def an(self, dag):
        graph = np.eye(self.num_feats)
        for p in range(1, self.num_feats):
            graph += np.linalg.matrix_power(dag, p)
        return graph

    def is_ibt(self, e, move="add"):
        dag = np.copy(self.dag)
        if move == "delete":
            dag[tuple(e)] = 0
            graph = self.an(dag)
        else:
            graph = self.an_graph
        c1 = graph[tuple(e)] > 0
        c2 = dag[tuple(e)] == 0 
        c3 = e[0] != e[1]
        return c1 and c2 and c3
    
    def is_pwc(self, e, move="add"):
        dag = np.copy(self.dag)
        if move == "delete":
            dag[tuple(e)] = 0
            graph = self.an(dag)
        else:
            graph = self.an_graph
        orphans = np.where(np.ones((1, self.num_feats)) @ dag == 0)[1]
        c1 = graph[tuple(e)] == graph.T[tuple(e)] == 0 # v\notin an(w), w\notin an(v)
        c2 = e[0] not in orphans # pa(v)\neq \emptyset
        c3 = graph[np.intersect1d(np.where(graph[:, e[1]] == 0)[0], orphans), e[0]].sum() == 0 # ma(v)\subseteq ma(w) via x \notin ma(w) => x \notin ma(v)
        return c1 and c2 and c3
    
    def is_wc(self, e):
        graph = self.an_graph
        orphans = np.where(np.ones((1, self.num_feats)) @ self.dag == 0)[1]
        
        pa_w = np.where(self.dag[:, e[1]] > 0)[0]
        pa_w_not_v = np.delete(pa_w, np.where(pa_w == e[0]))
        c1 = e[0] not in np.where(graph[:, pa_w_not_v] > 0)[0] # v\notin an(pa(w)\v)

        pa_v = np.where(self.dag[:, e[0]] > 0)[0]
        an_pa_v = np.where(graph[:, pa_v] > 0)[0]
        ma_pa_v =  np.intersect1d(orphans, an_pa_v)
        an_pa_w_not_v = np.where(graph[:, pa_w_not_v] > 0)[0]
        ma_pa_w_not_v = np.intersect1d(orphans, an_pa_w_not_v)
        c2 = {i for i in ma_pa_v} == {i for i in ma_pa_w_not_v}
        return c1 and c2
    
    def choose(self, stay, exit, move, action=None):
        if self.tuning_mode and action is None:
            self.balance[move].append(len(stay)>0 and len(exit)>0) # useful in testing
        ordered = stay + exit
        if len(ordered) == 0:
            raise ZeroDivisionError
        p = np.append(self.alpha_stay*np.ones(len(stay)), self.alpha_exit*np.ones(len(exit)))
        if p.sum() == 0:
            p = np.ones(len(ordered))
        p /= p.sum()
        if action and action["move"] == move:
            # Already fixed choice of edge / node pair (inv move transition)
            considered = action["considered"]
            considered_index = next(i for i in range(len(ordered)) if ordered[i]==list(considered))
        else:
            # Choose edge / node pair (forward move transition)
            considered_index = self.rng.choice(np.arange(len(ordered)), p=p)
        q = p[considered_index]
        self.q[move] *= q # probability that we chose considered given move
        if self.tuning_mode and action is None:
            self.stayed[move] = considered_index < len(stay) # useful in testing
        return tuple(ordered[considered_index])
    
    def consider_add(self, action=None):
        # Construct graph encoding which edges may be added
        # If i and j are not adjacent, and there is no path between j and i, may add i -> j
        graph = self.an_graph
        other_path = np.argwhere(graph.T > 0).tolist()
        eligible = [e for e in np.argwhere(self.dag == 0).tolist() if e not in other_path]
        if self.tuning_mode:
            stay = [e for e in eligible if self.is_ibt(e) or self.is_pwc(e)]
            exit = [e for e in eligible if e not in stay]
            choice = self.choose(stay=stay, exit=exit, move="add", action=action)
        else:
            choice = self.choose(stay=eligible, exit=[], move="add", action=action)
        return choice
    
    def consider_delete(self, action=None):
        # Any existing edge is eligible
        eligible = np.argwhere(self.dag == 1).tolist()
        if self.tuning_mode:
            stay = [e for e in eligible if self.is_ibt(e, move="delete") or self.is_pwc(e, move="delete")]
            exit = [e for e in eligible if e not in stay]
            choice = self.choose(stay=stay, exit=exit, move="delete", action=action)
        else:
            choice = self.choose(stay=eligible, exit=[], move="delete", action=action)            
        return choice

    def consider_reverse(self, action=None):
        # Find existing edges
        existing_edges = np.argwhere(self.dag == 1).tolist()
        # Find pairs of vertices with connecting path of length > 1
        graph = np.copy(self.an_graph)
        graph -= (self.dag + np.eye(self.num_feats)).astype(int)
        # Existing edges with no such path are eligible for reversal
        other_path = np.argwhere(graph > 0).tolist()
        eligible = [e for e in existing_edges if e not in other_path]
        if self.tuning_mode:
            stay = [e for e in eligible if self.is_wc(e)]
            exit = [e for e in eligible if e not in stay]
            choice = self.choose(stay=stay, exit=exit, move="reverse", action=action)
        else:
            choice = self.choose(stay=eligible, exit=[], move="reverse", action=action)            
        return choice

    def consider_stay(self, action=None):
        if self.tuning_mode and action is None:
            self.balance["stay"].append(False) # useful in testing
            self.stayed["stay"] = True # useful in testing
        return None

    def update_udg_an(self):
        self.an_graph = self.an(self.dag)
        self.udg = udg_from_dag(A=self.dag, an_graph=self.an_graph)
    
    def add(self, edge):
        self.dag[edge] = 1
        self.update_udg_an()
        return
    
    def delete(self, edge):
        self.dag[edge] = 0
        self.update_udg_an()
        return

    def reverse(self, edge):
        self.dag[edge] = 0
        rev_edge = (edge[1], edge[0])
        self.dag[rev_edge] = 1
        self.update_udg_an()
        return
    
    def stay(self, considered):
        """ Do nothing
        """
        return
    
    def consider(self, move, action=None):
        """
        Depending on move, chooses objects to act on. Updates probability self.q[move] with number of choices.
        """
        if move == "add":
            return self.consider_add(action)
        elif move == "delete":
            return self.consider_delete(action)
        elif move == "reverse":
            return self.consider_reverse(action)
        elif move == "stay":
            return self.consider_stay(action)

    def compute_transition_kernel(self, moves_dict, action=None):
        poss_moves = []
        p = []
        considered = {}
        self.q = {key: 1.0 for key in self.p.keys()} # modified in consider methods depending on eligible options
        for move in moves_dict.keys():
            try:
                considered[move] = self.consider(move, action)
                poss_moves += [move]
                p += [self.p[move]]
            except ZeroDivisionError:
                continue
        p = np.array(p)
        p /= p.sum()
        return np.array(poss_moves), considered, p
    
    def map_udg(self):
        """Computes the MAP in the UDG Markov chain
        """
        burned_chain = self.markov_chain_udg[self.burn_in::]
        udg_counts = Counter(hash(udg.tobytes()) for udg in burned_chain)
        most_freq_hash = max(udg_counts, key=udg_counts.get)
        most_freq_udg = next(udg for udg in burned_chain if hash(udg.tobytes()) == most_freq_hash)
        return most_freq_udg
    
    def hpd(self, interval_width=0.1):
        """Computes the HPD in the UDG Markov chain
        """
        burned_chain = self.markov_chain_udg[self.burn_in::]
        udg_counts = Counter(hash(udg.tobytes()) for udg in burned_chain)
        total_counts = sum(udg_counts.values())
        cum_prob = 0
        hpd_hash = []
        for item in udg_counts.most_common():
            cum_prob += item[1] / total_counts
            hpd_hash.append(item[0])
            if cum_prob >= interval_width:
                break
        hpd = []
        for udg in burned_chain:
            if len(hpd_hash) == 0:
                break
            hsh = hash(udg.tobytes())
            if hsh in hpd_hash:
                hpd.append(udg)
                hpd_hash.remove(hsh)
        return hpd
    
    def isin_hpd(self, true_udg, interval_width=0.8):
        """Checks if the true udg is in the HPD
        """
        burned_chain = self.markov_chain_udg[self.burn_in::]
        udg_counts = Counter(hash(udg.tobytes()) for udg in burned_chain)
        total_counts = sum(udg_counts.values())
        cum_prob = 0
        hpd_hash = []
        for item in udg_counts.most_common():
            cum_prob += item[1] / total_counts
            hpd_hash.append(item[0])
            if cum_prob >= interval_width:
                break
        return hash(true_udg.tobytes()) in hpd_hash
    
    def it_udg(self, significance_level=0.05, method=None):
        # As in NCFA, method="xicor" per paper
        if method is not None:
            ud_graph, p_vals = estimate_UDG(self.samples, method=method, significance_level=significance_level)
            np.fill_diagonal(ud_graph, val=True)
        else:
            # As in GrUES https://codeberg.org/alex-markham/GUES/src/tag/v0.3.0/scripts/reproduce_astat
            alpha = significance_level
            corr = np.corrcoef(self.samples, rowvar=False)
            dist = beta(
                self.num_samps / 2 - 1, self.num_samps / 2 - 1, loc=-1, scale=2
            )
            crit_val = abs(dist.ppf(alpha / 2))
            ud_graph = abs(corr) >= crit_val
        return ud_graph
    
    def isin_mchn(self, true, graph="dag", burn_in=None):
        if burn_in is None:
            burn_in = self.burn_in
        if graph == "dag":
            burned_chain = self.markov_chain_dag[burn_in::]
        elif graph == "udg":
            burned_chain = self.markov_chain_udg[burn_in::]
        true_idxs = [i for i in range(len(burned_chain)) if (burned_chain[i]==true).all()]
        return len(true_idxs) > 0

    def posterior(self, target_udg=None, plot=False, dest="results"):
        """Computes and plots the MCMC UDG posterior
        """
        def all_possible_udgs():
            udgs = []
            idxs = np.argwhere(np.triu(np.ones(self.num_feats), k=1)==1)
            all_values = list(product([1, 0], repeat=len(idxs)))
            for values in all_values:
                udg = np.eye(self.num_feats)
                for i in range(len(idxs)):
                    idx = tuple(idxs[i])
                    udg[idx] = udg[idx[::-1]] = values[i]
                udgs.append(udg.astype(bool))
            assert len(udgs) == 2**(np.triu(np.ones(self.num_feats), k=1)).sum()
            return udgs
        if target_udg is not None:
            targets = [target_udg]
        else:
            targets = all_possible_udgs()
        burned_chain = self.markov_chain_udg[self.burn_in::]
        udg_counts = Counter(hash(udg.tobytes()) for udg in burned_chain)
        total_counts = sum(udg_counts.values())
        post = {}
        for i, target in enumerate(targets):
            target_hsh = hash(target.tobytes())
            counts = 0
            for item in udg_counts.most_common():
                if target_hsh == item[0]:
                    counts = item[1]
                    break
            post[i] = (counts / total_counts)
        if plot:
            plot = pd.Series(post).plot.bar()
            plot.set(title= "Posterior probability per UDG", 
                    ylabel="Probability",
                    xlabel="UDG")
            fig = plot.get_figure()
            fig.tight_layout()
            fig.savefig(f"{dest}/posterior_d3.png", bbox_inches='tight')
        return post, targets
    