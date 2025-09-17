from simulate_data.multitrait_multitime_distribution import MTMTDistribution
from simulate_data.multilevel_simulator import SymmetricMTMT
from simulate_data.simulate_irt import ItemBank
import numpy as np
import pandas as pd
from typing import Iterable
from tqdm import tqdm

class SurveyDataset:
    def __init__(
            self,
            dimension_names: Iterable[str],
            base_trait_correlation_matrix: np.array,
            means_dimensions: np.array,
            n_waves: int=1,
            rho_time=None,
            sigma_time=None,
            var_shares_levels=None,
            units_levels=None,
            level_names=("org", "team", "individual"),
            item_types="auto",
    ):

        self.dim_names = dimension_names
        self.base_trait_correlation_matrix = base_trait_correlation_matrix
        self.means_dimensions = means_dimensions

        if n_waves > 1:
            assert (rho_time is not None) and (sigma_time is not None)
            Sigma_time = MTMTDistribution.ar1_cov(3, rho=0.6, sigma=1.0)
        self.population_flat = MTMTDistribution(
            means_dimensions,
            Sigma_time,
            base_trait_correlation_matrix,
            trait_names=dimension_names
        )
        self.hierarchy = SymmetricMTMT(
            base=self.population_flat,
            level_names=level_names,
            var_shares=var_shares_levels,
            default_branching=units_levels,
        )

        if item_types == "auto":
            self.item_dict = {}
            # generate 5 GRM items per dimension
            for latent_dim in dimension_names:
                self.item_dict[latent_dim] = ItemBank(latent_dims=latent_dim)
        else:
            raise NotImplementedError


    @property
    def item_summary(self):
        rows = []
        for dim, bank in self.item_dict.items():
            for idx, item in enumerate(bank.items, 1):

                rows.append({
                    "dimension": dim,
                    "item_type": bank.item_type,
                    "item_index": idx,
                    "discrimination": item.a.tolist(),
                    "thresholds": item.beta.tolist(),
                    "n_cats": len(item.beta) + 1,
                    "latent_dims": item.latent_dims
                })
        return pd.DataFrame(rows)

    def simulate(self, seed=None, data_form="both", normalize_empirical_theta=False):
        # simulate traits
        df_ind, df_meta = self.hierarchy.simulate(seed)
        # simulate items from traits
        df_wide = df_ind.pivot_table(index=["id_individual", "id_team", "id_org", "wave"],
                                     columns="trait",
                                     values="value").reset_index().rename_axis(None, axis=1)
        df_item_responses = df_wide[df_wide.columns[:df_wide.shape[1] - len(self.dim_names)]].copy()
        if normalize_empirical_theta:
            # normalize so overall mean of first wave is 0 and std is 1
            theta_w0 = df_wide.loc[df_wide.wave==0, self.dim_names]
            theta_mu = theta_w0.mean(axis=0)
            theta_sigma = theta_w0.std(axis=0)
            thetas_standardized = (df_wide[self.dim_names] - theta_mu) / theta_sigma
            df_wide[self.dim_names] = thetas_standardized

            # apply to df_meta
            for idx, val in theta_mu.items():
                mask = df_meta.columns.str.endswith(idx)  # all columns ending with that dimension name
                df_meta.loc[:, mask] = (df_meta.loc[:, mask] - val)/theta_sigma[idx]


        for dim in tqdm(self.dim_names):
            item_bank = self.item_dict[dim]
            resp_matrix = item_bank.simulate(df_wide[[dim]])
            item_names_for_dim = [f"{dim}_{i:02}" for i in range(item_bank.n_items)]
            df_item_responses[item_names_for_dim] = resp_matrix

        if data_form == "wide":
            return df_item_responses, df_meta, df_wide
        elif data_form == "long":
            return df_item_responses.melt(id_vars=["id_individual", "id_team", "id_org","wave"]), df_meta, df_wide
        elif data_form == "both":
            return (
                df_item_responses,
                df_item_responses.melt(id_vars=["id_individual", "id_team", "id_org","wave"]),
                df_meta,
                df_wide
            )
        else:
            raise ValueError("data_form must be one of 'wide', 'long', or 'both'")

