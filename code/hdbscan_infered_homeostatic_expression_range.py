#%% 
import pandas as pd
import numpy as np
import hdbscan
import statsmodels.stats.rates as st
from scipy.stats import binom
#%%

filtered_count_file = './../data/filtered_gene_bc_matrices/hg19/matrix.mtx'
filtered_gene_label = './../data/filtered_gene_bc_matrices/hg19/genes.tsv'
filtered_barcode_label = './../data/filtered_gene_bc_matrices/hg19/barcodes.tsv'

#%%

gene_label_tbl = (pd.read_csv(filtered_gene_label,comment='%',sep='\t',header=None)
 .rename(columns={0:'ID',1:'name'})
)

barcode_label_tbl = (pd.read_csv(filtered_barcode_label,comment='%',sep='\t',header=None)
 .rename(columns={0:'ID'})
)

count_tbl = (pd.read_csv(filtered_count_file,comment='%',sep=' ',header=None,skiprows=3)
 .rename(columns={0:'gene_idx',1:'barcode_idx',2:'read_count'})
)
#%%
cell_cov_tbl = (count_tbl
 .groupby('barcode_idx')
 .agg(tot_count= ('read_count','sum'))
 .reset_index())

#%%
gene_read_rate_tbl =(
    count_tbl
    .groupby('gene_idx')
    .agg(gene_read_count = ('read_count','sum'))
)
gene_read_rate_tbl = (gene_read_rate_tbl
                      .assign(gene_read_rate = lambda df: df.gene_read_count / count_tbl.read_count.sum())
                      .reset_index()
                      )

#%%
gene_of_interest_idx = gene_label_tbl.query("name == 'MS4A7'").index.to_list()[0] + 1

gene_of_interest_obs_count_tbl = (

    count_tbl
    .query("gene_idx == @gene_of_interest_idx")
    .merge(cell_cov_tbl)
    .assign(rel_count = lambda df: df.read_count / df.tot_count)
    .assign(lcount = lambda df: np.log10(df.rel_count))
    
)
min_bassin_size = np.ceil(gene_of_interest_obs_count_tbl.barcode_idx.nunique()/10).astype(int)
marker_clustering = hdbscan.HDBSCAN(min_cluster_size= np.max([min_bassin_size,50]),min_samples=1,allow_single_cluster=True,cluster_selection_method='eom')

marker_clustering.fit(gene_of_interest_obs_count_tbl.loc[:,['lcount']])
#%%
marker_clustering.condensed_tree_.plot()
#%%
(gene_of_interest_obs_count_tbl
 .assign(hdbscan_labels = marker_clustering.labels_)
 .assign(single_read = lambda df: df.read_count.lt(2))
 .groupby(['hdbscan_labels'])
 .agg(single_prop = ('single_read','mean'),
      ncell = ('barcode_idx','nunique'),
      avg_rel = ('rel_count','mean'))
 .reset_index()
 .assign(abundance = lambda df: df.ncell / gene_of_interest_obs_count_tbl.barcode_idx.nunique())

)

#%%
(gene_of_interest_obs_count_tbl
 .lcount
 .plot.kde())
#%%
(gene_of_interest_obs_count_tbl
 .assign(hdbscan_labels = marker_clustering.labels_)
 .assign(single_read = lambda df: df.read_count.lt(2))
 .groupby(['hdbscan_labels'])
 .lcount
 .plot.kde(legend=True)

)
#%%
(gene_of_interest_obs_count_tbl
 .assign(hdbscan_labels = marker_clustering.labels_)
 .query('hdbscan_labels == 0')
 .read_count.value_counts()
)
#%%
(gene_of_interest_obs_count_tbl
 .assign(hdbscan_labels = marker_clustering.labels_)
 .query('hdbscan_labels == 0')
 .plot
 .scatter(x='tot_count',y='read_count',logy=True,logx=True)
)
#%%
homeostatic_cell_id_list = (gene_of_interest_obs_count_tbl
 .assign(hdbscan_labels = marker_clustering.labels_)
 .query('hdbscan_labels == 0')
 .barcode_idx.drop_duplicates().to_list()
)

(
    count_tbl
    .query('barcode_idx in @homeostatic_cell_id_list')
    .gene_idx
    .value_counts()
    .reset_index()
    .rename(columns={'count':'homeo'})
    .merge(
           count_tbl
            .gene_idx
            .value_counts()
            .reset_index()
            .rename(columns={'count':'tot'})

    )
    .merge(
        count_tbl
        .merge(cell_cov_tbl)
        .assign(rel_count = lambda df: df.read_count / df.tot_count)
        .groupby('gene_idx')
        .agg(avg_rel = ('rel_count','mean'))
        .reset_index()

    )
    .assign(col_trx=lambda df: np.log10(df.avg_rel))
    .sort_values('avg_rel')
    .plot
    .scatter(x='homeo',y='tot',c='col_trx',logx=True,logy=True)
)
#%%
(    count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
    .query('gene_idx == @gene_of_interest_idx')
    .assign(rel_count = lambda df: df.read_count / df.tot_count)
    .assign(zero_count_proba = lambda df: df.apply(lambda row: binom.cdf(0, row.tot_count, row.gene_read_rate),axis=1),
            enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1))
    .assign(detectable = lambda df: df.zero_count_proba.lt(0.5),
            lcount = lambda df: np.log10(df.read_count),
            odds_ratio = lambda df: (df.read_count/df.tot_count) /df.gene_read_rate,
            pscore = lambda df: -np.log10(df.enrichment))
    .sort_values('odds_ratio')
    .enrichment
    .plot.kde()
    # .plot
    # .scatter(x='odds_ratio',y='pscore',c='lcount',logx=True)      
)


#%%
(
    count_tbl
    .query('barcode_idx in @homeostatic_cell_id_list')
    .gene_idx
    .value_counts()
    .reset_index()
    .rename(columns={'count':'homeo'})
    .merge(
           count_tbl
            .gene_idx
            .value_counts()
            .reset_index()
            .rename(columns={'count':'tot'})

    )
    .merge(
        count_tbl
        .merge(cell_cov_tbl)
        .assign(rel_count = lambda df: df.read_count / df.tot_count)
        .groupby('gene_idx')
        .agg(avg_rel = ('rel_count','mean'))
        .reset_index()

    )
    .assign(col_trx=lambda df: np.log10(df.avg_rel))
    .query('gene_idx == @')
    .sort_values('avg_rel')
    .plot
    .scatter(x='homeo',y='tot',c='col_trx',logx=True,logy=True)
)

#%%
(
    count_tbl
    .query('barcode_idx in @homeostatic_cell_id_list')
    .barcode_idx
    .value_counts()
    .reset_index()
    .rename(columns={'count':'homeo'})
    .homeo.plot.kde()
)
#%%
(count_tbl
    .query('barcode_idx in @homeostatic_cell_id_list')
    .merge(cell_cov_tbl)
    .assign(rel_count = lambda df: np.log10(df.read_count / df.tot_count))
    .rel_count.plot.kde()
)
#%%
df_wide = (count_tbl
    .query('barcode_idx in @homeostatic_cell_id_list')
    .merge(cell_cov_tbl)
    .assign(rel_count = lambda df: df.read_count / df.tot_count)

    .pivot_table(
        index='barcode_idx',
        columns='gene_idx',
        values='rel_count',
        fill_value=0  # IMPORTANT: Fill NaN values (zero counts) with 0
    )
        
)

cell_clusterer = hdbscan.HDBSCAN(min_cluster_size= 20,min_samples=1,allow_single_cluster=True,cluster_selection_method='eom',metric='manhattan')

cell_clusterer.fit(df_wide)
#%%
cell_clusterer.condensed_tree_.plot()
#%%
cell_clusterer.labels_
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 20))
plt.imshow(df_wide.to_numpy(), cmap='binary')
# %%
conf_int_arrays = st.confint_poisson(gene_of_interest_obs_count_tbl.read_count.to_numpy(), gene_of_interest_obs_count_tbl.tot_count.to_numpy(), method="exact-c", alpha=1-0.95)

single_read_conf_int_arrays = st.confint_poisson(np.ones(gene_of_interest_obs_count_tbl.shape[0]), gene_of_interest_obs_count_tbl.tot_count.to_numpy(), method="exact-c", alpha=1-0.95)

# %%
sum(conf_int_arrays[1] > single_read_conf_int_arrays[1])/gene_of_interest_obs_count_tbl.shape[0]
# %%
gene_of_interest_obs_count_tbl.read_count.plot.kde()
#%%
res_pvalue = st.test_poisson_2indep(gene_of_interest_obs_count_tbl.read_count.to_numpy(), gene_of_interest_obs_count_tbl.tot_count.to_numpy(),
                       np.ones(gene_of_interest_obs_count_tbl.shape[0]), gene_of_interest_obs_count_tbl.tot_count.to_numpy(),alternative='larger').pvalue

np.mean(res_pvalue < 0.5)

# %%
(gene_of_interest_obs_count_tbl
 .assign(noise_pvalue = res_pvalue)
 .assign(ok = lambda df: df.noise_pvalue.lt(0.5))
#  .query('~ok')
 .rel_count.plot.kde())
