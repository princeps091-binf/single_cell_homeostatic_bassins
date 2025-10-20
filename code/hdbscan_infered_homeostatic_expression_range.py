#%% 
import pandas as pd
import numpy as np
import hdbscan
import statsmodels.stats.rates as st
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
gene_stat_df = (
    count_tbl
    .merge(cell_cov_tbl)
    .assign(rel_count = lambda df: df.read_count / df.tot_count)
    .groupby('gene_idx')
    .agg(ncell = ('barcode_idx','nunique'),
        avg_rel = ('rel_count','mean'),
        std_rel = ('rel_count','std'),
        )
    .reset_index()
    .assign(CV2 = lambda df: (df.std_rel/df.avg_rel)**2)
)

#%%
gene_of_interest_idx = gene_label_tbl.query("name == 'PPBP'").index.to_list()[0] + 1

gene_of_interest_obs_count_tbl = (

    count_tbl
    .query("gene_idx == @gene_of_interest_idx")
    .merge(cell_cov_tbl)
    .assign(rel_count = lambda df: df.read_count / df.tot_count)
    .assign(lcount = lambda df: np.log10(df.rel_count))
    
)
marker_clustering = hdbscan.HDBSCAN(min_cluster_size= np.max([2,np.ceil(gene_of_interest_obs_count_tbl.barcode_idx.nunique()/20).astype(int)]))

marker_clustering.fit(gene_of_interest_obs_count_tbl.loc[:,['rel_count']])
#%%
(gene_of_interest_obs_count_tbl
 .assign(hdbscan_labels = marker_clustering.labels_)
 .assign(single_read = lambda df: df.read_count.lt(2))
 .groupby(['hdbscan_labels'])
 .agg(single_prop = ('single_read','mean'),
      ncell = ('barcode_idx','nunique'))
 .reset_index()
 .assign(abundance = lambda df: df.ncell / gene_of_interest_obs_count_tbl.barcode_idx.nunique())
 .query('single_prop < 0.5')

)
# %%


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
