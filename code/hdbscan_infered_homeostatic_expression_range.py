#%% 
import pandas as pd
import numpy as np
import hdbscan
import statsmodels.stats.rates as st
from scipy.stats import binom, false_discovery_control
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import cdist
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
gene_read_rate_tbl =(
    count_tbl
    .groupby('gene_idx')
    .agg(gene_read_count = ('read_count','sum'),
         ncell = ('barcode_idx','nunique'))
)
gene_read_rate_tbl = (gene_read_rate_tbl
                      .assign(gene_read_rate = lambda df: df.gene_read_count / count_tbl.read_count.sum())
                      .assign(detection_limit_sample = lambda df: 1/df.gene_read_rate)
                      .reset_index()
                      )
# %%

data_tbl = (    count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
)
#%%
def get_cell_enrich(cell_id,data_tbl):
    return (data_tbl
        .query('barcode_idx == @cell_id')
        .assign(cell_rate = lambda df: df.read_count/df.tot_count)
        .assign(cell_vs_bulk_OR= lambda df: df.cell_rate/df.gene_read_rate)
        .assign(enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1),
                fdr = lambda df: false_discovery_control(df.enrichment))
        .assign(pscore = lambda df: -np.log10(df.enrichment),
                prank = lambda df: df.pscore.rank(pct=True,ascending=False)
                )

        )  
#%%
with Pool(processes=10) as pool:
        # pool.map applies 'parallel_func' to every item in 'pairwise_combinations'
        df = pool.map(partial(get_cell_enrich,data_tbl=data_tbl), count_tbl.barcode_idx.drop_duplicates().to_list())

data_enrich_tbl = pd.concat(df)

#%%
gene_of_interest_idx = gene_label_tbl.query("name == 'CD3D'").index.to_list()[0] + 1

gene_of_interest_tbl = (

    data_enrich_tbl
    .query("gene_idx == @gene_of_interest_idx")
    
)
#%%
###########
# Compare every marker expressing cell against non-marker cells
# marker_cells
marker_cell_id = gene_of_interest_tbl.barcode_idx.unique()
# marker less cells
non_marker_cell_id = data_enrich_tbl.query('~(barcode_idx in @marker_cell_id)').barcode_idx.unique()
## 
data_enrich_wide_df = (data_enrich_tbl
.query('gene_idx != @gene_of_interest_idx')
.loc[:,['barcode_idx','gene_idx','enrichment']]
.pivot(index='barcode_idx',columns='gene_idx',values='enrichment')
.fillna(1)
)
#%%
marker_cell_profiles_array = data_enrich_wide_df.loc[marker_cell_id,:].to_numpy()
non_marker_cell_profiles_array = data_enrich_wide_df.loc[non_marker_cell_id,:].to_numpy()
distance_matrix = cdist(marker_cell_profiles_array,non_marker_cell_profiles_array,metric='cosine')
#%%
# np.quantile(distance_matrix,[0.1,0.5,0.9],axis=1)
(pd.DataFrame({'mdist':np.min(distance_matrix,axis=1),'barcode_idx':marker_cell_id})
 .merge(gene_of_interest_tbl.loc[:,['barcode_idx','enrichment','pscore','cell_vs_bulk_OR','tot_count']])
 .plot.scatter(x='cell_vs_bulk_OR',y='mdist',c='tot_count',logx=True,cmap='viridis',s=0.1)
)
#%%
gene_of_interest_tbl.cell_vs_bulk_OR.plot.kde()
#%%
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
