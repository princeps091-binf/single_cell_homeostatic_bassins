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
(    count_tbl
    .merge(cell_cov_tbl)
    .assign(rel_count = lambda df: df.read_count / df.tot_count)

    .plot
    .scatter(x='tot_count',y='rel_count',logx=True)

)
#%%
(    count_tbl
    .merge(cell_cov_tbl)
    .assign(single = lambda df: df.read_count.lt(2))
    .groupby(['gene_idx'])
    .agg(single_prop = ('single','mean'),
         ncell = ('barcode_idx','nunique'))
    .reset_index()
    .plot
    .scatter(y='ncell',x='single_prop',alpha=0.2,s=0.2)
)
#%%
(    count_tbl
    .merge(cell_cov_tbl)
    .assign(single = lambda df: df.read_count.lt(2))
    .groupby(['barcode_idx','tot_count'])
    .agg(prop_single = ('single','mean'),
         single_sum = ('single','sum'))
    .reset_index()
    .assign(single_tot_prop = lambda df: df.single_sum/df.tot_count)
    .sort_values('prop_single')
    .plot
    .scatter(x='tot_count',y='single_tot_prop',c='prop_single',logx=True)


)
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
(    count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
    .query('barcode_idx == 2000')
    .assign(zero_count_proba = lambda df: df.apply(lambda row: binom.cdf(0, row.tot_count, row.gene_read_rate),axis=1),
            enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1))
    .assign(detectable = lambda df: df.zero_count_proba.lt(0.5),
            lcount = lambda df: np.log10(df.read_count),
            odds_ratio = lambda df: (df.read_count/df.tot_count) /df.gene_read_rate,
            pscore = lambda df: -np.log10(df.enrichment))
    .plot
    .scatter(x='odds_ratio',y='pscore',s=2,logx=True)      
)
