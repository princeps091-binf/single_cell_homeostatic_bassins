#%% 
import pandas as pd
import numpy as np
import hdbscan
import statsmodels.stats.rates as st
from scipy.stats import binom
from scipy.stats import false_discovery_control
import xlmhglite
import itertools
from multiprocessing import Pool
from functools import partial
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
    .scatter(x='tot_count',y='rel_count',logy=True,logx=True,s=0.2)

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
(    count_tbl
    .merge(cell_cov_tbl)
    .assign(single = lambda df: df.read_count.lt(2))
    .assign(rel_count = lambda df: df.read_count / df.tot_count)

    .groupby(['barcode_idx','tot_count'])
    .agg(prop_single = ('single','mean'),
         single_sum = ('single','sum'),
         max_element = ('read_count','max')
         )
    .reset_index()
    .assign(single_tot_prop = lambda df: df.single_sum/df.tot_count)
    .sort_values('prop_single')
    .plot
    .scatter(x='tot_count',y='max_element',c='prop_single',logx=True)


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

#%%
expected_observations_tbl = (count_tbl
 .gene_idx.drop_duplicates().reset_index().loc[:,['gene_idx']]
.merge(count_tbl
       .barcode_idx.drop_duplicates().reset_index().loc[:,['barcode_idx']]
       ,how='cross')
.merge(cell_cov_tbl)
.merge(gene_read_rate_tbl.loc[:,['gene_idx','detection_limit_sample']])
.query('tot_count >= detection_limit_sample')
)

#%%
unobserved_but_expected_tbl = (    count_tbl
    .merge(expected_observations_tbl.assign(status='expected').loc[:,['gene_idx','barcode_idx','status']],
           how='outer')
    .query('status == "expected"')
    .fillna(0)
    .query('read_count < 1')
    .drop('status',axis=1)

)
#%%
# (pd.concat([count_tbl,unobserved_but_expected_tbl])
(count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
    .assign(cell_rate = lambda df: df.read_count/df.tot_count)
    .assign(cell_vs_bulk_OR= lambda df: df.cell_rate/df.gene_read_rate)
    .assign(rel_count = lambda df: df.read_count/df.tot_count)
    .groupby('gene_idx')
    .agg(gene_OR_mean = ('cell_vs_bulk_OR','mean'),
         gene_OR_sd = ('cell_vs_bulk_OR','std'),
         ncell = ('barcode_idx','nunique'),
         avg_rel = ('cell_rate','mean'),
         sd_rel = ('cell_rate','std'))
    .assign(CV2= lambda df: (df.sd_rel/df.avg_rel)**2,
            logn=lambda df: np.log10(df.ncell),
            lor = lambda df: np.log10(df.gene_OR_mean))
    .sort_values('CV2',ascending=False)
    # .plot.scatter(x='gene_OR_mean',y='CV2',c='logn',cmap='viridis',logx=True,logy=True,s=1)
    # .plot.scatter(x='avg_rel',y='CV2',c='logn',cmap='viridis',logx=True,logy=True,s=1)
    # .plot.scatter(x='gene_OR_mean',y='avg_rel',c='logn',cmap='viridis',logx=True,logy=True,s=1)
    .plot.scatter(x='CV2',y='ncell',c='CV2',cmap='viridis',logx=True,logy=True,s=1)

)
#%%
(count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
    .assign(cell_rate = lambda df: df.read_count/df.tot_count)
    .assign(cell_vs_bulk_OR= lambda df: df.cell_rate/df.gene_read_rate)
    .query('barcode_idx == 1679')
    .assign(enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1),
            fdr = lambda df: false_discovery_control(df.enrichment))
    .assign(pscore=lambda df: df.enrichment.rank(pct=True,ascending=False),
            cell_rate_score = lambda df: df.cell_rate.rank(pct=True),
            pcol = lambda df: pd.cut(df.fdr,bins=[0,1/cell_cov_tbl.shape[0],0.01,0.05,0.1,0.5,1]))

    .plot.scatter(x='cell_vs_bulk_OR',y='pscore',c='pcol',cmap='viridis_r',logx=True,s='read_count')
    # .plot.scatter(x='pscore',y='cell_rate_score',s=10,alpha=0.1)

)

#%%
gene_of_interest_idx = gene_label_tbl.query("name == 'FCER1A'").index.to_list()[0] + 1


(count_tbl
 .query('gene_idx == @gene_of_interest_idx')
 .loc[:,['barcode_idx']]
 .drop_duplicates()
 .merge(cell_cov_tbl)
 .sort_values('tot_count')
 )

cells_of_interest_list = (count_tbl
 .query('gene_idx == @gene_of_interest_idx')
 .loc[:,['barcode_idx']]
 .drop_duplicates()
 .merge(cell_cov_tbl)
 .sort_values('tot_count')
 .barcode_idx.drop_duplicates().to_list()
 )
print(len(cells_of_interest_list))

my_list = cells_of_interest_list
pairwise_combinations = list(itertools.combinations(my_list, 2))

#%%

cell_tbl = (
    count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
    .query('barcode_idx == 54')
    .assign(cell_rate = lambda df: df.read_count/df.tot_count)
    .assign(cell_vs_bulk_OR= lambda df: df.cell_rate/df.gene_read_rate)
    .assign(enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1),
            fdr = lambda df: false_discovery_control(df.enrichment))
    .assign(pscore = lambda df: -np.log10(df.enrichment),
            prank = lambda df: df.pscore.rank(pct=True,ascending=False)
            )
    .sort_values('fdr')

)
score_quantile = cell_tbl.pscore.quantile([0.25,0.75])
out_up_score_thresh = score_quantile.iloc[1] + 1.5*(score_quantile.iloc[1] - score_quantile.iloc[0])
print(cell_tbl.query('fdr < 1/@cell_tbl.shape[0]').shape[0]/cell_tbl.shape[0])
print(cell_tbl.query('pscore > @out_up_score_thresh').shape[0]/cell_tbl.shape[0])

# %%
cell_a_tbl = (
    count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
    .query('barcode_idx == 1457')
    .assign(cell_rate = lambda df: df.read_count/df.tot_count)
    .assign(cell_vs_bulk_OR= lambda df: df.cell_rate/df.gene_read_rate)
    .assign(enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1),
            fdr = lambda df: false_discovery_control(df.enrichment))
    .assign(pscore = lambda df: -np.log10(df.enrichment),
            prank = lambda df: df.pscore.rank(pct=True,ascending=False)
            )

)

cell_b_tbl = (
    count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
    .assign(cell_rate = lambda df: df.read_count/df.tot_count)
    .assign(cell_vs_bulk_OR= lambda df: df.cell_rate/df.gene_read_rate)
    .query('barcode_idx == 1634')
    .assign(enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1),
            fdr = lambda df: false_discovery_control(df.enrichment)
            )
    .assign(pscore = lambda df: -np.log10(df.enrichment),
            prank = lambda df: df.pscore.rank(pct=True,ascending=False)
)

)
a_b_gene_union_list = list(set(cell_b_tbl.gene_idx.to_list()).union(cell_a_tbl.gene_idx.to_list()))
a_b_gene_inter_list = list(set(cell_b_tbl.gene_idx.to_list()).intersection(cell_a_tbl.gene_idx.to_list()))
print(len(a_b_gene_inter_list)/len(a_b_gene_union_list))

cell_compare_tbl = (pd.DataFrame({'gene_idx':a_b_gene_union_list})
 .merge(cell_a_tbl.loc[:,['gene_idx','prank','enrichment']],how='left')
 .fillna(1)
 .rename(columns={'prank':'cell_a','enrichment':'a_pvalue'})
 .merge(cell_b_tbl.loc[:,['gene_idx','prank','enrichment']],how='left')
 .fillna(1)
 .rename(columns={'prank':'cell_b','enrichment':'b_pvalue'})
 .assign(avg_pval = lambda df: (df.b_pvalue + df.a_pvalue)/2,
         avg_cut = lambda df: pd.cut(df.avg_pval,[0,0.01,0.05,0.25,0.5,1.1]))

)

(cell_compare_tbl
 .plot
 .scatter(x='cell_a',y='cell_b',c='avg_cut',alpha=1,cmap='viridis_r',s=2)
)


v_a = cell_a_tbl.sort_values('pscore',ascending=False).assign(v=lambda df: np.where(df.gene_idx.isin(a_b_gene_inter_list),1,0)).v.to_numpy()
v_b = cell_b_tbl.sort_values('pscore',ascending=False).assign(v=lambda df: np.where(df.gene_idx.isin(a_b_gene_inter_list),1,0)).v.to_numpy()
# case for agreement with self
# v_a[-1] = 0
# v_b[-1] = 0

_, _, pval_a = xlmhglite.xlmhg_test(v_a, int(5), int(cell_a_tbl.shape[0]))
_, _, pval_b = xlmhglite.xlmhg_test(v_b, int(5), int(cell_b_tbl.shape[0]))

print(pval_a)
print(pval_b)
# %%
# %%

data_tbl = (    count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
)

def get_cell_similarity(pair,data_tbl):
    cell_a_tbl = (data_tbl
        .query('barcode_idx == @pair[0]')
        .assign(cell_rate = lambda df: df.read_count/df.tot_count)
        .assign(cell_vs_bulk_OR= lambda df: df.cell_rate/df.gene_read_rate)
        .assign(enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1),
                fdr = lambda df: false_discovery_control(df.enrichment))
        .assign(pscore = lambda df: -np.log10(df.enrichment),
                prank = lambda df: df.pscore.rank(pct=True,ascending=False)
                )

        )
    cell_b_tbl = (data_tbl
        .query('barcode_idx == @pair[1]')
        .assign(cell_rate = lambda df: df.read_count/df.tot_count)
        .assign(cell_vs_bulk_OR= lambda df: df.cell_rate/df.gene_read_rate)
        .assign(enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1),
                fdr = lambda df: false_discovery_control(df.enrichment))
        .assign(pscore = lambda df: -np.log10(df.enrichment),
                prank = lambda df: df.pscore.rank(pct=True,ascending=False)
                )

        )
    
    a_b_gene_union_list = list(set(cell_b_tbl.gene_idx.to_list()).union(cell_a_tbl.gene_idx.to_list()))
    a_b_gene_inter_list = list(set(cell_b_tbl.gene_idx.to_list()).intersection(cell_a_tbl.gene_idx.to_list()))

    v_a = cell_a_tbl.sort_values('pscore',ascending=False).assign(v=lambda df: np.where(df.gene_idx.isin(a_b_gene_inter_list),1,0)).v.to_numpy()
    v_b = cell_b_tbl.sort_values('pscore',ascending=False).assign(v=lambda df: np.where(df.gene_idx.isin(a_b_gene_inter_list),1,0)).v.to_numpy()

    _, _, pval_a = xlmhglite.xlmhg_test(v_a, int(5), int(cell_a_tbl.shape[0]))
    _, _, pval_b = xlmhglite.xlmhg_test(v_b, int(5), int(cell_b_tbl.shape[0]))
    return pd.DataFrame({'a':[pair[0]],'b':pair[1],'a_pvalue':[pval_a],'b_pvalue':[pval_b]})
#%%    

tmp_pair = pairwise_combinations[900]
get_cell_similarity(tmp_pair,data_tbl)

#%%
df = [get_cell_similarity(pair,data_tbl) for pair in pairwise_combinations]
#%%
with Pool(processes=5) as pool:
        # pool.map applies 'parallel_func' to every item in 'pairwise_combinations'
        df = pool.map(partial(get_cell_similarity,data_tbl=data_tbl), pairwise_combinations)
# %%
(pd.concat(df)
 .assign(avg_pvalue = lambda df: (df.a_pvalue + df.b_pvalue)/2)
 .sort_values('avg_pvalue')
#  .avg_pvalue
#  .plot.kde()
 )