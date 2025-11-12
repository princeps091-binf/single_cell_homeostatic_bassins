#%%
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.stats import false_discovery_control
import xlmhglite
import itertools
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances,manhattan_distances, cosine_distances
#%%
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

def get_cell_similarity(pair,data_tbl):
    cell_a_tbl = data_tbl.query('barcode_idx == @pair[0]')
    cell_b_tbl = data_tbl.query('barcode_idx == @pair[1]')
    
    a_b_gene_inter_list = list(set(cell_b_tbl.gene_idx.to_list()).intersection(cell_a_tbl.gene_idx.to_list()))

    v_a = cell_a_tbl.sort_values('pscore',ascending=False).assign(v=lambda df: np.where(df.gene_idx.isin(a_b_gene_inter_list),1,0)).v.to_numpy()
    v_b = cell_b_tbl.sort_values('pscore',ascending=False).assign(v=lambda df: np.where(df.gene_idx.isin(a_b_gene_inter_list),1,0)).v.to_numpy()

    _, _, pval_a = xlmhglite.xlmhg_test(v_a, int(1), int(cell_a_tbl.shape[0]))
    _, _, pval_b = xlmhglite.xlmhg_test(v_b, int(1), int(cell_b_tbl.shape[0]))
    return pd.DataFrame({'a':[pair[0]],'b':pair[1],'a_pvalue':[pval_a],'b_pvalue':[pval_b]})

#%%
with Pool(processes=10) as pool:
        # pool.map applies 'parallel_func' to every item in 'pairwise_combinations'
        df = pool.map(partial(get_cell_enrich,data_tbl=data_tbl), count_tbl.barcode_idx.drop_duplicates().to_list())
#%%
data_enrich_tbl = pd.concat(df)

#%%
gene_of_interest_idx = gene_label_tbl.query("name == 'CD79A'").index.to_list()[0] + 1

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
len(pairwise_combinations)

#%%
tmp_pair = pairwise_combinations[9]
get_cell_similarity(tmp_pair,data_enrich_tbl)

#%%
with Pool(processes=10) as pool:
        # pool.map applies 'parallel_func' to every item in 'pairwise_combinations'
        df = pool.map(partial(get_cell_similarity,data_tbl=data_enrich_tbl), pairwise_combinations)
# %%
tmp_biomarker_bassin_tbl = (pd.concat(df)
 .assign(avg_pvalue = lambda df: (df.a_pvalue + df.b_pvalue)/2)
 .assign(two_way= lambda df: df.a_pvalue.lt(0.5) * df.b_pvalue.lt(0.5))
 .assign(a_b = lambda df: df.a_pvalue.lt(df.b_pvalue))
 .assign(max_pvalue = lambda df: np.where(df.a_b,df.b_pvalue,df.a_pvalue))
#  .query('two_way')
)
 #%%
unique_items = sorted(list(set(tmp_biomarker_bassin_tbl['a']).union(set(tmp_biomarker_bassin_tbl['b']))))
dist_matrix_square = pd.DataFrame(1.0, index=unique_items, columns=unique_items)
for _, row in tmp_biomarker_bassin_tbl.iterrows():
    item1 = row['a']
    item2 = row['b']
    distance = row['max_pvalue']
    dist_matrix_square.loc[item1, item2] = distance
    dist_matrix_square.loc[item2, item1] = distance # Assuming symmetric distances
np.fill_diagonal(dist_matrix_square.to_numpy(), 0.0)

mhgt_condensed_dist_matrix = squareform(dist_matrix_square)
linked = linkage(mhgt_condensed_dist_matrix, method='ward') # You can choose other methods like 'complete', 'average', 'single'
# Extract the leaf order from the linkage matrix
# The optimal_leaf_ordering function reorders the leaves for better visualization
mght_ordered_linked = optimal_leaf_ordering(linked, mhgt_condensed_dist_matrix)
mght_leaf_order = dendrogram(mght_ordered_linked, no_plot=True)['leaves']
mght_reordered_matrix = dist_matrix_square.to_numpy()[mght_leaf_order, :]
mght_reordered_matrix = mght_reordered_matrix[:, mght_leaf_order]

#%%
fig, ax = plt.subplots(figsize=(7, 6))

# 2. Display the matrix data as an image
# 'cmap' sets the color scheme (e.g., 'viridis', 'plasma', 'coolwarm', 'Greys')
# 'interpolation' determines how pixels are drawn (nearest is usually best for matrices)
im = ax.imshow(mght_reordered_matrix, cmap='plasma_r', interpolation='nearest')

# %%
cell_a_tbl = (
    data_enrich_tbl
    .query('barcode_idx == 983')

)

cell_b_tbl = (
    data_enrich_tbl
    .query('barcode_idx == 2687')
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
         avg_cut = lambda df: pd.cut(df.avg_pval,[0,0.01,0.05,0.25,0.5,1]))

)

(cell_compare_tbl
 .plot
 .scatter(x='cell_a',y='cell_b',c='avg_cut',alpha=1,cmap='viridis_r',s=2)
)


v_a = cell_a_tbl.sort_values('pscore',ascending=False).assign(v=lambda df: np.where(df.gene_idx.isin(a_b_gene_inter_list),1,0)).v.to_numpy()
v_b = cell_b_tbl.sort_values('pscore',ascending=False).assign(v=lambda df: np.where(df.gene_idx.isin(a_b_gene_inter_list),1,0)).v.to_numpy()

_, _, pval_a = xlmhglite.xlmhg_test(v_a, int(5), int(cell_a_tbl.shape[0]))
_, _, pval_b = xlmhglite.xlmhg_test(v_b, int(5), int(cell_b_tbl.shape[0]))

print(pval_a)
print(pval_b)
#%%
tmp_count_matrix = (pd.pivot(data_enrich_tbl
 .query('barcode_idx in @cells_of_interest_list')
 .loc[:,['gene_idx','barcode_idx','enrichment','cell_rate']]
 .assign(log1p = lambda df: np.log10(df.cell_rate + 1)),
 index='barcode_idx',columns='gene_idx',values='log1p')
 .fillna(0)
 )

tmp_distance_matrix = euclidean_distances(tmp_count_matrix, tmp_count_matrix)
#%%
tmp_distance_matrix[np.where(~(tmp_distance_matrix == tmp_distance_matrix.T))]
#%%
condensed_dist_matrix = squareform(np.round(tmp_distance_matrix,decimals=5))
linked = linkage(condensed_dist_matrix, method='ward') # You can choose other methods like 'complete', 'average', 'single'
# Extract the leaf order from the linkage matrix
# The optimal_leaf_ordering function reorders the leaves for better visualization
ordered_linked = optimal_leaf_ordering(linked, condensed_dist_matrix)
leaf_order = dendrogram(ordered_linked, no_plot=True)['leaves']
reordered_matrix = tmp_distance_matrix[mght_leaf_order, :]
reordered_matrix = reordered_matrix[:, mght_leaf_order]

mght_values_reordered_matrix = dist_matrix_square.to_numpy()[leaf_order, :]
mght_values_reordered_matrix = mght_reordered_matrix[:, leaf_order]

#%%
fig, ax = plt.subplots(figsize=(7, 6))

# 2. Display the matrix data as an image
# 'cmap' sets the color scheme (e.g., 'viridis', 'plasma', 'coolwarm', 'Greys')
# 'interpolation' determines how pixels are drawn (nearest is usually best for matrices)
im = ax.imshow(reordered_matrix, cmap='plasma_r', interpolation='nearest')

# %%
(pd.DataFrame({'mght':mhgt_condensed_dist_matrix,'euclidean':condensed_dist_matrix})
 .plot
 .scatter(x='mght',y='euclidean',s=0.1,alpha=0.3))


# %%
