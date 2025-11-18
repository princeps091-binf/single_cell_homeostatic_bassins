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
from sklearn.manifold import MDS
import umap 
import scanpy as sc

#%%

filtered_count_folder = './../data/filtered_gene_bc_matrices/hg19/'

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
gene_of_interest_idx = gene_label_tbl.query("name == 'PPBP'").index.to_list()[0] + 1

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
#%%
mds = MDS(n_components=2,n_init=5, dissimilarity='precomputed', random_state=42)
X_transformed = mds.fit_transform(dist_matrix_square)
#%%
reordered_biomarker_trx_tbl = (data_enrich_tbl
 .query('barcode_idx in @unique_items')
 .query('gene_idx == @gene_of_interest_idx')
 .loc[:,['barcode_idx','gene_idx','enrichment']]
 .set_index('barcode_idx')
 .loc[unique_items]
)


#%%
plt_tbl = pd.DataFrame({'x':X_transformed[:, 0],
                        'y':X_transformed[:, 1],
                        'biomarker_lvl':reordered_biomarker_trx_tbl.enrichment.to_numpy()})
plt.figure(figsize=(8, 6))
plt.scatter(plt_tbl.x.to_numpy(), plt_tbl.y.to_numpy(),c=plt_tbl.biomarker_lvl.to_numpy(),cmap='viridis_r')
plt.title('MDS Embedding with Precomputed Distance Matrix')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()

#%%
reducer = umap.UMAP(metric='precomputed', random_state=42)
embedding = reducer.fit_transform(dist_matrix_square)
plt_tbl = pd.DataFrame({'x':embedding[:, 0],
                        'y':embedding[:, 1],
                        'biomarker_lvl':reordered_biomarker_trx_tbl.enrichment.to_numpy()})

plt.figure(figsize=(8, 6))
plt.scatter(plt_tbl.x.to_numpy(), plt_tbl.y.to_numpy(),c=plt_tbl.biomarker_lvl.to_numpy(),cmap='viridis_r')
plt.title('UMAP Embedding with Precomputed Distance Matrix')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()

# %%
# Current state-of-the-art only considers highly variable gene to
# determine cell similarity
adata = sc.read_10x_mtx(filtered_count_folder)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
# %%
cell_of_interest_id_list = (barcode_label_tbl
.assign(barcode_idx = lambda df: df.index + 1)
.query('barcode_idx in @cells_of_interest_list')
.ID.to_list()
)
#%%
pp_subspace_df = adata.to_df().loc[cell_of_interest_id_list,adata.var.query('highly_variable').index]
pp_subspace_long_df = pd.melt(pp_subspace_df.reset_index().rename(columns={'index':'ID'}),id_vars='ID',var_name='name',value_name='normalised_count')
#%%
tmp_distance_matrix = euclidean_distances(pp_subspace_df, pp_subspace_df)

# %%
hvg_condensed_dist_matrix = squareform(np.round(tmp_distance_matrix,decimals=5))
hvg_linked = linkage(hvg_condensed_dist_matrix, method='ward') # You can choose other methods like 'complete', 'average', 'single'
# Extract the leaf order from the linkage matrix
# The optimal_leaf_ordering function reorders the leaves for better visualization
hvg_ordered_linked = optimal_leaf_ordering(hvg_linked, hvg_condensed_dist_matrix)
leaf_order = dendrogram(hvg_ordered_linked, no_plot=True)['leaves']
hvg_reordered_matrix = tmp_distance_matrix[leaf_order, :]
hvg_reordered_matrix = hvg_reordered_matrix[:, leaf_order]

#%%
fig, ax = plt.subplots(figsize=(7, 6))

# 2. Display the matrix data as an image
# 'cmap' sets the color scheme (e.g., 'viridis', 'plasma', 'coolwarm', 'Greys')
# 'interpolation' determines how pixels are drawn (nearest is usually best for matrices)
im = ax.imshow(hvg_reordered_matrix, cmap='plasma_r', interpolation='nearest')
#%%
reordered_hvg_biomarker_trx_tbl = (data_enrich_tbl
 .query('barcode_idx in @unique_items')
 .query('gene_idx == @gene_of_interest_idx')
 .loc[:,['barcode_idx','gene_idx','enrichment']]
 .merge(barcode_label_tbl.assign(barcode_idx = lambda df: df.index + 1))
 .set_index('ID')
 .loc[cell_of_interest_id_list]
)
#%%
X_conventional_transformed = mds.fit_transform(tmp_distance_matrix)

plt_tbl = pd.DataFrame({'x':X_conventional_transformed[:, 0],
                        'y':X_conventional_transformed[:, 1],
                        'biomarker_lvl':reordered_hvg_biomarker_trx_tbl.enrichment.to_numpy()})
plt.figure(figsize=(8, 6))
plt.scatter(plt_tbl.x.to_numpy(), plt_tbl.y.to_numpy(),c=plt_tbl.biomarker_lvl.to_numpy(),cmap='viridis_r')
plt.title('MDS Embedding with Precomputed Distance Matrix')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()

# %%
hvg_reducer = umap.UMAP(metric='precomputed', random_state=42)
hvg_embedding = hvg_reducer.fit_transform(tmp_distance_matrix)
plt_tbl = pd.DataFrame({'x':hvg_embedding[:, 0],
                        'y':hvg_embedding[:, 1],
                        'biomarker_lvl':reordered_hvg_biomarker_trx_tbl.enrichment.to_numpy()})

plt.figure(figsize=(8, 6))
plt.scatter(plt_tbl.x.to_numpy(), plt_tbl.y.to_numpy(),c=plt_tbl.biomarker_lvl.to_numpy(),cmap='viridis_r')
plt.title('UMAP Embedding with Precomputed Distance Matrix')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()

# %%
