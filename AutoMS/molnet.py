# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:02:43 2023

@author: DELL
"""

import io
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from tqdm import tqdm
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

from bokeh.io import show
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, ResetTool, ColumnDataSource
from bokeh.models.graphs import from_networkx
from bokeh.palettes import Spectral4
from bokeh.plotting import figure

from sklearn.metrics import pairwise_distances


class MolNet:
    def __init__(self, feature_table_annotated, biomarker_table, group_info):
        self.smiles = feature_table_annotated['SMILES'].values
        self.names = feature_table_annotated['Annotated Name'].values
        self.group_info = group_info
        self.feature_table_annotated = feature_table_annotated
        self.biomarker_table = biomarker_table
        self.matrix = None
        self.G1 = None
        self.G2 = None
        self.calc_molfps = lambda x: np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))
        
        
    def compute_similarity_matrix(self, metric = 'jaccard'):
        smiles = self.smiles
        molfps = np.array([self.calc_molfps(s) for s in tqdm(smiles)])
        mol_sim_matrix = 1 - pairwise_distances(molfps, metric = metric)
        self.matrix = mol_sim_matrix
    
    
    def create_network(self, threshold = 0.5):
        names = self.names
        smiles = self.smiles
        mol_sim_matrix = self.matrix
        if mol_sim_matrix is None:
            raise ValueError('Please run compute_similarity_matrix first')
        
        G1 = nx.Graph()
        for i in range(mol_sim_matrix.shape[0]):
            for j in range(i+1, mol_sim_matrix.shape[0]):
                if mol_sim_matrix[i,j] >= threshold:
                    G1.add_edge(i, j, weight = mol_sim_matrix[i,j])
        
        for node in G1.nodes(data=True):
            i = node[0]
            G1.add_node(i, name = names[i], smiles = smiles[i])
        
        self.G1 = G1
        
        
    def plot_global_network(self):
        G1 = self.G1
        names = self.names
        smiles = self.smiles
        
        index = []
        for node in G1.nodes(data=True):
            index.append(node[0])
        index = np.array(index)
        
        if self.biomarker_table is not None:
            highlight_names = self.biomarker_table['Annotated Name'].values
            highlight_index = [i for i in index if names[i] in highlight_names]
        else:
            highlight_index = []
        node_indices = list(G1.nodes)
        node_colors = ["darkred" if node in highlight_index else "lightblue" for node in G1.nodes()]
        node_source = ColumnDataSource(data=dict(index=node_indices, fill_color=node_colors))
        
        name = {i: names[i] for i in index}
        smiles = {i: smiles[i] for i in index}
        
        plot = Plot(plot_width=1000, plot_height=700, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
        plot.add_tools(HoverTool(tooltips=[("index", "@index"), ("name", "@name"), ("smiles", "@smiles")]), TapTool(), BoxSelectTool(), ResetTool())

        graph_renderer = from_networkx(G1, nx.spring_layout, scale=1, center=(0,0))
        graph_renderer.node_renderer.data_source = node_source
        graph_renderer.node_renderer.glyph = Circle(size=10, fill_color={'field': 'fill_color'})
        graph_renderer.edge_renderer.glyph = MultiLine(line_alpha=0.8, line_width=1)

        graph_renderer.node_renderer.data_source.data['index'] = list(name.keys())
        graph_renderer.node_renderer.data_source.data['name'] = list(name.values())
        graph_renderer.node_renderer.data_source.data['smiles'] = list(smiles.values())

        plot.renderers.append(graph_renderer)
        show(plot)


    def get_subgraph(self, compound_name, maximum_compound = 12):
        G1 = self.G1
        names = self.names
        mol_sim_matrix = self.matrix
        
        i = np.where(names == compound_name)[0]
        if len(i) == 0:
            raise ValueError('{} is not in the annotated list'.format(compound_name))
        else:
            center_node = i[0]
        
        subgraph_nodes = set(nx.descendants(G1, center_node)) | {center_node}
        subgraph = G1.subgraph(subgraph_nodes)

        if len(subgraph.nodes) > maximum_compound:
            keep = np.argsort(-mol_sim_matrix[center_node,:])[0:maximum_compound]
            subgraph_nodes = set(nx.descendants(G1, center_node)) & set(keep)
            subgraph = G1.subgraph(subgraph_nodes)
 
        self.G2 = subgraph
        
        
    def plot_selected_subgraph(self):
        G2 = self.G2
        names = self.names
        if G2 is None:
            raise ValueError('No sub-graph is selected')
        group_info = self.group_info
        feature_table_annotated = self.feature_table_annotated
        feature_table_annotated = feature_table_annotated.reset_index(drop = True)
        
        thick = [(u, v) for (u, v, d) in G2.edges(data=True) if d["weight"] >= 0.8]
        medium = [(u, v) for (u, v, d) in G2.edges(data=True) if 0.6 < d["weight"] < 0.8]
        thin = [(u, v) for (u, v, d) in G2.edges(data=True) if d["weight"] <= 0.6]

        pos = nx.circular_layout(G2)
        pos = nx.spring_layout(G2, pos=pos)
        fig, ax = plt.subplots(figsize=(15, 12), dpi = 300)
        nx.draw_networkx_edges(G2, pos=pos, ax=ax, edgelist=thick, width=3, alpha=0.7, edge_color="black")
        nx.draw_networkx_edges(G2, pos=pos, ax=ax, edgelist=medium, width=2, alpha=0.7, edge_color="black")
        nx.draw_networkx_edges(G2, pos=pos, ax=ax, edgelist=thin, width=1, alpha=0.7, edge_color="black")
        ax.axis('off')
        
        tr_figure = ax.transData.transform
        tr_axes = fig.transFigure.inverted().transform

        struct_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.045 # adjust this value to change size of structure drawings
        struct_center = struct_size / 2.0

        for i, node in enumerate(G2.nodes(data = True)):
            rdimg = Draw.MolToImage(Chem.MolFromSmiles(node[1]['smiles']))
            pilimg = Image.frombytes('RGB', rdimg.size, rdimg.tobytes())
            xf, yf = tr_figure(pos[node[0]])
            xa, ya = tr_axes((xf, yf))
            # get overlapped axes and plot structure
            a = plt.axes([xa - struct_center, ya - struct_center, struct_size, struct_size])
            a.imshow(pilimg)
            plt.text(xa - struct_center + 0.02, ya - struct_center - 0.02, names[i])
            a.axis("off")
            
            if group_info is not None:
                group_values = [feature_table_annotated.loc[node[0], group_info[k]] for k in group_info.keys()]
                group_values = [np.median(v) for v in group_values]
                group_values = list(group_values / np.mean(group_values))
                b = plt.axes([xa - struct_center + 0.02, ya - struct_center - 0.02, struct_size / 2.5, struct_size / 2.5])
                b.imshow([group_values], cmap='bwr')
                border = Rectangle((-0.5, -0.5), len(group_values), 1, fill=False, edgecolor='black', linewidth=1.5)
                border.set_clip_on(False)
                b.add_patch(border)
                b.axis("off")
        plt.show()
        