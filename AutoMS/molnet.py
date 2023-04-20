# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:02:43 2023

@author: DELL
"""


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from bokeh.io import show
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, ResetTool
from bokeh.models.graphs import from_networkx
from bokeh.palettes import Spectral4
from bokeh.plotting import figure

from sklearn.metrics import pairwise_distances


class MolNet:
    def __init__(self, feature_table_annotated, group_info):
        self.smiles = feature_table_annotated['CanonicalSMILES'].values
        self.names = feature_table_annotated['Annotated Name'].values
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
        
        name = {i: names[i] for i in index}
        smiles = {i: smiles[i] for i in index}
        
        plot = Plot(plot_width=1000, plot_height=700, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
        plot.add_tools(HoverTool(tooltips=[("index", "@index"), ("name", "@name"), ("smiles", "@smiles")]), TapTool(), BoxSelectTool(), ResetTool())

        graph_renderer = from_networkx(G1, nx.spring_layout, scale=1, center=(0,0))
        graph_renderer.node_renderer.glyph = Circle(size=10, fill_color=Spectral4[0])
        graph_renderer.edge_renderer.glyph = MultiLine(line_alpha=0.8, line_width=1)

        graph_renderer.node_renderer.data_source.data['index'] = list(name.keys())
        graph_renderer.node_renderer.data_source.data['name'] = list(name.values())
        graph_renderer.node_renderer.data_source.data['smiles'] = list(smiles.values())

        plot.renderers.append(graph_renderer)
        show(plot)


    def get_subgraph(self, compound_name):
        G1 = self.G1
        names = self.names
        
        i = np.where(names == compound_name)[0]
        if len(i) == 0:
            print('{} is not in the annotated list')
        else:
            center_node = i[0]
        
        subgraph_nodes = set(nx.descendants(G1, center_node)) | {center_node}
        subgraph = G1.subgraph(subgraph_nodes)
        self.G2 = subgraph
        
        
    def plot_selected_subgraph(self):
        G2 = self.G2
        