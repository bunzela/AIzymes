import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
#import networkx as nx
from sklearn.decomposition import PCA
#import umap
#import torch
#import esm

# NOTES:
# Use self to access shared variables
# functions that use self must contain self, e.g.: plot_scores(self) 
# Only put super important variables into self! (for now please add nothing)

# Coding STRATEGY:
# Add plot_scores to class in AIzymes_013.py (must have a different name!)
# To initialze everything in the main script, do:

"""
import sys, os
if os.path.join(os.getcwd(), '../../src') not in sys.path: sys.path.append(os.path.join(os.getcwd(), '../../src'))
from AIzymes_013 import *
AIzymes = AIzymes_MAIN()
AIzymes.initialize(FOLDER_HOME = 'AB_refactor_standard_013', LOG="debug", PRINT_VAR=False) #LOG="debug/info"
"""
# For you, please have a look at --> pandas data frames
# This also loads self.all_scores_df
# For plotting, make a copy of this, or do: self.plot_scores_df Do not modify self.all_scores_df
# In the end, plotting will be done by executing: AIzymes.plot()



def plot_scores(self, combined_score_min=0, combined_score_max=1, combined_score_bin=0.01,
                interface_score_min=0, interface_score_max=1, interface_score_bin=0.01,
                total_score_min=0, total_score_max=1, total_score_bin=0.01,
                catalytic_score_min=0, catalytic_score_max=1, catalytic_score_bin=0.01
                ):
     
    print("plotting")
    mut_min=0
    mut_max=len(self.DESIGN.split(","))+1 
    
    # Break because file does not exist
    if not os.path.isfile(ALL_SCORES_CSV): return
    
    all_scores_df = pd.read_csv(ALL_SCORES_CSV)
    all_scores_df['sequence'] = all_scores_df['sequence'].astype(str)
    all_scores_df['design_method'] = all_scores_df['design_method'].astype(str)
    all_scores_df['score_taken_from'] = all_scores_df['score_taken_from'].astype(str)
            
    # Break because not enough data
    if len(all_scores_df.dropna(subset=['total_score'])) < 3: return
    

    if (ProteinMPNN_PROB > 0 or LMPNN_PROB > 0):
        #first not nan sequence from all_scores_df
        mut_max = len(all_scores_df[all_scores_df['sequence'] != 'nan']['sequence'].iloc[0])   
        
    # Plot data
    fig, axs = plt.subplots(3, 4, figsize=(15, 9))
    
    all_scores_df = all_scores_df.dropna(subset=['total_score'])
    catalytic_scores, total_scores, interface_scores, efield_scores, combined_scores = normalize_scores(all_scores_df, 
                                                                                         print_norm=True,
                                                                                         norm_all=True)
            
    plot_combined_score(axs[0,0], combined_scores, \
                        combined_score_min, combined_score_max, combined_score_bin)
    plot_interface_score(axs[0,1], interface_scores, \
                         interface_score_min, interface_score_max, interface_score_bin)
    plot_total_score(axs[0,2], total_scores, \
                     total_score_min, total_score_max, total_score_bin)
    plot_catalytic_score(axs[0,3], catalytic_scores, \
                         catalytic_score_min, catalytic_score_max, catalytic_score_bin)
    
    plot_boltzmann_histogram(axs[1,0], combined_scores, all_scores_df, \
                             combined_score_min, combined_score_max, combined_score_bin)
    plot_combined_score_v_index(axs[1,1], combined_scores, all_scores_df)
    plot_combined_score_v_generation_violin(axs[1,2], combined_scores, all_scores_df)
    plot_mutations_v_generation_violin(axs[1,3], all_scores_df, mut_min, mut_max)
        
    plot_efield_score(axs[2,0], efield_scores, \
                        catalytic_score_min, catalytic_score_max, catalytic_score_bin)
    plot_score_v_generation_violin(axs[2,1], 'total_score', all_scores_df)
    plot_score_v_generation_violin(axs[2,2], 'interface_score', all_scores_df)
    plot_score_v_generation_violin(axs[2,3], 'efield_score', all_scores_df)
    
    plt.tight_layout()
    plt.show()
    
    # fig, ax = plt.subplots(1, 1, figsize=(15, 2.5))
    # plot_tree(ax, combined_scores, all_scores_df, G)
    # plt.show()

def plot_summary():
    all_scores_df = pd.read_csv(ALL_SCORES_CSV)
    
    fig, axs = plt.subplots(3, 2, figsize=(20, 23))

    # Normalize scores
    catalytic_scores, total_scores, interface_scores, efield_scores, combined_scores = normalize_scores(all_scores_df, 
                                                                                         print_norm=True,
                                                                                         norm_all=True)
    
    # Plot interface vs total score colored by generation
    plot_interface_v_total_score_generation(axs[0,0], total_scores, interface_scores, all_scores_df['generation'])

    # Plot stacked histogram of interface scores by generation
    plot_stacked_histogram_by_generation(axs[0, 1], all_scores_df)

    # Plot stacked histogram of interface scores by catalytic residue index
        # Create a consistent color map for catalytic residues
    all_scores_df['cat_resi'] = pd.to_numeric(all_scores_df['cat_resi'], errors='coerce')
    unique_cat_resi = all_scores_df['cat_resi'].dropna().unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cat_resi)))
    color_map = {resi: colors[i] for i, resi in enumerate(unique_cat_resi)}
    plot_stacked_histogram_by_cat_resi(axs[1, 0], all_scores_df, color_map=color_map)

    # Plot stacked histogram of interface scores by catalytic residue index (excluding generation 0)
    all_scores_df_cleaned = all_scores_df[all_scores_df['generation'] != 0]  # Exclude generation 0
    plot_stacked_histogram_by_cat_resi(axs[1, 1], all_scores_df_cleaned, color_map=color_map)

    # Plot interface vs total score colored by catalytic residue index
    plot_interface_v_total_score_cat_resi(axs[2,0], total_scores, interface_scores, all_scores_df['cat_resi'])

    # Plot interface vs total score colored by catalytic residue name
    legend_elements = plot_interface_v_total_score_cat_resn(axs[2, 1], total_scores, interface_scores, all_scores_df['cat_resn'])
    
    axs[0,0].set_title('Total Scores vs Interface Scores by Generation')
    axs[0,1].set_title('Stacked Histogram of Interface Scores by Generation')
    axs[1,0].set_title('Stacked Histogram of Interface Scores by Catalytic Residue Index')
    axs[1,1].set_title('Stacked Histogram of Interface Scores by Catalytic Residue Index (Excluding Generation 0)')
    axs[2,0].set_title('Total Scores vs Interface Scores by Catalytic Residue Index')
    axs[2,1].set_title('Total Scores vs Interface Scores by Catalytic Residue Name')

    # Adjust legends
    # For cat_resi
    handles_cat_resi, labels_cat_resi = axs[1,1].get_legend_handles_labels()
    fig.legend(handles_cat_resi, labels_cat_resi, loc='upper right', bbox_to_anchor=(0.95, 0.65), title="Catalytic Residue Index")

    # For generation
    handles_generation, labels_generation = axs[0,1].get_legend_handles_labels()
    fig.legend(handles_generation, labels_generation, loc='upper right', bbox_to_anchor=(0.95, 0.98), title="Generation")

    # For cat_resn
    handles_cat_resn, labels_cat_resn = axs[2,1].get_legend_handles_labels()
    fig.legend(handles = legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.31), title="Catalytic Residue")

    # Adjust layout to make space for the legends on the right
    plt.tight_layout(rect=[0, 0, 0.85, 1])

def plot_interface_v_total_score_selection(ax, total_scores, interface_scores, selected_indices):
    """
    Plots a scatter plot of total_scores vs interface_scores and highlights the points
    corresponding to the selected indices.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to plot on.
    - total_scores (list or np.array): The total scores of the structures.
    - interface_scores (list or np.array): The interface scores of the structures.
    - selected_indices (list of int): Indices of the points to highlight.
    """
    
    # Create a mask for selected indices
    mask = np.ones(len(total_scores), dtype=bool)  # Initialize mask to include all points
    mask[selected_indices] = False  # Exclude selected indices
    
    # Plot all points excluding the selected ones on the given Axes object
    ax.scatter(total_scores[mask], interface_scores[mask], color='gray', alpha=0.3, label='All Points', s=1)
    
    # Highlight selected points on the given Axes object
    ax.scatter(total_scores[selected_indices], interface_scores[selected_indices], color='red', alpha=0.4, label='Selected Points', s=1)
    
    ax.set_title('Total Scores vs Interface Scores')
    ax.set_xlabel('Total Score')
    ax.set_ylabel('Interface Score')
    ax.legend()
    ax.grid(True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def plot_interface_v_total_score_cat_resi(ax, total_scores, interface_scores, cat_resi):
    from matplotlib.lines import Line2D
    """
    Plots a scatter plot of total_scores vs interface_scores and colors the points
    according to the catalytic residue number (cat_resi) for all data points, using categorical coloring.
    Adds a legend to represent each unique catalytic residue number with its corresponding color.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to plot on.
    - total_scores (list or np.array): The total scores of the structures.
    - interface_scores (list or np.array): The interface scores of the structures.
    - cat_resi (pd.Series or np.array): Catalytic residue numbers for all data points.
    """
    # Ensure cat_resi is a pandas Series for easier handling and remove NaN values
    if not isinstance(cat_resi, pd.Series):
        cat_resi = pd.Series(cat_resi)
    cat_resi = cat_resi.dropna()  # Drop NaN values
    
    # Proceed with the rest of the function after removing NaN values
    unique_cat_resi = cat_resi.unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cat_resi)))

    color_map = {resi: colors[i] for i, resi in enumerate(unique_cat_resi)}
    cat_resi_colors = cat_resi.map(color_map).values

    scatter = ax.scatter(total_scores[cat_resi.index], interface_scores[cat_resi.index], c=cat_resi_colors, alpha=0.4, s=2)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cat Resi {resi}',
                              markerfacecolor=color_map[resi], markersize=10) for resi in unique_cat_resi]
    #ax.legend(handles=legend_elements, title="Catalytic Residue", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title('Total Scores vs Interface Scores')
    ax.set_xlabel('Total Score')
    ax.set_ylabel('Interface Score')
    ax.grid(True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def plot_interface_v_total_score_cat_resn(ax, total_scores, interface_scores, cat_resn):
    from matplotlib.lines import Line2D
    """
    Plots a scatter plot of total_scores vs interface_scores and colors the points
    according to the catalytic residue name (cat_resn) for all data points, using categorical coloring.
    Adds a legend to represent each unique catalytic residue name with its corresponding color.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to plot on.
    - total_scores (list or np.array): The total scores of the structures.
    - interface_scores (list or np.array): The interface scores of the structures.
    - cat_resn (pd.Series or np.array): Catalytic residue names for all data points.
    """
    # Ensure cat_resn is a pandas Series for easier handling and remove NaN values
    if not isinstance(cat_resn, pd.Series):
        cat_resn = pd.Series(cat_resn)
    cat_resn = cat_resn.dropna()  # Drop NaN values
    
    # Proceed with the rest of the function after removing NaN values
    unique_cat_resn = cat_resn.unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cat_resn)))

    color_map = {resn: colors[i] for i, resn in enumerate(unique_cat_resn)}
    cat_resn_colors = cat_resn.map(color_map).values

    scatter = ax.scatter(total_scores[cat_resn.index], interface_scores[cat_resn.index], c=cat_resn_colors, alpha=0.4, s=2)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cat Resn {resn}',
                              markerfacecolor=color_map[resn], markersize=10) for resn in unique_cat_resn]
    #ax.legend(handles=legend_elements, title="Catalytic Residue", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title('Total Scores vs Interface Scores')
    ax.set_xlabel('Total Score')
    ax.set_ylabel('Interface Score')
    ax.grid(True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return legend_elements

def plot_interface_v_total_score_generation(ax, total_scores, interface_scores, generation):
    """
    Plots a scatter plot of total_scores vs interface_scores and colors the points
    according to the generation for all data points, using categorical coloring.
    Adds a legend to represent each unique generation with its corresponding color.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to plot on.
    - total_scores (list or np.array): The total scores of the structures.
    - interface_scores (list or np.array): The interface scores of the structures.
    - generation (pd.Series or np.array): Generation numbers for all data points.
    """
    if not isinstance(generation, pd.Series):
        generation = pd.Series(generation)
    generation = generation.dropna()  # Drop NaN values
    
    unique_generations = generation.unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_generations)))

    color_map = {gen: colors[i] for i, gen in enumerate(unique_generations)}
    
    # Loop through each generation to plot, adjusting alpha for generation 0
    for gen in unique_generations:
        gen_mask = generation == gen
        alpha_value = 0.2 if gen == 0 else 0.8  # More transparent for generation 0
        ax.scatter(total_scores[generation.index][gen_mask], interface_scores[generation.index][gen_mask], 
                   c=[color_map[gen]], alpha=alpha_value, s=2, label=f'Generation {gen}' if gen == 0 else None)
    #ax.legend(handles=legend_elements, title="Generation", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title('Total vs Interface Scores - Generation', fontsize=18)
    ax.set_xlabel('Total Score', fontsize=16)
    ax.set_ylabel('Interface Score', fontsize=16)
    ax.grid(True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.tick_params(axis='both', which='major', labelsize=16)

def plot_stacked_histogram_by_cat_resi(ax, all_scores_df, color_map=None, show_legend=False):
    """
    Plots a stacked bar plot of interface scores colored by cat_resi on the given Axes object,
    where each bar's segments represent counts of different cat_resi values in that bin.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to plot on.
    - all_scores_df (pd.DataFrame): DataFrame containing 'cat_resi' and 'interface_score' columns.
    - color_map (dict): Optional; A dictionary mapping catalytic residue indices to colors.
    - show_legend (bool): Optional; Whether to show the legend. Defaults to False.
    """
    # Drop rows with NaN in 'interface_score'
    all_scores_df_cleaned = all_scores_df.dropna(subset=['interface_score'])

    # Ensure cat_resi is numeric and drop NaN values
    all_scores_df_cleaned['cat_resi'] = pd.to_numeric(all_scores_df_cleaned['cat_resi'], errors='coerce').dropna()
    unique_cat_resi = all_scores_df_cleaned['cat_resi'].unique()

    # Use the provided color_map or generate a new one
    if color_map is None:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cat_resi)))  # Generate colors for unique cat_resi values
        color_map = {resi: colors[i] for i, resi in enumerate(unique_cat_resi)}  # Map cat_resi to colors

    # Define bins for the histogram
    bins = np.linspace(all_scores_df_cleaned['interface_score'].min(), all_scores_df_cleaned['interface_score'].max(), 21)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Calculate counts for each cat_resi in each bin
    counts = {resi: np.histogram(all_scores_df_cleaned[all_scores_df_cleaned['cat_resi'] == resi]['interface_score'], bins=bins)[0] for resi in unique_cat_resi}

    # Plot stacked bars for each bin
    bottom = np.zeros(len(bin_centers))
    for resi in unique_cat_resi:
        ax.bar(bin_centers, counts[resi], bottom=bottom, width=np.diff(bins), label=f'Cat Resi {resi}', color=color_map[resi], align='center')
        bottom += counts[resi]

    # Create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cat Resi {resi}',
                              markerfacecolor=color_map[resi], markersize=10) for resi in unique_cat_resi]
    if show_legend:
        ax.legend(handles=legend_elements, title="Catalytic Residue")

    ax.set_title('Stacked Histogram of Interface Scores')
    ax.set_xlabel('Interface Score')
    ax.set_ylabel('Count')

    ax.set_xlim(-32.5, -13.5)

def plot_stacked_histogram_by_cat_resn(ax, all_scores_df):
    """
    Plots a stacked bar plot of interface scores colored by cat_resn on the given Axes object,
    where each bar's segments represent counts of different cat_resn values in that bin.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to plot on.
    - all_scores_df (pd.DataFrame): DataFrame containing 'cat_resn' and 'interface_score' columns.
    """
    # Drop rows with NaN in 'interface_score'
    all_scores_df_cleaned = all_scores_df.dropna(subset=['interface_score'])

    # Ensure cat_resn is a string and drop NaN values
    all_scores_df_cleaned['cat_resn'] = all_scores_df_cleaned['cat_resn'].astype(str).dropna()
    unique_cat_resn = all_scores_df_cleaned['cat_resn'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cat_resn)))  # Generate colors for unique cat_resn values

    color_map = {resn: colors[i] for i, resn in enumerate(unique_cat_resn)}  # Map cat_resn to colors

    # Define bins for the histogram
    bins = np.linspace(all_scores_df_cleaned['interface_score'].min(), all_scores_df_cleaned['interface_score'].max(), 21)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Calculate counts for each cat_resn in each bin
    counts = {resn: np.histogram(all_scores_df_cleaned[all_scores_df_cleaned['cat_resn'] == resn]['interface_score'], bins=bins)[0] for resn in unique_cat_resn}

    # Plot stacked bars for each bin
    bottom = np.zeros(len(bin_centers))
    for resn in unique_cat_resn:
        ax.bar(bin_centers, counts[resn], bottom=bottom, width=np.diff(bins), label=f'Cat Resn {resn}', color=color_map[resn], align='center')
        bottom += counts[resn]
    
    ax.set_xlim(-32.5, -13.5)

    # Create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cat Resn {resn}',
                              markerfacecolor=color_map[resn], markersize=10) for resn in unique_cat_resn]
    #ax.legend(handles=legend_elements, title="Catalytic Residue")

    ax.set_title('Stacked Histogram of Interface Scores by Catalytic Residue')
    ax.set_xlabel('Interface Score')
    ax.set_ylabel('Count')

def plot_stacked_histogram_by_generation(ax, all_scores_df):
    """
    Plots a stacked bar plot of interface scores colored by generation on the given Axes object,
    where each bar's segments represent counts of different generation values in that bin.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to plot on.
    - all_scores_df (pd.DataFrame): DataFrame containing 'generation' and 'interface_score' columns.
    """
    all_scores_df_cleaned = all_scores_df.dropna(subset=['interface_score', 'generation'])

    all_scores_df_cleaned['generation'] = pd.to_numeric(all_scores_df_cleaned['generation'], errors='coerce').dropna()
    unique_generations = all_scores_df_cleaned['generation'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_generations)))

    color_map = {gen: colors[i] for i, gen in enumerate(unique_generations)}

    bins = np.linspace(all_scores_df_cleaned['interface_score'].min(), all_scores_df_cleaned['interface_score'].max(), 21)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    counts = {gen: np.histogram(all_scores_df_cleaned[all_scores_df_cleaned['generation'] == gen]['interface_score'], bins=bins)[0] for gen in unique_generations}

    bottom = np.zeros(len(bin_centers))
    for gen in unique_generations:
        ax.bar(bin_centers, counts[gen], bottom=bottom, width=np.diff(bins), label=f'Generation {gen}', color=color_map[gen], align='center')
        bottom += counts[gen]

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Generation {gen}',
                              markerfacecolor=color_map[gen], markersize=10) for gen in unique_generations]
    #ax.legend(handles=legend_elements, title="Generation")

    ax.set_title('Stacked Histogram of Interface Scores by Generation')
    ax.set_xlabel('Interface Score')
    ax.set_ylabel('Count')

    ax.set_xlim(-32.5, -13.5)

def plot_combined_score(ax, combined_scores, score_min, score_max, score_bin):
    
    ax.hist(combined_scores, bins=np.arange(score_min,score_max+score_bin,score_bin))
    ax.axvline(HIGHSCORE, color='b')
    ax.axvline(NEG_BEST, color='r')
    ax.set_xlim(score_min,score_max)
    ax.set_title('Histogram of Score')
    ax.set_xlabel('Combined Score')
    ax.set_ylabel('Frequency')
    
def plot_interface_score(ax, interface_scores, interface_score_min, interface_score_max, interface_score_bin):
        
    ax.hist(interface_scores, density=True,
            bins=np.arange(interface_score_min,interface_score_max+interface_score_bin,interface_score_bin))
    ax.set_xlim(interface_score_min,interface_score_max)
    ax.set_title('Histogram of Interface Score')
    ax.set_xlabel('Interface Score')
    ax.set_ylabel('Frequency')
    
def plot_total_score(ax, total_scores, total_score_min, total_score_max, total_score_bin):

    ax.hist(total_scores, density=True,
            bins=np.arange(total_score_min,total_score_max+total_score_bin,total_score_bin))
    ax.set_xlim(total_score_min,total_score_max)
    ax.set_title('Histogram of Total Score')
    ax.set_xlabel('Total Score')
    ax.set_ylabel('Frequency')
    
def plot_catalytic_score(ax, catalytic_scores, total_score_min, total_score_max, total_score_bin):

    ax.hist(catalytic_scores, density=True,
            bins=np.arange(total_score_min,total_score_max+total_score_bin,total_score_bin))
    ax.set_xlim(total_score_min,total_score_max)
    ax.set_title('Histogram of Catalytic Score')
    ax.set_xlabel('Catalytic Score')
    ax.set_ylabel('Frequency')
    
def plot_efield_score(ax, efield_scores, total_score_min, total_score_max, total_score_bin):

    ax.hist(efield_scores, density=True,
            bins=np.arange(total_score_min,total_score_max+total_score_bin,total_score_bin))
    ax.set_xlim(total_score_min,total_score_max)
    ax.set_title('Histogram of Efield Score')
    ax.set_xlabel('Efield Score')
    ax.set_ylabel('Frequency')

def plot_boltzmann_histogram(ax, combined_scores, all_scores_df, score_min, score_max, score_bin):

    _, _, _, _, combined_potentials = normalize_scores(all_scores_df, print_norm=False, norm_all=False, extension="score")
            
    if isinstance(KBT_BOLTZMANN, (float, int)):
        kbt_boltzmann = KBT_BOLTZMANN
    else:
        if len(KBT_BOLTZMANN) == 2:
            kbt_boltzmann = max(KBT_BOLTZMANN[0] * np.exp(-KBT_BOLTZMANN[1]*all_scores_df['index'].max()), 0.05)
    boltzmann_factors = np.exp(combined_potentials / (kbt_boltzmann)) 
    print(f"Min/Max boltzmann factors: {min(boltzmann_factors)}, {max(boltzmann_factors)}")
    probabilities     = boltzmann_factors / sum(boltzmann_factors) 
    
    random_scores     = np.random.choice(combined_potentials, size=10000, replace=True)
    boltzmann_scores  = np.random.choice(combined_potentials, size=10000, replace=True, p=probabilities)


    # Plot the first histogram
    ax.hist(random_scores, density=True, alpha=0.7, label='Random Sampling', \
            bins=np.arange(score_min-2,score_max+1+score_bin,score_bin))
    ax.text(0.05, 0.95, "normalized only to \n this dataset")
    ax.set_xlabel('Potential')
    ax.set_ylabel('Density (Normal)')
    ax.set_title(f'kbT = {kbt_boltzmann:.1e}')
    # Create a twin y-axis for the second histogram
    ax_dup = ax.twinx()
    ax_dup.hist(boltzmann_scores, density=True, alpha=0.7, color='orange', label='Boltzmann Sampling', \
                bins=np.arange(score_min-2,score_max+1+score_bin,score_bin))
    ax.set_xlim(score_min-2,score_max+1)
    ax_dup.set_ylabel('Density (Boltzmann)')
    ax_dup.tick_params(axis='y', labelcolor='orange')

def plot_interface_score_v_total_score(ax, all_scores_df, 
                                       total_score_min, total_score_max, interface_score_min, interface_score_max):

    ax.scatter(all_scores_df['total_score'], all_scores_df['interface_score'],
            c=all_scores_df['index'], cmap='coolwarm_r', s=5)
    correlation,_ = pearsonr(all_scores_df['total_score'], all_scores_df['interface_score'])
    xmin = all_scores_df['total_score'].min()
    xmax = all_scores_df['total_score'].max()
    z = np.polyfit(all_scores_df['total_score'], all_scores_df['interface_score'], 1)
    p = np.poly1d(z)
    x_trendline = np.linspace(xmin, xmax, 100) 
    ax.plot(x_trendline, p(x_trendline), "k")
    ax.set_title(f'Pearson r: {correlation:.2f}')
    ax.set_xlim(total_score_min,total_score_max)
    ax.set_ylim(interface_score_min,interface_score_max)
    ax.set_xlabel('Total Score')
    ax.set_ylabel('Interface Score')

def plot_combined_score_v_index(ax, combined_scores, all_scores_df):
    
    combined_scores = pd.Series(combined_scores)
    moving_avg = combined_scores.rolling(window=20).mean()
    ax.scatter(all_scores_df['index'], combined_scores, c='lightgrey', s=5) 
    ax.axhline(HIGHSCORE, color='b', alpha = 0.5)
    ax.axhline(NEG_BEST, color='r', alpha = 0.5)
    PARENTS = [i for i in os.listdir(f'{FOLDER_HOME}/{FOLDER_PARENT}') if i[-4:] == ".pdb"]
    ax.axvline(N_PARENT_JOBS*len(PARENTS), color='k')
    ax.plot(range(len(moving_avg)),moving_avg,c="k")
    ax.set_ylim(0,1)
    ax.set_xlim(0,MAX_DESIGNS)
    ax.set_title('Score vs Index')
    ax.set_xlabel('Index')
    ax.set_ylabel('Combined Score')    
    
def plot_combined_score_v_generation(ax, combined_scores, all_scores_df):
    
    all_scores_df['tmp'] = combined_scores
    all_scores_df = all_scores_df.dropna(subset=['tmp'])
    
    max_gen = int(all_scores_df['generation'].max())
    boxplot_data = [all_scores_df[all_scores_df['generation'] == gen]['tmp'] for gen in range(0,max_gen+1,1)]
    # Define properties for the outliers
    flierprops = dict(marker='o', markerfacecolor='green', markersize=3, linestyle='none')
    
    ax.boxplot(boxplot_data, positions=range(len(boxplot_data)), flierprops=flierprops)
    ax.axhline(HIGHSCORE, color='b')
    ax.axhline(NEG_BEST, color='r')
    ax.set_xticks(range(len(boxplot_data)))
    ax.set_xticklabels(range(0,len(boxplot_data),1))
    ax.set_ylim(0,1)
    ax.set_title('Combined Score vs Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Combined Score')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

def exponential_func(x, A, k, c):
    return c-A*np.exp(-k * x)

def plot_combined_score_v_generation_violin(ax, combined_scores, all_scores_df):
    
    all_scores_df['tmp'] = combined_scores
    all_scores_df = all_scores_df.dropna(subset=['tmp'])
    
    max_gen = int(all_scores_df['generation'].max())
    generations = np.arange(0, max_gen + 1)
    violin_data = [all_scores_df[all_scores_df['generation'] == gen]['tmp'] for gen in generations]
    
    # Create violin plots
    parts = ax.violinplot(violin_data, positions=generations, showmeans=False, showmedians=True)
    
    # Customizing the color of violin plots
    for pc in parts['bodies']:
        pc.set_facecolor('green')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customizing the color of the median lines
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts.get(partname)
        if vp:
            vp.set_edgecolor('tomato')
            vp.set_linewidth(0.5)
    
    vp = parts.get('cmedians')
    if vp:
        vp.set_edgecolor('tomato')
        vp.set_linewidth(2.0)
    
    # Fit the data to the exponential function
    #weights = np.linspace(1, 0.1, len(generations))
    weights = np.ones(len(generations))
    #weights[:1] = 0.3
    mean_scores = [np.mean(data) for data in violin_data]
    popt, pcov = curve_fit(exponential_func, generations, mean_scores, p0=(1, 0.1, 0.7), sigma=weights, maxfev=2000)
    
    # Plot the fitted curve
    fitted_curve = exponential_func(generations, *popt)
    ax.plot(generations, fitted_curve, 'r--', label=f'Fit: A*exp(-kt) - c\nA={popt[0]:.2f}, k={popt[1]:.2f}, c={popt[2]:.2f}')
    
    ax.axhline(HIGHSCORE, color='b', label='High Score', alpha = 0.5)
    ax.axhline(NEG_BEST, color='r', label='Negative Best', alpha = 0.5)

    # Select every second generation for ticks
    every_second_generation = generations[::2]
    ax.set_xticks(every_second_generation)
    ax.set_xticklabels(every_second_generation)

    ax.set_ylim(0, 1)
    ax.set_title('Combined Score vs Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Combined Score')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    ax.legend(loc='lower right')

def plot_score_v_generation_violin(ax, score_type, all_scores_df):
    all_scores_df = all_scores_df.dropna(subset=[score_type])
    
    max_gen = int(all_scores_df['generation'].max())
    generations = np.arange(0, max_gen + 1)
    violin_data = [all_scores_df[all_scores_df['generation'] == gen][score_type] for gen in generations]
    
    # Create violin plots
    parts = ax.violinplot(violin_data, positions=generations, showmeans=False, showmedians=True)
    
    # Customizing the color of violin plots
    for pc in parts['bodies']:
        pc.set_facecolor('green')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customizing the color of the median lines
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts.get(partname)
        if vp:
            vp.set_edgecolor('tomato')
            vp.set_linewidth(0.5)
    
    vp = parts.get('cmedians')
    if vp:
        vp.set_edgecolor('tomato')
        vp.set_linewidth(2.0)
    
    # Fit the data to the exponential function
    # weights = np.ones(len(generations))
    # mean_scores = [np.mean(data) for data in violin_data]
    # popt, pcov = curve_fit(exponential_func, generations, mean_scores, p0=(1, 0.1, 0.7), sigma=weights, maxfev=10000)
    
    # # Plot the fitted curve
    # fitted_curve = exponential_func(generations, *popt)
    # ax.plot(generations, fitted_curve, 'r--', label=f'Fit: A*exp(-kt) - c\nA={popt[0]:.2f}, k={popt[1]:.2f}, c={popt[2]:.2f}')
    
    # ax.axhline(HIGHSCORE, color='b', label='High Score', alpha=0.5)
    # ax.axhline(NEG_BEST, color='r', label='Negative Best', alpha=0.5)

    # Select every fourth generation for ticks
    every_fourth_generation = generations[::4]
    ax.set_xticks(every_fourth_generation)
    ax.set_xticklabels(every_fourth_generation)

    ax.set_title(f'{score_type.replace("_", " ").title()} vs Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel(f'{score_type.replace("_", " ").title()}')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

    
def plot_mutations_v_generation(ax, all_scores_df,  mut_min, mut_max):
    
    all_scores_df = all_scores_df.dropna(subset=['mutations'])
    
    max_gen = int(all_scores_df['generation'].max())
    boxplot_data = [all_scores_df[all_scores_df['generation'] == gen]['mutations'] for gen in range(0,max_gen+1,1)]
    # Define properties for the outliers
    flierprops = dict(marker='o', markerfacecolor='red', markersize=3, linestyle='none')
    
    ax.boxplot(boxplot_data, positions=range(len(boxplot_data)), flierprops=flierprops)
    ax.axhline(len(DESIGN.split(",")), color='r')
    ax.set_xticks(range(len(boxplot_data)))
    ax.set_xticklabels(range(0,len(boxplot_data),1))
    ax.set_ylim(mut_min,mut_max)
    ax.set_title('Mutations vs Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Number of Mutations')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

def plot_mutations_v_generation_violin(ax, all_scores_df, mut_min, mut_max):
    
    all_scores_df = all_scores_df.dropna(subset=['mutations'])
    
    max_gen = int(all_scores_df['generation'].max())
    generations = np.arange(0, max_gen + 1)
    violin_data = [all_scores_df[all_scores_df['generation'] == gen]['mutations'] for gen in generations]
    
    # Create violin plots
    parts = ax.violinplot(violin_data, positions=generations, showmeans=False, showmedians=True)
    
    # Customizing the color of violin plots
    for pc in parts['bodies']:
        pc.set_facecolor('green')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customizing the color of the median lines
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts.get(partname)
        if vp:
            vp.set_edgecolor('tomato')
            vp.set_linewidth(0.5)
    
    vp = parts.get('cmedians')
    if vp:
        vp.set_edgecolor('tomato')
        vp.set_linewidth(2.0)

    # Fit the data to the exponential function
    # mean_mutations = [np.mean(data) for data in violin_data]
    # weights = np.ones(len(generations))  # Uniform weights, adjust as needed
    # popt, pcov = curve_fit(exponential_func, generations, mean_mutations, p0=(1, 0.1, 0.7), sigma=weights, maxfev=2000)
    
    # # Plot the fitted curve
    # fitted_curve = exponential_func(generations, *popt)
    # ax.plot(generations, fitted_curve, 'r--', label=f'Fit: A*exp(-kt) + c\nA={popt[0]:.2f}, k={popt[1]:.2f}, c={popt[2]:.2f}')
    
    
    ax.axhline(len(DESIGN.split(",")), color='r', label='Design Length')

    # Select every second generation for ticks
    every_second_generation = generations[::2]
    ax.set_xticks(every_second_generation)
    ax.set_xticklabels(every_second_generation)
    
    ax.set_ylim(mut_min, mut_max)
    ax.set_title('Mutations vs Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Number of Mutations')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    ax.legend(loc='lower right')

def plot_delta_scores():
    all_scores_df = pd.read_csv(ALL_SCORES_CSV)
    all_scores_df = all_scores_df.dropna(subset=['total_score'])
    
    # Calculate combined scores using normalized scores
    _, _, _, _, combined_scores = normalize_scores(all_scores_df, print_norm=True, norm_all=True)
    
    # Add combined scores to the DataFrame
    all_scores_df['combined_score'] = combined_scores

    # Calculate delta scores
    all_scores_df['delta_combined'] = all_scores_df.apply(lambda row: row['combined_score'] - all_scores_df.loc[all_scores_df['index'] == int(float(row['parent_index'])), 'combined_score'].values[0] if row['parent_index'] != "Parent" else 0, axis=1)
    all_scores_df['delta_total'] = all_scores_df.apply(lambda row: row['total_score'] - all_scores_df.loc[all_scores_df['index'] == int(float(row['parent_index'])), 'total_score'].values[0] if row['parent_index'] != "Parent" else 0, axis=1)
    all_scores_df['delta_interface'] = all_scores_df.apply(lambda row: row['interface_score'] - all_scores_df.loc[all_scores_df['index'] == int(float(row['parent_index'])), 'interface_score'].values[0] if row['parent_index'] != "Parent" else 0, axis=1)
    all_scores_df['delta_efield'] = all_scores_df.apply(lambda row: row['efield_score'] - all_scores_df.loc[all_scores_df['index'] == int(float(row['parent_index'])), 'efield_score'].values[0] if row['parent_index'] != "Parent" else 0, axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    def plot_violin(ax, delta_scores, title, all_scores_df):
        all_scores_df['tmp'] = delta_scores
        all_scores_df = all_scores_df.dropna(subset=['tmp'])

        max_gen = int(all_scores_df['generation'].max())
        generations = np.arange(0, max_gen + 1)
        violin_data = [all_scores_df[all_scores_df['generation'] == gen]['tmp'] for gen in generations]

        # Create violin plots
        parts = ax.violinplot(violin_data, positions=generations, showmeans=False, showmedians=True)

        # Customizing the color of violin plots
        for pc in parts['bodies']:
            pc.set_facecolor('green')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # Customizing the color of the median lines
        for partname in ('cbars', 'cmins', 'cmaxes'):
            vp = parts.get(partname)
            if vp:
                vp.set_edgecolor('tomato')
                vp.set_linewidth(0.5)

        vp = parts.get('cmedians')
        if vp:
            vp.set_edgecolor('tomato')
            vp.set_linewidth(2.0)

        ax.set_title(title)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Delta Score')
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

    plot_violin(axs[0, 0], all_scores_df['delta_combined'], 'Delta Combined Score vs Generations', all_scores_df)
    plot_violin(axs[0, 1], all_scores_df['delta_total'], 'Delta Total Score vs Generations', all_scores_df)
    plot_violin(axs[1, 0], all_scores_df['delta_interface'], 'Delta Interface Score vs Generations', all_scores_df)
    plot_violin(axs[1, 1], all_scores_df['delta_efield'], 'Delta Efield Score vs Generations', all_scores_df)

    plt.tight_layout()
    plt.show()


def plot_tree_lin(leaf_nodes=None):
    all_scores_df = pd.read_csv(ALL_SCORES_CSV)
    _, _, _, _, combined_potentials = normalize_scores(all_scores_df, print_norm=False, norm_all=False, extension="potential")

    max_gen = int(all_scores_df['generation'].max())

    G = nx.DiGraph()

    for idx, row in all_scores_df.iterrows():
        G.add_node(idx, sequence=row['sequence'], interface_potential=row['interface_potential'], gen=int(row['generation']))
        if row['parent_index'] != "Parent":
            parent_idx = int(float(row['parent_index']))
            parent_sequence = all_scores_df.loc[all_scores_df.index == parent_idx, 'sequence'].values[0]
            current_sequence = row['sequence']
            # Calculate Hamming distance
            distance = hamming_distance(parent_sequence, current_sequence)
            # Add edge with Hamming distance as an attribute
            G.add_edge(parent_idx, idx, hamming_distance=distance)

    if leaf_nodes is not None:
        subgraph_nodes = set()
        for leaf in leaf_nodes:
            subgraph_nodes.update(nx.ancestors(G, leaf))
            subgraph_nodes.add(leaf)
        G = G.subgraph(subgraph_nodes)

    G_undirected = G.to_undirected()

    # Find connected components
    connected_components = list(nx.connected_components(G_undirected))

    largest_component = max(connected_components, key=len)
    # Create a subgraph of G using only the nodes in the largest component
    G_largest = G.subgraph(largest_component)

    def set_node_positions(G, node, pos, x, y, counts):
        pos[node] = (x, y)
        neighbors = list(G.successors(node))
        next_y = y - counts[node] / 2
        for neighbor in neighbors:
            set_node_positions(G, neighbor, pos, x + 1, next_y + counts[neighbor] / 2, counts)
            next_y += counts[neighbor]

    def count_descendants(G, node, counts):
        neighbors = list(G.successors(node))
        count = 1
        for neighbor in neighbors:
            count += count_descendants(G, neighbor, counts)
        counts[node] = count
        return count

    counts = {}
    root_node = list(largest_component)[0]
    count_descendants(G_largest, root_node, counts)

    pos = {}
    set_node_positions(G_largest, root_node, pos, 0, 0, counts)
    y_values = [y for x, y in pos.values()]
    y_span = max(y_values) - min(y_values)
    print(y_span)

    colors = combined_potentials
    colors[0] = np.nan
    normed_colors = [(x - np.nanmin(colors[1:])) / (np.nanmax(colors[1:]) - np.nanmin(colors[1:])) for x in colors]
    normed_colors = np.nan_to_num(normed_colors, nan=0)
    normed_colors = normed_colors**2

    # Convert positions to polar coordinates
    polar_pos = {node: ((x / (max(pos.values(), key=lambda p: p[0])[0] - min(pos.values(), key=lambda p: p[0])[0])) * 2 * np.pi, y) for node, (x, y) in pos.items()}

    # Convert polar coordinates to Cartesian coordinates for plotting
    cartesian_pos = {node: (radius * np.cos(angle), radius * np.sin(angle)) for node, (radius, angle) in polar_pos.items()}

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # Draw the graph with the positions set
    for start, end in G_largest.edges():
        color = plt.cm.coolwarm_r(normed_colors[end])
        if float(normed_colors[end]) == 0.0:
            color = [0., 0., 0., 1.]
        linewidth = 0.1 + 2 * normed_colors[end] * 0.01

        x0, y0 = cartesian_pos[start]
        x1, y1 = cartesian_pos[end]
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth)

    # Adjust axis labels and ticks for the swapped axes
    ax.axis('on')
    ax.set_title("Colored by Potential")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('equal')
    ax.grid(False)
    plt.show()

def plot_tree_nx_all():
    PARENT = '/net/bs-gridfs/export/grid/scratch/lmerlicek/design/Input/1ohp.pdb'
    from networkx.drawing.nx_agraph import graphviz_layout

    all_scores_df = pd.read_csv(ALL_SCORES_CSV)
    _, _, interface_potentials, _, combined_potentials = normalize_scores(all_scores_df, print_norm=False, norm_all=False, extension="potential")
    all_scores_df["interface_potential"] = interface_potentials
    G = nx.DiGraph()

    for _, row in all_scores_df.iterrows():
        index = int(float(row['index'])) + 1
        if not isinstance(row['sequence'], str):
            continue
        G.add_node(index, sequence=row['sequence'], interface_potential=row['interface_potential'], gen=int(row['generation']) + 1)
        if row['parent_index'] != "Parent":
            parent_idx = int(float(row['parent_index'])) + 1
            parent_sequence = all_scores_df.loc[all_scores_df.index == parent_idx - 1, 'sequence'].values[0]
            current_sequence = row['sequence']
            # Calculate Hamming distance
            distance = hamming_distance(parent_sequence, current_sequence)
            # Add edge with Hamming distance as an attribute
            G.add_edge(parent_idx, index, hamming_distance=distance)

    G_undirected = G.to_undirected()
    
    # Create a new root node
    G.add_node(0, sequence='root', interface_potential=0, gen=0)
    
    # Connect the new root node to all nodes of generation 1
    for node in G.nodes:
        if G.nodes[node]['gen'] == 1:
            parent_sequence = extract_sequence_from_pdb(PARENT)
            current_sequence = G.nodes[node]['sequence']
            distance = hamming_distance(parent_sequence, current_sequence)
            G.add_edge(0, node, hamming_distance=distance)

    # Use graphviz_layout to get the positions for a circular layout
    pos = graphviz_layout(G, prog="twopi", args="")

    # Normalize scores from 0 to 1
    scores = {node: all_scores_df.loc[all_scores_df['index'] == int(node)-1, 'interface_score'].values[0] for node in G.nodes if node != 0}
    min_score = min(scores.values())
    max_score = max(scores.values())
    normalized_scores = {node: (score - min_score) / (max_score - min_score) for node, score in scores.items()}

    # Get node colors based on index
    node_colors = [plt.cm.viridis(int(node) / len(G.nodes)) for node in G.nodes]

    # Mark generation 0 nodes with red
    gen_0_nodes = [node for node in G.nodes if G.nodes[node]['gen'] == 0]
    for node in gen_0_nodes:
        node_colors[list(G.nodes).index(node)] = 'red'

    # Normalize Hamming distances for edge colors
    hamming_distances = [G.edges[edge]['hamming_distance'] for edge in G.edges]
    # Normalize the Hamming distances
    min_hamming = min(hamming_distances)
    max_hamming = max(hamming_distances)
    normalized_hamming = [(dist - min_hamming) / (max_hamming - min_hamming) for dist in hamming_distances]

    # Plot the graph
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=5, node_color=node_colors, linewidths=0.01)
    
    # Draw edges with custom color based on normalized Hamming distance
    edge_colors = ['blue' if norm_dist == 0 else plt.cm.RdYlGn(norm_dist) for norm_dist in normalized_hamming]
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.2, edge_color=edge_colors, style='-', arrows=False)

    # Create a colorbar as a legend for Hamming distances
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=min_hamming, vmax=max_hamming))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Hamming Distance')

    ax.set_title("Colored by Index, Gen 0 in Red, Edges by Hamming Distance")
    ax.axis("equal")
    plt.show()

def calculate_rank_order(matrix):
    # Calculate the occurrence frequency of each amino acid in each column
    unique, counts = np.unique(matrix, return_counts=True)
    frequencies = dict(zip(unique, counts))
    
    # Sort amino acids in each column by their frequency, then alphabetically
    sorted_amino_acids = sorted(frequencies.items(), key=lambda x: (x[1], -ord(x[0])), reverse=True)
    
    # Assign rank order based on sorted position
    rank_order = {amino_acid: rank for rank, (amino_acid, _) in enumerate(sorted_amino_acids, start=1)}
    
    # Replace amino acids with their rank order
    rank_matrix = np.vectorize(rank_order.get)(matrix)
    
    return rank_matrix

def seq_to_rank_order_matrix(sequences):
    # Convert sequences to a 2D numpy array (matrix) of characters
    matrix = np.array([list(seq) for seq in sequences])
    
    # Initialize an empty matrix to store the rank order numbers
    rank_order_matrix = np.zeros(matrix.shape, dtype=int)
    
    # Calculate rank order for each column
    for i in range(matrix.shape[1]):  # Iterate over columns
        column = matrix[:, i]
        rank_order_matrix[:, i] = calculate_rank_order(column)
    
    return rank_order_matrix

def seq_to_numeric(seq):
    # Define a mapping for all 20 standard amino acids plus 'X' for unknown
    mapping = {
        'A': 1,  'C': 2,  'D': 3,  'E': 4,
        'F': 5,  'G': 6,  'H': 7,  'I': 8,
        'K': 9,  'L': 10, 'M': 11, 'N': 12,
        'P': 13, 'Q': 14, 'R': 15, 'S': 16,
        'T': 17, 'V': 18, 'W': 19, 'Y': 20,
        'X': 0   # 'X' for any unknown or non-standard amino acid
    }
    numeric_seq = [mapping[char] for char in seq]
    return numeric_seq

def plot_pca_umap():
    
    all_scores_df = pd.read_csv(ALL_SCORES_CSV)

    all_scores_df = all_scores_df.dropna(subset=['total_score'])
    all_scores_df = all_scores_df.dropna(subset=['catalytic_score'])
    all_scores_df = all_scores_df.dropna(subset=['interface_score'])
    
    numeric_seqs = seq_to_rank_order_matrix(all_scores_df['sequence'].tolist())
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(numeric_seqs)

    pca3 = PCA(n_components=3)
    pca_result3 = pca3.fit_transform(numeric_seqs)

    # Analyze PCA loadings for PC1
    # loadings = pca.components_.T[:, 0]  # Loadings for PC1
    # plt.figure(figsize=(10, 4))
    # plt.bar(range(len(loadings)), loadings)
    # plt.title('PCA Loadings for PC1')
    # plt.xlabel('Sequence Position')
    # plt.ylabel('Loading Value')
    # plt.show()

    # Perform UMAP
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(numeric_seqs)

    # Create a figure and a 2x2 grid of subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 24))  # Adjust the figure size as needed

    # Define a base font size
    base_font_size = 10  # Adjust here

    # Plot UMAP Interface score
    axs[0].scatter(umap_result[:, 0], umap_result[:, 1], c=all_scores_df['interface_score'], cmap='viridis', alpha=0.6, s=1)
    cbar = fig.colorbar(axs[0].collections[0], ax=axs[0], label='Interface Score')
    axs[0].set_title('UMAP of Sequences - Interface score', fontsize=base_font_size * 2)
    axs[0].set_xlabel('UMAP1', fontsize=base_font_size * 2)
    axs[0].set_ylabel('UMAP2', fontsize=base_font_size * 2)
    cbar.set_label('Interface Score', size=base_font_size * 2)

    # Filter the DataFrame to include only rows where 'total_score' is <= -340
    filtered_df = all_scores_df[all_scores_df['total_score'] <= -340]
    filtered_umap_result = umap_result[all_scores_df['total_score'] <= -340]

    # Now plot using the filtered data
    axs[1].scatter(filtered_umap_result[:, 0], filtered_umap_result[:, 1], c=filtered_df['total_score'], cmap='viridis', alpha=0.6, s=1)
    cbar = fig.colorbar(axs[1].collections[0], ax=axs[1], label='Total Score')
    axs[1].set_title('UMAP of Sequences - Total score', fontsize=base_font_size * 2)
    axs[1].set_xlabel('UMAP1', fontsize=base_font_size * 2)
    axs[1].set_ylabel('UMAP2', fontsize=base_font_size * 2)
    cbar.set_label('Total Score', size=base_font_size * 2)

    # Plot UMAP with 'index' as the color
    axs[2].scatter(umap_result[:, 0], umap_result[:, 1], c=all_scores_df['index'], cmap='viridis', alpha=0.6, s=1)
    cbar = fig.colorbar(axs[2].collections[0], ax=axs[2], label='Generation')
    axs[2].set_title('UMAP of Sequences - Generation', fontsize=base_font_size * 2)
    axs[2].set_xlabel('UMAP1', fontsize=base_font_size * 2)
    axs[2].set_ylabel('UMAP2', fontsize=base_font_size * 2)
    cbar.set_label('Generation', size=base_font_size * 2)

    plt.tight_layout()
    plt.show()



def plot_esm_umap():

    #ESM embeddings and UMAP
    def prepare_data(sequences):
        """ Convert a list of protein sequences to the model's input format. """
        batch_tokens = []
        for seq in sequences:
            tokens = torch.tensor([alphabet.encode(seq)], dtype=torch.long)
            batch_tokens.append(tokens)
        return torch.cat(batch_tokens)

    # 1. Load ESM model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()

    # Load and preprocess data
    all_scores_df = pd.read_csv(ALL_SCORES_CSV)
    all_scores_df.dropna(subset=['total_score', 'catalytic_score', 'interface_score', 'sequence'], inplace=True)

    # Extract sequences
    sequences = all_scores_df['sequence'].tolist()

    with torch.no_grad():
        tokens = prepare_data(sequences)
        results = model(tokens, repr_layers=[33])  # Specify the layer you want
        token_embeddings = results["representations"][33]

        # Mean pooling over positions
        sequence_embeddings = token_embeddings.mean(dim=1)
        
    embeddings_array = sequence_embeddings.cpu().numpy()

    # Perform UMAP
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(embeddings_array)

    # Create a figure and a 2x2 grid of subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 24))  # Adjust the figure size as needed

    # Define a base font size
    base_font_size = 10  # Adjust here

    # Plot UMAP Interface score
    axs[0].scatter(umap_result[:, 0], umap_result[:, 1], c=all_scores_df['interface_score'], cmap='viridis', alpha=0.6, s=1)
    cbar = fig.colorbar(axs[0].collections[0], ax=axs[0], label='Interface Score')
    axs[0].set_title('UMAP of Sequences - Interface score', fontsize=base_font_size * 2)
    axs[0].set_xlabel('UMAP1', fontsize=base_font_size * 2)
    axs[0].set_ylabel('UMAP2', fontsize=base_font_size * 2)
    cbar.set_label('Interface Score', size=base_font_size * 2)

    # Filter the DataFrame to include only rows where 'total_score' is <= -340
    filtered_df = all_scores_df[all_scores_df['total_score'] <= -340]
    filtered_umap_result = umap_result[all_scores_df['total_score'] <= -340]

    # Now plot using the filtered data
    axs[1].scatter(filtered_umap_result[:, 0], filtered_umap_result[:, 1], c=filtered_df['total_score'], cmap='viridis', alpha=0.6, s=1)
    cbar = fig.colorbar(axs[1].collections[0], ax=axs[1], label='Total Score')
    axs[1].set_title('UMAP of Sequences - Total score', fontsize=base_font_size * 2)
    axs[1].set_xlabel('UMAP1', fontsize=base_font_size * 2)
    axs[1].set_ylabel('UMAP2', fontsize=base_font_size * 2)
    cbar.set_label('Total Score', size=base_font_size * 2)

    # Plot UMAP with 'index' as the color
    axs[2].scatter(umap_result[:, 0], umap_result[:, 1], c=all_scores_df['index'], cmap='viridis', alpha=0.6, s=1)
    cbar = fig.colorbar(axs[2].collections[0], ax=axs[2], label='Generation')
    axs[2].set_title('UMAP of Sequences - Generation', fontsize=base_font_size * 2)
    axs[2].set_xlabel('UMAP1', fontsize=base_font_size * 2)
    axs[2].set_ylabel('UMAP2', fontsize=base_font_size * 2)
    cbar.set_label('Generation', size=base_font_size * 2)

    plt.tight_layout()
    plt.show()

def find_mutations(seq1, seq2):
    # Function to compare sequences and find mutation positions
    return [i for i, (a, b) in enumerate(zip(seq1, seq2)) if a != b]

def normalize_columnwise(matrix):
    min_vals = matrix.min(axis=0)
    max_vals = matrix.max(axis=0)
    # Avoid division by zero
    denom = np.where((max_vals - min_vals) == 0, 1, (max_vals - min_vals))
    normalized_matrix = (matrix - min_vals) / denom
    return normalized_matrix

def plot_mut_location():
    # Load the data
    all_scores_df = pd.read_csv(ALL_SCORES_CSV)

    all_scores_df = all_scores_df.dropna(subset=['sequence'])

    # Assuming the maximum length of sequences is 125
    max_length = 125
    max_generation = int(all_scores_df['generation'].max())

    # Initialize a matrix to hold mutation frequencies
    mutation_matrix = np.zeros((max_length, max_generation + 1))

    # Populate the mutation matrix
    for _, row in all_scores_df.iterrows():
        if pd.notnull(row['parent_index']) and row['parent_index'] != "Parent":  # Check if there's a valid parent
            parent_seq = all_scores_df.loc[all_scores_df['index'] == float(row['parent_index']), 'sequence'].values[0]
            mutations = find_mutations(row['sequence'], parent_seq)
            for pos in mutations:
                mutation_matrix[pos, int(row['generation'])] += 1

    # Normalize the mutation_matrix column-wise (i.e., each generation separately)
    normalized_mutation_matrix = normalize_columnwise(mutation_matrix)

   # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.imshow(normalized_mutation_matrix, aspect='auto', origin='lower', cmap='viridis', extent=[0, max_generation, 0, max_length])
    ax.set_xlabel('Generation')
    ax.set_ylabel('Position along AA chain')
    ax.set_title('Frequency of Mutation Over Generations')
    fig.colorbar(c, ax=ax, label='Normalized Frequency of Mutation')
    plt.show()
