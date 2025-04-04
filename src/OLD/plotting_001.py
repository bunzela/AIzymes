import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import networkx as nx
from sklearn.decomposition import PCA
from Bio import SeqIO
from networkx.drawing.nx_agraph import graphviz_layout
from transformers import EsmModel, EsmTokenizer
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import umap 
import torch
import esm
import math

from helper_002               import *

#plots = '/net/bs-gridfs/export/grid/scratch/vdegiorgi/AIzymes/design/VD/241025_VD_AlleyCat_cstweight_2.0/plots'
#os.makedirs(plots)


#Imports ALL_SCORES_CSV as the plot_scores_df dataframe
def load_plot_scores_df(self):
   
    # Condition to check if the ALL_SCORES_CSV file exists, otherwise it returns the function.
    if not os.path.isfile(f'{self.FOLDER_HOME}/all_scores.csv'): 
        print(f"ERROR: {self.FOLDER_HOME}/all_scores.csv does not exist!")
        return    
    
    self.plot_scores_df = pd.read_csv(self.ALL_SCORES_CSV)
    self.plot_scores_df = self.plot_scores_df.dropna(subset=['total_score'])
    self.plot_scores_df['sequence'] =  self.plot_scores_df['sequence'].astype(str)
    self.plot_scores_df['design_method'] =  self.plot_scores_df['design_method'].astype(str)
    self.plot_scores_df['score_taken_from'] =  self.plot_scores_df['score_taken_from'].astype(str)

#Define all the required plots for the main plot figure output
def plot_combined_score(self, ax, combined_scores, combined_score_min, combined_score_max, combined_score_bin):
    ax.hist(combined_scores, bins=np.arange(combined_score_min,combined_score_max+combined_score_bin,combined_score_bin))
    
    #Add the two values HIGHSCORE and NEG_BEST as reference values for the top ever design and the best ever negative control value.
    ax.axvline(self.HIGHSCORE_NEGBEST['HIGHSCORE_combined_score'], color='b', label='Highest Score')
    ax.axvline(self.HIGHSCORE_NEGBEST['NEGBEST_combined_score'], color='r', label='Negative Best')
    
    #Sets plot details
    ax.set_xlim(combined_score_min,combined_score_max)
    ax.set_title('Histogram of Combined Score')
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
    ax.hist(catalytic_scores, density=True, bins=np.arange(total_score_min,total_score_max+total_score_bin,total_score_bin))
    ax.set_xlim(total_score_min,total_score_max)
    ax.set_title('Histogram of Catalytic Score')
    ax.set_xlabel('Catalytic Score')
    ax.set_ylabel('Frequency')
    
def plot_efield_score(ax, efield_scores, total_score_min, total_score_max, total_score_bin):
    ax.hist(efield_scores, density=True, bins=np.arange(total_score_min,total_score_max+total_score_bin,total_score_bin))
    ax.set_xlim(total_score_min,total_score_max)
    ax.set_title('Histogram of Efield Score')
    ax.set_xlabel('Efield Score')
    ax.set_ylabel('Frequency')

def plot_identical_score(ax, identical_scores, total_score_min, total_score_max, total_score_bin):
    ax.hist(identical_scores, density=True, bins=np.arange(total_score_min,total_score_max+total_score_bin,total_score_bin))
    ax.set_xlim(total_score_min,total_score_max)
    ax.set_title('Histogram of Identical Score')
    ax.set_xlabel('Identical Score')
    ax.set_ylabel('Frequency')

def plot_boltzmann_histogram(self, ax, combined_scores, score_min, score_max, score_bin):

    # Generation of the combined potentials values
    _, _, _, _, _, combined_potentials = normalize_scores(self,
                                                       self.plot_scores_df, 
                                                       print_norm=False,
                                                       norm_all=False,
                                                       extension="score")

    scores = normalize_scores(self,
                              self.plot_scores_df,
                              print_norm=False,
                              norm_all=False,
                              extension="potential")
        
    combined_potentials = scores['combined_potential']   

    # Some issue with numpy exp when calculating boltzman factors.
    combined_potentials_list = [float(x) for x in combined_potentials]
    combined_potentials = np.array(combined_potentials_list)

    #Creates a numpy array with the kbt_weights of each index in the unblocked_all_scores dataframe
    kbt_values = self.plot_scores_df['kbt_boltzmann'].to_numpy()
    kbt_values = np.array(kbt_values, dtype=np.float64)
    
    #Calculates the weights using the Boltzmann formula which are then going to be used for the generation of the boltzmann_scores
    boltzmann_factors = np.exp(combined_potentials / kbt_values)
    print(f"Min/Max boltzmann factors: {min(boltzmann_factors)}, {max(boltzmann_factors)}")
    probabilities = boltzmann_factors / sum(boltzmann_factors)
    
    #Lists of values sampled from the combined potential either randomly (random_scores) or using weights (boltzmann_scores)
    random_scores = np.random.choice(combined_potentials, size=10000, replace=True)
    boltzmann_scores = np.random.choice(combined_potentials, size=10000, replace=True, p=probabilities)

    #Definition of the first plot using random sampling for the Boltzmann distribution
    ax.hist(random_scores, density=True, alpha=0.7, label='Random Sampling', \
            bins=np.arange(score_min-2,score_max+1+score_bin,score_bin))
    ax.text(0.05, 0.95, "normalized only to \n this dataset")
    ax.set_xlabel('Potential')
    ax.set_ylabel('Density (Normal)')
    #ax.set_title(f'kbT = {kbt_values:.1e}')
    
    #Definition of a twin axis for the second histogram using the Boltzmann sampling with the weights for the Boltzmann distribution
    ax_dup = ax.twinx()
    ax_dup.hist(boltzmann_scores, density=True, alpha=0.7, color='orange', label='Boltzmann Sampling', \
                bins=np.arange(score_min-2,score_max+1+score_bin,score_bin))
    ax.set_xlim(score_min-2,score_max+1)
    ax_dup.set_ylabel('Density (Boltzmann)')
    ax_dup.tick_params(axis='y', labelcolor='orange')
    
def plot_combined_score_v_index(self, ax, combined_scores):

    #Creates a scatter plot of the combined scores versus the index.
    combined_scores = pd.Series(combined_scores)
    ax.scatter(self.plot_scores_df['index'], combined_scores, c='lightgrey', s=5) 
    
    #Adds the two values HIGHSCORE and NEG_BEST as reference values for the top ever design and the best ever negative control value.
    ax.axhline(self.HIGHSCORE_NEGBEST['HIGHSCORE_combined_score'], color='b', label='Highest Score', alpha = 0.5)
    ax.axhline(self.HIGHSCORE_NEGBEST['NEGBEST_combined_score'], color='r', label='Negative Best', alpha = 0.5)
    

    # #Imports values from the PARENT files and creates a x-line on the plot indicating ????
    # full_path = os.path.join(self.FOLDER_HOME, self.FOLDER_PARENT)
    # PARENTS = [i for i in os.listdir(full_path) if i.endswith(".pdb") and os.path.isfile(os.path.join(full_path, i))]
    # ax.axvline(self.N_PARENT_JOBS*len(PARENTS), color='k')
    
    #Calculates the moving average in a window of 20 of the combined scores and creates a line plot with the moving average values.
    moving_avg = combined_scores.rolling(window=20).mean()
    ax.plot(range(len(moving_avg)),moving_avg,c="k")
    
    #Sets plot details
    ax.set_ylim(0,1)

    ax.set_xlim(0, self.plot_scores_df['index'].max())
    ax.set_title('Combined score vs Index')
    ax.set_xlabel('Index')
    ax.set_ylabel('Combined Score')
    ax.legend(loc='best')
    
def plot_score_v_generation_violin(self, ax, score_type, ylim=(0,1)):
    
    #Prepares data for the violin plot generation and creates it
    max_gen = int(self.plot_scores_df['generation'].max())
    generations = np.arange(0, max_gen + 1)

    scores = normalize_scores(self,
                              self.plot_scores_df,
                              print_norm=True,
                              norm_all=True)
        
    interface_scores = scores['interface_score']   
    total_scores = scores['total_score']    
    catalytic_scores = scores['catalytic_score']   
    efield_scores = scores['efield_score']   
    identical_scores = scores['identical_score']
    combined_scores = scores['combined_score']   
    
    if score_type=="total_score":
        normalized_violin_data = [total_scores[self.plot_scores_df['generation'] == gen] for gen in generations]
    elif score_type=="interface_score":
        normalized_violin_data = [interface_scores[self.plot_scores_df['generation'] == gen] for gen in generations]
    elif score_type=="catalytic_score":
        normalized_violin_data = [catalytic_scores[self.plot_scores_df['generation'] == gen] for gen in generations]
    elif score_type=="efield_score":
        normalized_violin_data = [efield_scores[self.plot_scores_df['generation'] == gen] for gen in generations]
    elif score_type=="identical_score":
        normalized_violin_data = [identical_scores[self.plot_scores_df['generation'] == gen] for gen in generations]
    elif score_type=="combined_score":
        normalized_violin_data = [combined_scores[self.plot_scores_df['generation'] == gen] for gen in generations]
    elif score_type == "final_score":
        normalized_violin_data = [ 
            (total_scores[self.plot_scores_df['generation'] == gen] + 
             interface_scores[self.plot_scores_df['generation'] == gen] + 
             catalytic_scores[self.plot_scores_df['generation'] == gen] + 
             efield_scores[self.plot_scores_df['generation'] == gen]) / 4
            for gen in generations]
    
    parts = ax.violinplot(normalized_violin_data, positions=generations)

    #Uncomment to generate the violin plot with the absolute rather than the normalized values of the scores (comment the lines above)
    '''
    violin_data = [self.plot_scores_df[self.plot_scores_df['generation'] == gen][score_type].dropna().values for gen in generations]
    parts = ax.violinplot(violin_data, positions=generations, showmeans=False, showmedians=True)
    
    '''
    # Customizing the color of violin plots 
    for pc in parts['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('none')
        pc.set_alpha(0.5)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts.get(partname)
        if vp:
            vp.set_edgecolor('black')
            vp.set_linewidth(0.1)
            
    # Add and customize mean points
    means = [np.mean(data) for data in normalized_violin_data]  
    inds = np.arange(1, len(means) + 1)  
    ax.scatter(inds, means, marker='o', color='orange', s=20, zorder=3)  

    #Adds the two values HIGHSCORE and NEG_BEST as reference values for the top ever design and the best ever negative control value.
    #ax.axhline(self.HIGHSCORE_NEGBEST[f'HIGHSCORE_{score_type}'], color='b', label='Highest Score', alpha=0.5)
    #ax.axhline(self.HIGHSCORE_NEGBEST[f'NEGBEST_{score_type}'], color='r', label='Negative Best', alpha=0.5)

    #Sets the plot details
    ax.set_title(f'{score_type.replace("_", " ").title()} vs Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel(f'{score_type.replace("_", " ").title()}')
    ax.set_ylim(ylim)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    #ax.legend(loc='best')
    every_fourth_generation = generations[::4]
    ax.set_xticks(every_fourth_generation)
    ax.set_xticklabels(every_fourth_generation)
    
def plot_mutations_v_generation_violin(self, ax, mut_min, mut_max):
    
    #Prepare data for the violin plot generation and creates it
    max_gen = int(self.plot_scores_df['generation'].max())
    generations = np.arange(0, max_gen + 1)
    violin_data = [self.plot_scores_df[self.plot_scores_df['generation'] == gen]['mutations'].dropna().values for gen in generations]
    parts = ax.violinplot(violin_data, positions=generations, showmeans=False, showmedians=True)
    ax.axhline(len(self.DESIGN.split(",")), color='r', label='Max # Mutations')
    
    # Customizing the color of violin plots and the median lines
    for pc in parts['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('none')
        pc.set_alpha(0.5)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts.get(partname)
        if vp:
            vp.set_edgecolor('black')
            vp.set_linewidth(0.1)
    vp = parts.get('cmedians')
    if vp:
        vp.set_edgecolor('tab:orange')
        vp.set_linewidth(2.0)

    # Fit the data to the exponential function and plot the fitted curve
    # mean_mutations = [np.mean(data) for data in violin_data]
    # weights = np.ones(len(generations))  # Uniform weights, adjust as needed
    # popt, pcov = curve_fit(exponential_func, generations, mean_mutations, p0=(1, 0.1, 0.7), sigma=weights, maxfev=2000)
    # fitted_curve = exponential_func(generations, *popt)
    # ax.plot(generations, fitted_curve, 'r--', label=f'Fit: A*exp(-kt) + c\nA={popt[0]:.2f}, k={popt[1]:.2f}, c={popt[2]:.2f}')
    
    #Sets plot details
    ax.set_ylim(mut_min, mut_max)
    ax.set_title('Mutations vs Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Number of Mutations')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    ax.legend(loc='lower right')
    every_second_generation = generations[::2]
    ax.set_xticks(every_second_generation)
    ax.set_xticklabels(every_second_generation)
    
def plot_hamming_distance_v_generation_violin(self, ax):
    
    #Retrieves the amino acid sequence of the ancestor protein
    with open(f"{self.FOLDER_HOME}/parent/{self.WT}.seq", "r") as f:
            self.ancestor_sequence = f.read()

    #Defines the data relative to the generations
    max_gen = int(self.plot_scores_df['generation'].max())
    generations = np.arange(0, max_gen + 1)
    
    #Creates an empty column in the plot_scores_df dataframe where the hamming distances will be uploaded
    self.plot_scores_df['Hamming Distance'] = 0  

    # Iterates through the rows and calculates the hamming distances values
    for index, row in self.plot_scores_df.iterrows():
        index = int(float(row['index'])) + 1
        if not isinstance(row['sequence'], str):
            continue
            
        #If the design belongs to the generation 0, the hamming distance is calculated using the original protein sequence
        if row['parent_index'] == "Parent":
            parent_sequence = self.ancestor_sequence
            
        #If the design belongs to the generation 1 or higher, the hamming distance is calculated using the corresponding parent sequence
        else:
            parent_idx = int(float(row['parent_index'])) + 1
            parent_sequence = self.plot_scores_df.loc[self.plot_scores_df.index == parent_idx - 1, 'sequence'].values[0]
        current_sequence = row['sequence']
        
        #Adds the calculated hamming distance values into the last column of the dataframe, called "Hamming Distance"
        self.plot_scores_df.at[index,"Hamming Distance"] = hamming_distance(parent_sequence, current_sequence)
        
    #Calculates the mean hamming distance for each generation
    mean_values = self.plot_scores_df.groupby('generation')['Hamming Distance'].mean()
    
    #Creates the data used for the generation of the violin plot and generates it
    violin_data_hm = [self.plot_scores_df[self.plot_scores_df['generation'] == gen]["Hamming Distance"].dropna().values for gen in generations]
    parts=ax.violinplot(violin_data_hm, positions=generations, showmeans=True, showmedians=False)
    
    # Customizing the color of violin plots 
    for pc in parts['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('none')
        pc.set_alpha(0.5)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts.get(partname)
        if vp:
            vp.set_edgecolor('black')
            vp.set_linewidth(0.1)
    vp = parts.get('cmeans')
    if vp:
        vp.set_edgecolor('tab:orange')
        vp.set_linewidth(2.0)
        
   #Generates the line plot connecting the mean values between violins
    ax.plot(generations, mean_values, linewidth=1.0)
    
    #Sets plot details
    ax.set_ylim(bottom=0)
    ax.set_title('Hamming Distance vs Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Hamming Distance')
    every_fourth_generation = generations[::4]
    ax.set_xticks(every_fourth_generation)
    ax.set_xticklabels(every_fourth_generation)
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
#Defines the plot_scores function which generates the 15 main plots and saves them in the plot folder
def plot_scores(self, combined_score_min=0, combined_score_max=1, combined_score_bin=0.01, 
                interface_score_min=0, interface_score_max=1, interface_score_bin=0.01,
                total_score_min=0, total_score_max=1, total_score_bin=0.01,
                catalytic_score_min=0, catalytic_score_max=1, catalytic_score_bin=0.01, 
                efield_score_min=0, efield_score_max=1, efield_score_bin=0.01,
                identical_score_min=0, identical_score_max=1, identical_score_bin=0.01,
                print_vals=True,
                ):
      
        mut_min=0
        mut_max=len(self.DESIGN.split(","))+1 
        
        #Retrieves the scores data from the ALL_SCORES_CSV file
        load_plot_scores_df(self)        
        
        #Checkpoint to see whether enough designs are completed, at least 3 are needed to generate the plots.
        if len(self.plot_scores_df) < 3: 
            print(f"ERROR: plot_scores_df does not contain enough data")
            print(self.plot_scores_df)
            return

        #to integrate??
        if (self.ProteinMPNN_PROB > 0 or self.LMPNN_PROB > 0):
            #first not nan sequence from all_scores_df
            mut_max = len(self.plot_scores_df[self.plot_scores_df['sequence'] != 'nan']['sequence'].iloc[0])    

        # Defines the scores variables by running the normalize_scores function.
        scores = normalize_scores(self,
                                    self.plot_scores_df,
                                    print_norm=True,
                                    norm_all=True)
        
        interface_scores = scores['interface_score']   
        total_scores = scores['total_score']    
        catalytic_scores = scores['catalytic_score']   
        efield_scores = scores['efield_score']   
        identical_scores = scores['identical_score']
        combined_scores = scores['combined_score']    
        
        # Print normalized values
        if print_vals:            
            print(f'interface_score: {round(min(interface_scores),3)} , {round(max(interface_scores),3)}')
            print(f'total_score: {round(min(total_scores),3)} , {round(max(total_scores),3)}')
            print(f'efield_score: {round(min(efield_scores),3)} , {round(max(efield_scores),3)}')
            print(f'combined_score: {round(min(combined_scores),3)} , {round(max(combined_scores),3)}')

        # Creates a figure with the 15 subplots and generates the different plots. Their position in the figure is defined by the axs index.
        fig, axs = plt.subplots(3, 5, figsize=(15, 11))

        plot_interface_score(axs[0,0], interface_scores, interface_score_min, interface_score_max, interface_score_bin)

        plot_total_score(axs[0,1], total_scores,  total_score_min, total_score_max, total_score_bin)

        plot_catalytic_score(axs[0,2], catalytic_scores, catalytic_score_min, catalytic_score_max, catalytic_score_bin)

        plot_efield_score(axs[0,3], efield_scores, efield_score_min, efield_score_max, efield_score_bin)

        plot_identical_score(axs[0,4], identical_scores, identical_score_min, identical_score_max, identical_score_bin)

        plot_score_v_generation_violin(self, axs[1,0], 'interface_score', ylim=(0.4,0.8))

        plot_score_v_generation_violin(self, axs[1,1], 'total_score', ylim=(0.7, 0.9))

        plot_score_v_generation_violin(self, axs[1,2], 'catalytic_score', ylim=(0.8,1))

        plot_score_v_generation_violin(self, axs[1,3], 'efield_score', ylim=(0.2,0.8))

        plot_score_v_generation_violin(self, axs[1,4], 'identical_score')

        plot_combined_score(self, axs[2,0], combined_scores, combined_score_min, combined_score_max, combined_score_bin)

        plot_score_v_generation_violin(self, axs[2,1], 'combined_score',ylim=(0.4,0.9))

        plot_combined_score_v_index(self, axs[2,2], combined_scores)

        plot_score_v_generation_violin(self, axs[2,3], 'final_score', ylim=(0.4,1))
    
        plot_hamming_distance_v_generation_violin(self, axs[2,4])



        #Additional plot types to interchange
        #plot_mutations_v_generation_violin(self, axs[2,0], plot_scores_df, mut_min, mut_max)
        #plot_interface_score_v_total_score(axs[-,-],plot_scores_df, total_score_min, total_score_max, interface_score_min, interface_score_max)
        #plot_boltzmann_histogram(self, axs[2,3], combined_scores, combined_score_min, combined_score_max, combined_score_bin)

        plt.tight_layout()

        plt.savefig(os.path.join(self.FOLDER_PLOT, 'main_plots.png'), format='png')    
        #plt.savefig(os.path.join(plots, 'main_plots.png'), format='png')

        plt.show()
        
#Defines the kbt_cst_weights_plotting_function which generates two line plots and saves them in the plot folder
def kbt_cst_weights_plotting_function(self):
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('kbt and cst weights across generations', fontsize=15, weight='bold')    

    # Extract generation,kbt and cst_weight columns
    gen_kbt_cst_df = self.plot_scores_df[['generation', 'kbt_boltzmann', 'cst_weight']].dropna()
    
    # Compute the average cst_weight per unique generation
    avg_kbt = gen_kbt_cst_df.groupby('generation')['kbt_boltzmann'].mean()
    avg_cst_weights = gen_kbt_cst_df.groupby('generation')['cst_weight'].mean()
    generations = np.arange(0, int(self.plot_scores_df['generation'].max()) + 1)  

    # kbt plot 
    kbt_plot = axs[0].plot(generations, avg_kbt)
    
    axs[0].set_title("kbt Weights vs Generations")
    axs[0].set_xlabel("Generations")
    every_third_generation = generations[::3]
    axs[0].set_xticks(every_third_generation)
    axs[0].set_xticklabels(every_third_generation)
    axs[0].set_ylabel("kbt Weights")
    

    # cst plot
    cst_plot = axs[1].plot(generations, avg_cst_weights)
    axs[1].set_title("cst Weights vs Generations")
    axs[1].set_xlabel("Generations")
    every_third_generation = generations[::3]
    axs[1].set_xticks(every_third_generation)
    axs[1].set_xticklabels(every_third_generation)
    axs[1].set_ylabel("cst weights")
    
    plt.tight_layout()
    plt.savefig(os.path.join(self.FOLDER_PLOT, 'kbt_cst_weights_plot.png'), format='png')  

    plt.show()
      
def plot_tree(self, 
              max_generation=0, 
              radius_offset=0, 
              max_angle=450, 
              color="hamming_distance",
              generation_labels=None, 
              start_angle_label=0, 
              filename=None,
              large_plot=True):
    
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    # Read the input CSV
    all_scores_df = pd.read_csv(self.ALL_SCORES_CSV)
    scores = normalize_scores(self, all_scores_df)
        
    interface_scores = scores['interface_score']   
    total_scores = scores['total_score']    
    catalytic_scores = scores['catalytic_score']   
    efield_scores = scores['efield_score']   
    identical_scores = scores['identical_score']
    combined_scores = scores['combined_score']   

    if max_generation == 0:
        max_generation = int(all_scores_df['generation'].max() + 1)  
        print('max_generation',max_generation)
        
    # Create a directed graph
    G = nx.DiGraph()
    G.add_node(0, 
               sequence="X",
               generation=0,
               hamming_distance=np.nan,
               catalytic_score=np.nan, 
               total_score=np.nan, 
               interface_score=np.nan, 
               efield_score=np.nan, 
               combined_score=np.nan) 

    # Add nodes and edges
    for idx, row in all_scores_df.iterrows():
        if pd.isna(row['parent_index']):
            continue
        if row['generation'] + 1 > max_generation and max_generation != 0:
            continue

        if row['parent_index'] != "Parent":
            parent_idx = int(float(row['parent_index'])) + 1
            parent_sequence = all_scores_df.loc[all_scores_df.index == parent_idx - 1, 'sequence'].values[0]
            current_sequence = row['sequence']
            distance = hamming_distance(parent_sequence, current_sequence)
        else:
            parent_idx = 0
            distance = row['mutations']
        
        if not isinstance(distance, (int, float)) or np.isnan(distance):
            print(f"Illegal hamming_distance value detected: {distance} for variant {idx}")
            continue

        G.add_node(idx + 1, sequence=row['sequence'], 
                   generation=row['generation'] + 1,
                   hamming_distance=distance,
                   catalytic_score=catalytic_scores[idx], 
                   total_score=total_scores[idx], 
                   interface_score=interface_scores[idx], 
                   efield_score=efield_scores[idx], 
                   combined_score=combined_scores[idx])
        
        G.add_edge(parent_idx,
                   idx + 1, 
                   hamming_distance=distance,
                   catalytic_score=catalytic_scores[idx], 
                   total_score=total_scores[idx], 
                   interface_score=interface_scores[idx], 
                   efield_score=efield_scores[idx], 
                   combined_score=combined_scores[idx])

    # Calculate subtree sizes
    def calculate_subtree_sizes(graph, node, subtree_sizes):
        size = 1  # Count the current node
        for child in graph.successors(node):
            size += calculate_subtree_sizes(graph, child, subtree_sizes)
        subtree_sizes[node] = size
        return size

    subtree_sizes = {}
    calculate_subtree_sizes(G, 0, subtree_sizes)

    # Assign polar coordinates with grouping
    pos_polar = {}

    def assign_polar_positions(node, radius=0, angle_start=0, angle_range=max_angle):
        """
        Recursively assigns polar coordinates (radius, angle) to nodes, sorting branches by size.
        - `node`: Current node to position.
        - `radius`: Radial distance from the center (root node).
        - `angle_start`: Starting angle for this node's children.
        - `angle_range`: Total angular range for this node's children.
        """
        # Compute current node angle
        node_angle = angle_start
        pos_polar[node] = (radius, math.radians(node_angle))

        # Recurse for children, sorted by subtree size
        children = list(G.successors(node))
        if children:
            children.sort(key=lambda x: subtree_sizes[x])  # Sort children by subtree size
            num_children = len(children)
            child_angle_range = angle_range / num_children  # Subdivide angle range
            for i, child in enumerate(children):
                child_angle_start = angle_start + i * child_angle_range
                assign_polar_positions(
                    child, radius + 1, child_angle_start, child_angle_range
                )

    # Assign positions starting from the root
    assign_polar_positions(0, radius=0, angle_start=0, angle_range=max_angle)

    print("pos assigned")

    # Adjust angles to be uniformly distributed
    def adjust_uniform_angles(pos_polar):
        unique_angles = sorted({math.degrees(angle) for _, angle in pos_polar.values()})
        num_angles = len(unique_angles)
        uniform_angles = [max_angle / num_angles * i for i in range(num_angles)]

        angle_mapping = {unique_angles[i]: uniform_angles[i] for i in range(num_angles)}

        for node, (radius, angle) in pos_polar.items():
            pos_polar[node] = (radius, math.radians(angle_mapping[math.degrees(angle)]))

    adjust_uniform_angles(pos_polar)

    for node, (radius, angle) in pos_polar.items():
        if radius != 0:
            pos_polar[node] = (radius + radius_offset, angle)

    def polar_to_cartesian(radius, angle):
        return radius * math.cos(angle), radius * math.sin(angle)

    # Make colors
    if color=="hamming_distance":
    
        min_color = 0
        max_color = int(max([G.edges[edge]['hamming_distance'] for edge in G.edges])) 
        custom_cmap = LinearSegmentedColormap.from_list("WhiteToBlue", [(1, 1, 1), (0, 0, 1)]) 
        bounds = np.linspace(min_color, max_color + 1, max_color + 1) 
        norm = BoundaryNorm(bounds, ncolors=len(bounds) - 1, clip=True) 

        custom_cmap = ['salmon'] + [custom_cmap(i / max_color) for i in range(1, max_color)]
        custom_cmap = ListedColormap(custom_cmap)
        bounds = np.linspace(min_color, max_color + 1, max_color + 1) 
        norm = BoundaryNorm(bounds, ncolors=len(bounds) - 1, clip=True)  
        
        color_values = [G.edges[edge]['hamming_distance'] for edge in G.edges]
    
    else:

        values = [G.edges[edge][color] for edge in G.edges]
        min_value = min(values)
        max_value = max(values)
        values = [(val - min_value) / (max_value - min_value) for val in values]

        min_color = 0
        max_color = 1

        # Define a continuous colormap (White to Blue)
        custom_cmap = LinearSegmentedColormap.from_list("WhiteToBlue", [(1, 1, 1), (0, 0, 1)]) 

        # Normalize scores to range [0, 1]
        norm = Normalize(vmin=min_color, vmax=max_color)

        # Map edge 'combined_score' values to colors
        color_values = [G.edges[edge]['combined_score'] for edge in G.edges]
                
    edge_colors = [custom_cmap(norm(dist)) for dist in color_values]

    print("plot started")

    if large_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig, ax = plt.subplots(figsize=(4, 4))




    for node in G.nodes:
        # Get all children of the current node
        children = list(G.successors(node))

        if len(children) > 1:
            # Find the child with the maximum angular distance
            parent_angle = pos_polar[node][1]
            child_angles = [(child, abs(pos_polar[child][1] - parent_angle)) for child in children]
            furthest_child = max(child_angles, key=lambda x: x[1])[0]

            # Plot the arc for the furthest child
            r0, theta0 = pos_polar[node]
            r1, theta1 = pos_polar[furthest_child]

            # Draw the arc
            arc_radius = r1 - 1.0
            arc_theta = np.linspace(theta0, theta1, 10)
            arc_x = arc_radius * np.cos(arc_theta)
            arc_y = arc_radius * np.sin(arc_theta)
            #ax.plot(arc_x, arc_y, color="grey", linewidth=1.0, zorder=4)
            ax.plot(arc_x, arc_y, color="black", linewidth=1.5, zorder=3)

    # Plot edges
    for edge_idx, edge in enumerate(G.edges):
        start, end = edge
        r0, theta0 = pos_polar[start]
        r1, theta1 = pos_polar[end]

        x_mid2, y_mid2 = polar_to_cartesian(r1 - 1.0, theta1)
        x1, y1 = polar_to_cartesian(r1, theta1)
        #ax.plot([x_mid2, x1], [y_mid2, y1], color=edge_colors[edge_idx], linewidth=1.0, zorder=3)
        ax.plot([x_mid2, x1], [y_mid2, y1], color='black', linewidth=1.5, zorder=4)
        ax.scatter(x1, y1, s=20, c="k", zorder=5)
        ax.scatter(x1, y1, s=5, c=[edge_colors[edge_idx]], zorder=6)

    ax.scatter(0, 0, s=25, c="k", zorder=10, alpha=0.5)

    for radius in range(1, max_generation + radius_offset  + 1):  # Range from 1 to max_radius inclusive
        circle = plt.Circle(
            (0, 0), 
            radius, 
            color='lightgrey', 
            linewidth=1.0, 
            fill=False,
            linestyle=(0, (1, 5)), 
            zorder=-3
        )
        ax.add_artist(circle)

        # Add generation labels if generation_labels is provided and the current radius is labeled
        if generation_labels and radius in generation_labels:
            if radius == generation_labels[-1]:
                label_text = f"generation {radius}"
            else: 
                label_text = f"{radius}"
            char_spacing = 0.3  # Spacing between characters in real space
            start_angle = start_angle_label  # Start angle for the label (top-center part of the circle)

            # Calculate the total length of the label in real space
            label_length = (len(label_text) - 1) * char_spacing

            # Adjust starting position to center the label along the top-center arc
            start_arc_length = (math.pi * radius * (start_angle / 360)) - (label_length / 2)

            for i, char in enumerate(label_text):
                # Calculate arc length and angle for each character
                arc_length = start_arc_length + i * char_spacing
                angle = 180 - (arc_length / (math.pi * radius) * 360)

                # Calculate character position
                x = radius * math.cos(math.radians(angle))
                y = radius * math.sin(math.radians(angle))

                # Plot white background box
                ax.text(
                    x, y,
                    " ",  # Empty text for background
                    ha="center",
                    va="center",
                    rotation=angle - 90,
                    rotation_mode="anchor",
                    fontsize=8,
                    zorder=-2,  # Lower z-order for the background
                    bbox=dict(boxstyle="round,pad=0.5", edgecolor="none", facecolor="white")
                )

                # Plot the actual text
                ax.text(
                    x, y,
                    char,
                    ha="center",
                    va="center",
                    rotation=angle - 90,
                    rotation_mode="anchor",
                    fontsize=8,
                    color="grey",
                    zorder=-1  # Higher z-order for the text
                )

    ax.set_xlim(-(max_generation + radius_offset + 0.5), max_generation + radius_offset + 0.5)
    ax.set_ylim(-(max_generation + radius_offset + 0.5), max_generation + radius_offset + 0.5)
    ax.axis("off")
    if filename:
        plt.savefig(f'{filename}.png', bbox_inches="tight", dpi=300)
    else:
        plt.show()

    plt.show()

    # Add colorbar for Hamming distances
    fig, ax = plt.subplots(figsize=(1, 5))
    ax.axis("off")
    if color == "hamming_distance":
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=max_color))
    else:
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)  # Attach the colorbar to the specific axes
    if color == "combined_score":
        cbar.set_label('combined score')
    if color == "hamming_distance":
        cbar.set_label('Hamming Distance')
    if filename:
        plt.savefig(f'{filename}_colorbar.png', bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.show()
    
    from types import SimpleNamespace

    # Create a simple object
    self = SimpleNamespace() 




#Defines the landscape_plotting function which generates the landscape plot and saves it in the plot folder

def landscape_plotting_function(self):
    
     #Checks for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Defines the esm2 model to be used
    esm2_model = "facebook/esm2_t6_8M_UR50D"

    #Retrieves the protein seqeunces from the all_scores_csv file and performs tokenization
    sequences_df = self.plot_scores_df['sequence']
    tokenizer = EsmTokenizer.from_pretrained(esm2_model)
    tokenized_sequence = [tokenizer(sequence, return_tensors='pt') for sequence in sequences_df]
    
    #Performs ESM embedding
    
    #Loads pre-trained esm model into the choosen device (CPU or GPU)
    model = EsmModel.from_pretrained(esm2_model).to(device)
    model.eval()

    #Performs ESM embedding and saves output into the all_hidden_states numpy array.
    
    #The all_hidden_states numpy array is an array of vectors, containing all the hidden states for each sequence token. 
    def get_last_hidden_states(model, tokenized_sequences):
        with torch.no_grad():
            all_hidden_states = []
            for seq in tokenized_sequences:
                seq = {k: v.to(device) for k, v in seq.items()}
                output = model(**seq)
                hidden_states = output.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()
                all_hidden_states.append(hidden_states)
            return np.array(all_hidden_states)
    all_hidden_states = get_last_hidden_states(model, tokenized_sequence)
    
    print("ESM Embedding done")

    # Dimensionality reduction of all_hidden_states
    umap_model = umap.UMAP()  
    umap_df = pd.DataFrame(umap_model.fit_transform(all_hidden_states), columns=['umap_x', 'umap_y'], index=sequences_df)

    # Merge the UMAP values with the plot_scores_df to generate the landscape_plot_df needed for plotting
    landscape_plot_df = pd.merge(self.plot_scores_df, umap_df, on='sequence', how='left')
    
    #Define the features to be plotted in the z-coordinate and corrects their value if necessary
    features = ['interface_score', 'total_score', 'efield_score', 'generation']
    invert   = ['interface_score', 'total_score']
    
    for feature in features:
        if feature not in invert: continue
        landscape_plot_df[feature] = -landscape_plot_df[feature]
    
    #Creates a figure and a 2x4 grid for the landscape subplot and set the title of the plot
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle(f'ESM2 embeddings UMAP colored by all_scores metrics and generation', fontsize=15, weight='bold')
    
    #For each feature, it generates the corresponding z-coordinate values and creates the scatter and contour plot along with the UMAP values (x,y)
    for idx, feature in enumerate(features):
        x = landscape_plot_df['umap_x']
        y = landscape_plot_df['umap_y']
        z = landscape_plot_df[feature]
        vmin, vmax = z.min(), z.max()
        
        #Scatter Plots Generation
        #Generates the scatter plot displaying the UMAP values on the x- and y-axis and the corresponding metric score for the z-coordinate.
        if feature=='total_score':
            scatter = axs[0, 0].scatter(x, y, c=z, cmap='bwr', s=2, vmin=vmin, vmax=vmax, zorder=10)
            axs[0, 0].set_title(feature)
            fig.colorbar(scatter, ax=axs[0, 0], orientation='vertical')
        if feature=='generation':
            scatter = axs[0, 1].scatter(x, y, c=z, cmap='bwr', s=2, vmin=vmin, vmax=vmax, zorder=10)
            axs[0, 1].set_title(feature)
            fig.colorbar(scatter, ax=axs[0, 1], orientation='vertical')
        if feature=='interface_score':
            scatter = axs[1, 0].scatter(x, y, c=z, cmap='bwr', s=2, vmin=vmin, vmax=vmax, zorder=10)
            axs[1, 0].set_title(feature)
            fig.colorbar(scatter, ax=axs[1, 0], orientation='vertical')
        if feature=='efield_score':
            scatter = axs[1, 1].scatter(x, y, c=z, cmap='bwr', s=2, vmin=vmin, vmax=vmax, zorder=10)
            axs[1,1].set_title(feature)
            fig.colorbar(scatter, ax=axs[1, 1], orientation='vertical')

        #scatter = axs[0, idx].scatter(x, y, c=z, cmap='bwr', s=2, vmin=vmin, vmax=vmax, zorder=10)
        #axs[0, idx].set_title(feature)

        #Extends the limits of the x and y axes for the landscape scatter plot by adding a 10%-margin to the minimum and maximum values. 
        x_margin = (x.max() - x.min()) * 0.1
        y_margin = (y.max() - y.min()) * 0.1
        x_min, x_max = x.min() - x_margin, x.max() + x_margin
        y_min, y_max = y.min() - y_margin, y.max() + y_margin
        
        '''
        #Contour Plots Generation
        #Creates grid for interpolation and calculates the corresponding zz values at the define grid points using linear interpolation
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        zz = griddata((x, y), z, (xx, yy), method='linear')
        zz = gaussian_filter(zz, sigma=3)
        
        #Generates the contour plot displaying the UMAP values on the x- and y-axis and the corresponding metric score for the z-coordinate.
        contour = axs[1, idx].contourf(xx, yy, zz, levels=10, cmap='bwr', zorder=1, vmin=vmin, vmax=vmax)

        #Computes the distances between each grid point and the nearest data point
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        data_points = np.column_stack([x, y])
        distances = cdist(grid_points, data_points, metric='euclidean').min(axis=1)
        
        #Filters out grid points that are more than 0.5 away from the nearest data point and sets them equal to the minimum zz value in the dataset.
        distance_mask = distances.reshape(xx.shape) > 0.5
        zz[distance_mask] = np.nan
        zz[np.isnan(zz)] = np.nanmin(zz)
        '''

        # Add a colorbar for the features in each scatter and contour plot
        #fig.colorbar(scatter, ax=axs[0, idx], orientation='vertical')
        #fig.colorbar(contour, ax=axs[1, idx], orientation='vertical')


    plt.tight_layout()
    #plt.savefig(os.path.join(plots, 'landscape_plot.png'), format='png')
    plt.savefig(os.path.join(self.FOLDER_PLOT, 'landscape_plot.png'), format='png')

    plt.show()

    
    
    
    
    
    
    
    
    

       
    
    
    
    
    
    

####################################################### ##    TO WORK ON    #######################################################

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

def plot_delta_scores():
    all_scores_df = pd.read_csv(ALL_SCORES_CSV)
    all_scores_df = all_scores_df.dropna(subset=['total_score'])
    
    # Calculate combined scores using normalized scores
    scores = normalize_scores(self,
                                self.plot_scores_df,
                                print_norm=True,
                                norm_all=True)
    
    combined_scores = scores['combined_score']   
    
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
            pc.set_edgecolor('none')
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
