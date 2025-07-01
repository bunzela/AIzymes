import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

from helper_002               import normalize_scores, hamming_distance, save_cat_res_into_all_scores_df, save_all_scores_df, exponential_func
from plotting_tree_001        import *

# Data loding function --------------------------------------------------------------------------------------------------------------------------

def make_plots(self):
    
    load_plot_scores_df(self)

    ### TEMPORARY WORK AROUND TO FACILITATE PLOTTING
    if not hasattr(self, "WEIGHT_TOTAL"): self.WEIGHT_TOTAL = 1.0
    if not hasattr(self, "WEIGHT_CATALYTIC"): self.WEIGHT_CATALYTIC = 1.0
    if not hasattr(self, "WEIGHT_INTERFACE"): self.WEIGHT_INTERFACE = 1.0
    if not hasattr(self, "WEIGHT_IDENTICAL"): self.WEIGHT_IDENTICAL = 1.0
    if not hasattr(self, "WEIGHT_EFIELD"): self.WEIGHT_EFIELD = 1.0
    if not hasattr(self, "WEIGHT_REDOX"): self.WEIGHT_REDOX = 1.0

    if self.PRINT_VALS:
        print_vals(self)
        
    if self.STATISTICS:
        plot_statistics(self)
        
    if self.SCORES_V_INDEX:
        plot_scores_v_index(self)
        
    if self.SCORES_V_GEN:
        plot_scores_v_generation(self)
        
    if self.SCORES_HIST:
        plot_scores_hist(self)
                    
    if self.PLOT_TREE:
        plot_tree(self,color=self.TREE_SCORE)

    if self.RESOURCE_LOG:
        plot_resource_log(self)
        
    if self.landscape_plot:
        landscape_plotting_function(self)

    combine_images(self)

def load_plot_scores_df(self):

    # Condition to check if the ALL_SCORES_CSV file exists, otherwise it returns the function.
    if not os.path.isfile(f'{self.FOLDER_HOME}/all_scores.csv'): 
        print(f"ERROR: {self.FOLDER_HOME}/all_scores.csv does not exist!")
        sys.exit()  
    
    # Load and modify scores_df
    self.all_scores_df = pd.read_csv(self.ALL_SCORES_CSV)
    self.plot_scores_df = self.all_scores_df.dropna(subset=[f'{self.SELECTED_SCORES[0]}_score'])
    
    # Check if there is enough data
    if len(self.plot_scores_df)<10:
        print(f"ERROR: {self.FOLDER_HOME}/all_scores.csv does not contain enough finished designs!")
        sys.exit()  
        
    # Calculate scores
    self.scores = normalize_scores(self, 
                                   self.plot_scores_df, 
                                   norm_all=True, 
                                   extension="score") 
    self.scores["final_score"] = (self.NORM["final_score"][0]-self.scores["final_score"])/\
                                 (self.NORM["final_score"][0]-self.NORM["final_score"][1])

    # Use combined_score as during Boltzmann selection
    scores_tmp = normalize_scores(self, 
                                   self.plot_scores_df, 
                                   norm_all=False, 
                                   extension="score") 
    self.scores["combined_score"] = scores_tmp["combined_score"]

    if "redox" in self.SELECTED_SCORES:
        self.scores['BioDC_redox'] = self.plot_scores_df["BioDC_redox"]
        self.HIGHSCORE['BioDC_redox'] = self.TARGET_REDOX

# Combined plot functions --------------------------------------------------------------------------------------------------------------------------

def plot_statistics(self):

    fig, axes = plt.subplots(1, 4, figsize=((len(self.SELECTED_SCORES) + 1) * self.PLOT_SIZE, self.PLOT_SIZE))
    
    # --- Chart 1: Completed vs. Not Finished ---
    finished_count = self.all_scores_df[f'{self.SELECTED_SCORES[0]}_score'].notna().sum()
    crashed_count = 0
    non_finished_count = 0

    na_rows = self.all_scores_df[self.all_scores_df[f'{self.SELECTED_SCORES[0]}_score'].isna()]
    for index, row in na_rows.iterrows():
        error_files = glob.glob(os.path.join(self.FOLDER_HOME, str(index), "scripts", "*.err"))
        has_error = False
        for err_file in error_files:
            with open(err_file, "r", errors="ignore") as f:
                if any("error" in line.lower() for line in f):
                    has_error = True
                    break 
        if not error_files or not has_error:
            non_finished_count += 1
        else:
            crashed_count += 1
            
    data = [finished_count, non_finished_count, crashed_count]
    labels = [f'Finished ({finished_count})', f'Unfinished ({non_finished_count})', f'Failed ({crashed_count})' ]
    colors = [(*color, 0.5) for color in reversed(plt.cm.Set1.colors[:3])]
    wedges, _ = axes[0].pie(data, autopct=None, colors=colors, startangle=90)
    axes[0].legend(wedges, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
    axes[0].set_title("Completed")
    
    # --- Chart 2: Design Methods ---
    if 'design_method' in self.all_scores_df.columns:
        design_method_counts = self.all_scores_df['design_method'].value_counts()
        data = design_method_counts.values
        labels = [f'{label} ({count})' for label, count in zip(design_method_counts.index, design_method_counts.values)]
        colors = [(*color, 0.5) for color in plt.cm.Set1.colors[:len(design_method_counts)]]
        wedges, _ = axes[1].pie(data, autopct=None, colors=colors, startangle=90)
        axes[1].legend(wedges, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
        axes[1].set_title("Methods")
    else:
        axes[1].set_visible(False)
    
    # --- Chart 3: Blocked (Running) ---
    if 'blocked' in self.all_scores_df.columns:
        blocked_counts = self.all_scores_df['blocked'].value_counts()
        # Exclude "unblocked"
        blocked_counts = blocked_counts[blocked_counts.index != "unblocked"]
        if not blocked_counts.empty:
            data = blocked_counts.values
            labels = [f'{label} ({count})' for label, count in zip(blocked_counts.index, blocked_counts.values)]
            colors = [(*color, 0.5) for color in plt.cm.Set1.colors[:len(blocked_counts)]]
            wedges, _ = axes[2].pie(data, autopct=None, colors=colors, startangle=90)
            axes[2].legend(wedges, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
            axes[2].set_title("Running")
        else:
            axes[2].set_title("Nothing Running")
    else:
        axes[2].set_visible(False)
            
    # --- Chart 4: Next Steps ---
    if 'next_steps' in self.all_scores_df.columns:
        self.all_scores_df['next_step'] = self.all_scores_df['next_steps'].dropna().apply(lambda x: x.split(',')[0])
        next_step_counts = self.all_scores_df['next_step'].value_counts()
        data = next_step_counts.values
        labels = [f'{label} ({count})' for label, count in zip(next_step_counts.index, next_step_counts.values)]
        colors = [(*color, 0.5) for color in plt.cm.Set1.colors[:len(next_step_counts)]]
        wedges, _ = axes[3].pie(data, autopct=None, colors=colors, startangle=90)
        axes[3].legend(wedges, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
        axes[3].set_title("Next Step")
    else:
        axes[3].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(self.FOLDER_PLOT, "plot_statistics.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_scores_hist(self):
    
    fig, axes = plt.subplots(1, len(self.SELECTED_SCORES) + 1,
                             figsize=((len(self.SELECTED_SCORES) + 1) * self.PLOT_SIZE, self.PLOT_SIZE))
    
    for idx, ax in enumerate(axes):
        
        if idx == 0:
            score_type = "combined"
        else:
            score_type = self.SELECTED_SCORES[idx - 1]

        plot_score_hist(self, ax, f'{score_type}_score')
        
    plt.tight_layout()
    save_path = os.path.join(self.FOLDER_PLOT, "plot_scores_hist.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_scores_v_index(self):
    
    fig, axes = plt.subplots(1, len(self.SELECTED_SCORES) + 1,
                             figsize=((len(self.SELECTED_SCORES) + 1) * self.PLOT_SIZE, self.PLOT_SIZE))
    
    for idx, ax in enumerate(axes):
        
        if idx == 0:
            score_type = "final"
        else:
            score_type = self.SELECTED_SCORES[idx - 1]
            
        if score_type in ["redox"]:
            score_plotted = "BioDC_redox"
        else:
            score_plotted = f'{score_type}_score'
            
        plot_score_v_index(self, ax, score_plotted)
        
    plt.tight_layout()
    save_path = os.path.join(self.FOLDER_PLOT, "plot_scores_v_index.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_scores_v_generation(self):
    
    fig, axes = plt.subplots(1, len(self.SELECTED_SCORES) + 1,
                             figsize=((len(self.SELECTED_SCORES) + 1) * self.PLOT_SIZE, self.PLOT_SIZE))
    
    for idx, ax in enumerate(axes):
        if idx == 0:
            score_type = "final"
        else:
            score_type = self.SELECTED_SCORES[idx - 1]
            
        if score_type in ["redox"]:
            score_plotted = "BioDC_redox"
        else:
            score_plotted = f'{score_type}_score'
            
        plot_score_v_generation_violin(self, ax, score_plotted)
        
    plt.tight_layout()
    save_path = os.path.join(self.FOLDER_PLOT, "plot_scores_v_generation.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_mutations_v_generation(self):
    
    fig, axes = plt.subplots(1, len(self.SELECTED_SCORES) + 1,
                             figsize=((len(self.SELECTED_SCORES) + 1) * self.PLOT_SIZE, self.PLOT_SIZE))

    plot_mutations_v_generation_violin(self, axes[0])
    plot_hamming_distance_v_generation_violin(self, axes[1])
    for idx in range(2,len(self.SELECTED_SCORES) + 1, 1):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(self.FOLDER_PLOT, "plot_mutations_v_generation.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
def print_vals(self):
    rows = []
    for score_type in [score for score in self.scores if "score" in score]:

        row = {
            "score_type": score_type,
            "min": min(self.scores[score_type]),
            "max": max(self.scores[score_type]),
            "highscore": self.HIGHSCORE.get(f'{score_type}', None),
            "negbest": self.NEGBEST.get(f'{score_type}', None)
        }
        if "combined" not in score_type:
            row["norm_min"] = self.NORM[score_type][0]
            row["norm_max"] = self.NORM[score_type][1]
        else:
            row["norm_min"] = None
            row["norm_max"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    df["min"] = df["min"].map(lambda x: f"{x:.2f}")
    df["max"] = df["max"].map(lambda x: f"{x:.2f}")
    df = df[["score_type","min", "max", "norm_min", "norm_max", "highscore", "negbest"]]
    
    # Render the DataFrame as a table using matplotlib
    fig, ax = plt.subplots(figsize=((len(self.SELECTED_SCORES) + 1) * self.PLOT_SIZE, len(rows) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.tight_layout()
    save_path = os.path.join(self.FOLDER_PLOT, "print_vals.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def combine_images(self):
    # List the file names you saved from the above functions
    file_names = ["print_vals.png" ]

    if self.STATISTICS:     file_names += [ "plot_statistics.png"]
    if self.SCORES_HIST:    file_names += [ "plot_scores_hist.png"]
    if self.SCORES_V_INDEX: file_names += [ "plot_scores_v_index.png"]
    if self.SCORES_V_GEN:   file_names += [ "plot_scores_v_generation.png"]

    file_paths = [os.path.join(self.FOLDER_PLOT, name) for name in file_names]

    # Open images using PIL
    images = [Image.open(fp) for fp in file_paths if os.path.exists(fp)]
    if not images:
        print("No images found to combine.")
        return

    # Compute the total size (vertical stacking)
    widths, heights = zip(*(im.size for im in images))
    total_width = max(widths)
    total_height = sum(heights)

    # Create a new blank image with white background
    combined_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # Paste each image into the combined image
    y_offset = 0
    for im in images:
        combined_image.paste(im, (0, y_offset))
        y_offset += im.size[1]
    
    # Save the combined image
    combined_path = os.path.join(self.FOLDER_PLOT, "results.png")
    combined_image.save(combined_path)
    display(combined_image)

def plot_resource_log(self):

    # Load perfomrance data
    self.resource_log_df = pd.read_csv(self.RESOURCE_LOG_CSV)

    # Compute the relative time (in seconds) based on the first row
    time = self.resource_log_df['time'] - self.resource_log_df['time'].iloc[0]

    # Create 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(3 * self.PLOT_SIZE, self.PLOT_SIZE))
    
    # --- Plot 1: CPU and GPU usage as percentages ---
    # CPU usage percentage (divide by self.MAX_JOBS)
    cpu_usage_pct = self.resource_log_df['cpus_used'] / self.MAX_JOBS * 100

    # GPU usage percentage (divide by self.MAX_GPUS, if any GPUs available)
    if self.MAX_GPUS > 0:
        gpu_usage_pct = self.resource_log_df['gpus_used'] / self.MAX_GPUS * 100
    else:
        gpu_usage_pct = [0] * len(self.resource_log_df)

    axes[0].plot(time, cpu_usage_pct, label='CPU', marker='o')
    axes[0].plot(time, gpu_usage_pct, label='GPU', marker='o')
    axes[0].set_ylabel("Usage (%)")
    axes[0].set_title("CPU and GPU Usage")
    axes[0].set_ylim(0,100)
    axes[0].set_xlim(left=0)
    axes[0].legend()

    # --- Plot 2: Design count ---
    total_designs = self.resource_log_df['total_designs']
    unfinished_designs = self.resource_log_df['unfinished_designs']
    finished_designs = self.resource_log_df['finished_designs']
    failed_designs = self.resource_log_df['failed_designs']
    axes[1].plot(time, total_designs, label='Total', marker='o')
    axes[1].plot(time, finished_designs, label='Finished', marker='o')
    axes[1].plot(time, unfinished_designs, label='Unfinished', marker='o')
    axes[1].plot(time, failed_designs, label='Failed', marker='o')
    axes[1].set_ylabel("Number of Designs")
    axes[1].set_xlabel("Relative Time (seconds)")
    axes[1].set_title("Total Designs")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlim(left=0)
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(self.FOLDER_PLOT, "resource_log_performance.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
# Sub-plot functions --------------------------------------------------------------------------------------------------------------------------

def plot_score_v_index(self, ax, score_type):

    # Prepares data for the plot
    normalized_scores = self.scores[score_type]
    
    # Plot score
    ax.scatter(self.plot_scores_df.index, normalized_scores, c='lightgrey', s=5)     
    moving_avg = pd.Series(normalized_scores).rolling(window=int(len(normalized_scores)/50)).max()    
    ax.plot(self.plot_scores_df.index,moving_avg,c="k")
   
    if score_type == "BioDC_redox":
        ylim = (-1,1)
    else:
        ylim = (0,1)

    # Adds the two values HIGHSCORE and NEG_BEST as reference values for the top ever design and the best ever negative control value.
    if score_type in self.HIGHSCORE: ax.axhline(self.HIGHSCORE[score_type], color='b', label='Highest Score', alpha=0.5)
    if score_type in self.NEGBEST:   ax.axhline(self.NEGBEST[score_type], color='r', label='Negative Best', alpha=0.5)

    # Sets the plot details
    ax.set_title(score_type.replace("_", " "))
    ax.set_xlabel('index')
    ax.set_ylim(ylim)
    ax.set_xlim(left=0)
    if score_type not in ["BioDC_redox"]:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
 
def plot_score_hist(self, ax, score_type):

    # Prepares data for the plot
    normalized_scores = self.scores[score_type]

    if score_type != "combined_score":
        xmin = -0.1
        xmax = 1.1
    else:
        xmin = np.percentile(normalized_scores, 10)-0.5
        xmax = np.max(normalized_scores)+0.1



def plot_score_hist(self, ax, score_type):
    
    normalized_scores = np.array(self.scores[score_type])
    normalized_scores = normalized_scores[np.isfinite(normalized_scores)]
    cut_min, cut_max = -2, 2
    normalized_scores = normalized_scores[(normalized_scores >= cut_min) & (normalized_scores <= cut_max)]

    if score_type != "combined_score":
        xmin, xmax = -0.1, 1.1
    else:
        xmin = max(np.percentile(normalized_scores, 5) - 0.5, cut_min)
        xmax = min(np.max(normalized_scores) + 0.1, cut_max)
        
    # Plot score
    ax.hist(normalized_scores, bins=np.arange(xmin,xmax,0.02),color='grey')

    if score_type == "combined_score":

        # Calculate current kbt value        
        generation=self.plot_scores_df['generation'].max()
        if isinstance(self.KBT_BOLTZMANN, (float, int)):
            kbt_boltzmann = self.KBT_BOLTZMANN
        elif len(self.KBT_BOLTZMANN) == 2:
            kbt_boltzmann = self.KBT_BOLTZMANN[0] * np.exp(-self.KBT_BOLTZMANN[1]*generation)
        elif len(self.KBT_BOLTZMANN) == 3:
            kbt_boltzmann = (self.KBT_BOLTZMANN[0]-self.KBT_BOLTZMANN[2])*np.exp(-self.KBT_BOLTZMANN[1]*generation)+self.KBT_BOLTZMANN[2]
        
        boltzmann_factors = np.exp(normalized_scores / kbt_boltzmann)
        probabilities = boltzmann_factors / sum(boltzmann_factors)
        boltzmann_scores = np.random.choice(normalized_scores, size=100000, replace=True, p=probabilities)
        ax_dup = ax.twinx()
        ax_dup.hist(boltzmann_scores, bins=np.arange(xmin,xmax,0.02), density=True, alpha=0.7, color='orange', label=f'kbt = {kbt_boltzmann:.2f}')
        ax_dup.tick_params(axis='y', which='both', right=False, labelright=False)
        ax_dup.legend()  

    # Adds the two values HIGHSCORE and NEG_BEST as reference values for the top ever design and the best ever negative control value.
    if score_type != "combined_score":
        if score_type in self.HIGHSCORE: ax.axvline(self.HIGHSCORE[score_type], color='b', label='Highest Score', alpha=0.5)
        if score_type in self.NEGBEST:   ax.axvline(self.NEGBEST[score_type], color='r', label='Negative Best', alpha=0.5)

    # Sets the plot details
    ax.set_title(score_type.replace("_", " "))
    ax.set_xlabel(score_type)
    ax.set_ylim(bottom=0)
    if score_type != "combined_score":
        ax.set_xlim(0,1)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
def plot_score_v_generation_violin(self, ax, score_type):
    
    # Prepares data for the plot
    normalized_scores = self.scores[score_type]
    generations = self.plot_scores_df['generation']

    # Group scores by generation using an explicit for loop.
    scores_v_generations = []
    for generation in range(0, max(generations)+1, 1):
        current_gen_scores = [] 
        for score, gen in zip(normalized_scores, generations):
            if gen == generation:
                current_gen_scores.append(score)
        scores_v_generations.append(current_gen_scores)
        
    # Plot score
    parts = ax.violinplot(scores_v_generations, positions=range(0,max(generations)+1,1))

    # Customizing the color of violin plots 
    for pc in parts['bodies']:
        pc.set_facecolor('grey')
        pc.set_edgecolor('none')
        pc.set_alpha(0.5)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts.get(partname)
        vp.set_visible(False)
            
    # Add mean points
    means = [np.mean(data) for data in scores_v_generations]  
    inds = np.arange(0, len(means), 1)  
    ax.scatter(inds, means, marker='o', color='k', s=20, zorder=3)  

    if score_type ==  "BioDC_redox":
        ylim = (-1,1)
    else:
        ylim = (0,1)

    #Adds the two values HIGHSCORE and NEG_BEST as reference values for the top ever design and the best ever negative control value.
    if score_type in self.HIGHSCORE: ax.axhline(self.HIGHSCORE[score_type], color='b', label='K3', alpha=0.5)
    if score_type in self.NEGBEST:   ax.axhline(self.NEGBEST[score_type], color='r', label='WT', alpha=0.5)

    #Sets the plot details
    ax.set_title(score_type.replace("_", " "))
    ax.set_xlabel('generation')
    ax.set_ylim(ylim)
    ax.set_xlim(left=-0.3)
    ax.set_xticks(range(0, max(generations)+1))
    if score_type not in ["BioDC_redox"]:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)