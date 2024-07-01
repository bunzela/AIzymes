import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def plot_pearson(observed_preds, test_ys, scores, model_name):
    
    fig, axs = plt.subplots(1, len(scores), figsize=(12, 4))
    p = ['#9C6987', '#A690CC', '#8FBFE0', '#00F6FF', '#390099', '#FF5400', '#FFBD00', '#ADA59E']

    sns.set_style("white")
    plt.style.use('tableau-colorblind10')
    plt.suptitle(f'{model_name}', fontsize = 14, weight = 'bold')
    for i in range(len(scores)):
        corr_res =  pearsonr(test_ys[i].cpu().numpy(), observed_preds[i].mean.cpu().numpy())
        axs[i].errorbar(test_ys[i].cpu().numpy(), observed_preds[i].mean.cpu().numpy(), 
                              yerr=observed_preds[i].stddev.cpu().numpy(), fmt='o', 
                              ecolor = p[i],
                              elinewidth = 0.85,
                              markersize = 3.5,
                              c = 'black'
                             )
        axs[i].title.set_text(f'{scores[i]}, Corr=%.2f, p-val.=%.3f' % (corr_res.statistic, corr_res.pvalue))
        axs[i].set_xlabel('Actual')
        axs[i].set_ylabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(f"./out/{model_name}_r2.pdf", bbox_inches='tight')
    plt.savefig(f"./out/{model_name}_r2.png", bbox_inches='tight')
    

def plot_metrics_per_iter(val_per_iter, scores, model_name):
    fig, axs = plt.subplots(3, len(scores), figsize=(12, 9), sharex=True)
    p = ['#9C6987', '#A690CC', '#8FBFE0', '#00F6FF', '#390099', '#FF5400', '#FFBD00', '#ADA59E']

    sns.set_style("white")
    plt.style.use('tableau-colorblind10')
    plt.suptitle(f'{model_name}', fontsize = 14, weight = 'bold')
    
    for i in range(len(scores)):

        axs[i][0].plot(val_per_iter[i]['train_iter'], val_per_iter[i]['loss'], 
                       c = p[i],
                       linewidth=2.5)
        axs[i][1].plot(val_per_iter[i]['train_iter'], val_per_iter[i]['noise'], '--',
                       c = p[i],
                       linewidth=2.5)
        axs[i][2].plot(val_per_iter[i]['train_iter'], val_per_iter[i]['mse'], '-.r',
                       c = p[i],
                       linewidth=2.5)

        axs[i][0].title.set_text(f'{scores[i]} - loss per train_iter')
        axs[i][1].title.set_text(f'{scores[i]} - noise per train_iter')
        axs[i][2].title.set_text(f'{scores[i]} - MSE per train_iter')

        axs[i][0].set_xlabel('train_iter')
        axs[i][0].set_ylabel('Loss')
            
        axs[i][1].set_xlabel('train_iter')
        axs[i][1].set_ylabel('Noise')
            
        axs[i][2].set_xlabel('train_iter')
        axs[i][2].set_ylabel('MSE')
    
    plt.tight_layout()
    plt.savefig(f"./out/{model_name}_metrics.pdf", bbox_inches='tight')
    plt.savefig(f"./out/{model_name}_metrics.png", bbox_inches='tight')

    
def plot_embeddings_metrics(df, projection, save_to):
    
    if projection.lower() == 'umap':
        x = 'x_tsne'
        y = 'y_tsne'
        
    elif projection.lower() == 'tsne':
        x = 'x_tsne'
        y = 'y_tsne'
        
    elif projection.lower() == 'pca':
        x = 'x_pca'
        y = 'y_pca'
        
    
    sns.set_style("white")
    plt.style.use('tableau-colorblind10')

    p = sns.color_palette()

    fig, axs = plt.subplots(2, 3, sharex = True, figsize=(12, 6))
    fig.suptitle(f'ESM2 embeddings ({projection}) colored by all_scores metrics', fontsize = 15, weight = 'bold')

    ax1=sns.scatterplot(data=df, x=x, y=y, hue = 'interface_score', color = p[0], ax = axs[0,0], edgecolor = None, s=1)
    ax1.title.set_text('interface_score')
    ax2=sns.scatterplot(data=df, x=x, y=y, hue = 'catalytic_score', color = p[1], ax = axs[0,1], edgecolor = None, s=1)
    ax2.title.set_text('catalytic_score')
    ax3=sns.scatterplot(data=df, x=x, y=y, hue = 'total_score', color = p[2], ax = axs[0,2], edgecolor = None, s=1)
    ax3.title.set_text('total_score')
    ax4=sns.scatterplot(data=df, x=x, y=y, hue = 'mutations', color = p[3], ax = axs[1,0], edgecolor = None, s=1)
    ax4.title.set_text('mutations')
    ax5=sns.scatterplot(data=df, x=x, y=y, hue = 'generation', color = p[4], ax = axs[1,1], edgecolor = None, s=1)
    ax5.title.set_text('generation')
    ax6=sns.scatterplot(data=df, x=x, y=y, hue = 'cat', color = p[5], ax = axs[1,2], edgecolor = None, s=1)
    ax6.title.set_text('cat')

    ax1.legend(title = 'interface_score: ', bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    ax2.legend(title = 'catalytic_score: ', bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    ax3.legend(title = 'total_score: ', bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    ax4.legend(title = 'mutations: ', bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    ax5.legend(title = 'generation: ', bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    ax6.legend(title = 'cat: ', bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)


    fig.tight_layout()
    plt.savefig(save_to, bbox_inches='tight')
