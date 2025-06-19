import os
import numpy as np
from Bio.PDB import PDBParser
import ipywidgets as widgets
from IPython.display import display
from bqplot import Figure, Scatter, LinearScale, Axis
import py3Dmol
from IPython.display import clear_output

from helper_002 import normalize_scores

def make_structure_viewer(self):
    out = widgets.Output(layout=widgets.Layout(height='300px', flex='1 1 auto', overflow='hidden'))
    with out:
        pdb = self.all_scores_df.loc[self.index, "final_variant"]
        pdb_file = os.path.abspath(f"{pdb}.pdb")
        display(widgets.HTML(f"<span style='font-size:12px;color:#666;'>{os.path.basename(pdb_file)}</span>"))
        if not os.path.isfile(pdb_file):
            raise FileNotFoundError(f"{pdb_file} not found")

        # Compute principal axis
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("X", pdb_file)
        coords = np.array([a.get_coord() for a in struct.get_atoms()])
        centroid = coords.mean(axis=0)
        cov = np.cov(coords - centroid, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        principal = evecs[:, np.argmax(evals)]

        # Rotation to x-axis
        x_axis = np.array([1,0,0])
        v = np.cross(principal, x_axis)
        s = np.linalg.norm(v); c = np.dot(principal, x_axis)
        if s < 1e-6:
            angle_deg, axis_obj = 0.0, {'x':1,'y':0,'z':0}
        else:
            axis = v/s
            angle_deg = np.degrees(np.arccos(np.clip(c, -1, 1)))
            axis_obj = {'x':axis[0],'y':axis[1],'z':axis[2]}

        # 3Dmol render
        view = py3Dmol.view(width=600, height=300)
        view.addModel(open(pdb_file).read(), "pdb")
        view.setStyle({}, {"cartoon":{"color":"white"}})
        view.translate(-centroid[0], -centroid[1], -centroid[2])
        view.rotate(angle_deg, axis_obj)
        view.translate(centroid[0], centroid[1], centroid[2])

        # Highlight ligand/design
        backbone = ["C","N","O"]
        if getattr(self, 'LIGAND', None):
            view.addStyle(
                {"resn":self.LIGAND, "not":{"atom":backbone}},
                {"stick":{"radius":0.5}}
            )
        design = [r.strip() for r in getattr(self,'DESIGN',"").split(",") if r.strip()]
        if design:
            view.addStyle(
                {"resi":design, "not":{"atom":backbone}},
                {"stick":{"radius":0.3}}
            )
        rest_sel = {"and":[
            {"not":{"atom":backbone}},
            {"not":{"resi":design}},
            {"not":{"resn":getattr(self,'LIGAND',"")}}
        ]}
        view.addStyle(rest_sel, {"stick":{"radius":0.1}})
        view.zoomTo()
        view.show()
    return out

def make_score_plot(self):
    df = self.plot_scores_df
    x = np.arange(len(df))
    y = df[self.SCORE].values

    x_sc, y_sc = LinearScale(), LinearScale()

    # 1) grey base
    base = Scatter(x=x, y=y, scales={'x':x_sc,'y':y_sc},
                   default_size=32, colors=['#888888'], tooltip=widgets.Label())

    # 2) blue selected
    if self.index in df.index:
        i = df.index.get_loc(self.index)
        sel_x = [i]
        sel_y = [df.loc[self.index, self.SCORE]]
    else:
        sel_x = []
        sel_y = []

    sel = Scatter(x=sel_x, y=sel_y, scales={'x':x_sc,'y':y_sc},
                  default_size=48, colors=['#0000FF'], tooltip=widgets.Label())

    # Hover on both
    def hover_cb(_, pt):
        idx = df.index[pt['index']]
        v = df.loc[idx, self.SCORE]
        txt = f"Index: {idx}\n{self.SCORE}: {v:.3g}"
        base.tooltip.value = sel.tooltip.value = txt
    base.on_hover(hover_cb)
    sel.on_hover(hover_cb)

    # Click on both
    def click_cb(_, pt):
        idx = df.index[pt['index']]
        self.index_slider.value = idx
    base.on_element_click(click_cb)
    sel.on_element_click(click_cb)

    axes = [
        Axis(scale=x_sc, visible=False),  # hide x-axis
        Axis(label=self.SCORE, scale=y_sc, orientation='vertical')
    ]

    return Figure(marks=[base, sel],
                  axes=axes,
                  title=self.SCORE,
                  layout=widgets.Layout(width='300px', height='300px'))

def display_variants(self):
    # filter and sort
    self.plot_scores_df = self.all_scores_df[self.all_scores_df[f'{self.SELECTED_SCORES[0]}_score'].notna()]

    # remove outliers beyond ±5 standard deviations
    for score in self.SELECTED_SCORES:
        score_values = self.plot_scores_df[f'{score}_score']
        mean = score_values.mean()
        std = score_values.std()
        self.plot_scores_df = self.plot_scores_df[(score_values >= mean - 5 * std) & (score_values <= mean + 5 * std)]

    # Calculate normalized scores using your normalize_scores function.
    scores = normalize_scores(self, unblocked_all_scores_df=self.plot_scores_df, norm_all=False)
    for score in scores:
        self.plot_scores_df[score] = scores[score]
        
    # Remove bad structures below 3 standard deviations
    for score in ['final']+[i for i in self.SELECTED_SCORES if i not in ['identical','catalytic']]:
        score = f'{score}_score'
        mean = self.plot_scores_df[score].mean()
        std = self.plot_scores_df[score].std()
        self.plot_scores_df = self.plot_scores_df[(self.plot_scores_df[score] >= mean - 3 * std)]

    # Sort df
    self.plot_scores_df = self.plot_scores_df.sort_values(by=self.SCORE, ascending=True)
    opts = self.plot_scores_df.index.tolist()

    # Pre-select best scoring (first in sorted list)
    if self.index == None: 
        self.index = opts[-1]

    # controls
    if not hasattr(self, 'index_slider'):
        
        self.index_slider = widgets.SelectionSlider(
            options=opts, value=self.index,
            description="Index:", continuous_update=False,
            layout=widgets.Layout(flex='1 1 auto')
        )
        first_btn = widgets.Button(description='⏮ First')
        prev_btn = widgets.Button(description='◀ Prev')
        next_btn = widgets.Button(description='Next ▶')
        last_btn = widgets.Button(description='Last ⏭') 

        def on_first(_):
            if opts:
                self.index_slider.value = opts[0]
        
        def on_last(_):
            if opts:
                self.index_slider.value = opts[-1]
        
        def on_prev(_):
            if self.index in opts:
                i = opts.index(self.index)
            else:
                i = 0
            if i > 0:
                self.index_slider.value = opts[i - 1]
        
        def on_next(_):
            if self.index in opts:
                i = opts.index(self.index)
            else:
                i = 0
            if i < len(opts) - 1:
                self.index_slider.value = opts[i + 1]

        first_btn.on_click(on_first)
        last_btn.on_click(on_last)
        prev_btn.on_click(on_prev)
        next_btn.on_click(on_next)
        
        control = widgets.HBox(
            [self.index_slider, first_btn, prev_btn, next_btn, last_btn],
            layout=widgets.Layout(width='100%')
        )

        def slider_cb(change):
            if change['name'] == 'value' and change['new'] != self.index:
                self.index = change['new']
                self.hbox.children = (
                    make_structure_viewer(self),
                    make_score_plot(self)
                )

        self.index_slider.observe(slider_cb)
        self.controls = control
    else:
        self.index_slider.options = opts
        self.index_slider.value = self.index

    # panels
    struct = make_structure_viewer(self)
    plot = make_score_plot(self)

    if not hasattr(self, 'hbox'):
        self.hbox = widgets.HBox([struct, plot],
            layout=widgets.Layout(
                height='300px', width='100%',
                display='flex', flex_flow='row nowrap',
                align_items='flex-start'
            ))
    else:
        self.hbox.children = (struct, plot)

    # main
    clear_output(wait=True) 
    self.container = widgets.VBox([self.controls, self.hbox],
                                  layout=widgets.Layout(width='100%'))
    display(self.container)
