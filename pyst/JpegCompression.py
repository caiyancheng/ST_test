from math import log10, log2
import abc
from pyst.synthetic_test import SyntheticTest
import numpy as np
import os
import matplotlib.pyplot as plt
from pyst.utils import *
import torch 
from pycvvdp.video_source_file import video_source_image_frames
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry, vvdp_display_photometry
import json
from scipy.optimize import minimize, root_scalar, minimize_scalar
from matplotlib.lines import Line2D


class JpegCompressionTest(SyntheticTest):
    """
    The test of contrast constancy across spatial frequencies
    """

    # We are only varying the frequencies, as the ref contrasts are pre-defined!

    def __init__(self, device=None):

        super().__init__(device)

        self.ref_folder = os.path.join('jpeg_dataset', 'ppm_ref_images')
        self.test_folder = os.path.join('jpeg_dataset', 'test_images')

        self.display_photometry = vvdp_display_photometry.load('standard_fhd', [])
        self.display_geometry = vvdp_display_geometry.load('standard_fhd', [])

        self.ref_images = np.array(['i23', 'i03', 'i21', 'i15'])
        self.jpeg_levels =  torch.linspace(0, 100, 101, device=self.device)

        self.N_images = len(self.ref_images)
        self.N_levels = len(self.jpeg_levels)

        self.ref_images_indices, self.jpeg_levels_indices = torch.meshgrid(torch.arange(self.N_images), torch.arange(self.N_levels), indexing='xy')

        self.ref_images_indices = self.ref_images_indices.flatten()
        self.jpeg_levels_indices = self.jpeg_levels_indices.flatten()

        self._preview_folder = 'jpeg_compression'
        self._preview_as = 'image'

    def __len__(self):
        return self.N_levels * self.N_images

    def get_test_condition_parameters(self, index):

        ref_im = self.ref_images[self.ref_images_indices[index]]
        jpeg_level = self.jpeg_levels[self.jpeg_levels_indices[index]]

        return ref_im, str(int(jpeg_level))
    
    def get_test_condition(self, index):

        # Here we are simply generating the reference and test conditions to which we are matching, this is used for
        # preview, but not for the optimization! This is not similar to the other tests!

        ref_im, jpeg_level = self.get_test_condition_parameters(index)
        ref_path = os.path.join(self.ref_folder, ref_im+'.ppm')
        test_path = os.path.join(self.test_folder, ref_im+'_'+jpeg_level+'.jpeg')

        return video_source_image_frames(test_path, ref_path, display_photometry=self.display_photometry)


    def get_condition(self, index):
        return self.get_test_condition(index), self.display_photometry, self.display_geometry

    def get_rows_conditions(self):
        return [self.ref_images[self.ref_images_indices.cpu().numpy()], self.jpeg_levels[self.jpeg_levels_indices]]
    
    def size(self):
        return self.N_images, self.N_levels

    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as
    
    def get_ticks(self):
        x = [20, 40, 60, 80, 100]
        return x

    def get_row_header(self):
        return ['Ref Image', 'JPEG Compression Level']

    def units(self):
        return ['lin', 'lin'], [None, None]
    

    def plot(self, predictions, reverse_color_order=False, title=None, output_filename=None, axis=None, is_first_column=False, fontsize=12):

        conditions = self.get_rows_conditions()
        condition_names = self.get_row_header()

        y_condition = conditions[0].reshape(self.size()).cpu().numpy()
        y_condition_name = condition_names[0]

        x_condition = conditions[1].reshape(self.size()).cpu().numpy()
        x_condition_name = condition_names[1]

        predictions = predictions.reshape(self.size())

        _, _, gt_contrasts = self.get_gt_results()
        gt_contrasts = np.array(gt_contrasts)

        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=(8, 4))

        # Plot the results
        for i in range(len(predictions)):
            #GT
            axis.plot(x_condition[i], gt_contrasts[i], linewidth=2, linestyle='--', marker='o', color=self.colors[i])
            #Prediction
            axis.plot(x_condition[i], predictions[i], linewidth=2, marker='x', color=self.colors[i])

        axis.grid(alpha=0.4, axis='x')

        scales, units = self.units()

        if units[1]:
            axis.set_xlabel(x_condition_name+' ['+units[1]+']', fontsize=fontsize)
        else:
            axis.set_xlabel(x_condition_name, fontsize=fontsize)

        if is_first_column:
            if units[0]:
                axis.set_ylabel(y_condition_name+' ['+units[1]+']', fontsize=fontsize)
            else:
                axis.set_ylabel(y_condition_name, fontsize=fontsize)

            label = ' '.join(self.latex_name().split(' - ')[:2])
            label += '\n' + ' '.join(self.latex_name().split(' - ')[2:])

            if output_filename is None:
                axis.text(-0.28, 0.5, label, fontsize=fontsize+2, fontweight='bold', transform=axis.transAxes, ha='center', va='center', rotation=90)
        
        axis.set_xscale(scales[1])
        axis.set_yscale(scales[0])
        
        x_ticks, y_ticks = self.get_ticks()

        formatted_xticks = [f'{tick:g}' for tick in x_ticks]

        axis.set_xticks(x_ticks, formatted_xticks, fontsize=fontsize-2)

        if is_first_column:
            axis.set_yticks(y_ticks, y_ticks, fontsize=fontsize-2)
        else:
            axis.set_yticks([])

        axis.set_xlim([np.min(x_ticks), np.max(x_ticks)])
        axis.set_ylim([np.min(y_ticks), np.max(y_ticks)])

        custom_legend = [
            Line2D([0], [0], linestyle='-', marker='x', color='black', label='Metric Prediction'),
            Line2D([0], [0], linestyle='--', marker='o', color='black', label='Ground-truth')
        ]

        axis.legend(handles=custom_legend, loc=4, fontsize=fontsize-2)

        if title is not None:
            axis.set_title(title, fontsize=fontsize+2)

        if output_filename is not None:
            fig.savefig(output_filename, dpi=200, bbox_inches='tight')
        
        if axis is None:
            plt.close(fig)
    
    def plotly(self, predictions, reverse_color_order=False, title='', fig_id=None):
        # The code will return the JS script that should work with plotly

        js_command = ""        

        predictions = predictions.reshape((self.N_levels, self.N_images))
        predictions = np.mean(predictions, axis=1)
        #predictions = (predictions - np.mean(predictions)) / np.std(predictions)
        predictions = np.diff(predictions) 
        predictions = (predictions) / (np.max(predictions) - np.min(predictions))

        if reverse_color_order:
            predictions = -1 * predictions
        
        condition_names = self.get_row_header()
        units_scales, units_names = self.units()


        # Now Making the figure!

        y_condition_name = 'Derivative of prediction (Normalized)'
        if units_names[0] is not None:
            y_condition_name += " ["+units_names[0]+"]"

        x_condition_name = condition_names[1]
        if units_names[1] is not None:
            x_condition_name += " ["+units_names[1]+"]"

        x_tick_vals = self.get_ticks()

        x_condition = self.jpeg_levels.cpu().numpy()[2:]

        # Create the necessary variables now - Just lines basically

        trace = 'var trace = { '

        x = 'x: ['
        for j, val in enumerate(x_condition):
            if j>0:
                x += ', '
            x += str(val)
        x += ']'

        y = 'y: ['
        for j, val in enumerate(predictions):
            if j>0:
                y += ', '
            y += str(val)
        y += ']'

        trace += x + ', ' + y + ", mode: 'lines', line: {width:2}"
        trace += '};'

        trace += '\n'

        trace += 'var data = [trace];\n'

        js_command = trace

        # Layout of the Figure

        x = "["
        for i, el in enumerate(x_tick_vals):
            if i>0:
                x += ", "
            x += str(el)
        x += "]"

        axes_info = "xaxis: {type:'lin', range: ["+str(np.min(x_condition))+","+str(np.max(x_condition))+"], tickvals: "+x+", ticktext: "+x+", title:'"+x_condition_name+"'}, yaxis: {type:'lin', title:'"+y_condition_name+"'}"
        edge_info = "shapes: [{type:'rect', x0: 0, y0: 0, x1:1, y1:1, xref: 'paper', yref: 'paper', line: {color: 'black', width:1},}]"
        margin_info = r"margin: {l:50, r:40, t:50, b:40}"

        js_command += "\nvar layout = {\ntitle: {text: '"+title+"', y: 0.95}, width:450, height:400, "+axes_info+",\n"+edge_info+",\n"+margin_info+"\n};"
        
        if fig_id:
            js_command += "\nPlotly.newPlot('"+fig_id+"', data, layout);\n\n"

        return js_command

    def short_name(self):
        return 'JPEG Compression Test'
    
    # def latex_name(self):
    #     return 'Contrast - Matching - Spatial - Freq.'



