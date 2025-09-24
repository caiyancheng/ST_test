import time

import pandas as pd
import logging
from tqdm import trange

from metrics.de2000 import de2000
from pyst.GaborPatch import *
from pyst.ContrastMasking import *
from pyst.ContrastMatching import *
from pyst.NonSmoothness import *
from pyst.NonSmoothness_extroploate import *
from metrics import *
import gc, torch
import sys

if __name__ == '__main__':

    level = logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    """ We will need at some point to add arguments, you simply call this with all tests and metrics for those tests, so we can parallalize it easily or run it with ssh"""

    test_index = 2

    if test_index == 0:
        tests = [GaborPatchSpatialFreqAchTest, GaborPatchLuminanceAchTest, GaborPatchSizeAchTest,
                 ContrastMakingCoherentAchTest, ContrastMakingIncoherentAchTest, ContrastConstancySpatialFreqAchTest]
        metrics = [cvvdp_met, fvvdp_met, ldr_flip_metric, funque_met, vmaf_met, psnry_metric, ssim_metric,
                   ms_ssim_metric, gmsd_metric, ms_gmsd_metric, vifp_metric, fsim_metric, vsi_metric, mdsi_metric,
                   dss_metric, haarpsi_metric, mad_metric, nlpd_metric, wadiqam_metric, lpips_metric, lpips_vgg_metric,
                   dists_metric, ahiq_metric, topiq_metric, hyab, de2000, ictcp, msswd_metric, stlpips_metric,
                   deepdc_metric]
        # metrics = [scielab_metric, hdrvdp3_metric, speedqa_metric]

    elif test_index == 1:
        tests = [GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest, ContrastConstancyColorTest]
        metrics = [cvvdp_met, ldr_flip_metric, fsim_metric, vsi_metric, mdsi_metric, haarpsi_metric, wadiqam_metric,
                   lpips_metric, lpips_vgg_metric, dists_metric, ahiq_metric, topiq_metric, hyab, de2000, ictcp,
                   msswd_metric, stlpips_metric, deepdc_metric]
        # metrics = [scielab_metric]
        tests = [GaborPatchSpatialFreqRGTest, GaborPatchSpatialFreqYVTest]

    elif test_index == 2:
        # tests = [GaborPatchSpatialFreqAchTransientTest, DiskTemporalFreqAchTest]
        # tests = [NonSmoothnessSinGratingTest]
        tests = [NonSmoothnessExtrapolate]
        metrics = [vmaf_met] #, vmaf_met, funque_met, speedqa_metric]
        # metrics = [cvvdp_met, fvvdp_met, vmaf_met, funque_met, speedqa_metric]
        # metrics = [cvvdp_met, fvvdp_met, vmaf_met, funque_met]
        # metrics = [cvvdp_met]
        # tests = [GaborPatchSpatialFreqAchTransientTest]

    output_folder = 'results'

    metric_classes = []
    for j, M in enumerate(metrics):
        metric_classes.append(M())
    with torch.no_grad():
        for T in tests:
            Test = T()
            test_name = Test.short_name()
            print(test_name)

            logging.info(f"Processing {test_name} - with {len(Test)} conditions")

            full_output_folder = os.path.join(output_folder, Test.get_preview_folder())

            if not os.path.exists(full_output_folder):
                os.makedirs(full_output_folder)

            scores = np.zeros((len(Test), len(metrics)))
            metric_names = []

            for i in trange(len(Test)):
                # Get condition
                vid_source, display_photometry, display_geometry = Test.get_condition(i)

                condition_scores = []

                for j, M in enumerate(metrics):

                    try:
                        # Metric = M()
                        Metric = metric_classes[j]
                        metric_name = M.name()
                        print(metric_name)

                        if i == 0:
                            metric_names.append(metric_name)

                        Metric.set_display_model(display_photometry=display_photometry,
                                                 display_geometry=display_geometry)

                        if test_name in ['Contrast Matching - Across Spatial Frequencies',
                                         'Contrast Matching - Across Spatial Frequencies - Band Limited Noise']:
                            score = Test.run_on_test_condition(i, Metric)
                        else:
                            with torch.no_grad():  # Do not accumulate gradients
                                score = Metric.predict_video_source(vid_source)

                        try:
                            if len(score) == 2:
                                score, stats = score
                        except:
                            pass

                        scores[i, j] = score

                    except:
                        logging.error(f'Failed on test {test_name}, condition {i}, and the {metric_name} metric')
                        scores[i, j] = np.nan
                        pass
                del vid_source, display_photometry, display_geometry
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Here, we need to take the header of the class, which will give us information of what does the metric provide to us
            # Then, we want to get these information

            header = Test.get_row_header()
            conditions = Test.get_rows_conditions()

            df = pd.DataFrame()
            df.index.name = 'Index'
            for i in range(len(header)):
                try:
                    df[header[i]] = conditions[i].cpu().numpy()
                except:
                    df[header[i]] = conditions[i]

            for j in range(len(metrics)):
                df['Prediction'] = scores[:, j]

                filename = os.path.join(full_output_folder, metric_names[j] + '.csv')
                df.to_csv(filename)


