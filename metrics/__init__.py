import logging

try:
    from metrics.piq_metrics.dss_metric import dss_metric
    from metrics.piq_metrics.fsim_metric import fsim_metric
    from metrics.piq_metrics.gmsd_metric import gmsd_metric
    from metrics.piq_metrics.haarpsi_metric import haarpsi_metric
    from metrics.piq_metrics.iw_ssim_metric import iw_ssim_metric
    from metrics.piq_metrics.mdsi_metric import mdsi_metric
    from metrics.piq_metrics.ms_gmsd_metric import ms_gmsd_metric
    from metrics.piq_metrics.vif_metric import vifp_metric
    from metrics.piq_metrics.vsi_metric import vsi_metric
except ImportError as e:
    logging.warning( "Failed to import PIQ metrics. Run `pip install piq`.")

try:
    from metrics.pyiqa_metrics.topiq_metric import topiq_metric
    from metrics.pyiqa_metrics.pieapp_metric import pieapp_metric
    from metrics.pyiqa_metrics.nlpd_metric import nlpd_metric
    from metrics.pyiqa_metrics.dists_metric import dists_metric
    from metrics.pyiqa_metrics.ckdn_metric import ckdn_metric
    from metrics.pyiqa_metrics.ahiq_metric import ahiq_metric
    from metrics.pyiqa_metrics.mad_metric import mad_metric
    from metrics.pyiqa_metrics.ms_ssim_metric import ms_ssim_metric
    from metrics.pyiqa_metrics.vif_metric import vif_metric
    from metrics.pyiqa_metrics.wadiqam_metric import wadiqam_metric
    from metrics.pyiqa_metrics.lpips_vgg_metric import lpips_vgg_metric
    from metrics.pyiqa_metrics.lpips_alex_metric import lpips_metric
    from metrics.pyiqa_metrics.stlpips_vgg_metric import stlpips_vgg_metric
    from metrics.pyiqa_metrics.stlpips_alex_metric import stlpips_metric
    from metrics.pyiqa_metrics.msswd_metric import msswd_metric
    from metrics.pyiqa_metrics.deepdc_metric import deepdc_metric
    from metrics.pyiqa_metrics.brisque_metric import brisque_metric
    from metrics.pyiqa_metrics.hyperiqa_metric import hyperiqa_metric
    from metrics.pyiqa_metrics.arniqa_metric import arniqa_metric
    from metrics.pyiqa_metrics.nima_metric import nima_metric
    from metrics.pyiqa_metrics.paq2piq_metric import paq2piq_metric
    from metrics.pyiqa_metrics.musiq_metric import musiq_metric
    from metrics.pyiqa_metrics.tres_metric import tres_metric
except ImportError as e:
    logging.warning( "Failed to import PYIQA metrics. Run `pip install pyiqa`.")

from metrics.psnr_metric import psnry_metric
from metrics.hdr_flip import hdr_flip_metric
from metrics.ldr_flip import ldr_flip_metric

try:
    from metrics.ssim_metric import ssim_metric
except ImportError as e:
    logging.warning( "Failed to import SSIM from pycvvdp. Ensure that you have the full installation of ColorVideoVDP.")

from metrics.vmaf_metric import vmaf_met
from metrics.vmaf_neg_metric import vmaf_neg_met
from metrics.funque_metric import funque_met

try:
    from metrics.cvvdp_metric import cvvdp_met
except ImportError as e:
    logging.warning( "Failed to import pycvvdp. Ensure that you have the full installation of ColorVideoVDP.")

try:
    from metrics.fvvdp_metric import fvvdp_met
except ImportError as e:
    logging.warning( "Failed to import pyfvvdp. Ensure that you have the full installation of FovVideoVDP.")

from metrics.de2000 import de2000
from metrics.dolby_ictcp import ictcp
from metrics.hyab import hyab

from metrics.matlab_metrics.hdr_vdp_3 import hdrvdp3_metric
from metrics.matlab_metrics.scielab import scielab_metric
from metrics.matlab_metrics.strred import strred_metric
from metrics.matlab_metrics.speedqa import speedqa_metric