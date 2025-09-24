from pyfvvdp import fvvdp

class fvvdp_met(fvvdp):

    @staticmethod
    def name():
        return 'FovVideoVDP'

    @staticmethod
    def is_lower_better():
        return False

    @staticmethod
    def predictions_range():
        return 0, 10