from pycvvdp import cvvdp

class cvvdp_met(cvvdp):

    @staticmethod
    def name():
        return 'ColorVideoVDP'
    
    @staticmethod
    def is_lower_better():
        return False
    
    @staticmethod
    def predictions_range():
        return 0, 10