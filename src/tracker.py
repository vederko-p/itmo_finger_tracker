
class PositionTracker:
    def __init__(self, n_to_hold: int):
        self.n_to_hold = n_to_hold
        self.cords = []

    def append(self, xy: tuple):
        self.cords = (self.cords + [xy])[-self.n_to_hold:]
