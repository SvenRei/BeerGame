from collections import defaultdict

class TransitPipeline:
    def __init__(self):
        self.pipeline = defaultdict(int)

    def add_shipment(self, current_step, quantity, lead_time):
        arrival_step = current_step + lead_time
        self.pipeline[arrival_step] += quantity

    def receive_shipment(self, current_step):
        return self.pipeline.pop(current_step, 0)