class TransitPipeline:
    def __init__(self):
        # A dictionary where the key is the absolute arrival step, 
        # and the value is the total quantity of goods arriving.
        self.pipeline = {}

    def add_shipment(self, current_step, quantity, lead_time):
        arrival_step = current_step + lead_time
        
        # FIX: Ensure the dictionary key exists before attempting to use +=
        if arrival_step not in self.pipeline:
            self.pipeline[arrival_step] = 0
            
        # Add the physical quantity to the arrival week
        self.pipeline[arrival_step] += quantity

    def receive_shipment(self, current_step):
        # Retrieve the goods arriving this week. 
        # Using .pop() removes the old data from memory, keeping the simulation fast,
        # and defaults to 0 if no shipment was scheduled.
        return self.pipeline.pop(current_step, 0)