class UpdateFunction:
    def __init__(self, function, input_channel_ids, affected_channels):
        self.function = function
        self.input_channel_ids = input_channel_ids
        self.affected_channels = affected_channels