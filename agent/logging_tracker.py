import torch

class LoggingTracker:
    def __init__(self, writer):
        self.writer = writer

        self.track_list = []

    def step(self, out, y):
        distances = (y - out)**2
        distances = torch.mean(distances, dim=0)

        self.track_list.append(distances)


    def record(self, epoch):
        track_list = torch.stack(self.track_list)
        track_list = torch.mean(track_list, dim=0)
        track_list = track_list.detach().clone().cpu().tolist()
        for i, track_feature in enumerate(track_list):
            self.writer.add_scalar(f"Train/_feature_{i}/", track_feature, epoch)

