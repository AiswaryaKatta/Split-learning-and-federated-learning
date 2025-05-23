from codecarbon import EmissionsTracker

class CarbonTracker:
    def __init__(self, project_name="ML Training", output_dir="emissions_logs"):
        self.tracker = EmissionsTracker(project_name=project_name, output_dir=output_dir)

    def __enter__(self):
        self.tracker.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.stop()
