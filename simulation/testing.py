class Test:
    def perform_task(self, delta_time):
        self.task = "Wandering"
        self.apply_task(delta_time)

    def apply_task(self, delta_time):
        print(self.task)
    perform_task()