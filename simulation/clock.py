class Clock:
    def __init__(self,day_length:int): #day_length in secconds
        self.day_length = day_length
        self.timer: int = self.day_length
        self.new_day = True
        self.day_counter = int(1)
    def tick(self,delta_time): #advances the clock
        if self.timer > 0:
            self.timer -= delta_time
            
        else:
            self.day_counter+=1
            self.timer = self.day_length
            self.new_day = True