import time
import entity_structures
import clock
import resources
last_tick_time = time.time()
tick_speed = 60
clock = clock.Clock(180)

food = resources.Resource(None,"food")


print(a.task)

while True:
    sleep = 1 / tick_speed - (time.time() - last_tick_time)
    last_tick_time = time.time()
    if sleep > 0:
        time.sleep(sleep)
        clock.tick()

