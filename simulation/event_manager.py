import random
from simulation import entity_structures, resources, clock
from dataclasses import dataclass

@dataclass
class EventManager():
    entity_manager: entity_structures.EntityManager
    resource_manager: resources.ResourceManager
    clock: clock.Clock
    area_width: int
    area_height: int
    day_1 = True
    
    def __post_init__(self):
        # Initialize the first day
        self.initialize_first_day()
        if self.entity_manager.clock.day_counter == 1 or self.entity_manager.clock.day_counter == 0:
            self.day_1 = True
    
    def initialize_first_day(self):
        if self.day_1:
            self.day_1 = False
            self.spawn_resources()

    def update(self, delta_time):
        if self.clock.new_day:
            self.spawn_resources()

    def spawn_resources(self):
        for resource_name, resource in self.resource_manager.resources.items():
            spawn_chance = resource[self.resource_manager.SPAWNCHANCE]
            if random.randint(0, 100) <= spawn_chance:
                spawn_amount = resource[self.resource_manager.SPAWNAMOUNT]
                for i in range(spawn_amount):
                    new_resource = resources.Resource(
                        position=entity_structures.Vector2(random.randint(1, self.area_width), random.randint(0, self.area_height)),
                        entity_manager=self.entity_manager,
                        texture_name=random.choice(resource[self.resource_manager.TEXTURES]),
                        name=resource_name,
                        quantity=resource[self.resource_manager.QUANTITY],
                        last_accessed_day=self.entity_manager.clock.day_counter
                    )
                    self.entity_manager.entities.append(new_resource)
                    
