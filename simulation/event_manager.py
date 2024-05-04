import random
from simulation import entity_structures, resources,clock
from dataclasses import dataclass
@dataclass
class EventManager():
    entity_manager: entity_structures.EntityManager
    resource_manager: resources.ResourceManager
    clock: clock.Clock
    area_width: int
    area_height: int
    def update(self,delta_time):
        if self.clock.new_day:        
            for entity in self.entity_manager.entities:
                if entity.entity_type == entity_structures.EntityType.resource:
                    entity.destroy()
                    pass
            for resource_name, resource in self.resource_manager.resources.items():  #TODO: add config option for this
                if random.randint(0,100) <= resource[self.resource_manager.SPAWNCHANCE]:
                    for i in range(0,resource[self.resource_manager.SPAWNAMOUNT]):
                        resources.Resource(entity_structures.Vector2(random.randint(1,self.area_width),random.randint(0,self.area_height)),self.entity_manager,random.choice(resource[self.resource_manager.TEXTURES]),resource_name,resource[self.resource_manager.QUANTITY])