from simulation import entity_structures
from dataclasses import dataclass
import os
import json
@dataclass
class Resource(entity_structures.Entity):
    name: str
    quantity: int
    def __post_init__(self):
        super().__post_init__()
        self.entity_type = entity_structures.EntityType.resource
@dataclass
class ResourceManager():
    path_to_data: str
    def __post_init__(self):
        self.resources = list()
        self.TEXTURES = "textures"
        self.SPAWNCHANCE = "spawn_chance"
        self.SPAWNAMOUNT = "spawn_amount" 
        self.QUANTITY = "quantity" #how much each resource gives on collection
        with open(os.path.join(self.path_to_data,"resources.json")) as resource_json:
            self.resources = json.load(resource_json)
@dataclass
class AnimalResourceRequirements:
    neededForSurvival: bool
    neededForReproduction: bool
    priority: int #animals who have multiple requirements will go to the closest resource with the highest priority
    dailyUsageRate: list # rate is randomly chosen from list, TODO: Investigate weighting this
    reproductionUsageRate: list
    def decode_dict(dictionary):
        final_dictionary = dict()
        for key, value in dictionary.items():
            final_dictionary[key] = AnimalResourceRequirements.decode(value)
        return final_dictionary
    def decode(dictionary):
        return AnimalResourceRequirements(dictionary["neededForSurvival"],dictionary["neededForReproduction"],dictionary["priority"],dictionary["dailyUsageRate"],dictionary["reproductionUsageRate"])

        
