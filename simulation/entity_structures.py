from typing import List, Dict
from dataclasses import dataclass
from enum import Enum, IntEnum
import random
import math
import simulation.clock
import simulation.output
import arcade
import torch
import torch.nn as nn
import torch.optim as optim
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Select gpu for torch


@dataclass
class Vector2:
    x: int
    y: int
    def distance_to(self,other_vector2):
        return math.hypot(self.x - other_vector2.x, self.y - other_vector2.y)
    
    def closest(self,vector2_list):
        return sorted(vector2_list,key=lambda x: self.distance_to(x),reverse=True)
    

class State(IntEnum):
    low_living_resources = 0,
    chased = 1,
    low_reproduction_resources = 2,
    high_reproduction_resources = 3


class Task(IntEnum):
    wander = 0
    gather = 1
    reproduce = 2
    hunt = 3
    escape = 4


class EntityType(Enum):
    none = -1
    animal = 0
    resource = 1


@dataclass
class EntityManager():
    entities: list
    map_size: Vector2
    clock: simulation.clock.Clock
    stats: simulation.output.Stats
    textures: list

    def __post_init__(self):
        self.sprite_list = arcade.SpriteList(True)


@dataclass
class Entity:
    position: Vector2     
    entity_manager: EntityManager
    texture_name: str
    def update(self,delta_time):
        pass

    def __post_init__(self):
        self.entity_manager.entities.append(self)
        self.entity_type = EntityType.none        
        self.sprite = arcade.Sprite(self.entity_manager.textures[self.texture_name],1,self.position.x,self.position.y)
        self.entity_manager.sprite_list.append(self.sprite)

    def destroy(self):
        #may need some tweaking for when editing list while itterating
        self.entity_manager.entities.remove(self)
        try:
            self.entity_manager.sprite_list.remove(self.sprite)
        except:
            pass


class TaskPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TaskPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

@dataclass
class RLAnimal(Entity):
    import simulation.resources
    animal_type: str
    age: int
    max_age: int
    max_days_before_reproduction: int
    children: list 
    parents: list
    task: Task
    resource_requirements : dict 
    speed: int
    prey: dict #pass as none to indicate herbivore
    resource_on_death: str #its name   
    resource_count_on_death: int
    reproduction_reward: int
    living_reward: int
    gathering_reward: int
    hunting_reward: int
    death_by_hunger_reward: int
    experimentation_factor: int
    experimentation_factor_decay: int
    max_hunt_per_day: int
    def __post_init__(self):
        super().__post_init__()
        self.input_size = 4  # State size
        self.hidden_size = 64
        self.output_size = 5  # Number of tasks
        self.model = TaskPredictor(self.input_size, self.hidden_size, self.output_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.entity_manager.stats.populations[self.animal_type] += 1
        self.resource_count = dict.fromkeys(self.resource_requirements,0)
        self.days_before_reproduction = self.max_days_before_reproduction
        self.target = self
        self.task = Task.wander
        self.age = 0
        self.chased = False
        self.hunt_per_day = 0
        self.states = [False, False, False, False]
        self.table = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

    def initialize_simulation(self, entity_manager):
        self.initialize_animals(entity_manager)  # This function loads models for each animal
        self.simulation_loop(entity_manager, total_steps=1, save_interval=1)  # Start the simulation loop

    @classmethod
    def initialize_animals(self, animal, entity_manager):
        # Load configurations
        import simulation.resources
        deer_config = self.load_animal_config('./data/animals/deer.json')
        lion_config = self.load_animal_config('./data/animals/lion.json')
        
        # Create deer instances
        for _ in range(deer_config["starting_number"]):
            deer = RLAnimal(Vector2(random.randint(0, entity_manager.map_size.x), random.randint(0, entity_manager.map_size.y)), entity_manager, animal["texture"],animal["animal_type"],0,animal["max_age"],animal["max_days_before_reproduction"],None,None,Task.wander,simulation.resources.AnimalResourceRequirements.decode_dict(animal["resource_requirements"]),random.randint(animal["base_speed"][0],animal["base_speed"][1]),animal["prey"],animal["resource_on_death"],animal["resource_count_on_death"],animal["reproduction_reward"],animal["living_reward"],animal["gathering_reward"],animal["hunting_reward"],animal["death_by_hunger_reward"],animal["experimentation_factor"],animal["experimentation_factor_decay"],animal["max_hunt_per_day"])
            model_path = f"./data/model/{deer.animal_type}_model.pth"
            try:
                deer.load_model(model_path)
            except FileNotFoundError:
                print("Deer model not found, initializing with new model.")
        
        
        # Create lion instances
        for _ in range(lion_config["starting_number"]):
            lion = RLAnimal(Vector2(random.randint(0, entity_manager.map_size.x), random.randint(0, entity_manager.map_size.y)), entity_manager, animal["texture"],animal["animal_type"],0,animal["max_age"],animal["max_days_before_reproduction"],None,None,Task.wander,simulation.resources.AnimalResourceRequirements.decode_dict(animal["resource_requirements"]),random.randint(animal["base_speed"][0],animal["base_speed"][1]),animal["prey"],animal["resource_on_death"],animal["resource_count_on_death"],animal["reproduction_reward"],animal["living_reward"],animal["gathering_reward"],animal["hunting_reward"],animal["death_by_hunger_reward"],animal["experimentation_factor"],animal["experimentation_factor_decay"],animal["max_hunt_per_day"])
            model_path = f"./data/model/{lion.animal_type}_model.pth"
            try:
                lion.load_model(model_path)
            except FileNotFoundError:
                print("Lion model not found, initializing with new model.")
        print("Initialized animals")

    def update(self, delta_time):
        super().update(delta_time)
        self.perform_task(delta_time)
        self.sprite.center_x = self.position.x
        self.sprite.center_y = self.position.y

        if self.entity_manager.clock.new_day and self.entity_manager.clock.day_counter >1:
            self.hunt_per_day = 0
            self.update_task(delta_time)

            for resource_name, resource_requirement in self.resource_requirements.items():
                if (self.resource_count[resource_name] < resource_requirement.dailyUsageRate[1]):
                    self.resource_count[resource_name] -= resource_requirement.dailyUsageRate[0]
                else:
                    self.resource_count[resource_name] -= random.randint(resource_requirement.dailyUsageRate[0], resource_requirement.dailyUsageRate[1],)
                if (self.resource_count[resource_name] < 0 and resource_requirement.neededForSurvival):
                    for i in range(len(self.states)):
                        if self.states[i]:
                            self.table[i][self.task] -= self.death_by_hunger_reward
                            if self.children:
                                for child in self.children:
                                    child.table[i][self.task] -= self.death_by_hunger_reward
                    self.destroy()

            if self.days_before_reproduction > 0:
                self.days_before_reproduction -= 1
            self.age += 1
            self.experimentation_factor -= self.experimentation_factor_decay
            
            for i in range(len(self.states)):
                if self.states[i]:
                    self.table[i][self.task] += self.living_reward
            
            if self.age > self.max_age:
                self.destroy()

        if self.task == Task.wander:
            if (not isinstance(self.target, Vector2) or self.position.distance_to(self.target) < 100):
                self.target = Vector2(random.randint(0, self.entity_manager.map_size.x), random.randint(0, self.entity_manager.map_size.y),)
            if self.pathfind_until(self.target, delta_time, 32):
                self.target = Vector2(random.randint(0, self.entity_manager.map_size.x), random.randint(0, self.entity_manager.map_size.y),)
            else:
                self.update_task(delta_time)

        elif self.task == Task.gather:
            resource_requirements = sorted(
                self.resource_requirements.items(), key=lambda x: x[1].priority)
            targets = []
            resource_counter = 0

            while not targets:
                if resource_counter > len(resource_requirements) - 1:
                    break  # no valid targets found
                for entity in self.entity_manager.entities:
                    if isinstance(entity, simulation.resources.Resource):
                        if entity.name == resource_requirements[resource_counter][0]:
                            targets.append(entity)
                resource_counter += 1
            
            if not targets:
                self.update_task(delta_time) # this keeps looping until a valid target is found
            
            else:
                targets.sort(
                    key=lambda x: x.position.distance_to(self.position), reverse=False)
                if (self.target != targets[0] or not isinstance(self.target, simulation.resources.Resource) or self.position.distance_to(self.target.position) < 100):
                    self.target = targets[0]
                if self.pathfind_until(self.target.position, delta_time, 32):
                    self.resource_count[self.target.name] += self.target.quantity
                    self.target.destroy()
                    self.update_task(delta_time)
                    for i in range(len(self.states)):
                        if self.states[i]:
                            self.table[i][self.task] += self.gathering_reward

        elif self.task == Task.reproduce:
            if self.days_before_reproduction > 0:
                self.update_task(delta_time)
                return
            for resource_name, resource_requirement in self.resource_requirements.items():
                if (self.resource_count[resource_name] < resource_requirement.reproductionUsageRate[1]):
                    self.update_task(delta_time)
                    return

            targets = []
            for entity in self.entity_manager.entities:
                if (isinstance(entity, Animal) and id(entity) != id(self) and entity.animal_type == self.animal_type):
                    targets.append(entity)
            if not targets:
                self.update_task(delta_time)
                return
            else:
                targets.sort(
                    key=lambda x: x.position.distance_to(self.position), reverse=False)
                if (self.target != targets[0] and (self.children is None or self.target not in self.children)) or id(self.target) == id(self):
                    self.target = targets[0]

                if self.pathfind_until(self.target.position, delta_time, 32):
                        child = Animal(Vector2(random.randint(0,self.entity_manager.map_size.x),random.randint(0,self.entity_manager.map_size.y)),self.entity_manager,self.texture_name,self.animal_type,0,self.max_age,self.max_days_before_reproduction,list(),[self,self,targets[0]],Task.wander,self.resource_requirements,random.choice([self.speed,targets[0].speed]),self.prey,self.resource_on_death,self.resource_count_on_death,self.reproduction_reward,self.living_reward,self.gathering_reward,self.hunting_reward,self.death_by_hunger_reward,self.experimentation_factor,self.experimentation_factor_decay,self.max_hunt_per_day)
                        if self.children == None:
                            self.children = [child]

                        else:
                            self.children.append(child)

                        for resource_name,resource_requirement in self.resource_requirements.items():
                            usage = random.randint(resource_requirement.reproductionUsageRate[0],resource_requirement.reproductionUsageRate[1])
                            child.resource_count[resource_name] = usage
                            self.resource_count[resource_name] -= usage

                        self.days_before_reproduction = self.max_days_before_reproduction
                        self.update_task(delta_time)
                        child.update_task(delta_time)
                        child.days_before_reproduction = self.max_days_before_reproduction 
                        for i in range(0,len(self.states)):
                            if self.states[i]:
                                self.table[i][self.task] += self.reproduction_reward

        elif self.task == Task.hunt:
            if self.hunt_per_day >= self.max_hunt_per_day:
                self.update_task(delta_time)
                return

            prey = sorted(self.prey.items(), key=lambda x: x[1])
            targets = []
            prey_counter = 0
            while not targets:
                    if prey_counter > len(prey)-1:
                        break #no valid targets found
                    for entity in self.entity_manager.entities:
                        if isinstance(entity, Animal):   
                            if entity.animal_type == prey[prey_counter][0]:
                                targets.append(entity)
                    prey_counter += 1
                    #this keeps looping until a valid target is found
            if not targets:
                self.update_task(delta_time)
                pass
            elif self.hunt_per_day < self.max_hunt_per_day:
                targets.sort(
                    key=lambda x: x.position.distance_to(self.position), reverse=False)
                if (self.target != targets[0] and self.target not in self.entity_manager.entities) or self.target == self:
                    self.target = targets[0]
                    self.target.states[State.chased] = False

                else:
                    self.target.states[State.chased] = True
                    self.target.update_task(delta_time)
                    if self.pathfind_until(self.target.position,delta_time,32) :
                        if self.target.resource_on_death in self.resource_count:
                            self.resource_count[self.target.resource_on_death] += self.target.resource_count_on_death
                        else:
                            self.resource_count[self.target.resource_on_death] = self.target.resource_count_on_death
                    self.target.destroy()
                    self.hunt_per_day += 1
                    for i in range(0,len(self.states)):
                        if self.states[i]:
                            self.table[i][self.task] += self.hunting_reward
                            self.update_task(delta_time)

            elif self.task == Task.escape:
                if not self.chased:
                    self.update_task(delta_time)
                predators = []
                for entity in self.entity_manager.entities:
                    if isinstance(entity, Animal) and entity.target == self:
                        predators.append(entity)
                if not predators:
                    self.update_task(delta_time)
                    return
                predators = sorted(
                    predators, key=lambda x: x.position.distance_to(self.position)
                )
                # determine corners
                centre = Vector2(
                    self.entity_manager.map_size.x / 2,
                    self.entity_manager.map_size.y / 2,
                )
                corners = [
                    Vector2(0, 0),
                    Vector2(self.entity_manager.map_size.x, 0),
                    Vector2(0, self.entity_manager.map_size.y),
                    Vector2(
                        self.entity_manager.map_size.x,
                        self.entity_manager.map_size.y,
                    ),
                ]
                predators_corners = sorted(
                    corners,
                    key=lambda x: x.distance_to(predators[0].position),
                    reverse=True,
                )
                final_corner = Vector2(0, 0)
                if predators_corners[0] == corners[0]:
                    final_corner = corners[3]
                    self.target = Vector2(
                        final_corner.x - predators[0].position.x,
                        final_corner.y - predators[0].position.y,
                    )
                elif predators_corners[0] == corners[1]:
                    final_corner = corners[2]
                    self.target = Vector2(
                        final_corner.x + predators[0].position.x,
                        final_corner.y - predators[0].position.y,
                    )
                elif predators_corners[0] == corners[2]:
                    final_corner = corners[1]
                    self.target = Vector2(
                        final_corner.x - predators[0].position.x,
                        final_corner.y + predators[0].position.y,
                    )
                elif predators_corners[0] == corners[3]:
                    final_corner = corners[0]
                    self.target = Vector2(
                        final_corner.x - predators[0].position.x,
                        final_corner.y - predators[0].position.y,
                    )
                self.pathfind_until(self.target, delta_time, 32)

    def perform_task(self, delta_time):
        state_vector = torch.tensor([self.states], dtype=torch.float).to(device)
        task_probabilities = self.model(state_vector)
        self.task = Task(task_probabilities.argmax().item())
        print(self.task, self.animal_type)
        self.apply_task(delta_time)  

    def apply_task(self, delta_time):
        self.task = Task.wander
        if self.task == Task.wander:
            self.wander(delta_time)
        elif self.task == Task.gather:
            self.gather(delta_time)
        elif self.task == Task.reproduce:
            self.reproduce(delta_time)
        elif self.task == Task.hunt:
            self.hunt(delta_time)
        elif self.task == Task.escape:
            self.escape(delta_time)
        else:
            raise ValueError(f"Unhandled task {self.task}")

    def wander(self, delta_time):
        if not type(self.target) is Vector2:
            self.target = Vector2(random.randint(0,self.entity_manager.map_size.x),random.randint(0,self.entity_manager.map_size.y))
        if self.pathfind_until(self.target,delta_time,32):
            self.target = Vector2(random.randint(0,self.entity_manager.map_size.x),random.randint(0,self.entity_manager.map_size.y))
        else:
            self.update_task(delta_time)
    
    def gather(self, delta_time):
        import simulation.resources
        #consider which resource is highest priority
        resource_requirements = sorted(self.resource_requirements.items(),key=lambda x: x[1].priority)
        targets = list()
        resource_counter = 0
        while not targets:
            if resource_counter > len(resource_requirements)-1:
                break #no valid targets found
            for entity in self.entity_manager.entities:
                if type(entity) is simulation.resources.Resource:   
                    if entity.name == resource_requirements[resource_counter][0]:
                        targets.append(entity)
            resource_counter += 1
            #this keeps looping until a valid target is found
        if not targets:
            self.update_task(delta_time)
        else:
            targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
            if self.target != targets[0] or not type(self.target) is simulation.resources.Resource:
                self.target = targets[0]
            if self.pathfind_until(self.target.position,delta_time,32): #TODO: make this texture_size for bounding_box
                self.resource_count[self.target.name] += self.target.quantity
                self.target.destroy()
                self.update_task(delta_time)
                for i in range(0,len(self.states)):
                    if self.states[i]:
                        self.table[i][self.task] += self.gathering_reward

    def reproduce(self, delta_time):
        if self.days_before_reproduction > 0:
            self.update_task(delta_time)
            return
        for resource_name,resource_requirement in self.resource_requirements.items():
            if self.resource_count[resource_name] < resource_requirement.reproductionUsageRate[1]:
                self.update_task(delta_time)
                return


        targets = list()
        for entity in self.entity_manager.entities:
            if type(entity) is RLAnimal and not id(entity) is id(self) and entity.animal_type == self.animal_type:   
                targets.append(entity)
        if not targets:
            self.update_task(delta_time)
            return
        else:
            targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
            if (self.target != targets[0] and (self.children == None or not self.target in self.children)) or id(self.target) is id( self):
                self.target = targets[0]
                
                
            if self.pathfind_until(self.target.position,delta_time,32): #TODO: make this texture_size for bounding_box

                child = RLAnimal(Vector2(random.randint(0,self.entity_manager.map_size.x),random.randint(0,self.entity_manager.map_size.y)),self.entity_manager,self.texture_name,self.animal_type,0,self.max_age,self.max_days_before_reproduction,list(),[self,self,targets[0]],Task.wander,self.resource_requirements,random.choice([self.speed,targets[0].speed]),self.prey,self.resource_on_death,self.resource_count_on_death,self.reproduction_reward,self.living_reward,self.gathering_reward,self.hunting_reward,self.death_by_hunger_reward,self.experimentation_factor,self.experimentation_factor_decay,self.max_hunt_per_day)
                if self.children == None:
                    self.children = [child]
                else:
                    self.children.append(child)
                for resource_name,resource_requirement in self.resource_requirements.items():
                    usage = random.randint(resource_requirement.reproductionUsageRate[0],resource_requirement.reproductionUsageRate[1])
                    child.resource_count[resource_name] = usage
                    self.resource_count[resource_name] -= usage

                self.days_before_reproduction = self.max_days_before_reproduction
                self.update_task(delta_time)
                child.update_task(delta_time)
                child.days_before_reproduction = self.max_days_before_reproduction 
                for i in range(0,len(self.states)):
                    if self.states[i]:
                        self.table[i][self.task] += self.reproduction_reward

    def hunt(self, delta_time):
        if self.hunt_per_day >= self.max_hunt_per_day:
            self.update_task(delta_time)
            return
        if self.prey == None:
            self.task = Task.escape
            return
        
        prey = sorted(self.prey.items(),key=lambda x: x[1])
        targets = list()
        prey_counter = 0
        while not targets:
            if prey_counter > len(prey)-1:
                break #no valid targets found
            for entity in self.entity_manager.entities:
                if type(entity) is RLAnimal:   
                    if entity.animal_type == prey[prey_counter][0]:
                        targets.append(entity)
            prey_counter += 1
            #this keeps looping until a valid target is found
        if not targets:
            self.update_task(delta_time)
            pass
        elif self.hunt_per_day < self.max_hunt_per_day:
            targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
            if (self.target != targets[0] and not self.target in self.entity_manager.entities) or self.target == self:
                self.target = targets[0]
                self.target.states[State.chased] = False 

            else:
                self.target.states[State.chased] = True
                self.target.update_task(delta_time)
            if self.pathfind_until(self.target.position,delta_time,32) :
                if self.target.resource_on_death in self.resource_count:
                    self.resource_count[self.target.resource_on_death] += self.target.resource_count_on_death
                else:
                    self.resource_count[self.target.resource_on_death] = self.target.resource_count_on_death
                self.target.destroy()
                self.hunt_per_day += 1
                for i in range(0,len(self.states)):
                    if self.states[i]:
                        self.table[i][self.task] += self.hunting_reward
                self.update_task(delta_time)

    def escape(self, delta_time):
        if not self.chased:
            self.update_task(delta_time)
        predators = list()
        for entity in self.entity_manager.entities:
            if type(entity) is type(RLAnimal) and entity.target == self:
                predators.append(entity)
        if not predators:
            self.update_task(delta_time)
            return
        predators = sorted(predators,key=lambda x: x.position.distance_to(self.position))
        #determine corners
        centre = Vector2(self.entity_manager.map_size.x /2,  self.entity_manager.map_size.y /2)
        corners = [Vector2(0,0),Vector2(self.entity_manager.map_size.x,0),Vector2(0,self.entity_manager.map_size.y),Vector2(self.entity_manager.map_size.x,self.entity_manager.map_size.y)]
        predators_corners = sorted(corners,key=lambda x: x.distance_to(predators[0].position),reverse=True) 
        final_corner = Vector2(0,0)
        if predators_corners[0] == corners[0]:
            final_corner = corners[3]
            self.target = Vector2(final_corner.x - predators[0].position.x,final_corner.y - predators[0].position.y)
        elif predators_corners[0] == corners[1]:
            final_corner = corners[2]
            self.target = Vector2(final_corner.x + predators[0].position.x,final_corner.y - predators[0].position.y)
        elif predators_corners[0] == corners[2]:
            final_corner = corners[1]
            self.target = Vector2(final_corner.x - predators[0].position.x,final_corner.y + predators[0].position.y)
        elif predators_corners[0] == corners[3]:
            final_corner = corners[0]
            self.target = Vector2(final_corner.x - predators[0].position.x,final_corner.y - predators[0].position.y)
        self.pathfind_until(self.target,delta_time,32)

    def learn_from_action(self, reward):
        self.optimizer.zero_grad()
        predicted_task_probabilities = self.model(torch.tensor([self.states], dtype=torch.float).to(device))
        loss = self.criterion(predicted_task_probabilities, torch.tensor([self.task], dtype=torch.long).to(device))
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.model.state_dict(), "./data/model")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=device))
   
    def load_animal_config(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return {}

    def destroy(self):
        if self in self.entity_manager.entities:
            Entity.destroy(self)
            self.entity_manager.stats.populations[self.animal_type] -= 1

    def update_task(self, delta_time):
        current_state = -1
        survival_count = 0
        reproduction_count = 0
        if self.chased:
            self.states[State.chased] = True
        else:
            self.states[State.chased] = False

        for resource_name, resource_requirement in self.resource_requirements.items():
            if (
                resource_requirement.neededForSurvival
                and self.resource_count[resource_name]
                < resource_requirement.dailyUsageRate[0]
            ):
                survival_count += 1
            if (
                resource_requirement.neededForReproduction
                and self.resource_count[resource_name]
                < resource_requirement.reproductionUsageRate[1]):
                reproduction_count += 1

        if survival_count > 0:
            self.states[State.low_living_resources] = True
        else:
            self.states[State.low_living_resources] = False

        if reproduction_count > 0:
            self.states[State.low_reproduction_resources] = True
            self.states[State.high_reproduction_resources] = False
        else:
            self.states[State.low_reproduction_resources] = False
            self.states[State.high_reproduction_resources] = True
        
        if self.states[State.chased]:
            current_state = State.chased
        else:
            true_states = [state for state in self.states if state]
            current_state = random.choice(true_states)
        

    def pathfind_to(self, goal, delta_time):
        snapped_goal = Vector2(
            self.speed * round(goal.x / self.speed),
            self.speed * round(goal.y / self.speed),
        )
        # naive pathfinding, good for now

        if self.position.x > goal.x:
            self.position.x -= self.speed * delta_time
        elif self.position.x < goal.x:
            self.position.x += self.speed * delta_time
        if self.position.y > goal.y:
            self.position.y -= self.speed * delta_time
        elif self.position.y < goal.y:
            self.position.y += self.speed * delta_time
            
    def pathfind_until(self,goal,delta_time,bounding_box):
        
        if abs(self.position.x-goal.x) <= bounding_box and abs(self.position.y-goal.y) <= bounding_box:
            return True
        else:
            return False

    def simulation_loop(delta_time, entity_manager, total_steps, save_interval):
        try:
            for step in range(total_steps):
                for entity in entity_manager.entities:
                    if isinstance(entity, RLAnimal):
                        entity.update(delta_time)  
                        if step % save_interval == 0:
                            entity.save_model(f"./model_states/{entity.animal_type}_model.pth")
                            print("Saved model")
        except Exception:
            print("Could not save model")
