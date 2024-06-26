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

""" 
USE ENERGY 0-100 percent
Energy for hunting, each hunt(minus a bit of energy) state long distnace, short distance 
Energy for escaping
ADD HIDING FOR ALTERNATIVE CHOICE TO ESCAPING, HIDING LAST ONE DAY AND IT HIDES ON THE FOOD
REPRODUCING USES ENERGY
WANDERING USES ENERGY
DEER CAN LAST ARONUND 3 D AND LIONS ABT 10-7 D
 """

@dataclass
class Vector2:
    x: int
    y: int
    def distance_to(self,other_vector2):
        return math.hypot(self.x - other_vector2.x, self.y - other_vector2.y)
    
    def closest(self,vector2_list):
        return sorted(vector2_list,key=lambda x: self.distance_to(x),reverse=True)
    

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

    def assign_task_for_all_animals(self, delta_time):
        for entity in self.entities:
            if isinstance(entity, RLAnimal):
                entity.animal_update(delta_time)

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
    children: list 
    parents: list
    task: list
    resource_requirements : dict 
    speed: int
    prey: dict #pass as none to indicate herbivore
    resource_on_death: str #its name   
    resource_count_on_death: int
    
    #Rewards
    reproduction_reward: int
    living_reward: int
    gathering_reward: int
    hunting_reward: int
    death_by_hunger_reward: int

    def __post_init__(self):
        super().__post_init__()
        self.input_size = 4  # State size
        self.hidden_size = 64
        self.output_size = 5  # Task size

        self.model_deer = TaskPredictor(self.input_size, self.hidden_size, self.output_size).to(device)
        self.optimizer_deer = optim.Adam(self.model_deer.parameters(), lr=0.01)
        self.criterion_deer = nn.BCEWithLogitsLoss()
        
        self.model_lion = TaskPredictor(self.input_size, self.hidden_size, self.output_size).to(device)
        self.optimizer_lion = optim.Adam(self.model_lion.parameters(), lr=0.01)
        self.criterion_lion = nn.BCEWithLogitsLoss()

        self.entity_manager.stats.populations[self.animal_type] += 1
        self.resource_count = dict.fromkeys(self.resource_requirements,0)
        self.target = None
        
        #Task: 1Wander, 2gather, 3reproduce, 4hunt, 5escape, 6hide
        self.task = [int(0), int(0), int(0), int(0),int(0), int(0)]
        #States: living energy, repruction energy, chased, hunt_distance
        self.states = [float(1), int(0), int(0), int(0)]
        

    @classmethod
    def initialize_animals(self, animal, entity_manager):
        # Load configurations
        import simulation.resources
        model_path_deer = f"./data/model/Deer_model.pth" 
        model_path_lion = f"./data/model/Lion_model.pth"

        for _ in range(animal["starting_number"]):
            RLAnimal(Vector2(random.randint(0,entity_manager.map_size.x), random.randint(0, entity_manager.map_size.y)),
                    entity_manager = entity_manager,
                    texture_name = animal["texture"],
                    animal_type = animal["animal_type"],
                    age = int(0),
                    max_age = animal["max_age"],
                    children = None,
                    parents = None,
                    task = None,
                    resource_requirements = simulation.resources.AnimalResourceRequirements.decode_dict(animal["resource_requirements"]),
                    speed = random.randint(animal["base_speed"][0],animal["base_speed"][1]),
                    prey = animal["prey"],
                    resource_on_death = animal["resource_on_death"],
                    resource_count_on_death = animal["resource_count_on_death"],
                    reproduction_reward = animal["reproduction_reward"],
                    living_reward = animal["living_reward"],
                    gathering_reward = animal["gathering_reward"],
                    hunting_reward = animal["hunting_reward"],
                    death_by_hunger_reward = animal["death_by_hunger_reward"]
            )

        try:
            pass
        #    self.model_deer = self.load_model(self,model_path_deer)
        #    self.model_lion = self.load_model(self,model_path_lion)
        #    print(f"{animal} model found, initializing...")
        except FileNotFoundError:
                print(f"{animal} model not found, initializing with new model.")
        #self.simulation_loop(entity_manager)


    def update(self, delta_time):
        self.sprite.center_x = self.position.x
        self.sprite.center_y = self.position.y

        if self.entity_manager.clock.new_day:
            print("NEW DAY")
            self.entity_manager.clock.new_day = False
            self.entity_manager.assign_task_for_all_animals(delta_time)
        self.perform_task(delta_time)
        



    def animal_update(self, delta_time):
        
        for resource_name, resource_requirement in self.resource_requirements.items():
            if self.states[0] > 0:
                self.states[0] -= resource_requirement.DailyEnergyUsageRate
                self.age +=1
            
            if self.states[0] <= 0:
                self.states[0] = 0
                self.destroy()
                return
    
            if self.states[0] < resource_requirement.ReproductionEnergyUsage:
                self.states[1] = int(0)
            else:
                self.states[1] = int(1)

            if self.age > self.max_age:
                self.destroy()
                return
            
            if self.animal_type == "Deer":
                self.states[2] = 0  # Reset chased state
                for entity in self.entity_manager.entities:
                    if isinstance(entity, RLAnimal) and entity.target == self:
                        if random.random() < 0.9:
                            self.states[2] = 1  
                            break
                        else:
                            return
                        
            
            
            
            print(f"Animal Type: {self.animal_type}, Age: {self.age}, States: {self.states}")
        
        
        
        state_vector = torch.tensor([self.states], dtype=torch.float).to(device)
        if self.animal_type == "Deer":
            task_probabilities = self.model_deer(state_vector)
        elif self.animal_type == "Lion":
            task_probabilities = self.model_lion(state_vector)
        else:
            task_probabilities = torch.zeros(self.output_size).to(device)
            print("Unidentified Animal for task assign")
        
        self.task = torch.argmax(task_probabilities).item()
        print(f"Assigned task {self.task} to {self.animal_type}")


    def perform_task(self, delta_time):
        self.task = int(0)
        if self.animal_type == "Lion":
            self.task = 3
        
        
        if self.task == 0:
            self.wander(delta_time)
        elif self.task == 1:
            self.gather(delta_time)
        elif self.task == 2:
            self.reproduce(delta_time)
        elif self.task == 3:
            self.hunt(delta_time)
        elif self.task == 4:
            self.escape(delta_time)
        elif self.task == 5:
            self.hide(delta_time)
        else:
            raise ValueError(f"Unhandled task {self.task}")

    def wander(self, delta_time):
        
        if self.target is None or random.random() < 0.02:  # 2% chance to pick a new target each update
            self.target = Vector2(random.randint(0, self.entity_manager.map_size.x),
                                random.randint(0, self.entity_manager.map_size.y))

        # Calculate the direction towards the target
        direction = Vector2(self.target.x - self.position.x, self.target.y - self.position.y)
        distance = math.hypot(direction.x, direction.y)

        # Add random to the direction
        if distance > 0:
            direction.x /= distance
            direction.y /= distance

            # Randomly adjust direction to avoid straight lines
            direction.x += random.uniform(-0.1, 0.1)
            direction.y += random.uniform(-0.1, 0.1)

            # Normalize the direction again 
            distance = math.hypot(direction.x, direction.y)
            direction.x /= distance
            direction.y /= distance

        self.position.x += direction.x * self.speed * delta_time
        self.position.y += direction.y * self.speed * delta_time
        
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
            return
        else:
            targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
            if self.target != targets[0] or not type(self.target) is simulation.resources.Resource:
                self.target = targets[0]
            if self.pathfind_until(self.target.position,delta_time,32): #TODO: make this texture_size for bounding_box
                self.resource_count[self.target.name] += self.target.quantity
                self.target.destroy()
                for i in range(0,len(self.states)):
                    if self.states[i]:
                        self.table[i][self.task] += self.gathering_reward

    def reproduce(self, delta_time):
        for resource_name,resource_requirement in self.resource_requirements.items():
            if self.resource_count[resource_name] < resource_requirement.ReproductionEnergyUsage:
                return


        targets = list()
        for entity in self.entity_manager.entities:
            if type(entity) is RLAnimal and not id(entity) is id(self) and entity.animal_type == self.animal_type:   
                targets.append(entity)
        if not targets:
            return
        else:
            targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
            if (self.target != targets[0] and (self.children == None or not self.target in self.children)) or id(self.target) is id( self):
                self.target = targets[0]
                
                
            if self.pathfind_until(self.target.position,delta_time,32): #TODO: make this texture_size for bounding_box

                child = RLAnimal(Vector2(random.randint(0,self.entity_manager.map_size.x), random.randint(0, self.entity_manager.map_size.y)),
                    entity_manager = self.entity_manager,
                    texture_name = self.texture_name,
                    animal_type = self.animal_type,
                    age = int(0),
                    max_age = self.max_age,
                    children = list(),
                    parents = [self,self,targets[0]],
                    task = self.wander(delta_time),
                    resource_requirements = self.resource_requirements,
                    speed = random.choice([self.speed,targets[0].speed]),
                    prey = self.prey,
                    resource_on_death = self.resource_on_death,
                    resource_count_on_death = self.resource_count_on_death,
                    reproduction_reward = self.reproduction_reward,
                    living_reward = self.living_reward,
                    gathering_reward = self.gathering_reward,
                    hunting_reward = self.hunting_reward,
                    death_by_hunger_reward = self.death_by_hunger_reward
            )
                
                if self.children == None:
                    self.children = [child]
                else:
                    self.children.append(child)
                for resource_name,resource_requirement in self.resource_requirements.items():
                    usage = resource_requirement.ReproductionEnergyUsage
                    child.resource_count[resource_name] = usage
                    self.resource_count[resource_name] -= usage

                self.wander(delta_time)
                child.wander(delta_time)

                for i in range(0,len(self.states)):
                    if self.states[i]:
                        self.table[i][self.task] += self.reproduction_reward

    def hunt(self, delta_time):
        if self.prey == None:
            self.escape(delta_time)
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
            return

        
        targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
        if (self.target != targets[0] and not self.target in self.entity_manager.entities) or self.target == self:
            self.target = targets[0]
            self.target.states[2] = int(0)

        else:
            self.target.states[2] = int(1)
            
        if self.pathfind_until(self.target.position,delta_time,32) :
            if self.target.resource_on_death in self.resource_count:
                self.resource_count[self.target.resource_on_death] += self.target.resource_count_on_death
            else:
                self.resource_count[self.target.resource_on_death] = self.target.resource_count_on_death
            self.target.destroy()
            
            # for i in range(0,len(self.states)):
            #     if self.states[i]:
            #         self.table[i][self.task] += self.hunting_reward
                

    def escape(self, delta_time):
        if self.states[2] == int(0):
            return
        predators = list()
        for entity in self.entity_manager.entities:
            if type(entity) is type(RLAnimal) and entity.target == self:
                predators.append(entity)
        if not predators:
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

    def hide(self, delta_time):
        pass

    def learn_from_action(self, reward):
        self.optimizer.zero_grad()
        predicted_task_probabilities = self.model(torch.tensor([self.states], dtype=torch.float).to(device))
        loss = self.criterion(predicted_task_probabilities, torch.tensor([self.task], dtype=torch.long).to(device))
        loss.backward()
        self.optimizer.step()
   
    def destroy(self):
        if self in self.entity_manager.entities:
            Entity.destroy(self)
            self.entity_manager.stats.populations[self.animal_type] -= 1

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
            
    def pathfind_until(self, goal, delta_time, bounding_box):
        direction = Vector2(goal.x - self.position.x, goal.y - self.position.y)
        distance = math.hypot(direction.x, direction.y)
        if distance <= bounding_box:
            return True

        if distance > 0:
            direction.x /= distance
            direction.y /= distance

        self.position.x += direction.x * self.speed * delta_time
        self.position.y += direction.y * self.speed * delta_time

        return False

        
    # def save_model(self,x):
    #     torch.save(self.model.state_dict(),x)

    # def load_model(self, model_path):
    #     model = TaskPredictor(self.input_size, self.hidden_size, self.output_size)
    #     state_dict = torch.load(model_path)
    #     model.load_state_dict(state_dict)
    #     return model
        
    def simulation_loop(self, entity_manager):
        while True:
            for entity in self.entities:
                entity.update(self.delta_time)
        
        
        #for entity in entity_manager.entities:
            
            
            # if isinstance(entity, RLAnimal):
            #     entity.save_model(f"./data/model/{entity.animal_type}_model.pth")
            #     print("Model Saved")
