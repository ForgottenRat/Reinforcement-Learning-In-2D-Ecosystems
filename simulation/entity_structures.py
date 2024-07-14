from dataclasses import dataclass
from typing import List, Dict
from enum import Enum, IntEnum
from collections import deque
import random
import math
import simulation.clock
import simulation.output
import arcade
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Select gpu for torch

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
                entity.assign_task(delta_time)

    def update(self, delta_time):
        for entity in self.entities:
            entity.update(delta_time)


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
        self.sprite = arcade.Sprite(self.entity_manager.textures[self.texture_name],1.35,self.position.x,self.position.y)
        self.entity_manager.sprite_list.append(self.sprite)

    def destroy(self):
        self.entity_manager.entities.remove(self)
        try:
            self.entity_manager.sprite_list.remove(self.sprite)
        except:
            pass


class DQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def save_model(self, file_path):
        try:
            torch.save(self.state_dict(), file_path)
            print(f"Model saved to {file_path}")
        
        except:
            print("CRITICAL ERROR FAILED TO SAVE MODEL")

    def load_model(self, file_path):
        try:
            self.load_state_dict(torch.load(file_path))
            print(f"Model loaded from {file_path}")
        
        except:
            print("NO MODEL TO LOAD OR FAILED TO LOAD MODEL", file_path)

    def is_model_loaded(self):
        return any(param.numel() > 0 for param in self.parameters())

@dataclass
class RLAnimal(Entity):
    import simulation.resources
    
    #Model
    model_deer = None
    optimizer_deer = None
    criterion_deer = None
    model_lion = None
    optimizer_lion = None
    criterion_lion = None
    models_loaded = False

    #Animal
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
    hiding_reward: int

    def __post_init__(self):
        super().__post_init__()
        self.input_size = 3  # State size
        self.hidden_size = 64
        self.output_size = 6  # Task size

        if not RLAnimal.models_loaded:
            RLAnimal.model_deer = DQNetwork(self.input_size, self.hidden_size, self.output_size).to(device)
            RLAnimal.optimizer_deer = optim.Adam(RLAnimal.model_deer.parameters(), lr=0.01)
            RLAnimal.criterion_deer = nn.BCEWithLogitsLoss()

            RLAnimal.model_lion = DQNetwork(self.input_size, self.hidden_size, self.output_size).to(device)
            RLAnimal.optimizer_lion = optim.Adam(RLAnimal.model_lion.parameters(), lr=0.01)
            RLAnimal.criterion_lion = nn.BCEWithLogitsLoss()

            RLAnimal.model_deer.load_model("./data/model/Deer_model.pth")
            RLAnimal.model_lion.load_model("./data/model/Lion_model.pth")

            if RLAnimal.model_deer.is_model_loaded() and RLAnimal.model_lion.is_model_loaded():
                RLAnimal.models_loaded = True
            else:
                print("Error loading models.")
                RLAnimal.models_loaded = False

        self.entity_manager.stats.populations[self.animal_type] += 1
        self.resource_count = dict.fromkeys(self.resource_requirements,0)
        self.target = None

        self.task_history = deque(maxlen=2)
        self.task_history.append(-1)
        self.task_history.append(-1)
        #Task: 1Wander, 2gather, 3reproduce, 4hunt, 5escape, 6hide
        self.task = [int(0), int(0), int(0), int(0),int(0), int(0)]
        #States: living energy, repruction energy, chased
        self.states = [float(1), int(0), int(0)]
        self.hiding = False

        self.epsilon: float = 0.1  # Exploration rate
        
        
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
                    death_by_hunger_reward = animal["death_by_hunger_reward"],
                    hiding_reward = animal["hiding_reward"]
            )
        

    def update(self, delta_time):
        self.sprite.center_x = self.position.x
        self.sprite.center_y = self.position.y

        for entity in self.entity_manager.entities:
            if isinstance(entity, RLAnimal):
                entity.animal_update(delta_time)
        if self.entity_manager.clock.new_day:
            print("#=====-+-- Day number:", self.entity_manager.clock.day_counter,"--+-====#")
            self.entity_manager.clock.new_day = False
            self.entity_manager.assign_task_for_all_animals(delta_time)
        self.perform_task(delta_time)
       
    def learn_from_action(self, reward, next_state, done):
            state_vector = torch.tensor([self.states], dtype=torch.float).to(device)
            next_state_vector = torch.tensor([next_state], dtype=torch.float).to(device)

            if self.animal_type == "Deer":
                model = self.model_deer
                optimizer = self.optimizer_deer
            elif self.animal_type == "Lion":
                model = self.model_lion
                optimizer = self.optimizer_lion
            else:
                return
            
            q_values = model(state_vector)
            next_q_values = model(next_state_vector)
            
            max_next_q_value = next_q_values.max(1)[0].detach()
            target_q_value = reward + (0.99 * max_next_q_value * (1 - done))

            loss = torch.nn.functional.mse_loss(q_values[0][self.task], target_q_value)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                if self.animal_type == "Deer":
                    model.save_model(f"./data/model/Deer_model.pth")
                elif self.animal_type == "Lion":
                    model.save_model(f"./data/model/Lion_model.pth")

    def animal_update(self, delta_time):
        
        for resource_name, resource_requirement in self.resource_requirements.items():
            if self.states[0] > 1:
                self.states[0] = 1
            
            if self.entity_manager.clock.new_day:
                if self.states[0] > 0:
                    self.states[0] -= resource_requirement.DailyEnergyUsageRate
                    self.age +=1
                    self.learn_from_action(self.living_reward, self.states, False)
                    if self.entity_manager.clock.day_counter == 1:
                        self.age = 1
                    
                    
            if self.states[0] <= 0:
                self.states[0] = 0
                self.learn_from_action(self.death_by_hunger_reward,self.states,True)
                self.destroy()
                return
    
            if self.states[0] < resource_requirement.ReproductionEnergyUsage:
                self.states[1] = int(0)
            
            else:
                self.states[1] = int(1)

            if self.age > self.max_age:
                self.learn_from_action(0, self.states, True)
                self.destroy()
                return
            
            if self.animal_type == "Deer" and self.states[2] == 1:
                for entity in self.entity_manager.entities:
                    if isinstance(entity, RLAnimal) and entity.target == self:
                        if random.random() > 0.8:
                            self.states[2] = int(1)  
                            break
                        else:
                            self.states[2] = int(0)
                            return
            print(self.states, self.task)

    def assign_task(self, delta_time):
        state_vector = torch.tensor([self.states], dtype=torch.float).to(device)
        if self.animal_type == "Deer":
            task_probabilities = self.model_deer(state_vector)
        elif self.animal_type == "Lion":
            task_probabilities = self.model_lion(state_vector)
        else:
            task_probabilities = torch.zeros(self.output_size).to(device)
            print("Unidentified Animal for task assign")

        if random.random() < self.epsilon:
            # Explore: select a random task
            self.task = random.randint(0, self.output_size - 1)
        else:
            # Exploit: select the task with the highest Q-value
            topk_values, topk_indices = torch.topk(task_probabilities, 2)
            primary_task = topk_indices[0][0].item() 
            secondary_task = topk_indices[0][1].item() 
            
            if primary_task not in self.task_history:
                self.task = primary_task
            else:
                self.task = secondary_task

        self.task_history.append(self.task)
        self.perform_task(delta_time)
        self.states[0] = round(self.states[0],2)
        print(f"> {self.animal_type}-> Task: {self.task}, Age: {self.age}, States: {self.states}")

    def perform_task(self, delta_time):
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
            print(f"-  -  -  -  -  -  -  -  - E: Task Not Found {self.task} {self.animal_type}")
            self.wander(delta_time)

    def wander(self, delta_time):
        wander_energy_cost = float(0.001)
        
        if self.target is None or not isinstance(self.target, Vector2) or random.random() < 0.02:  # 2% chance to pick a new target each update
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
        self.states[0] -= wander_energy_cost
        
    def gather(self, delta_time):
        gather_energy_gain = 0.35
        
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
            self.learn_from_action(0, self.states, False)
            return
        else:
            targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
            if self.target != targets[0] or not type(self.target) is simulation.resources.Resource:
                self.target = targets[0]
            if self.pathfind_until(self.target.position,delta_time,45): #TODO: make this texture_size for bounding_box
                self.states[0] += gather_energy_gain
                self.target.destroy()
                self.learn_from_action(self.gathering_reward, self.states, False)
                
    def reproduce(self, delta_time):
        import simulation.resources
        
        for resource_name, resource_requirement in self.resource_requirements.items():
            if self.states[0] < resource_requirement.ReproductionEnergyUsage:
                self.assign_task(delta_time)
                return
            
        usage = resource_requirement.ReproductionEnergyUsage
        child_energy = 0.15
        mate_usage = 0.05

        # Select potential mates
        potential_mates = [entity for entity in self.entity_manager.entities 
                        if isinstance(entity, RLAnimal) 
                        and entity.animal_type == self.animal_type 
                        and entity is not self 
                        and entity.states[1] == 1
                        and entity.age > 3]  # Ensure the mate is also ready to reproduce

        if not potential_mates:
            self.assign_task(delta_time)
            return

        # Sort mates by proximity and select the closest
        potential_mates.sort(key=lambda x: x.position.distance_to(self.position))
        mate = potential_mates[0]

        # Move towards the mate
        if not self.pathfind_until(mate.position, delta_time, 65):
            self.assign_task(delta_time)
            return

        # Create a child if within range
        if self.position.distance_to(mate.position) <= 65:
            try:
                child_position = Vector2(self.position.x+25, self.position.y+25)
            except:
                child_position = Vector2(
                random.randint(0, self.entity_manager.map_size.x), 
                random.randint(0, self.entity_manager.map_size.y)
                )
            
            # Inherit attributes with some variability
            child_speed = random.choice([self.speed, mate.speed]) + random.uniform(-0.1, 0.1)
            
            child = RLAnimal(
                position=child_position,
                entity_manager=self.entity_manager,
                texture_name=self.texture_name,
                animal_type=self.animal_type,
                age=int(0),
                max_age=self.max_age,
                children=[],
                parents=[self, mate],
                task=None,
                resource_requirements=self.resource_requirements,
                speed=child_speed,
                prey=self.prey,
                resource_on_death=self.resource_on_death,
                resource_count_on_death=self.resource_count_on_death,
                reproduction_reward=self.reproduction_reward,
                living_reward=self.living_reward,
                gathering_reward=self.gathering_reward,
                hunting_reward=self.hunting_reward,
                death_by_hunger_reward=self.death_by_hunger_reward,
                hiding_reward = self.hiding_reward
            )
            
            if self.children is None:
                self.children = [child]
            else:
                self.children.append(child)

            for resource_name, resource_requirement in self.resource_requirements.items():
                usage = resource_requirement.ReproductionEnergyUsage
                child.states[0] = child_energy
                self.states[0] -= usage
                mate.states[0] -= mate_usage
            
            child
            # Assign initial task to the child and parents
            child.assign_task(delta_time)
            self.assign_task(delta_time)
            mate.assign_task(delta_time)
            self.entity_manager.entities.append(child)
            
            print(f"- - - - - - - - - - - - - - - ->Animal {self.animal_type} reproduced")
            self.learn_from_action(self.reproduction_reward, self.states, False)

    def hunt(self, delta_time):
        hiding_energy_loss = 0.005
        fail_energy_loss = 0.1
        hunt_energy_gain = 0.5
        hunt_energy_loss = 0.015
        hunt_hiding_distance = 60
        
        
        if self.prey == None:
            self.learn_from_action(0, self.states, False)
            self.assign_task(delta_time)
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
            self.learn_from_action(0, self.states, False)
            self.assign_task(delta_time)
            return

        
        targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
        if (self.target != targets[0] and not self.target in self.entity_manager.entities) or self.target == self:
            self.target = targets[0]
            self.target.states[2] = int(0)
        else:
            self.target.states[2] = int(1)
        
        if self.target.hiding:  # Check if the prey is hiding
            distance_to_prey = self.position.distance_to(self.target.position)
            if distance_to_prey > hunt_hiding_distance:
                self.pathfind_until(self.target.position, delta_time, hunt_hiding_distance)
            else:
                self.states[0] -= hiding_energy_loss
            return

        if self.pathfind_until(self.target.position, delta_time, 32):
            self.states[0] += hunt_energy_gain
            self.target.destroy()
            self.learn_from_action(self.hunting_reward, self.states, False)
            self.assign_task(delta_time)

        else:
            self.states[0] -= fail_energy_loss
            self.learn_from_action(0, self.states, False)
            self.assign_task(delta_time)
                    
    def escape(self, delta_time):
        escape_energy_cost = 0.006
        
        if self.states[2] == int(0):
            return
        predators = list()
        for entity in self.entity_manager.entities:
            if type(entity) is type(RLAnimal) and entity.target == self:
                predators.append(entity)
        if not predators:
            self.assign_task(delta_time)
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
        self.states[0] -= escape_energy_cost *delta_time
    
    def hide(self, delta_time):
        hide_energy_gain = 0.15
        hide_duration = 4

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
        if not targets:
            self.assign_task(delta_time)
            return
        else:
            targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
            if self.target != targets[0] or not type(self.target) is simulation.resources.Resource:
                self.target = targets[0]
                if self.states[2] == int(1) and random.random() > 0.4:
                    self.states[2] == int(0)
                    

            if self.pathfind_until(self.target.position,delta_time,45): #TODO: make this texture_size for bounding_box
                if not self.hiding:
                    self.hiding = True
                    self.hide_timer_main = hide_duration
                if self.hide_timer_main > 0: 
                    self.hide_timer_main -= delta_time
                elif self.hide_timer_main < 0:
                    self.states[0] += hide_energy_gain
                    self.hiding = False
                    self.target.destroy()
                    self.hide_timer_main = 0 
                    self.learn_from_action(hide_energy_gain, self.states, False)
                    self.assign_task(delta_time) 
    
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

        
