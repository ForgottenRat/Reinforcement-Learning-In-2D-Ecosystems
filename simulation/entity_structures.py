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
import os
import json
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Select gpu for torch

@dataclass
class SimulationSummary:
    days: int
    daily_population: List[Dict[str, int]]
    daily_animal_data: List[Dict[str, List[Dict[str, any]]]]


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
    summary: SimulationSummary = None

    
    def __post_init__(self):
        self.sprite_list = arcade.SpriteList(use_spatial_hash=False)
        self.summary = SimulationSummary(days=0, daily_population=[], daily_animal_data=[])

    def check_population(self):
        for population in self.stats.populations.values():
            if population == 0:
                self.generate_summary_json()
                return False
        return True
    
    def assign_task_for_all_animals(self, delta_time):
        for entity in self.entities:
            if isinstance(entity, RLAnimal):
                entity.assign_task(delta_time)

    def update(self, delta_time):
        self.remove_old_resources()
    
    def remove_old_resources(self):
        current_day = self.clock.day_counter
        lifespan = 2  # Lifespan of resources in days
        for entity in list(self.entities):  # Create a copy of the list to avoid modification during iteration
            if isinstance(entity, simulation.resources.Resource):
                if current_day - entity.last_accessed_day > lifespan and random.random()>0.5:
                    entity.destroy()
                
            

    def record_daily_data(self):
        day_data = {
            'day': self.clock.day_counter,
            'population': {k: v for k, v in self.stats.populations.items()},
            'animals': []
        }

        for entity in self.entities:
            if isinstance(entity, RLAnimal):
                animal_data = {
                    'animal_type': entity.animal_type,
                    'animal_id': entity.id,
                    'states': entity.states,
                    'task': entity.task
                }
                day_data['animals'].append(animal_data)
        
        self.summary.daily_animal_data.append(day_data)
        self.summary.days += 1

    def generate_summary_json(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"./output/simulation_summary_{current_time}.json"
        with open(file_name, 'w') as json_file:
            json.dump(self.summary.__dict__, json_file, indent=4)
        print(f"Simulation summary saved to {file_name}")


    


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
        self.sprite = arcade.Sprite(self.entity_manager.textures[self.texture_name],1.35,self.position.x,self.position.y, hit_box_algorithm=None)
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
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, output_size)
        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def save_model(self, file_path):
        try:
            torch.save(self.state_dict(), file_path)
        
        except:
            print("FAILED TO SAVE MODEL")

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
    model_deer: None
    optimizer_deer: None
    criterion_deer: None
    model_lion: None
    optimizer_lion: None
    criterion_lion: None
    id: str
    
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
    escaping_reward: int

    #Reproduction Control
    reproduction_count: int = 0
    last_reproduction_day: int = -1


    def __post_init__(self):
        super().__post_init__()
        

        self.entity_manager.stats.populations[self.animal_type] += 1
        self.resource_count = dict.fromkeys(self.resource_requirements,0)
        self.target = None

        self.task_history = deque(maxlen=2)
        self.task_history.append(-1)
        self.task_history.append(-1)
        #Task: 0Wander, 1gather, 2reproduce, 3hunt, 4escape, 5hide
        self.task = [int(0), int(0), int(0), int(0),int(0), int(0)]
        #States: 0Living energy, 1Rep En, 2Chased, 3Age, 4Rep Urg, 5Food Prox, 6Predator Prox, 7Prey Prox
        self.states = [float(1), int(0), int(0), int(0), float(0), float(0), float(0), float(0)]
        self.hiding = False
        self.epsilon: float = 0.9  # Exploration rate
        self.reproduction_timer_main: float = 0.0
        
    @classmethod
    def initialize_animals(self, animal, entity_manager):
        # Load configurations
        import simulation.resources
        self.input_size = 8  # State size
        self.hidden_size = 64
        self.output_size = 6  # Task size
        path_deer = "./data/model/Deer_model.pth"
        model_deer = DQNetwork(self.input_size, self.hidden_size, self.output_size).to(device)
        
        model_lion = DQNetwork(self.input_size, self.hidden_size, self.output_size).to(device)
        if os.path.exists(path_deer):
            model_deer_w = torch.load(path_deer)
            model_deer.load_state_dict(model_deer_w)

        path_lion = "./data/model/Lion_model.pth"
        if os.path.exists(path_lion):
            model_lion_w = torch.load(path_lion)
            model_lion.load_state_dict(model_lion_w)
        
        model_deer.eval()
        optimizer_deer = optim.SGD(model_deer.parameters(), lr=0.01, momentum=0.9)
        criterion_deer = nn.BCEWithLogitsLoss()

        model_lion.eval()
        optimizer_lion = optim.SGD(model_lion.parameters(), lr=0.01, momentum=0.9)
        criterion_lion = nn.BCEWithLogitsLoss()

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
                    hiding_reward = animal["hiding_reward"],
                    escaping_reward= animal["escaping_reward"],
                    model_deer = model_deer,
                    model_lion = model_lion,
                    optimizer_deer = optimizer_deer,
                    optimizer_lion = optimizer_lion,
                    criterion_deer = criterion_deer,
                    criterion_lion = criterion_lion,
                    id = str(random.randint(0,999))
            )
        
    def update(self, delta_time):
        self.sprite.position = (self.position.x, self.position.y)
        for entity in self.entity_manager.entities:
            if isinstance(entity, RLAnimal):
                entity.animal_update(delta_time)
        if self.entity_manager.clock.new_day:
            print("#=====-+-- Day number:", self.entity_manager.clock.day_counter,"--+-====#")
            self.entity_manager.clock.new_day = False
            self.entity_manager.assign_task_for_all_animals(delta_time)
            self.entity_manager.record_daily_data()
            self.entity_manager.update(delta_time)
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

        q_values = model(state_vector)  # Shape: [1, 6]
        next_q_values = model(next_state_vector)  # Shape: [1, 6]

        max_next_q_value = next_q_values.max(1)[0].detach()  # Shape: []
        target_q_value = reward + (0.99 * max_next_q_value * (1 - done))  # Shape: []

        # Ensure q_values[0, self.task] and target_q_value have the same shape
        target_q_value = target_q_value.unsqueeze(0)  # Shape: [1]
        q_value = q_values[0, self.task].unsqueeze(0)  # Shape: [1]

        loss = torch.nn.functional.mse_loss(q_value, target_q_value)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        if done:
            if self.animal_type == "Deer":
                model.save_model(f"./data/model/Deer_model.pth")
            elif self.animal_type == "Lion":
                model.save_model(f"./data/model/Lion_model.pth")
        
        # print("Q VALUES:", q_values)
        # print(f"Updated Q-value for task {self.task}: {q_value}")
        # print(f"Target Q-value: {target_q_value}")
        # print(f"Loss: {loss.item()}")

    def animal_update(self, delta_time):
        if self.animal_type == "Deer":
            model = self.model_deer
        elif self.animal_type == "Lion":
            model = self.model_lion
        else:
            return
        
        for resource_name, resource_requirement in self.resource_requirements.items():
            if self.states[0] > 1:
                self.states[0] = 1
            
            if self.entity_manager.clock.new_day:
                if self.states[0] > 0:
                    self.states[0] -= resource_requirement.DailyEnergyUsageRate
                    self.states[3] += 1
                    self.states[4] += 0.1
                    self.age +=1
                    self.learn_from_action(self.living_reward, self.states, False)
                    if self.entity_manager.clock.day_counter == 1:
                        self.age = 1
                # Define population thresholds for reproduction adjustments
                population_threshold = 6  # Minimum population level to boost reproduction urge

                # Different high population thresholds for deer and lion
                high_population_deer = 34
                high_population_lion = 11

                # Reproduction urge adjustments
                reproduction_urge_boost = 0.4  # Amount to increase the reproduction urge
                reproduction_urge_decrease = 0.2  # Amount to decrease the reproduction urge

                # Get the current population for the animal type
                current_population = self.entity_manager.stats.populations[self.animal_type]

                # Adjust reproduction urge based on population size
                if current_population < population_threshold:
                    # Boost reproduction urge when population is low
                    self.states[4] += reproduction_urge_boost

                # Check for high population and decrease reproduction urge accordingly
                if self.animal_type == "Deer" and current_population > high_population_deer:
                    self.states[4] -= reproduction_urge_decrease

                if self.animal_type == "Lion" and current_population > high_population_lion:
                    self.states[4] -= reproduction_urge_decrease

                # Ensure the reproduction urge stays within bounds
                if self.states[4] > 1.0:
                    self.states[4] = 1.0
                if self.states[4] < 0.0:
                    self.states[4] = 0.0  # Optional: ensure it doesn't go below 0
                    
            if self.states[0] <= 0:
                self.states[0] = 0
                self.learn_from_action(self.death_by_hunger_reward,self.states,True)
                self.destroy()
                return


            # Check reproduction conditions and availability of a mate
            available_mate = any(
                isinstance(entity, RLAnimal) and
                entity.animal_type == self.animal_type and
                entity is not self and
                entity.age >= 3 and
                entity.states[1] >= 1
                for entity in self.entity_manager.entities
            )

            if (
                self.states[0] > resource_requirement.ReproductionEnergyUsage and
                self.reproduction_count <= 1 and
                (self.entity_manager.clock.day_counter + 1) > self.last_reproduction_day and
                self.age >= 5
            ):
                self.states[1] = int(1)
            else:
                self.states[1] = int(0)



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

        # Update food proximity
        food_sources = [entity for entity in self.entity_manager.entities
                        if isinstance(entity, simulation.resources.Resource)]  # Adjust condition to match your food resource entities
        if food_sources:
            closest_food_distance = min(food.position.distance_to(self.position) for food in food_sources)
            self.states[5] = closest_food_distance
        else:
            self.states[5] = float(-1)  # No food available

        # Update predator proximity
        predators = [entity for entity in self.entity_manager.entities
                    if isinstance(entity, RLAnimal) and entity.animal_type != self.animal_type
                    and entity.prey and self.animal_type in entity.prey]  # Adjust condition to match your predator entities
        if predators:
            closest_predator_distance = min(predator.position.distance_to(self.position) for predator in predators)
            self.states[6] = closest_predator_distance
        else:
            self.states[6] = float(-1)  # No predators nearby

        # Update prey proximity
        prey_list = [entity for entity in self.entity_manager.entities
                    if isinstance(entity, RLAnimal) and entity.animal_type in self.prey]  # Adjust condition to match your prey entities
        if prey_list:
            closest_prey_distance = min(prey.position.distance_to(self.position) for prey in prey_list)
            self.states[7] = closest_prey_distance
        else:
            self.states[7] = float(-1)  # No prey nearby

        for population in self.entity_manager.stats.populations.values():
            if population == 0:
                if self.animal_type == "Deer":
                    model.save_model(f"./data/model/Deer_model.pth")
                elif self.animal_type == "Lion":
                    model.save_model(f"./data/model/Lion_model.pth")

    def assign_task(self, delta_time):
        state_vector = torch.tensor([self.states], dtype=torch.float).to(device)
        if self.animal_type == "Deer":
            task_probabilities = self.model_deer(state_vector)
        elif self.animal_type == "Lion":
            task_probabilities = self.model_lion(state_vector)
        else:
            task_probabilities = torch.zeros(self.output_size).to(device)
            print("Unidentified Animal for task assign")

        file_name = "cycle_num"
        cycle_num = 0
        if os.path.exists(file_name):
            # Open the file in read mode ('r')
            with open(file_name, 'r') as file:
                # Read the contents of the file
                content = file.read()
                # Convert the read string back to an integer
                cycle_num = int(content)
                # Optionally, print the read integer to verify

        if cycle_num!=0:
            epsilon_temp = self.epsilon / (cycle_num/15)

        if random.random() < epsilon_temp:
            # Explore: select a random task
            self.task = random.randint(0, 5)
        else:
            # Exploit: select the task with the highest Q-value
            topk_values, topk_indices = torch.topk(task_probabilities, 2)
            primary_task = topk_indices[0][0].item() 
            # secondary_task = topk_indices[0][1].item() 
            # if primary_task not in self.task_history:
            self.task = primary_task
            # else:
            #     self.task = secondary_task
        
        #Reproduction Urge
        if self.states[4] >= 0.6:
            if random.random() < 0.7:
                self.task = 2
        
        # self.task_history.append(self.task)
        self.states = [round(state, 2) for state in self.states]
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
        wander_energy_cost = float(0.0001)
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
        gather_energy_gain = 0.25
        
        if self.animal_type == "Lion":
            self.learn_from_action(self.gathering_reward, self.states, False)
            self.task = 0
        
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
        x = 0 # Keep track of how many it ate
        if not targets:
            self.learn_from_action(-10, self.states, False)
            return
        
        else:
            targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
            if self.target != targets[0] or not type(self.target) is simulation.resources.Resource:
                self.target = targets[0]
            if self.pathfind_until(self.target.position,delta_time,45): #TODO: make this texture_size for bounding_box
                x += 1
                self.states[0] += gather_energy_gain
                self.target.destroy()
                self.learn_from_action(self.gathering_reward, self.states, False)
                if x > 1:
                    self.task = 0
                
    def reproduce(self, delta_time):
        import simulation.resources
        
        if self.states[4] < 0.3:
            self.learn_from_action(-10, self.states, False)
            self.task = 0

        # Start the reproduction timer if it hasn't been started yet
        if self.reproduction_timer_main > 0:
            self.reproduction_timer_main -= delta_time
            if self.reproduction_timer_main <= 0:
                self.reproduction_timer_main = 0
            self.task = 0
            return

        for resource_name, resource_requirement in self.resource_requirements.items():
            usage = resource_requirement.ReproductionEnergyUsage
        if self.states[1] == 0:
            self.learn_from_action(-10, self.states, False)
            self.task = 0
            return
            
        child_energy = 0.5

        # Select potential mates
        potential_mates = [entity for entity in self.entity_manager.entities 
                        if isinstance(entity, RLAnimal) 
                        and entity.animal_type == self.animal_type 
                        and entity is not self 
                        and entity.states[1] == 1
                        and entity.age >= 3]  # Ensure the mate is also ready to reproduce

        if not potential_mates:
            self.learn_from_action(-10, self.states, False)
            self.task = 0
            return

        # Sort mates by proximity and select the closest
        potential_mates.sort(key=lambda x: x.position.distance_to(self.position))
        mate = potential_mates[0]

        # Move towards the mate
        if mate not in self.entity_manager.entities:
            self.task = 0
            return

        # Create a child if within range
        if self.pathfind_until(mate.position, delta_time, 32):
            self.states[4] -= 0.5
            try:
                child_position = Vector2(self.position.x+25, self.position.y+25)
            except:
                child_position = Vector2(
                random.randint(0, self.entity_manager.map_size.x), 
                random.randint(0, self.entity_manager.map_size.y)
                )
            
            # Inherit attributes with some variability
            child_speed = random.choice([self.speed, mate.speed]) + random.uniform(-0.15, 0.1)
            child_id = str(self.id) + "_" + str(mate.id) + "_" + str(random.randint(0, 999))

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
                hiding_reward = self.hiding_reward,
                escaping_reward = self.escaping_reward,
                model_deer = self.model_deer,
                model_lion = self.model_lion,
                optimizer_deer = self.optimizer_deer,
                optimizer_lion = self.optimizer_lion,
                criterion_deer = self.criterion_deer,
                criterion_lion = self.criterion_lion,
                id = child_id
            )
            
            if self.children is None:
                self.children = [child]
            else:
                self.children.append(child)

            for resource_name, resource_requirement in self.resource_requirements.items():
                usage = resource_requirement.ReproductionEnergyUsage
                child.states[0] = child_energy
                self.states[0] -= usage
                mate.states[0] -= usage
            
            # Assign initial task to the child and parents
            child.task = 0
            self.task = 0
            mate.task = 0

            self.reproduction_count += 1
            self.last_reproduction_day = self.entity_manager.clock.day_counter
            
            # Set the reproduction timer to 5 seconds (similar to how hiding works)
            self.reproduction_timer_main = 5.0
            
            print(f"- - - - - - - - - - - - - - - ->Animal {self.animal_type} reproduced")
            self.learn_from_action(self.reproduction_reward, self.states, False)

    def hunt(self, delta_time):
        hiding_energy_loss = 0.0001
        fail_energy_loss = 0.05
        hunt_energy_gain = 0.55
        hunt_energy_loss = 0.0001
        hunt_hiding_distance = 85
        day_count = self.entity_manager.clock.day_counter
    
        if self.prey == None:
            self.learn_from_action(self.hunting_reward, self.states, False)
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
            self.learn_from_action(-10, self.states, False)
            self.task = 0
            return

        
        targets.sort(key=lambda x:x.position.distance_to(self.position),reverse=False)
        if (self.target != targets[0] and not self.target in self.entity_manager.entities) or self.target == self:
            self.target = targets[0]
            self.target.states[2] = int(0)
        else:
            self.target.states[2] = int(1)
        
        if self.target.hiding and random.random() > 0.004:  # Check if the prey is hiding
            distance_to_prey = self.position.distance_to(self.target.position)
            if distance_to_prey > hunt_hiding_distance:
                self.pathfind_until(self.target.position, delta_time, hunt_hiding_distance)
                self.states[0] -= hunt_energy_loss

        else:
            if self.pathfind_until(self.target.position, delta_time, 32):
                self.states[0] += hunt_energy_gain
                self.target.learn_from_action(-10, self.target.states, False)
                self.target.destroy()
                self.learn_from_action(self.hunting_reward, self.states, False)
                self.task = 0
            
            elif day_count < self.entity_manager.clock.day_counter or self.target not in self.entity_manager.entities:
                self.states[0] -= fail_energy_loss
                self.learn_from_action(0, self.states, False)
                self.task = 0

            else:
                self.states[0] -= hunt_energy_loss

            
                    
    def escape(self, delta_time):
        escape_energy_cost = 0.002
        
        if self.animal_type == "Lion":
            self.learn_from_action(self.escaping_reward, self.states, False)
            self.task = 0
            return

        if self.states[2] == int(0):
            self.learn_from_action(-10, self.states, False)
            self.task = 0
            return
        predators = list()
        for entity in self.entity_manager.entities:
            if type(entity) is type(RLAnimal) and entity.target == self:
                predators.append(entity)
        if not predators:
            self.learn_from_action(-10, self.states, False)
            self.task = 0
            self.states[2] == int(0)
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

        if all(predator.position.distance_to(self.position) > 45 for predator in predators):
            self.learn_from_action(self.escaping_reward, self.states, False)
            self.states[2] = int(0)
            self.task = 0
    
    def hide(self, delta_time):
        hide_energy_gain = 0.35
        hide_duration = 7
        if self.animal_type == "Lion":
            self.learn_from_action(self.escaping_reward, self.states, False)
            self.task = 0
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
            self.learn_from_action(-10, self.states, False)
            self.task = 0
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
                    self.learn_from_action(self.hiding_reward, self.states, False)
                    self.task = 4
    
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


