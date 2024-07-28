from simulation import entity_structures, resources, event_manager, output, clock
import matplotlib.pyplot as plt
import logging
import json
import arcade
import random
import os
from arcade.gui import *
import time
from datetime import datetime


with open("./data/settings.json") as settings_json: # Loading JSON settings file
    settings_json = json.load(settings_json)

font_path = "./data/fonts"
fonts = ["Ticketing", "novem___", "arcadeclassic", "lunchds"] #Font names
WINDOW_WIDTH = int(settings_json["window_width"])
WINDOW_HEIGHT = int(settings_json["window_height"])
WINDOW_TITLE = str(settings_json["window_title"])
MAP_WIDTH = int(settings_json["map_width"])
MAP_HEIGHT = int(settings_json["map_height"])
CLOCK_SPEED = int(settings_json["clock_speed"])
TRAINING_MODE = str(settings_json["training_mode"])
SPRITE_SCALE = 1
random.seed()
arcade.enable_timings()  # Enables Timing For FPS & STATS


def font_loader(fonts, font_path):
    for font in fonts:
        font_file = os.path.join(font_path, font + ".ttf")
        if os.path.exists(font_file):
            arcade.load_font(font_file)
            logging.info(f"Successfully Loaded Font: {font}.ttf")
        else:
            logging.error(f"Font file does not exist: {font}.ttf")


class Cloud(arcade.Sprite):
    def __init__(self, texture, scale):
        super().__init__(texture, scale=scale)

        self.change_x = random.uniform(0.3, 1.5) # Velocity
        self.position = (
            random.randrange(0, 2400),
            random.randrange(WINDOW_HEIGHT-320, WINDOW_HEIGHT-160)
        )

    def update(self):
        self.position = (
            self.position[0] + self.change_x,
            self.position[1] + self.change_y
        )
        if self.position[0] > WINDOW_WIDTH:
            self.center_x = 0
            self.center_y = random.randrange(WINDOW_HEIGHT-320, WINDOW_HEIGHT-160)


class MenuView(arcade.View):  # MENU VIEW
    def __init__(self):
        super().__init__()
        self.stats = output.Stats()

        self.texture_path = "./data/texture/MenuBackgrounds/"  # texture path

        backgrounds = ["menu_background (1).png"]  # Texture File Names 

        background = random.choice(backgrounds) # Loads Background Texture
        self.background_texture = arcade.load_texture(
            self.texture_path + background)

        self.heading_text = arcade.Text( # Creating text object for menu heading
            "Reinforcement Learning In 2D Ecosystems", self.window.width/2, self.window.height/2+250     ,
            arcade.color.WHITE, font_size=65,
            anchor_x="center",
            anchor_y="bottom",
            font_name="Ticketing")


        # Button Style
        button_style = {
            "normal": UIFlatButton.UIStyle(
                font_size=16,
                font_name="lunchtime doubly so",
                font_color=arcade.color.NAVAJO_WHITE,
                bg=arcade.color.TRANSPARENT_BLACK,
                border=None,
                border_width=2
            ),
            "hover": UIFlatButton.UIStyle(
                font_size=18,
                font_name="lunchtime doubly so",
                font_color=arcade.color.BLACK,
                bg=arcade.color.ANTIQUE_WHITE,
                border=arcade.color.EERIE_BLACK,
                border_width=3
            ),
            "press": UIFlatButton.UIStyle(
                font_size=18,
                font_name="lunchtime doubly so",
                font_color=arcade.color.WARM_BLACK,
                bg=arcade.color.LIGHT_GRAY,
                border=arcade.color.DAVY_GREY,
                border_width=3
            )
        }

        # UI Manager For Buttons
        self.ui_manager = arcade.gui.UIManager()
        self.ui_manager.enable()

        # Creates Vertical Box
        self.v_box = arcade.gui.widgets.layout.UIBoxLayout(
            space_between=15, align="center")  # Vertical Box

        # Creates Buttons
        simulation_button = arcade.gui.widgets.buttons.UIFlatButton(
            text="Start Simulation", width=450, height=53, style=button_style)
        exit_button = arcade.gui.widgets.buttons.UIFlatButton(
            text="Exit", width=450, height=53, style=button_style)

        # Adds Buttons To The Vertical Box
        self.v_box.add(simulation_button)
        self.v_box.add(exit_button)

        # Creates Widget
        ui_anchor_layout = arcade.gui.widgets.layout.UIAnchorLayout()
        ui_anchor_layout.add(child=self.v_box)
        self.ui_manager.add(ui_anchor_layout)

        # Button Click Events
        simulation_button.on_click = self.on_click_StartSimulation
        exit_button.on_click = self.on_click_quit

        #cute cloudies
        self.cloud_list = None
        self.cloud_textures = [self.texture_path+"cloud_texture (1).png", self.texture_path+"cloud_texture (2).png", self.texture_path+"cloud_texture (3).png"]
        self.arr_len = len(self.cloud_textures)
    
    def add_clouds(self, amount): 
        for i in range(amount):
            cloud = Cloud(self.cloud_textures[(random.randrange(0, self.arr_len))], random.uniform(1, 3))
            self.cloud_list.append(cloud)

    def setup(self):
        self.cloud_list = arcade.SpriteList(use_spatial_hash=True)
        self.add_clouds(15)
        arcade.schedule(self.update, 1/60)
    
    def update(self, delta_time):
        self.cloud_list.update()

    def on_draw(self):
        arcade.start_render()
        arcade.draw_texture_rectangle(
            self.window.width / 2, self.window.height / 2, self.window.width, self.window.height, self.background_texture)  # Draws Wallpaper
        self.cloud_list.draw()
        self.heading_text.draw()  # Renders Heading Text
        self.ui_manager.draw()  # Renders Buttons

    # Button Functions
    def on_click_StartSimulation(self, event):
        print("View Change To SimulationView")
        self.ui_manager.disable()  # Unloads buttons
        simulation_view = SimulationView(self.stats)          
        self.window.show_view(simulation_view)  # Changes View

    def on_click_quit(self, event):
        print("Quit button pressed")
        arcade.close_window()  # Quits Arcade


class SimulationView(arcade.View):
    def __init__(self, stats):
        super().__init__()
        self.stats = stats
        self.setup()  # Initialize the simulation

    def setup(self):
        self.fps_text = None
        self.arcade_texture_list = dict()
        self.path_to_data = os.path.join(".", "data")
        self.sprite_texture_path = "./data/texture/SpriteTexture/"  # Path To Sprite Texture
        with open(os.path.join(self.sprite_texture_path, "texture_list.json")) as texture_json:
            texture_list = json.load(texture_json)

        for texture in texture_list:
            self.arcade_texture_list[texture] = arcade.load_texture(os.path.join(
                self.sprite_texture_path, texture_list[texture]["texture_name"]))
            self.arcade_texture_list[texture].size = (texture_list[texture]["width"], texture_list[texture]["height"])
        self.clock = clock.Clock(CLOCK_SPEED)
        self.entity_manager = entity_structures.EntityManager(list(), entity_structures.Vector2(MAP_WIDTH, MAP_HEIGHT), self.clock, self.stats, self.arcade_texture_list)
        self.simulation_texture = arcade.load_texture(os.path.join(self.path_to_data, "texture", "SpriteTexture", "simulation_background.png"))
        self.resource_manager = resources.ResourceManager(self.path_to_data)
        for path in os.listdir(os.path.join(self.path_to_data, "animals")):
            with open(os.path.join(self.path_to_data, "animals", path)) as animal:
                decoded_animal = json.load(animal)
                self.stats.populations[decoded_animal["animal_type"]] = 0
                self.stats.populations_per_day[decoded_animal["animal_type"]] = list()
                entity_structures.RLAnimal.initialize_animals(decoded_animal, self.entity_manager)
        
        # event manager
        self.event_manager = event_manager.EventManager(self.entity_manager, self.resource_manager, self.clock, MAP_WIDTH, MAP_HEIGHT)

        self.population_text = arcade.Text(text="", font_size=25, x=480, y=WINDOW_HEIGHT-33, color=arcade.color.WHITE)
        self.fps_text = arcade.Text(text="", font_size=25, x=18, y=WINDOW_HEIGHT-33, color=arcade.color.BLACK)
        self.day_counter_text = arcade.Text(text="", font_size=25, x=200, y=WINDOW_HEIGHT-33, color=arcade.color.BLACK)
        if TRAINING_MODE == "True":
            self.training_text = arcade.Text(text="Training Mode Active", font_size=25, x=1200, y=WINDOW_HEIGHT-33, color=arcade.color.RED)
        else:
            self.training_text = arcade.Text(text="Training Mode InActive", font_size=25, x=1200, y=WINDOW_HEIGHT-33, color=arcade.color.RED)

    def on_draw(self):
        self.clear()
        arcade.start_render()
        arcade.draw_texture_rectangle(
            self.window.width / 2, self.window.height / 2, self.window.width, self.window.height, self.simulation_texture)
        self.entity_manager.sprite_list.draw()

        # Draw text objects
        self.population_text.draw()
        self.fps_text.draw()
        self.day_counter_text.draw()
        self.training_text.draw()
    
    def on_update(self, delta_time):
        for entity in self.entity_manager.entities:
            entity.update(delta_time)

        self.clock.tick(delta_time)
        if self.clock.new_day:
            for key, value in self.entity_manager.stats.populations.items():
                self.entity_manager.stats.populations_per_day[key].append(value)
        self.event_manager.update(delta_time)

        # Check populations and switch to menu view if necessary
        if TRAINING_MODE == "True":
            if not self.entity_manager.check_population():
                self.restart_simulation()

        # Update text content
        population_text = ""
        for animal_name, animal_population in self.stats.populations.items():
            population_text += f"{animal_name}: {animal_population}   "
        self.population_text.text = population_text
        self.fps_text.text = f"FPS: {round(arcade.get_fps())}"
        self.day_counter_text.text = f"Day Counter: {self.clock.day_counter}"


    def restart_simulation(self):
        print(''' -----> Restarting Simulation <-----
       -Training Mode enabled
              
              ''')
        self.save_population_data()
        self.setup()  # Reinitialize the simulation
    
    def save_population_data(self):
        # Get current time
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Create file name
        file_name = f'./output/population_data_{current_time}.json'

        # Gather data
        population_data = {
            'final_populations': self.stats.populations,
            'populations_per_day': self.stats.populations_per_day,
            'day_counter': self.clock.day_counter
        }

        # Save to JSON file
        with open(file_name, 'w') as json_file:
            json.dump(population_data, json_file, indent=4)

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            self.MenuView_Change()

    def MenuView_Change(self):
        print("View Change To MenuView")
        menu_view = MenuView()
        menu_view.setup()
        self.window.show_view(menu_view)


def main():  # MAIN FUNCTION
    window = arcade.Window(  # Creates window
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        title=WINDOW_TITLE,
        antialiasing=False,
        enable_polling=True,
        fullscreen=False     
        )

    font_loader(fonts, font_path)
    menu_view = MenuView()
    menu_view.setup()
    window.show_view(menu_view)  # Changes View To Menu
    arcade.run()

if __name__ == "__main__":
    main()