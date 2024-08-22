#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:21:52 2022

@author: panos
"""
"""
Program to simulate the orbital motion of N bodies using the Beeman numerical 
integration scheme
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Importing relevant modules

class Simulation(object):
    
    def read_input_data(self):
        """
        Method for reading in data from text file and for initialising list of 
        bodies to be simulated
        """
        self.G = 6.67408e-11 # Set G
        
        #filename = str(input("Enter the filename: "))
        filename = "projectdata.txt"
        filein = open(filename, "r")
        file = filein.readlines() 
        
        data = []
        
        for line in file:  
            if line.startswith("#"): 

                continue # Skipping lines starting with a #
            
            line = line.rstrip("\n") # Removing \n to avoid error in later code
            tokens = line.split("=")
            
            data.append(tokens[1]) # Append the data into data list
           
        filein.close()
        
        self.num_step = int(data[0])
        self.timestep = int(data[1])
        self.sun_mass = float(data[2]) # Setting values to be used throughout the simulation
        
        parameters = 6 # Number of parameters for each object
        self.split_data = [] # List of lists to hold data for each body
        
        self.counter = 0 # Number of bodies      
        self.bodies = [] # List to hold body objects
        
        for i in range(3, len(data), parameters): # Loop through data list skipping first 2 elements with step of 6 to append to split_data
            
            self.split_data.append(data[i : i + parameters]) # Index list at i to i + 5 to cover all parameters
            self.counter = self.counter + 1
        
        for i in range(0, self.counter):
            
            planet = Body() # Creating a body instance
        
            self.bodies.append(planet)
            self.bodies[i].initialise(self.split_data[i], self.G, self.sun_mass) # Call initialise method 
            
        for j in range(0, self.counter):
            
            self.bodies[j].initial_acceleration(self.bodies[j].name, self.bodies[j].position, self.bodies, self.G, self.counter)
            # Initialising acceleration
        for j in range(0, self.counter):
            
            self.bodies[j].initial_PE(self.bodies[j].name, self.bodies[j].position, self.bodies, self.G, self.counter)
            # Initialising potential energy
         
            
    def step_forward(self, t):
        """
        Method for updating position, acceleration and velocity in "lockstep" 
        for each time step in the simulation. Additionally updating history 
        of object positions
        """
        for j in range(self.counter):
            
            self.prev_y = self.bodies[j].position[1] # Set self.prev_y to previous y coord (for num orbits)
            
            self.bodies[j].position = self.bodies[j].update_position(self.bodies[j].current_acceleration, self.bodies[j].previous_acceleration, self.timestep)
            
            self.new_y = self.bodies[j].position[1] # Set self.new_y to new y coord (for num orbits)
            
            self.bodies[j].position_history = np.vstack((self.bodies[j].position_history, self.bodies[j].position)) 
            # Add to position history list to be used for animation
            
            self.bodies[j].time_elapsed = self.bodies[j].orbital_period(self.bodies[j], self.timestep, self.prev_y, self.new_y)
            # Find period if self.prev_y < 0 and self.new_y > 0
            
        for j in range(self.counter):
            
            self.bodies[j].new_acceleration = self.calc_acceleration(self.bodies[j].name, self.bodies[j].position)
        
        self.current_ke = 0
        for j in range(self.counter):       
                     
            self.bodies[j].velocity = self.bodies[j].update_velocity(self.bodies[j].new_acceleration, self.bodies[j].current_acceleration, self.bodies[j].previous_acceleration, self.timestep)
            v = np.linalg.norm(self.bodies[j].velocity)
            self.bodies[j].ke = self.bodies[j].calc_KE(self.bodies[j].mass, v) # Find ke from new velocity
            self.current_ke = self.current_ke + self.bodies[j].ke # Add to total system ke at this timestep
            
        for j in range(self.counter):
            
            self.bodies[j].previous_acceleration = self.bodies[j].current_acceleration
            self.bodies[j].current_acceleration = self.bodies[j].new_acceleration
            # Set current to previous and new to current for each body            
            
        self.current_pe = 0
        for j in range(self.counter):
            
            self.bodies[j].pe = self.calc_PE(self.bodies[j].name, self.bodies[j].position, self.bodies[j].mass)
            self.current_pe = self.current_pe + self.bodies[j].pe
            # Find and add pe of each body to total system pe at that timestep
        
        
    def run_simulation(self):
        """
        Method for running the simulation, writing to a file the orbital 
        periods, satellite travel/return time and closest distance to Mars
        """  
       
        self.total_energy = []
        self.time = []
        with open('Total_energy.txt', 'w') as file:
            for t in range(self.num_step):
            
                self.step_forward(t)
                self.calc_total_energy(self.current_ke, self.current_pe)
                self.time.append(t * self.timestep)
                
                if (t/500).is_integer() == True : 
                    file.writelines("The total system energy is :" + str(self.current_total) + " Joules.")
                    file.write('\n')             
                    # Write to file total system energy every 500 iterations
                    
        periods = [] # List of periods for each body    
        
        for i in range(self.counter): 
            
            if self.bodies[i].num_orbits == 0: # So not to write period for body with no number of orbits
                continue
            
            periods.append(self.bodies[i].time_elapsed/(self.bodies[i].num_orbits*60*60*24*365.25))
            # Add to list of periods the period of body in Earth years
            
        self.satellite_sim(self.bodies[3].position_history, self.bodies[4].position_history, self.bodies[5].position_history)
        # Call satellite sim method with Earth, Mars and Satellite position histories
        
        with open('Project_output_data.txt', 'w') as file:
            
            for i in range(len(periods)): # To write number of bodies' periods to file
                
                print("The orbital period of " + self.bodies[int(i+1)].name + " is: " + str(periods[i]) + " Earth years.")
                
                file.writelines("The orbital period of " + self.bodies[int(i+1)].name + " is " + str(periods[i]) + " Earth years.")
                file.write('\n')                             
                
            file.writelines("The travel time for the satellite to get to Mars from Earth is " + str(self.travel_time) + " months.")
            file.write('\n')
            file.writelines("The closest distance the satellite gets to Mars is " + str(self.min_d) + " meters.")
            file.write('\n') 
            
            if self.min_d2 <= (self.bodies[5].x_coord-self.bodies[3].x_coord): # Return condition is if min distance <= to starting position
                file.writelines("The return time of the satellite is " + str(self.return_time) + " Earth years.")
               
            else:
                file.writelines("The satellite does not return to Earth in the duration of this simulation.")
 
    def calc_PE(self, name, position, mass):
        """
        Method for calculating the potential energy of each body due to all 
        other bodies in the system for each timestep
        """
        self.pe = 0
        for i in range(self.counter):
            
            if name != self.bodies[i].name: # Not calculate pe of one body due to itself
                
                r = position - self.bodies[i].position
                
                abs_r = np.linalg.norm(r)
                
                self.pe = self.pe -(1/2) * self.G * mass * self.bodies[i].mass / (abs_r)
        
        return self.pe
    
    def calc_total_energy(self, ke, pe):
        """
        Method for calculating the total energy of the system at each timestep
        of the simulation
        """
        self.current_total = ke + pe
        self.total_energy.append(self.current_total)
           
    
    def calc_acceleration(self, name, position):
        """
        Method for calculating the acceleration of a body due to all other
        bodies in the system using the Gravitational acceleration formula
        """       
        self.acceleration = np.array([0,0])
        for i in range(self.counter): # Calc acceleration for body i due to rest of bodies j
                
            if name != self.bodies[i].name: # Avoiding calc acceleration due to itself
                                                    
                r = position - self.bodies[i].position # Accessing bodies' .position attribute
                
                abs_r = np.linalg.norm(r)
             
                self.acceleration = self.acceleration - self.G * self.bodies[i].mass / ((abs_r) ** 3) * r
            
        return self.acceleration
     
    def animate(self, i):
        """
        Method for creating animation
        """
        for j in range(self.counter):
            
            self.patches[j].center = self.bodies[j].position_history[i]
            # Setting patches.center to body j's position at iteration i
        return self.patches
    
    def display(self):
        """
        Method for running animation using animate method and for displaying 
        the animation 
        """
        figure = plt.figure() 
        axes = plt.axes() 
        # Creating figure and axes to plot animation on
        
        self.patches = []
        
        for i in range(self.counter):
            
            self.patches.append(plt.Circle((self.bodies[i].position_history[0]), self.bodies[i].radius, color = self.bodies[i].colour, label = self.bodies[i].name, animated = True))
        
        for i in range(len(self.patches)):
            axes.add_patch(self.patches[i]) 
            
        plt.axis("scaled")
        axes.set_title("Simulation of The Solar System")
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        axes.set_facecolor("black")
        
        plt.xlim([-3e11, 3e11])
        plt.ylim([-3e11, 3e11])       
        # Setting axes parameters
        
        self.anim = FuncAnimation(figure, self.animate, frames = self.num_step, repeat = True, interval = 1, blit = True)
        axes.legend(fontsize=5, loc = "upper right")
        plt.show()
        
        plt.figure()
        axes2 = plt.axes()
        axes2.plot(self.time, self.total_energy) # Plot time against total energy
        axes2.set_xlabel("Time (s)")
        axes2.set_ylabel("Total Energy of System (J)")
        axes2.set_title("Total Energy of System vs Time")
        
        plt.show()
        
    def satellite_sim(self, earth_list, mars_list, satellite_list ):
        """
        Method for the satellite simulation, finding the travel 
        time from Earth to Mars, the minimum distance it gets 
        to Mars and return time to Earth, if ever
        """
        self.sat_mars_d = []
        self.sat_earth_d = []
        
        for i in range(len(mars_list)):
            
            mars_pos = mars_list[i] # Set mars_pos to i'th element in Mars position history list
            
            sat_pos = satellite_list[i] # Set sat_pos to i'th element in Satellite position history list
            
            distance = mars_pos - sat_pos # Distance between them both
            self.sat_mars_d.append(np.linalg.norm(distance)) # Append norm of distance to list of distances
        
        self.min_d = min(self.sat_mars_d) # Use min() function to find the min value in the list of distances
        iteration = self.sat_mars_d.index(self.min_d) # Use index() function to find index of min distance
        time = iteration * self.timestep # Find time elapsed at the index/ iteration that simulation is on
        self.travel_time = time/(60*60*24*30.437) # Convert from seconds into months (30.437 is avg days in a month)       
    
        for i in range(iteration, len(earth_list)): # Loop only from after satellite reaches Mars
            
            sat_pos2 = satellite_list[i]
            earth_pos = earth_list[i]
            distance2 = sat_pos2 - earth_pos
            self.sat_earth_d.append(np.linalg.norm(distance2))
            
        self.min_d2 = min(self.sat_earth_d)
        iteration2 = self.sat_earth_d.index(self.min_d2)
        time2 = iteration2 * self.timestep
        self.return_time = time2/(60*60*24*30.437*12)
        # Above use same logic as for Satellite and Mars
        
        if self.min_d2 <= (self.bodies[5].x_coord-self.bodies[3].x_coord): # Condition for if Satellite returns to Earth
            
            return self.min_d2
            return self.return_time
        
        return self.travel_time
        return self.min_d   
    
class Body(object):
    
    def initialise(self, sublist, G, sun_mass): 
        """
        Method for initialising planet (object) attributes using data read in 
        from file
        """
        self.name = str(sublist[0])
        self.mass = float(sublist[1])
        self.x_coord = float(sublist[2])
        self.y_coord = float(sublist[3])
        self.colour = str(sublist[4])
        self.radius = float(sublist[5]) # Radius of the patch to be animated
        
        if self.name != "SATELLITE": # Conditional for setting Satellite launch velocity
            
            self.x_velocity = 0  
        else:
            self.x_velocity = 11495 # Initial satellite x velocity in m/s

        self.position_history = []
                
        if self.x_coord == 0: # Avoid /0 error for sun
            self.y_velocity = 0 
            
        else:            
            self.y_velocity = math.sqrt(G * sun_mass/ self.x_coord)
        
        if self.name == "SATELLITE": # Conditional for setting Satellite launch velocity
            
            self.y_velocity = 30000 # Initial satellite y velocity in m/s
            
        self.position = np.array([self.x_coord, self.y_coord])
        
        self.velocity = np.array([self.x_velocity, self.y_velocity])
        
        self.new_acceleration = np.array([0,0])
        
        self.position_history = np.array([self.x_coord, self.y_coord])
        
        self.time_elapsed = 0
        
        self.num_orbits = 0
        
        self.niter = 0 # Number of iterations since last orbit completion 
        
        self.ke = (1/2) * self.mass * np.linalg.norm(np.array([self.x_velocity, self.y_velocity]))
        # Above lines setting initial attributes for each body
        
    def initial_acceleration(self, name, position, body_list, G, counter):
        """
        Method for calculating the initial acceleration of the body using the 
        Gravitational acceleration formula
        """
        self.acceleration = np.array([0,0])
        for i in range(counter): # Calc acceleration for body i due to rest of bodies j
                
            if name != body_list[i].name: # Avoiding calc acceleration due to itself
                                                    
                r = position - body_list[i].position # Accessing bodies' .position attribute
                
                abs_r = np.linalg.norm(r)
       
                self.acceleration = self.acceleration - G * body_list[i].mass / ((abs_r) ** 3) * r
            
        self.previous_acceleration = self.acceleration
        self.current_acceleration = self.previous_acceleration
        # At t=0, set previous acceleration to current acceleration
        
    def update_position(self, current_acceleration, previous_acceleration, timestep):
        """
        Method for updating the position of the body using Beenam numerical
        integration scheme
        """
        self.position = self.position + self.velocity * timestep + (1/6) * \
            (4 * current_acceleration - previous_acceleration) * timestep ** 2
            
        return self.position
        
    def update_velocity(self, new_acceleration, current_acceleration, previous_acceleration, timestep):
        """
        Method for updating the velocity of the body using Beenam numerical
        integration scheme
        """
        self.velocity = self.velocity + (1/6) * \
            (2 * new_acceleration + 5 * current_acceleration - previous_acceleration) * timestep
            
        return self.velocity
    
    def orbital_period(self, body, timestep, prev_y, new_y):    
        """
        Method for calculating the orbital period of each body in the 
        simulation
        """
        if prev_y < 0 and new_y > 0:
 
            self.num_orbits = self.num_orbits + 1

            self.time_elapsed = self.time_elapsed + self.niter * timestep
            self.niter = 0
        else:
            
            self.niter = self.niter + 1 # Updating number of iterations since last orbit completion 
   
        return self.time_elapsed
        return self.num_orbits
        return self.niter
    
    def calc_KE(self, mass, velocity):
        """
        Method for calculating the kinetic energy of each body
        """
        self.ke = (1/2) * mass * (velocity ** 2)
    
        return self.ke
    
    def initial_PE(self, name, position, body_list, G, counter):
        """
        Method for calculating the initial gravitational potential energy of 
        each body due to all the other bodies
        """
        self.pe = 0
        
        for i in range(counter):
            
            if name != body_list[i].name:
                
                r = position-body_list[i].position
                abs_r = np.linalg.norm(r)
                
                self.pe = self.pe -(1/2) * G * self.mass * body_list[i].mass / (abs_r)

        return self.pe