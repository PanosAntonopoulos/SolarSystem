#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:11:37 2022

@author: panos
"""
"""
Test program to run simulation 
"""

from FinalProject import Simulation # Importing the class

def main():
    
    test = Simulation() # Create Simulation object
    test.read_input_data() # Read in inpput data
    test.run_simulation() # Run the simulation
    test.display() # Display the results
    
main() 