# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
Agent Class
Deals with supervised learning for diagnosing statuses
"""
from ..ai_components.learners_interface import LearnersInterface
from ..ai_components.planners_interface import PlannersInterface
from ..reasoning.complex_reasoning_interface import ComplexReasoningInterface
from ..reasoning.diagnosis import Diagnosis

class Agent:
    def __init__(self, vehicle, learners_plugin_list, planners_plugin_list, complex_plugin_list):

        self.vehicle_rep = vehicle
        self.mission_status = self.vehicle_rep.get_status()
        self.bayesian_status = self.vehicle_rep.get_bayesian_status()

        # AI Interfaces
        self.learning_systems = LearnersInterface(self.vehicle_rep.get_headers(),learners_plugin_list)
        self.planning_systems = PlannersInterface(self.vehicle_rep.get_headers(),planners_plugin_list)
        self.complex_reasoning_systems = ComplexReasoningInterface(self.vehicle_rep.get_headers(),complex_plugin_list)

      
    def render_reasoning(self):
        return self.complex_reasoning_systems.render_reasoning()

    def reason(self, frame):
        self.vehicle_rep.update(frame) 
        self.learning_systems.update(frame, self.vehicle_rep.get_state_information(['status']))
        self.planning_systems.update(self.vehicle_rep.get_state_information('PDDL_state')) 
        
        
        aggregate_high_level_info = {'vehicle_rep' : self.vehicle_rep.get_state_information(),
                                     'learning_systems' : self.learning_systems.render_reasoning(),
                                     'planning_systems' : self.planning_systems.render_reasoning()}

        self.complex_reasoning_systems.update(aggregate_high_level_info)
        
        return self.render_reasoning()
        # Does this need further separation?
  

    def diagnose(self, time_step):
        """ Grab the mnemonics from the """
        learning_system_results = self.learning_systems.render_reasoning()
        diagnosis = Diagnosis(time_step, 
                              learning_system_results,
                              self.bayesian_status,
                              self.vehicle_rep.get_current_faulting_mnemonics())
        return diagnosis.perform_diagnosis()

