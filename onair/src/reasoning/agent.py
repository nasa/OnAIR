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
    def __init__(self, vehicle, learners_plugin_dict, planners_plugin_dict, complex_plugin_dict):

        self.vehicle_rep = vehicle
        self.mission_status = self.vehicle_rep.get_status()
        self.bayesian_status = self.vehicle_rep.get_bayesian_status()

        # AI Interfaces
        self.learning_systems = LearnersInterface(self.vehicle_rep.get_headers(),learners_plugin_dict)
        self.planning_systems = PlannersInterface(self.vehicle_rep.get_headers(),planners_plugin_dict)
        self.complex_reasoning_systems = ComplexReasoningInterface(self.vehicle_rep.get_headers(),complex_plugin_dict)

    def reason(self, frame):
        aggregate_high_level_info = {}
        self.vehicle_rep.update(frame)
        aggregate_high_level_info['vehicle_rep'] = self.vehicle_rep.get_state_information()
        self.learning_systems.update(self.vehicle_rep.curr_data, aggregate_high_level_info)
        aggregate_high_level_info['learning_systems'] = self.learning_systems.render_reasoning()
        self.planning_systems.update(aggregate_high_level_info)
        aggregate_high_level_info['planning_systems'] = self.planning_systems.render_reasoning()

        return self.complex_reasoning_systems.update_and_render_reasoning(aggregate_high_level_info)

    def diagnose(self, time_step):
        """ Grab the mnemonics from the """
        learning_system_results = self.learning_systems.render_reasoning()
        diagnosis = Diagnosis(time_step,
                              learning_system_results,
                              self.bayesian_status,
                              self.vehicle_rep.get_current_faulting_mnemonics())
        return diagnosis.perform_diagnosis()
