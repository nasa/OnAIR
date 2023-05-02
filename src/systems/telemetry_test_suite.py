 
"""
TelemetryTestSuite Class
Handles telemetry mnemonic testing
"""

# from collections import Counter
from src.systems.status import Status
from collections import Counter

class TelemetryTestSuite:
    def __init__(self, headers=[], tests=[]):
        self.dataFields = headers
        self.tests = tests
        self.latest_results = None
        self.epsilon = 0.00001 # should define this intelligently 
        self.all_tests = {'STATE' : self.state,
                          'FEASIBILITY' : self.feasibility,
                          'NOOP' : self.noop}

    ################################################
    ################  Running Tests  ############### 

    def execute_suite(self, updated_frame, sync_data={}):
        results = []
        for i in range(len(updated_frame)):
            results.append(self.run_tests(i, updated_frame[i], sync_data))
        self.latest_results = results

    def run_tests(self, header_index, test_val, sync_data):
        status = []
        tests = self.tests[header_index]

        for test in tests:
            test_name = test[0]
            test_data = test[1:]

            stat, mass_assignments = self.all_tests[test_name](test_val, test_data, self.epsilon)
            status.append(stat) # tuple

        bayesian = self.calc_single_status(status)
        # TODO: Add all mass assignments?
        return Status(self.dataFields[header_index], bayesian[0], bayesian[1])


    def get_latest_result(self, fieldName):
        if self.latest_results == None:
            return None
        hdr_index = self.dataFields.index(fieldName)
        return self.latest_results[hdr_index]

    ################################################
    ################   Test Suites  ################ 

    def state(self, val, test_params, epsilon):
        mass_assignments = []
        val = float(val)
        green = test_params[0]
        yellow = test_params[1]
        red = test_params[2]

        if int(val) in green:
            stat = 'GREEN'
            mass_assignments.append(({stat}, 1.0))
            return stat, mass_assignments
        if int(val) in yellow:
            stat = 'YELLOW'
            mass_assignments.append(({stat}, 1.0))
            return stat, mass_assignments
        if int(val) in red:
            stat = 'RED'
            mass_assignments.append(({stat}, 1.0))
            return stat, mass_assignments
        
        stat = '---'
        mass_assignments.append(({'RED', 'YELLOW', 'GREEN'}, 1.0))
        return stat, mass_assignments

    def feasibility(self, val, test_params, epsilon):
        '''
        Test_Params : threshold ranges an attribute should fall in
        # if len(test_params == 4) then the thresholds are as follows:
        # before [0] is red
        # between [0] - [1] is yellow
        # between [1] - [2] is green 
        # between [2] - [3] is yellow
        # after [3] is red

        # if len(test_params == 2) then the thresholds are as follows:
        # before [0] is red
        # between [0] - [1] is green
        # after [1] is red
        '''
        assert( (len(test_params) == 2) or (len(test_params) == 4))

        stat = '---'
        mass_assignments = []

        val = float(val)

        lowest_bound = test_params[0]
        highest_bound = test_params[len(test_params)-1]
        deltas = [abs(test_params[i+1] - test_params[i]) for i in range(0, len(test_params) - 1)]
        delta = epsilon * min(deltas)

        statuses = ['RED', 'YELLOW', 'GREEN', 'YELLOW', 'RED']
        if len(test_params) == 2:
            statuses = ['RED', 'GREEN', 'RED']

        #Lower boundary values   stat: red   stat_right:  green / yellow
        if val <= lowest_bound: 
            stat = statuses[0]
            right_stat = statuses[1]

            l_range = lowest_bound - delta 

            if val == lowest_bound:
                mass_assignments.append(({stat, right_stat}, 1.0))
            else:
                if val < l_range:
                    mass_assignments.append(({stat}, 1.0))
                else:
                    mass = abs(lowest_bound - val)/delta
                    mass_assignments.append(({stat}, mass))

                    red_yellow_mass = 1.0 - mass
                    mass_assignments.append(({stat, right_stat}, red_yellow_mass))
        # Upper boundary values     stat : red  stat_left: green/yellow      
        elif val >= highest_bound:
            stat = statuses[len(statuses)-1]
            left_stat = statuses[len(statuses)-2]

            u_range = highest_bound + delta 

            if val == highest_bound:
                mass_assignments.append(({left_stat, stat}, 1.0))
            else:
                if val > u_range:
                    mass_assignments.append(({stat}, 1.0))
                else:
                    mass = abs(highest_bound - val)/delta
                    mass_assignments.append(({stat}, mass))

                    red_yellow_mass = 1.0 - mass
                    mass_assignments.append(({left_stat, stat}, red_yellow_mass))
        #Between boundaries
        else:
            for i in range(0, len(test_params) - 1): #This may need to change... 
                l_bound = test_params[i]
                u_bound = test_params[i+1]

                left_stat = statuses[i] 
                temp_mid_stat = statuses[i+1]  
                right_stat = statuses[i+2] 

                lb_buffer = l_bound + delta 
                ub_buffer = u_bound - delta 

                if l_bound < val < u_bound:
                    # Lower bound 
                    if val < lb_buffer:
                        stat = temp_mid_stat
                        mass = abs(l_bound - val)/delta                        
                        mass_assignments.append(({stat}, mass))
                        mass_assignments.append(({left_stat, stat}, 1.0 - mass))
                    # Upper bound 
                    elif val > ub_buffer:
                        stat = temp_mid_stat
                        mass = abs(u_bound - val)/delta
                        mass_assignments.append(({stat}, mass))
                        mass_assignments.append(({stat, right_stat}, 1.0 - mass))
                    else:
                        stat = temp_mid_stat
                        mass_assignments.append(({stat}, 1.0))
                else:
                    if val == l_bound:
                        stat = temp_mid_stat
                        mass_assignments.append(({left_stat, stat}, 1.0))
                    #elif val == u_bound:
                    #   stat = temp_mid_stat
                    #    mass_assignments.append(({stat, right_stat}, 1.0))
        return stat, mass_assignments

    def noop(self, val, test_params, epsilon):
        stat = 'GREEN'
        mass_assignments = [({stat}, 1.0)]
        return stat, mass_assignments


    ################################################
    ############## Combining statuses ############## 
    def calc_single_status(self, status_list, mode='strict'):
        occurrences = Counter(status_list)
        max_occurrence = occurrences.most_common(1)[0][0]

        if mode == 'strict':
            if occurrences['RED'] > 0:
                conf = occurrences['RED']/len(status_list)
                return 'RED', conf
            else:
                return max_occurrence, 1.0 # return max 

        if mode == 'distr':
            conf = occurrences[max_occurrence]/len(status_list)
            return max_occurrence, conf

        elif mode == 'max':
            return max_occurrence, 1.0
        else: 
            return max_occurrence, 1.0 # return max 

    def get_suite_status(self):
        status_strings = [res.get_status() for res in self.latest_results]
        return self.calc_single_status(status_strings) 

    def get_status_specific_mnemonics(self, status='RED'):
        names = [res.get_name() for res in self.latest_results if res.get_status() == status]
        return names


