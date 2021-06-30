"""
TelemetryTestSuite Class
Handles telemetry mnemonic testing
"""

# from collections import Counter
from src.data_handling.data_source import DataSource
from src.systems.status import Status
from collections import Counter


class TelemetryTestSuite:
    def __init__(self, headers=[], tests=[]):
        self.dataFields = headers
        self.tests = tests
        self.latest_results = None
        self.epsilon = 0.00001 # should define this intelligently 
        self.all_tests = {'SYNC' : self.sync,
                    'ROTATIONAL' : self.rotational, 
                         'STATE' : self.state,
                   'FEASIBILITY' : self.feasibility, 
                           'NOOP': self.noop}

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

            if test_name == 'SYNC':
                test_data = [sync_data[var] for var in test_data if var in sync_data.keys()]

            stat, mass_assignments = self.all_tests[test_name](test_val, test_data, self.epsilon)
            status.append(stat) # tuple

        bayesian = self.calc_single_status(status)
        # Add all mass assignments?
        return Status(self.dataFields[header_index], bayesian[0], bayesian[1])


    def get_latest_result(self, fieldName):
        if self.latest_results == None:
            return None
        hdr_index = self.dataFields.index(fieldName)
        return self.latest_results[hdr_index]

    ################################################
    ################   Test Suites  ################ 

    def sync(self, val, test_params, epsilon):
        
        mass_assignments = [] # list of tuples
        if len(test_params) == 0:
            stat = 'RED'
            mass_assignments.append(({stat}, 1.0))
            return stat, mass_assignments
        if val == test_params[0]:
            stat = 'GREEN'
            mass_assignments.append(({stat}, 1.0))
            return stat, mass_assignments

        return '---', mass_assignments

    def rotational(self, val, test_params, epsilon):
        stat = 'YELLOW'
        mass_assignments = []
        return stat, mass_assignments

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

        # if val in test_params:
        #     index = test_params.index(val)
        #     left_status = 


        if val <= lowest_bound: 
            stat = statuses[0]
            left_stat = statuses[1]
            l_range = lowest_bound - delta 

            if val == lowest_bound:
                mass_assignments.append(({stat, left_stat}, 1.0))
            else:
                if val < l_range:
                    mass = 1.0
                    mass_assignments.append(({stat}, mass))
                else:
                    mass = abs(lowest_bound - val)/delta
                    red_yellow_mass = 1.0 - mass
                    mass_assignments.append(({stat}, mass))
                    mass_assignments.append(({stat, left_stat}, red_yellow_mass))
                    
        elif val >= highest_bound:
            stat = statuses[len(statuses)-1]
            right_stat = statuses[len(statuses)-2]
            u_range = highest_bound + delta 

            if val == highest_bound:
                mass_assignments.append(({stat, right_stat}, 1.0))
            else:
                if val > u_range:
                    mass = 1.0
                    mass_assignments.append(({stat}, mass))
                else:
                    mass = abs(highest_bound - val)/delta
                    red_yellow_mass = 1.0 - mass
                    mass_assignments.append(({stat}, mass))
                    mass_assignments.append(({stat, right_stat}, red_yellow_mass))

        else:
            for i in range(0, len(test_params) - 1): #This may need to change... 
                l_bound = test_params[i]
                u_bound = test_params[i+1]
                left_stat = statuses[i]
                stat = statuses[i+1]
                right_stat = statuses[i+2]
                lb_buffer = l_bound + delta 
                ub_buffer = u_bound - delta 

                if l_bound < val < u_bound:
                    # Lower bound 
                    if val < lb_buffer:
                        mass = abs(l_bound - val)/delta
                        mass_assignments.append(({stat}, mass))
                        mass_assignments.append(({stat, left_stat}, 1.0 - mass))
                    # Upper bound 
                    elif val > ub_buffer:
                        mass = abs(u_bound - val)/delta
                        mass_assignments.append(({stat}, mass))
                        mass_assignments.append(({stat, right_stat}, 1.0 - mass))
                    else:
                        mass = 1.0
                        mass_assignments.append(({stat}, mass))
                else:
                    if val == l_bound:
                        mass = 1.0
                        mass_assignments.append(({left_stat, stat}, mass))
                    # elif val == u_bound:
                    #     mass = 1.0
                    #     mass_assignments.append(({right_stat, stat}, mass))

        return stat, mass_assignments

    def noop(self, val, test_params, epsilon):
        stat = 'GREEN'
        mass_assignments = [({stat}, 1.0)]
        return stat, mass_assignments


    ################################################
    ############## Combining statuses ############## 
    def calc_single_status(self, status_list, mode='max'):
        occurences = Counter(status_list)
        max_occurence = occurences.most_common(1)[0][0]

        if mode == 'strict':
            if occurences['RED'] > 0:
                conf = occurences['RED']/len(status_list)
                # conf = -1.0
                return 'RED', conf

        if mode == 'distr':
            conf = occurences[max_occurence]/len(status_list)
            return max_occurence, conf

        elif mode == 'max':
            return max_occurence, 1.0
        else: 
            return max_occurence, 1.0 # return max 

    def get_suite_status(self):
        return self.calc_single_status([res.get_status() for res in self.latest_results]) 


# class TestResult:
#     def __init__(self, stat, bayesian_conf):
#         self.stat = stat
#         self.bayesian_conf  = bayesian_conf

#     def get_all_test_results(self):
#         return self.stat, self.bayesian_conf

#     def get_stat(self):
#         return self.stat

#     def get_bayesian_conf(self):
#         return self.bayesian_conf
