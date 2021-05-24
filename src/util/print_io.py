"""
print_io.py
Helper script used by sim.py to print out simulation data with pretty colors
"""

#############################     COLORS    #############################
# Static class to hold color constants
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Global colors dictionary
scolors = {'HEADER' : bcolors.HEADER,
           'OKBLUE' : bcolors.OKBLUE,
           'OKGREEN' : bcolors.OKGREEN,
           'WARNING' : bcolors.WARNING,
           'FAIL' : bcolors.FAIL,
           'ENDC' : bcolors.ENDC,
           'BOLD' : bcolors.BOLD,
           'UNDERLINE' : bcolors.UNDERLINE}

# Global dictionary for STATUS -> COLOR
status_colors = {'GREEN' : bcolors.OKGREEN,
                 'YELLOW' : bcolors.WARNING,
                 'RED' : bcolors.FAIL,
                 '---' : bcolors.OKBLUE}

#############################      I/O     #############################
# Print that the simulation started
def print_sim_header():
    print(bcolors.HEADER + bcolors.BOLD+ "\n***************************************************")
    print("************    SIMULATION STARTED     ************")
    print("***************************************************\n\n" + bcolors.ENDC)

# Print when a new step is starting
def print_sim_step(step_num):
    print(bcolors.HEADER + bcolors.BOLD + "\n--------------------- STEP " + str(step_num) + " ---------------------\n" + bcolors.ENDC)

# Print a line to separate things
def print_seperator(color=bcolors.HEADER):
    print(color + bcolors.BOLD + "\n------------------------------------------------\n" + bcolors.ENDC)

# Print header update
def update_header(msg, clr=bcolors.BOLD):
    print(clr + "--------- " + msg + ' update' + bcolors.ENDC)

# Print header update
def print_msg(msg, clrs=['HEADER']):
    for clr in clrs:
        print(scolors[clr])
    print("---- " + msg + bcolors.ENDC)

# Print interpreted mission status
def print_interpreted_status(brain):
    print_seperator(bcolors.OKBLUE)
    print("INTERPRETED MISSION STATUS: " + str(format_status(brain.interpreted_status)))
    print_seperator(bcolors.OKBLUE)

# Print diagnosis info
def print_diagnosis(diagnosis):
    status_list = diagnosis.get_status_list()
    tree_traversal = diagnosis.fault_tree
    activations = diagnosis.current_activations
    print_seperator()
    print(bcolors.HEADER + bcolors.BOLD + "DIAGNOSIS INFO: \n" + bcolors.ENDC)
    for status in status_list:
        stat = status[1]
        print(status[0] + ': ' + format_status(stat))

    # print(bcolors.HEADER + bcolors.BOLD + "\nFAULT PATHS: \n" + bcolors.ENDC)
    # for path in tree_traversal:
    #     for level in range(len(path)):
    #         print('  ' + '-'*3*level + bcolors.FAIL + path[level] + bcolors.ENDC)
    #     print('\n')

    print(bcolors.HEADER + bcolors.BOLD + "\nCURRENT ACTIVATIONS: \n" + bcolors.ENDC)
    if len(activations) > 0:
        for activation in activations:
            print('---' + str(activation))
    print_seperator()

# Print subsystem status
def subsystem_status_str(ss):
    s = bcolors.BOLD + '[' + str(ss.type)+ '] : ' + bcolors.ENDC
    stat = ss.get_status()
    s = s + '\n' + status_colors[stat] + ' ---- ' + str(stat) + bcolors.ENDC + ' (' + str(ss.uncertainty) + ')'
    return s + '\n'

# Print out subsystem information
def subsystem_str(ss):
    s = bcolors.BOLD + ss.type + '\n' + bcolors.ENDC
    s = s + '--[headers] '
    for h in ss.headers:
        s = s + '\n---' + str(h)
    s = s + '\n--[tests] '
    for t in ss.tests:
        s = s + '\n---' + str(t)
    s = s + '\n--[test data] '
    for d in ss.test_data:
        s = s + '\n---' + str(d)
    return s

# Print out headers
def headers_string(headers):
    s = ''
    for hdr in headers:
        s = s + '\n  -- ' + hdr
    return s

# Print out status
def format_status(stat):
    if type(stat) == str:
        return status_colors[stat] + stat + scolors['ENDC']
    else: 
        s = '('
        for status in stat:
            s = s + format_status(status) + ', '
        s = s[:-2] + ')'
        return s    

