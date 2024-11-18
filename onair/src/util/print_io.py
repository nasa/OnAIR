# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.print_string. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
print_io.py
Helper script used by sim.py to print out simulation data with pretty colors
"""


#############################     COLORS    #############################
# Static class to hold color constants
BCOLORS = {
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}


# Global colors dictionary
SCOLORS = {
    "HEADER": BCOLORS["HEADER"],
    "OKBLUE": BCOLORS["OKBLUE"],
    "OKGREEN": BCOLORS["OKGREEN"],
    "WARNING": BCOLORS["WARNING"],
    "FAIL": BCOLORS["FAIL"],
    "ENDC": BCOLORS["ENDC"],
    "BOLD": BCOLORS["BOLD"],
    "UNDERLINE": BCOLORS["UNDERLINE"],
}

# Global dictionary for STATUS -> COLOR
STATUS_COLORS = {
    "GREEN": BCOLORS["OKGREEN"],
    "YELLOW": BCOLORS["WARNING"],
    "RED": BCOLORS["FAIL"],
    "---": BCOLORS["OKBLUE"],
}


#############################      I/O     #############################
def print_sim_header():
    """
    Print that the simulation started
    """
    print(
        BCOLORS["HEADER"]
        + BCOLORS["BOLD"]
        + "\n***************************************************"
    )
    print("************    SIMULATION STARTED     ************")
    print("***************************************************" + BCOLORS["ENDC"])


def print_sim_step(step_num):
    """
    Print when a new sim step is starting
    """
    print(
        BCOLORS["HEADER"]
        + BCOLORS["BOLD"]
        + "\n--------------------- STEP "
        + str(step_num)
        + " ---------------------\n"
        + BCOLORS["ENDC"]
    )


def print_separator(color=BCOLORS["HEADER"]):
    """
    Print a line to separate things
    """
    print(
        color
        + BCOLORS["BOLD"]
        + "\n------------------------------------------------\n"
        + BCOLORS["ENDC"]
    )


def update_header(msg, clr=BCOLORS["BOLD"]):
    """
    Print header update
    """
    print(clr + "--------- " + msg + " update" + BCOLORS["ENDC"])


def print_msg(msg, clrs=None):
    """
    Print a message
    """
    if clrs is None:
        clrs = ["HEADER"]
    for clr in clrs:
        print(SCOLORS[clr])
    print("---- " + msg + BCOLORS["ENDC"])


def print_system_status(agent, data=None):
    """
    Print interpreted system status
    """
    # print_separator(BCOLORS["OKBLUE"])
    if data is not None:
        print("CURRENT DATA: " + str(data))
    print("INTERPRETED SYSTEM STATUS: " + str(format_status(agent.mission_status)))
    # print_separator(BCOLORS["OKBLUE"])


def print_diagnosis(diagnosis):
    """
    Print diagnosis info
    """
    status_list = diagnosis.get_status_list()
    activations = diagnosis.current_activations
    print_separator()
    print(BCOLORS["HEADER"] + BCOLORS["BOLD"] + "DIAGNOSIS INFO: \n" + BCOLORS["ENDC"])
    for status in status_list:
        stat = status[1]
        print(status[0] + ": " + format_status(stat))

    print(
        BCOLORS["HEADER"]
        + BCOLORS["BOLD"]
        + "\nCURRENT ACTIVATIONS: \n"
        + BCOLORS["ENDC"]
    )
    if len(activations) > 0:
        for activation in activations:
            print("---" + str(activation))
    print_separator()


def subsystem_status_str(subsystem):
    """
    Print subsystem status
    """
    print_string = (
        BCOLORS["BOLD"] + "[" + str(subsystem.type) + "] : " + BCOLORS["ENDC"]
    )
    stat = subsystem.get_status()
    print_string = (
        print_string
        + "\n"
        + STATUS_COLORS[stat]
        + " ---- "
        + str(stat)
        + BCOLORS["ENDC"]
        + " ("
        + str(subsystem.uncertainty)
        + ")"
    )
    return print_string + "\n"


def subsystem_str(subsystem):
    """
    Print out subsystem information
    """
    print_string = BCOLORS["BOLD"] + subsystem.type + "\n" + BCOLORS["ENDC"]
    print_string = print_string + "--[headers] "
    for header in subsystem.headers:
        print_string = print_string + "\n---" + str(header)
    print_string = print_string + "\n--[tests] "
    for test in subsystem.tests:
        print_string = print_string + "\n---" + str(test)
    print_string = print_string + "\n--[test data] "
    for data in subsystem.test_data:
        print_string = print_string + "\n---" + str(data)
    return print_string


def headers_string(headers):
    """
    Print out headers
    """
    print_string = ""
    for hdr in headers:
        print_string = print_string + "\n  -- " + hdr
    return print_string


def format_status(stat):
    """
    Print out status
    """
    if isinstance(stat, str):
        return STATUS_COLORS[stat] + stat + SCOLORS["ENDC"]
    print_string = "("
    for status in stat:
        print_string = print_string + format_status(status) + ", "
    print_string = print_string[:-2] + ")"
    return print_string
