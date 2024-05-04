import numpy as np
import json, os, docker, scipy

class TrafficModel:
    def __init__(self, params_file):
        # params_file can be a dict or a path to a json file
        if isinstance(params_file, str):
            with open(params_file, 'r') as fp:
                params = json.load(fp)
        else:
            params = params_file

        self.const_int_vars = [
            "max_street_length", 
            "min_street_length",
            "sidewalk_height",
            "crosswalk_pos",
            "crosswalk_width",
            "max_speed_car",
            "max_speed_ped", 
            "car_width",
            "car_height", 
            "car_y"
        ]
        self.int_vars = [
            "max_accel_car",
            "slippery_range_start",
            "slippery_range_end",
            "car_x_init",
            "car_v_init"
        ]
        self.float_vars = [
            "slippery_factor"
        ]

        if self.check_params(params):
            self.params = params
        else:
            assert False, "Some parameters are not fine"

    def check_params(self, params):
        
        for i in self.const_int_vars:
            if i not in params:
                return False
            if not isinstance(params[i], int):
                return False
        
        return True

    def produce_mdp(self):
        mdp_string = "mdp\n\n"
        for i in self.const_int_vars:
            mdp_string += f"const int {i} = {self.params[i]};\n"
        
        mdp_string += f"formula is_slippery = (car_x > {self.params['slippery_range_start']}) & (car_x < {self.params['slippery_range_end']});\n"
        mdp_string += f"formula crash = (turn=1) &  (car_v > 0) & ((ped_x >= car_x+car_width-car_v) & (ped_x <= car_x + car_width)) & ((ped_y >= car_y - car_height/2) & (ped_y <= car_y + car_height/2));\n" # TODO: check correctness of the formula
        mdp_string += f"formula not_let_pass = (car_x > ped_x) & (ped_y < sidewalk_height + crosswalk_width -1);\n"

        mdp_string += f"formula accelerate = true;\n"
        mdp_string += f"formula brake = true;\n"
        mdp_string += f"formula noop = true;\n"

        mdp_string += "global turn : [0..1] init 0;\n"
        mdp_string += "global crashed : [0..1] init 0;\n"




        

        acc_probs = np.zeros(self.params["max_accel_car"]+1)
        acc_probs[0] = 0
        acc_probs[1] = 0.5
        for i in range(2,len(acc_probs)):
            acc_probs[i] = 0.5**(i-1)
        acc_probs = acc_probs/np.sum(acc_probs)

        acc_probs_slippery = acc_probs.copy()
        for i in range(1,len(acc_probs_slippery)):
            acc_probs_slippery[i] = acc_probs[i]/(np.power(2, i+self.params["slippery_factor"])-1)
        acc_probs_slippery[0] = 1-np.sum(acc_probs_slippery[1:])
        

        ## MODULE CAR
        mdp_string += "\nmodule Car\n"
        mdp_string += f"  car_x : [min_street_length..max_street_length] init {self.params['car_x_init']};\n"
        mdp_string += f"  car_v : [0..max_speed_car] init {self.params['car_v_init']};\n"
        mdp_string += f"  finished : [0..1] init 0;\n"

        basic_guard = "  [] (turn = 0) & (finished = 0) & (car_x < max_street_length) & (crashed = 0)"

        mdp_string += f"{basic_guard} & (!is_slippery) & (accelerate) -> \n"
        for i in range(len(acc_probs)):
            mdp_string += f"  {acc_probs[i]} : (car_v' = min(max_speed_car, car_v + {i})) & (car_x' = min(max_street_length, car_x + car_v + {i})) & (turn' = 1) + "
        mdp_string = mdp_string[:-3] + ";\n\n"

        mdp_string += f"{basic_guard} & (is_slippery) & (accelerate) -> \n"
        for i in range(len(acc_probs)):
            mdp_string += f"  {acc_probs_slippery[i]} : (car_v' = min(max_speed_car, car_v + {i})) & (car_x' = min(max_street_length, car_x + car_v + {i})) & (turn' = 1) + "
        mdp_string = mdp_string[:-3] + ";\n\n"

        mdp_string += f"{basic_guard} & (!is_slippery) & (accelerate) -> \n"
        for i in range(len(acc_probs)):
            mdp_string += f"  {acc_probs[i]} : (car_v' = min(max_speed_car, car_v + {i})) & (car_x' = min(max_street_length, car_x + min(max_speed_car, car_v + {i}))) & (turn' = 1) + "
        mdp_string = mdp_string[:-3] + ";\n\n"

        mdp_string += f"{basic_guard} & (is_slippery) & (accelerate) -> \n"
        for i in range(len(acc_probs)):
            mdp_string += f"  {acc_probs_slippery[i]} : (car_v' = min(max_speed_car, car_v + {i})) & (car_x' = min(max_street_length, car_x + min(max_speed_car, car_v + {i}))) & (turn' = 1) + "
        mdp_string = mdp_string[:-3] + ";\n\n"

        mdp_string += f"{basic_guard} & (!is_slippery) & (brake) -> \n"
        for i in range(len(acc_probs)):
            mdp_string += f"  {acc_probs[i]} : (car_v' = max(0, car_v - {i})) & (car_x' = min(max_street_length, car_x + max(0, car_v - {i}))) & (turn' = 1) + "
        mdp_string = mdp_string[:-3] + ";\n\n"

        mdp_string += f"{basic_guard} & (is_slippery) & (brake) -> \n"
        for i in range(len(acc_probs)):
            mdp_string += f"  {acc_probs_slippery[i]} : (car_v' = max(0, car_v - {i})) & (car_x' = min(max_street_length, car_x + max(0, car_v - {i}))) & (turn' = 1) + "
        mdp_string = mdp_string[:-3] + ";\n\n"


        mdp_string += f"{basic_guard} & (noop) -> \n"
        
        mdp_string += f"  0.5 : (car_v' = max(0, car_v - 1)) & (car_x' = min(max_street_length, car_x + car_v)) & (turn' = 1) + " 
        mdp_string += "0.5 : (car_x' =  min(max_street_length, car_x + car_v)) & (turn' = 1);\n\n"

        mdp_string += "  [] (turn = 0) & (finished = 0) & ((car_x = max_street_length) | (crashed=1)) -> (finished'=1);\n"
        mdp_string += "  [] (turn = 0) & (finished = 1) -> true;\n"
        mdp_string += "\nendmodule\n\n"

        mdp_string += "formula on_road = (ped_y >= sidewalk_height) & (ped_y < sidewalk_height + crosswalk_width);\n"
        mdp_string += "formula before_cross = (ped_y < sidewalk_height);\n"
        mdp_string += "formula after_cross = ped_y = sidewalk_height + crosswalk_width;\n"

        pedestrian_speeds = np.zeros(self.params["max_speed_ped"]+1)
        n = self.params["max_speed_ped"]
        mu = self.params["avg_ped_vel"]
        sigma = self.params["std_ped_vel"]**2
        alpha = -mu*(mu**2 - n*mu + sigma)/(mu**2 - n*mu + n*sigma)
        beta = (mu - n)*(mu**2 - n*mu + sigma)/(mu**2 - n*mu + n*sigma)
        assert alpha > 0, "Alpha < 0, make sure that (mu/n)(n-mu) < sigma < mu(n-mu)"
        assert beta > 0, "Beta < 0, make sure that (mu/n)(n-mu) < sigma < mu(n-mu)"
        randomvar = scipy.stats.betabinom(n, alpha, beta)
        for i in range(len(pedestrian_speeds)):
            pedestrian_speeds[i] = randomvar.pmf(i)


        mdp_string += "\nmodule Ped\n"
        mdp_string += f"  ped_x : [min_street_length..max_street_length] init {self.params['ped_x_init']};\n"
        mdp_string += f"  ped_y : [0..sidewalk_height + crosswalk_width] init {self.params['ped_y_init']};\n\n"

        mdp_string += f"[] (turn = 1) & (crash) -> (crashed'=1) & (turn'=0);\n"

        basic_guard = "  [] (turn = 1) & (!crash)"
        
        mdp_string += f"{basic_guard} & (after_cross) -> \n"
        p_left = 0.3
        for i in range(len(pedestrian_speeds)):
            mdp_string += f"  {p_left*pedestrian_speeds[i]} : (ped_x' = max(min_street_length, ped_x - {i})) & (turn' = 0) + "
            mdp_string += f"{(1-p_left)*pedestrian_speeds[i]} : (ped_x' = min(max_street_length, ped_x + {i})) & (turn' = 0) + "
        mdp_string = mdp_string[:-3] + ";\n\n"

        mdp_string += f"{basic_guard} & (before_cross) -> \n"
        p_left = 0.3
        p_up = 0.35
        for i in range(len(pedestrian_speeds)):
            for j in range(len(pedestrian_speeds)):
                mdp_string += f"  {p_left*p_up*pedestrian_speeds[i]*pedestrian_speeds[j]} : "
                mdp_string += f"(ped_x' = max(min_street_length, ped_x - {i})) & "
                mdp_string += f"(ped_y' = min(sidewalk_height + crosswalk_width, ped_y + {j})) & (turn' = 0) + "
                mdp_string += f"  {(1-p_left)*p_up*pedestrian_speeds[i]*pedestrian_speeds[j]} : "
                mdp_string += f"(ped_x' = min(max_street_length, ped_x + {i})) & "
                mdp_string += f"(ped_y' = min(sidewalk_height + crosswalk_width, ped_y + {j})) & (turn' = 0) + "
                mdp_string += f"  {p_left*(1-p_up)*pedestrian_speeds[i]*pedestrian_speeds[j]} : "
                mdp_string += f"(ped_x' = max(min_street_length, ped_x - {i})) & "
                mdp_string += f"(ped_y' = max(0, ped_y - {j})) & (turn' = 0) + "
                mdp_string += f"  {(1-p_left)*(1-p_up)*pedestrian_speeds[i]*pedestrian_speeds[j]} : "
                mdp_string += f"(ped_x' = min(max_street_length, ped_x + {i})) & "
                mdp_string += f"(ped_y' = max(0, ped_y - {j})) & (turn' = 0) + "
        mdp_string = mdp_string[:-3] + ";\n\n"

        mdp_string += f"{basic_guard} & (on_road) -> \n"
        p_left = 0.5
        p_up = 0.85
        for i in range(len(pedestrian_speeds)):
            for j in range(len(pedestrian_speeds)):
                mdp_string += f"  {p_left*p_up*pedestrian_speeds[i]*pedestrian_speeds[j]} : "
                mdp_string += f"(ped_x' = max(min_street_length, ped_x - {i})) & "
                mdp_string += f"(ped_y' = min(sidewalk_height + crosswalk_width, ped_y + {j})) & (turn' = 0) + "
                mdp_string += f"  {(1-p_left)*p_up*pedestrian_speeds[i]*pedestrian_speeds[j]} : "
                mdp_string += f"(ped_x' = min(max_street_length, ped_x + {i})) & "
                mdp_string += f"(ped_y' = min(sidewalk_height + crosswalk_width, ped_y + {j})) & (turn' = 0) + "
                mdp_string += f"  {p_left*(1-p_up)*pedestrian_speeds[i]*pedestrian_speeds[j]} : "
                mdp_string += f"(ped_x' = max(min_street_length, ped_x - {i})) & "
                mdp_string += f"(ped_y' = max(0, ped_y - {j})) & (turn' = 0) + "
                mdp_string += f"  {(1-p_left)*(1-p_up)*pedestrian_speeds[i]*pedestrian_speeds[j]} : "
                mdp_string += f"(ped_x' = min(max_street_length, ped_x + {i})) & "
                mdp_string += f"(ped_y' = max(0, ped_y - {j})) & (turn' = 0) + "
        mdp_string = mdp_string[:-3] + ";\n\n"

        




        mdp_string += "\nendmodule\n\n"
        return mdp_string

def main():
    TM = TrafficModel("params_files/params_example.json")
    mdp_string = TM.produce_mdp()
    mdp_temp_path = "tmp/mdp.pm"
    with open(mdp_temp_path, 'w') as fp:
        fp.write(mdp_string)

    prism_path = "/home/fcano/IAIKWork/Accountability/prism-4.7-linux64/bin/prism"
    path_length = 100
    trace_filepath = "trace.txt"
    os.system("{} {} -simpath {} {} >{}".format(prism_path, mdp_temp_path, path_length, trace_filepath, "merda.txt"))
    

    client = docker.from_env()     
    aux = client.containers.run("lposch/tempest-devel-traces:latest", f"storm --prism {mdp_temp_path} --prop prism_files/mdp_props.pm", volumes = {os.getcwd(): {'bind': '/mnt/vol1', 'mode': 'rw'}}, working_dir = "/mnt/vol1", stderr = True)
    outstr = aux.decode("utf-8")
    print(outstr)
    # os.remove('aux.pm')

    # aux = client.containers.run("lposch/tempest-devel-traces:latest", "storm --prism aux.pm --prop prism_files/mdp_props.pm --trace-input trace.txt --exportresult mdpprops.json --buildstateval", volumes = {os.getcwd(): {'bind': '/mnt/vol1', 'mode': 'rw'}}, working_dir = "/mnt/vol1", stderr = True)

if __name__ == "__main__":
    main()
