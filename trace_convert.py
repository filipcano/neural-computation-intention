import re
import json


def main(input_prism_trace = "trace.txt", output_tempest_trace = 'tmp/path.txt', USE_VISIBILITY=False):

    file = open(input_prism_trace, "r")
    lines = [line.strip().replace('-', 'Pedestrian') for line in file.readlines()]
    file.close()

    label_ped = re.compile("Pedestrian")
    label_car = re.compile("Car")

        # make each array item into sep array
    path_lines = [item.split(" ") for item in lines]
    labels = path_lines[0]
    path_lines = path_lines[1:]
    dict = {}
    for i, label in enumerate(labels):
        dict[label] = [row[i] for row in path_lines]

    dict.pop('action')
    dict.pop('step')

    count = 0
    arr = []
    # trace_dict = {}
    length = len(dict['turn'])
    for j, line in enumerate(lines):
        # print([dict[key][j] for key in dict])
        label_match1 = label_ped.search(line)
        label_match2 = label_car.search(line)
        if (label_match1 == None) | (label_match2 == None):
            lines[j] = ""
        if ((label_match1 != None) | (label_match2 != None)) & (j <= length):
            
            if [f"{key}={dict[key][j-1]}" for key in dict][0] == 'turn=1':  #turn=2 checks when car turn has just ended
                # trace_dict.update(dict)
                count += 1
                arr.append([f"{key}={dict[key][j-1]}" for key in dict])
                # trace_dict.update([dict[key] for key in dict])


    for i,input in enumerate(arr):
        for j,str in enumerate(input):
            lines[i] += f"{str}"
            if j != len(arr[i]) - 1:
                lines[i] += " & "

    
    while '' in lines:
        lines.remove('')
    # print(lines)

    replaced = '\n'.join(lines)
    if not USE_VISIBILITY:
        replaced = re.sub("\&\s*visibility\s*\=\s*[0-1]\s*", "", replaced)
        replaced = re.sub("\&\s*seen_ped\s*\=\s*[0-1]\s*", "", replaced)

    with open(output_tempest_trace, "w") as f:
        f.write(replaced)




if __name__ == "__main__":
    main()