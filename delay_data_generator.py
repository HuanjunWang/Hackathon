from datetime import datetime
import math

def read_file(file_name):
    lines = []
    with open(file_name, "r") as f:
        for line in f:
            if 'PGSL-DLDATA-IND' not in line:
                continue
            lines.append(line)
    return lines


def save_result(result_dict, file_name="delay.txt"):
    result = []

    for ip_arr in result_dict.values():
        for msg in ip_arr:
            result.append(msg[2])

    import pickle
    with open(file_name, 'wb') as fp:
        pickle.dump(result, fp)


def filter_and_calculate(lines):
    time_per_frame = 15 / 26 * 8

    start_time = {}
    start_frame = {}
    min_diff = {}
    result = {}

    for line in lines:
        line_l = line.split()
        ip = line_l[4] + line_l[5]
        frame = int(line_l[9])
        time = datetime.strptime(line_l[3], "%H:%M:%S.%f000")

        if ip not in start_time:
            start_time[ip] = time
        if ip not in start_frame:
            start_frame[ip] = frame

        time_diff = (time - start_time[ip]).total_seconds() * 1000
        diff = frame - (start_frame[ip] + time_diff / time_per_frame)

        if (ip in min_diff and diff < min_diff[ip]) or ip not in min_diff:
            min_diff[ip] = diff
        msgs = [str(ip), str(frame), diff]

        if ip in result:
            result[ip].append(msgs)
        else:
            result[ip] = [msgs]

    for k, v in result.items():
        min_d = min_diff[k]
        for msg in v:
            msg[2] = int(math.floor(msg[2] - min_d))

    return result


def main():
    lines = read_file("58.txt")
    result_list = filter_and_calculate(lines)
    save_result(result_list)


if __name__ == "__main__":
    import cProfile

    cProfile.run("main()")
