from analyse import load_data
import time
import numpy as np
import sys


def save_data():

    reader = load_data()
    start_time = time.time()
    pre_save_time = None
    numbers = []
    saved = False
    running = True
    pre_time = 0
    try:
        while True:
            reader.load_online_data()
            time_dist = int(time.time()-start_time)
            # per sec
            if pre_time != time_dist:
                pre_time = time_dist

                # print(reader.dfbase)

                if running:
                    reader.get1s_online()
                    # print(reader.dfbase.tail())
                    pre_save_time = reader.save_base(pre_save_time)
                    numbers = []

                if time_dist % 5 == 1:
                    running, num = check_signals()
                    if not running:
                        b_continue, numbers = check_continue(numbers, num)

                        if b_continue:
                            write_running_signal()
                            running = True
                            saved = False

                        elif not saved:
                            if reader.dfbase is not None:
                                print('force save')
                                reader.save_base(0, True)
                                reader.dfbase = None
                                saved = True

            time.sleep(0.05)
    except KeyboardInterrupt:
        reader.save_base(0, True)
        print('Abnormal exit')
        sys.exit(0)


def check_signals(fn='workspace/data/control/signals'):
    with open(fn, 'r') as f:
        lines = f.readlines()

        f.close()

    try:
        running = lines[0].strip('\n').split(':')[1]
        num = lines[1].split(':')[1]
    except IndexError:
        return False, 0
    
    if running == 'True':
        running = True
    else:
        running = False

    return running, int(num)


def check_continue(numbers, num):
    numbers.append(num)
    numbers = numbers[-5:]
    if np.std(numbers) < 0.01 and len(numbers) > 4:
        return True, numbers
    else:
        return False, numbers


def write_running_signal(fn='workspace/data/control/signals'):
    with open(fn, 'w') as f:
        f.write('save:True\n')
        f.write('count:0')
        f.close()


if __name__ == '__main__':
    save_data()

