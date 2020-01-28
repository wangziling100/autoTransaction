import subprocess
import time


# the current workspace has the same path with the in your docker container,
# so you can run your script directly without changing path

root = 'workspace'
script_file = '/script/'
service = 'forex '
run_command = 'python '
bringup_command = 'sudo docker-compose up'
base_command = 'sudo docker-compose run '+service+run_command+root+script_file
torch_command = 'sudo docker-compose run pytorch '+run_command+root+script_file
end_command = 'sudo docker-compose down'
print(base_command+'test.py')

commands = [
    # here define your commands, they will be conducted orderly.
    # type of element is [cmd, cmd type, shell]
    # cmd type: 
    # 1. synchronous
    # 2. asynchronous
    # [base_command + 'test.py', 'synchronous', True],
    #[base_command + 'cnn_example.py', 'synchronous', True],
    [bringup_command+' hub', 'asynchronous', True],
    [10, 'sleep', None],
    [bringup_command+' chrome', 'asynchronous', True],
    [10, 'sleep', None],
    [base_command + 'collect.py', 'asynchronous', True],
    [20, 'sleep', None], 
    [torch_command + 'save_as_historical_data.py', 'synchronous', True],
    [end_command, 'synchronous', True]
        ]

# procedure process
for cmd, type, shell in commands:
    print(cmd)
    if type == 'synchronous':
        subprocess.call(cmd, shell=shell)
    elif type == 'asynchronous':
        subprocess.Popen(cmd, shell=shell)

    elif type == 'sleep':
        time.sleep(cmd)



