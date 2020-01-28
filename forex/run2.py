import subprocess
import time
from multiprocessing import Process
from datetime import datetime


def start_hub():
    with open('workspace/log/process.log', 'a') as f:
        f.write(str(datetime.now())+': ')
        f.write('start service hub\n')
        f.close()
    cmd = 'sudo docker-compose up hub'
    subprocess.call(cmd, shell=True)


def start_chrome():
    with open('workspace/log/process.log', 'a') as f:
        f.write(str(datetime.now())+': ')
        f.write('start service chrome\n')
        f.close()
    cmd = 'sudo docker-compose up chrome'
    subprocess.call(cmd, shell=True)


def collect_service():
    with open('workspace/log/process.log', 'a') as f:
        f.write(str(datetime.now())+': ')
        f.write('start service collect\n')
        f.close()
    cmd = 'sudo docker-compose run forex python workspace/script/collect.py'
    subprocess.call(cmd, shell=True)


def store_data_service():
    with open('workspace/log/process.log', 'a') as f:
        f.write(str(datetime.now())+': ')
        f.write('start service store\n')
        f.close()
    cmd = 'sudo docker-compose run pytorch python workspace/script/save_as_historical_data.py'
    subprocess.call(cmd, shell=True)


def stop_containers():
    cmd = 'sudo docker-compose down'
    subprocess.call(cmd, shell=True)


def stop_all(services):
    with open('workspace/log/process.log', 'a') as f:
        f.write(str(datetime.now())+': ')
        f.write('stop all services\n')
        f.close()
    for service in services:
        service.terminate()
        time.sleep(10)
        del service

    stop_containers()
    time.sleep(60)


def start_service_by_name(name):
    if name == 'hub':
        ret = Process(target=start_hub, name='hub')
    elif name == 'chrome':
        ret = Process(target=start_chrome, name='chrome')
    elif name == 'collect':
        ret = Process(target=collect_service, name='collect')
    elif name == 'store':
        ret = Process(target=store_data_service, name='store')

    ret.start()
    return ret


def start_all_by_name(base_names, names):

    with open('workspace/log/process.log', 'a') as f:
        f.write(str(datetime.now())+': ')
        f.write('start all services\n')
        f.close()

    base_services = []
    services = []
    for name in base_names:
        service = start_service_by_name(name)
        base_services.append(service)
        time.sleep(10)

    for name in names:
        service = start_service_by_name(name)
        services.append(service)
        time.sleep(10)

    return base_services, services


def run():
    # init
    base_service_names = ['hub', 'chrome']
    service_names = ['collect', #'store'
            ]
    with open('workspace/log/process.log', 'w') as f:
        f.write('------ start services -------\n')
        f.write(str(datetime.now())+'\n')
        f.close()

    base_services, services = start_all_by_name(base_service_names,
            service_names)

    is_stop = False
    # superviser
    while True:
        for service in base_services:
            if not service.is_alive():
                with open('workspace/log/process.log', 'a') as f:
                    f.write('--------------------\n')
                    f.write('service '+service.name+' is stopped\n')
                    f.close()
                stop_all(services+base_services)
                is_stop = True
                time.sleep(5)
                break

            time.sleep(0.5)

        for service in services:
            if is_stop:
                break
            if not service.is_alive():
                with open('workspace/log/process.log', 'a') as f:
                    f.write('--------------------\n')
                    f.write(str(datetime.now())+': ')
                    f.write('service '+service.name+' is stopped\n')
                    f.write(str(datetime.now())+': ')
                    f.write('service '+service.name+' is restarting...\n')
                    f.close()
                name = service.name
                service.terminate()
                time.sleep(5)
                services.remove(service)
                del service
                p = start_service_by_name(name)
                services.append(p)
                time.sleep(20)
            time.sleep(0.5)

        if is_stop:
            base_services, services = start_all_by_name(
                    base_service_names, service_names)
            is_stop = False


if __name__ == '__main__':
    p = Process(target=run)
    p.start()
