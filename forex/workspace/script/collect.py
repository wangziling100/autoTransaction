from action import action
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from util import Log
from selenium.webdriver import ChromeOptions
from multiprocessing import Process
import time
import sys


def collect_price(browser, name, logger):
    while True:
        try:
            browser.collect(name, 1800000000000)
            sys.exit(1)
        except Exception as e:
            logger.error(str(e))
            sys.exit(1)


def collect_prof_loss(browser, logger):
    while True:
        try:
            browser.collect_prof_loss(1000000000000, 1)
        except Exception as e:
            logger.error(str(e))


def run():
    options = ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    chrome = webdriver.Remote(
            command_executor='http://172.28.0.13:5555/wd/hub',
            desired_capabilities=DesiredCapabilities.CHROME, options=options)
    logger = Log()

    my_act = action(chrome)
    p_price = Process(target=collect_price, args=(my_act, 'EURUSD', logger, ))
    # p_prof_loss = Process(target=collect_prof_loss, args=(my_act, logger, ))

    my_act.login()
    p_price.start()
    time.sleep(5)
    # p_prof_loss.start()
    p_price.join()
    # p_prof_loss.join()

    """
    while True:
        try:
            my_act.collect('EURUSD', 1800000000000)
            # my_act.collect_cc('Bitcoin', 100000000000)
        except Exception as e:
            logger.error(str(e))
    """


if __name__ == '__main__':
    run()






