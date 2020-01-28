from action import action, prepare
from multiprocessing import Process
import time
import sys


def collect_total_money(browser):
    browser.collect_prof_loss(10000000000, 10)


def do_action(browser, name='EURUSD'):
    if name == 'EURUSD':
        print('load page')
        while True:
            check_success(browser.get_forex_page)
            try:
                check_success(browser.search_index)
                break
            except Exception as e:
                print(e)
                print("can't load index, try is again...")
        print('loading page finished')

    browser.save_source()
    browser.take_action(10000000000000, 1)


def request_responser(browser):
    browser.search_index()
    browser.request_responser()


def check_success(p, args=None):
    while True:
        if args is None:
            is_loaded = p()
        else:
            is_loaded = p(args)

        if is_loaded:
            break
        else:
            print("can't load the elements, please wait ...")


if __name__ == '__main__':

    browser = prepare('account1', verbose=True)
    p_collect = Process(target=collect_total_money, args=(browser, ))
    p_do = Process(target=do_action, args=(browser, ))
    p_responser = Process(target=request_responser, args=(browser, ))
    time.sleep(1)
    p_collect.start()
    time.sleep(1)
    p_do.start()
    time.sleep(1)
    p_responser.start()

    should_stop = False
    while True:
        if not p_collect.is_alive():
            should_stop = True

        if not p_do.is_alive():
            should_stop = True

        if not p_responser.is_alive():
            should_stop = True

        if should_stop:
            p_collect.terminate()
            time.sleep(5)
            p_do.terminate()
            time.sleep(5)
            p_responser.terminate()
            time.sleep(5)
            del p_collect
            del p_do
            del p_responser
            sys.exit(1)

        else:
            time.sleep(10)

