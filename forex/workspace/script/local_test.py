from selenium import webdriver
import os
from action import action
import random
import time
from selenium.common.exceptions import ElementClickInterceptedException, NoSuchElementException
import traceback


def prepare():
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'geckodriver')
    browser = webdriver.Firefox(executable_path=path)
    return browser


def test_random_actions(actor, n_min=60):
    actions = ['buy', 'sell', 'close']
    try:
        p_l = actor.browser.find_element_by_xpath('//span[@data-win-bind="textContent: TotalNetProfitLoss Converters.formatMoneyAmount; winControl.innerHTML: TotalNetProfitLossExplained Converters.emptyConverter"]')
    except NoSuchElementException as e:
        time.sleep(1)
        print(e)
        traceback.print_exc()

    prof = 20*10**-5
    loss = 100*10**-5

    for _ in range(n_min):
        action = random.choice(actions)
        try:
            p_l.click()
        except ElementClickInterceptedException as e:
            print(e)

        if action == 'buy':
            actor.buy(prof_lim=prof, loss_lim=loss)
        elif action == 'sell':
            actor.sell(prof_lim=prof, loss_lim=loss)

        else:
            actor.close()

        time.sleep(60)


if __name__ == '__main__':

    browser = prepare()
    actor = action(browser)
    actor.login('account1', verbose=True)
    actor.get_forex_page()
    actor.search_index('EURUSD')
    test_random_actions(actor)

