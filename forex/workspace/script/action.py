from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
import time
from datetime import datetime, timedelta
from selenium.webdriver import ChromeOptions
from selenium.common.exceptions import StaleElementReferenceException, ElementClickInterceptedException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import WebDriverException
import subprocess
import random
import os
import re
try:
    from util import Log, IO
    local_test_mode = False
except:
    local_test_mode = True
    pass
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
import json


class action:

    def __init__(self, browser, name='EURUSD'):
        self.browser = browser
        self.wait = WebDriverWait(browser, 300)
        self.last = ''
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.path += '/'
        self.name = name
        if not local_test_mode:
            self.logger = Log()

    def extract(self, info):
        pass

    def load_url(self, url):
        self.browser.get(url)

    def save_source(self, fn='workspace/data/tmp.html'):
        with open(fn, 'w') as f:
            f.write(self.browser.page_source)

    def save(self, obj, fn='workspace/data/print.txt'):
        with open(fn, 'w') as f:
            f.write(obj)

    def login(self, account='account', verbose=False):
        if local_test_mode:
            pardir = os.path.join(self.path, os.pardir)
            fn = pardir + '/data/' + account + '.txt'
        else:
            fn = 'workspace/data/' + account + '.txt'

        with open(fn, 'r') as f:
            lines = f.readlines()
            user_index = [i for i in range(len(lines))]
            index = random.choice(user_index)
            user, pwd = lines[index].split(' ')
        self.load_url('https://app.plus500.com/')
        self.wait.until(EC.presence_of_element_located((By.ID, 'demoMode')))
        demoMode = self.browser.find_element_by_id('demoMode')
        demoMode.click()
        self.wait.until(EC.presence_of_element_located((By.ID, 'newUserCancel')))
        demoMode = self.browser.find_element_by_id('newUserCancel')
        demoMode.click()
        self.wait.until(EC.presence_of_element_located((By.ID, 'email')))
        demoMode = self.browser.find_element_by_id('email')
        demoMode.send_keys(user)
        demoMode = self.browser.find_element_by_id('password')
        time.sleep(1)
        demoMode.send_keys(pwd)
        if verbose:
            print(user, pwd)
        time.sleep(1)
        try:
            demoMode = self.browser.find_element_by_id('submitLogin')
            # demoMode.screenshot('workspace/data/sub.png')
            demoMode.click()
        except StaleElementReferenceException:
            time.sleep(5)
            return
        except NoSuchElementException:
            time.sleep(5)
            return
        except WebDriverException:
            time.sleep(5)
            print('opps')
            return
        time.sleep(5)
        self.wait.until(EC.presence_of_element_located((By.XPATH, '//span[@data-win-bind="textContent: AvailableBalance Converters.formatMoneyAmountFloor; winControl.innerHTML: AvailableBalanceExplained Converters.emptyConverter"]')))

    def get_forex_page(self):

        for _ in range(5):
            try:
                button = self.browser.find_element_by_xpath('//*[contains(text(), "Popular Pairs")]')
                button.screenshot('workspace/data/popular_pairs.png')
                button.click()
                time.sleep(1)
                return True
            except NoSuchElementException:
                self.save_source()
                time.sleep(1)
            except ElementClickInterceptedException:
                self.save_source()
                time.sleep(1)

        return False

    def get_crypto_page(self):
        button = self.browser.find_element_by_xpath('//a[contains(text(), "Crypto Currencies")]')
        button.screenshot('workspace/data/crypto_crurencies.png')
        button.click()
        time.sleep(1)

    def collect_cc(self, name='Bitcoin', period=10, frequency=0.1, limit=1200, length=3600):
        # collect crypto currencies data
        root_p = '//span[@class="drag"]'
        pair_p = root_p+'/following-sibling::span[1]'
        parent_p = root_p+'/parent::div/following-sibling::'
        change_p = parent_p+'div[@class="change"]'
        buy_p = parent_p+'div[@class="buy"]'
        sell_p = parent_p+'div[@class="sell"]'

        pairs = self.browser.find_elements_by_xpath(pair_p)
        changes = self.browser.find_elements_by_xpath(change_p)
        sells = self.browser.find_elements_by_xpath(sell_p)
        buys = self.browser.find_elements_by_xpath(buy_p)
        
        start_time = time.time()
        index = None
        fn = 'workspace/data/'+name+'_new_record'
        # extract data
        while True:
            time_dist = time.time()-start_time
            if time_dist > period:
                break
            if index is None:
                pairs = [p.text for p in pairs]
                index = pairs.index(name)

            data = [name, changes[index].text, \
                    sells[index].text, \
                    buys[index].text]
            print(buys)

            self._add_data(data, fn)
            time.sleep(frequency)

    def collect(self, pair='EURUSD', period=10, frequency=0.2, limit=1200, length=3600):
        pair_ = pair[:3]+'/'+pair[3:]
        success = self.get_forex_page()
        if not success:
            raise
        start_time = time.time()
        index = None
        fn = 'workspace/data/'+pair+'_new_record'
        # extract data
        cnt = 0
        pre_data = []
        # elements
        root_p = '//span[@class="drag"]'
        pair_p = root_p+'/following-sibling::span[1]'
        parent_p = root_p+'/parent::div/following-sibling::'
        change_p = parent_p+'div[@class="change"]'
        buy_p = parent_p+'div[@class="buy"]'
        sell_p = parent_p+'div[@class="sell"]'
        p_l = self.browser.find_element_by_xpath('//span[@data-win-bind="textContent: AvailableBalance Converters.formatMoneyAmountFloor; winControl.innerHTML: AvailableBalanceExplained Converters.emptyConverter"]')

        # high_low_p = parent_p+'div[@class="high-low"]'
        pairs = self.browser.find_elements_by_xpath(pair_p)
        changes = self.browser.find_elements_by_xpath(change_p)
        sells = self.browser.find_elements_by_xpath(sell_p)
        buys = self.browser.find_elements_by_xpath(buy_p)
        success = self.search_index(pair)
        if not success:
            raise

        while True:
            time_dist = time.time()-start_time
            if time_dist > period:
                break
            if index is None:
                pairs = [p.text for p in pairs]
                index = pairs.index(pair_)
            data = [pair_, changes[index].text, \
                    sells[index].text, \
                    buys[index].text]
            
            cnt += 1
            if cnt % 100 == 1:
                try:
                    p_l.click()
                except ElementClickInterceptedException as e:
                    print(e)
                    raise
                    
                if data == pre_data:
                    print('get the same data')
                    self.get_forex_page()
                    time.sleep(1)
                    pairs = self.browser.find_elements_by_xpath(pair_p)
                    changes = self.browser.find_elements_by_xpath(change_p)
                    sells = self.browser.find_elements_by_xpath(sell_p)
                    buys = self.browser.find_elements_by_xpath(buy_p)
                    print('search index')
                    success = self.search_index(pair)
                    if not success:
                        raise

                pre_data = data

            self._add_data(data, fn)

            time.sleep(frequency)

    def collect_prof_loss(self, period, freq):
        for i in range(10):
            try:
                p_l = self.browser.find_element_by_xpath('//span[@data-win-bind="textContent: TotalNetProfitLoss Converters.formatMoneyAmount; winControl.innerHTML: TotalNetProfitLossExplained Converters.emptyConverter"]')
                break
            except NoSuchElementException:
                self.save_source()
                time.sleep(1)

        start_time = time.time()
        while True:
            time_dist = time.time()-start_time
            try:
                p_l.click()
            except ElementClickInterceptedException as e:
                print(e)
                raise
            if time_dist > period:
                break
            self.logger.write_available('EURUSD', p_l.text)
            time.sleep(freq)

    def collect_available(self, period, freq):
        self.save_source()
        for i in range(10):
            try:
                p_l = self.browser.find_element_by_xpath('//span[@data-win-bind="textContent: AvailableBalance Converters.formatMoneyAmountFloor; winControl.innerHTML: AvailableBalanceExplained Converters.emptyConverter"]')
            except NoSuchElementException as e:
                self.save_source()
                time.sleep(1)
                continue
            break
        start_time = time.time()
        while True:
            time_dist = time.time() - start_time
            if time_dist > period:
                break
            self.logger.write_available('EURUSD', p_l.text)
            time.sleep(freq)

    def name_map(self, name):
        if name == 'EURUSD':
            return 'EUR/USD'
        else:
            return name

    def search_index(self, name='EURUSD'):
        root_p = '//span[@class="drag"]'
        pair_p = root_p+'/following-sibling::span[1]'
        for _ in range(5):
            try:
                pairs = self.browser.find_elements_by_xpath(pair_p)
                pairs = [p.text for p in pairs]
                self.index = pairs.index(self.name_map(name))
                return True
            except Exception as e:
                print(e)

        return False

    def get_opening_price(self, name='EURUSD'):
        root_p = '//span[@class="drag"]'
        parent_p = root_p+'/parent::div/following-sibling::'
        change_p = parent_p+'div[@class="change"]'
        sell_p = parent_p+'div[@class="sell"]'

        changes = self.browser.find_elements_by_xpath(change_p)
        sells = self.browser.find_elements_by_xpath(sell_p)
        change = changes[self.index].text
        change = float(change.strip('%'))/100
        sell = float(sells[self.index].text)
        opening_price = sell*(1+change)
        return opening_price

    def set_close_at_loss(self, limit=None):
        try:
            checkbox = self.browser.find_element_by_id('close-at-loss-checkbox')
            checkbox.click()
            time.sleep(0.1)
            inputs = self.browser.find_element_by_id('close-at-loss-rate')
            if limit is not None:
                while True:
                    inputs.send_keys(Keys.CONTROL+'a')
                    time.sleep(0.2)
                    inputs.send_keys(Keys.DELETE)
                    time.sleep(0.2)
                    inputs.send_keys(limit)
                    time.sleep(0.2)
                    value = inputs.get_attribute('value')
                    if (value) == limit:
                        break
            return True
        except Exception as e:
            print(e)
            return False

    def set_close_at_prof(self, limit=None):
        try:
            checkbox = self.browser.find_element_by_id('close-at-profit-checkbox')
            checkbox.click()
            time.sleep(0.1)
            inputs = self.browser.find_element_by_id('close-at-profit-rate')
            if limit is not None:
                while True:
                    inputs.send_keys(Keys.CONTROL+'a')
                    time.sleep(0.2)
                    inputs.send_keys(Keys.DELETE)
                    time.sleep(0.2)
                    inputs.send_keys(limit)
                    time.sleep(0.2)
                    value = inputs.get_attribute('value')
                    if value == limit:
                        break
            return True
        except Exception:
            return False

    def buy(self, name='EURUSD', unit=1000, prof_lim=None, loss_lim=None):
        print('buy function')
        button = self.browser.find_elements_by_xpath('//button[@class="buySellButton" and contains(text(), "Buy")]')[self.index]
        button.click()
        self.wait.until(EC.presence_of_element_located((By.XPATH, '//button[@id="trade-button"]')))
        try:
            inputs = self.browser.find_element_by_xpath('//input[@id="amount-input"]')
            while True:
                inputs.send_keys(Keys.CONTROL+'a')
                time.sleep(0.3)
                inputs.send_keys(Keys.DELETE)
                time.sleep(0.3)
                inputs.send_keys(unit)
                time.sleep(0.3)
                value = inputs.get_attribute('value')
                value = re.sub('\D', '', value)
                if float(value) == float(unit):
                    break
            if not self.set_close_at_loss(loss_lim):
                return False
            if not self.set_close_at_prof(prof_lim):
                return False

            button = self.browser.find_element_by_xpath('//button[@id="trade-button"]')
            button.click()
        except NoSuchElementException:
            print('no such element in buy function')
            return False
        except StaleElementReferenceException:
            print("can't attach the element")
            return False
        time.sleep(1)
        if not local_test_mode:
            self.logger.write_available('EURUSD', 'buy')
        return True

    def sell(self, unit=1000, prof_lim=None, loss_lim=None):
        print('sell function')
        button = self.browser.find_elements_by_xpath('//button[@class="buySellButton" and contains(text(), "Sell")]')[self.index]
        button.click()
        self.wait.until(EC.presence_of_element_located((By.XPATH, '//button[@id="trade-button"]')))
        try:
            inputs = self.browser.find_element_by_xpath('//input[@id="amount-input"]')
            while True:
                inputs.send_keys(Keys.CONTROL+'a')
                time.sleep(0.2)
                inputs.send_keys(Keys.DELETE)
                time.sleep(0.2)
                inputs.send_keys(unit)
                time.sleep(0.2)
                value = inputs.get_attribute('value')
                value = re.sub('\D', '', value)
                if float(value) == float(unit):
                    break
            if not self.set_close_at_loss(loss_lim):
                return False
            if not self.set_close_at_prof(prof_lim):
                return False
            button = self.browser.find_element_by_xpath('//button[@id="trade-button"]')
            button.click()
        except NoSuchElementException:
            print('no such element in sell fucntion')
            return False
        except StaleElementReferenceException:
            print("can't attach the element")
            return False

        time.sleep(1)
        if not local_test_mode:
            self.logger.write_available('EURUSD', 'sell')
        return True

    def close(self):
        print('close function')
        for _ in range(20):
            try:
                buttons = self.browser.find_elements_by_xpath('//button[@class="close-position icon-times"]')
            except Exception as e:
                self.save_source()
                print(e)
                time.sleep(1)
                continue
            break

        time.sleep(0.3)

        for button in buttons:
            
            try:
                button.click()
            except StaleElementReferenceException:
                return False

            time.sleep(0.2)
            for _ in range(20):
                try:
                    close_bt = self.browser.find_element_by_xpath('//button[@id="closePositionBtn"]')
                except Exception as e:
                    print(e)
                    time.sleep(1)
            close_bt.click()
            time.sleep(0.2)
            if not local_test_mode:
                self.logger.write_available('EURUSD', 'close')
        return True

    def take_action(self, period, freq):
        fn = 'workspace/data/decision'
        start_time = time.time()
        pre_action = None
        while True:
            time_dist = time.time()-start_time
            if time_dist > period:
                print('take action stop')
                break

            with open(fn, 'r') as f:
                line = f.readline()
                line = line.strip('\n')
                line = line.split('|')
                if self.last == line[0]:
                    time.sleep(freq)
                    continue
                else:
                    self.last = line[0]
                if len(line) < 2:
                    print(line)
                    action = line[1]
                else:
                    action = line[1]
                    upper = line[2]
                    lower = line[3]

            # print('pre action is:', pre_action)

            if pre_action is None:
                pre_action = 'wait'
                time.sleep(freq)
                continue
            elif pre_action == 'sell fail':
                action = 'sell'

            elif pre_action == 'buy fail':
                action = 'buy'

            else:
                if pre_action != action and action != 'wait':
                    print('take action: close')
                    success = self.close()
                    if not success:
                        pre_action = 'close fail'
                    else:
                        pre_action = action

                else:
                    action = 'wait'

            if action == 'sell':
                print('take_action: sell')
                success = self.sell(prof_lim=lower, loss_lim=upper)
                if not success:
                    pre_action = 'sell fail'
            elif action == 'buy':
                print('take_action: buy')
                success = self.buy(prof_lim=upper, loss_lim=lower)
                if not success:
                    pre_action = 'buy fail'

            time.sleep(freq)

    def request_responser(self, request_fn='workspace/data/control/request', sleep_time=10):
        # request format
        # request: True/False
        # open price: True/False
        while True:
            try:
                with open(request_fn, 'r') as f:
                    request = json.load(f)
                    f.close()
                if request['request'] is False:
                    time.sleep(sleep_time)
                    continue
                reqs = request['order']
            except Exception:
                time.sleep(sleep_time)
                continue
            data = {}

            for r in reqs:
                if r == 'open price':
                    data[r] = self.get_opening_price(self.name)
                if r == 'search index':
                    self.search_index(self.name)

            while True:
                success = IO.write_response(data)
                if success:
                    success = IO.close_request()
                    if success:
                        break
                time.sleep(1)
            time.sleep(sleep_time)

    def _add_data(self, data, fn):
        # data: list of data
        data = [datetime.now()] + data
        data = ' '.join(str(v) for v in data)
        cmd = 'echo "' + data + '" > ' + fn
        subprocess.call(cmd, shell=True)

    def __del__(self):
        self.browser.quit()


def prepare(account='account', verbose=False):
    options = ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    chrome = webdriver.Remote(
            command_executor='http://172.28.0.13:5555/wd/hub',
            desired_capabilities=DesiredCapabilities.CHROME, 
            options=options)

    my_act = action(chrome)
    my_act.login(account, verbose)
    return my_act


class action2:
    def __init__(self, name):
        self.name = name
        self.url = 'https:/financialmodelingprep.com/api/v3/forex/'+name

    def load_forex_data(self):
        url = 'https://financialmodelingprep.com/api/v3/forex/'+self.name

        while True:
            response = urlopen(url)

            data = response.read().decode('utf-8')
            data = json.loads(data)
            price = data['bid']
            print(price)
            time.sleep(1)

    def get_online_price(self):
        response = urlopen(self.url)
        data = response.read().decode('utf-8')
        data = json.load(data)
        price = data['bid']
        t = data['date']
        return t, price

    def get_historical_forex_rete(self, date=None):
        # get end of day conversion rate
        if date is None:
            date = datetime.now()
            diff = timedelta(days=1)
            last_day = date-diff
            last_day = str(last_day.date())
        else:
            last_day = date

        base = self.name[:3]
        target = self.name[3:]
        token = 'Vjz0NBTpOQ6Hj6IXiwO29TJ9CodOs7o04IzkLvdJEyz3ZXztz8NU3dMZ25aC'
        url = 'https://api.worldtradingdata.com/api/v1/forex_single_day?base='+base+'&convert_to='+target+'&date='+last_day+'&api_token='+token
        response = urlopen(url)
        data = response.read().decode('utf-8')
        data = json.loads(data)
        print(data['data'][target])
        return data['data'][target]


def run():

    my_act = prepare()
    # my_act.collect('EURUSD', 10)
    my_act.collect_cc('Bitcoin', 10)


def test():
    my_act = prepare()
    my_act.search_index()
    my_act.request_responser()


if __name__ == '__main__': 
    # run()
    test()
