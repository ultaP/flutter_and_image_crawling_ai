from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import urllib.request
import os
import time

# 검색어와 다운로드 폴더 이름 쌍 리스트
search_pairs = [
    #('순대국', 'sundae_soup'),
    ('김치찌개', 'kimchi_stew'),
]

# Selenium 설정
service = Service(ChromeDriverManager(driver_version='125').install())
driver = webdriver.Chrome(service=service)
wait = WebDriverWait(driver, 20)
actions = ActionChains(driver)

for search_key, download_folder_name in search_pairs:
    # 다운로드 폴더 설정
    download_folder = os.path.join('C:\\img_c\\downloaded_images', download_folder_name)
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # 네이버 지도 접속
    driver.get('https://map.naver.com/')

    # 검색어 입력 후 엔터
    search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input.input_search')))
    search_box.send_keys('서울 ' + search_key)  # 여기에 검색어 입력
    search_box.send_keys(Keys.RETURN)

    # 검색 결과 대기
    wait.until(EC.presence_of_element_located((By.ID, 'searchIframe')))
    driver.switch_to.frame('searchIframe')

    # 필터링된 li 태그 선택 및 조건에 맞는 요소 찾기
    filtered_results = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//li[@data-laim-exp-id="undefinedundefined" and not(.//div[@class="cZnHG"])]')))[:10]

    for idx, result in enumerate(filtered_results):
        try:
            # tzwk0 클래스 클릭
            tzwk0_element = result.find_element(By.CSS_SELECTOR, '.tzwk0')
            actions.move_to_element(tzwk0_element).click().perform()

            # 기본 컨텍스트로 돌아가기
            driver.switch_to.default_content()
            time.sleep(1)
            wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, 'entryIframe')))

            # 사진 탭 클릭
            wait.until(EC.element_to_be_clickable((By.XPATH, '//span[text()="사진"]'))).click()

            # 스크롤
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1)
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    wzrbN_elements = driver.find_elements(By.CSS_SELECTOR, '.wzrbN')
                    if new_height == last_height or len(wzrbN_elements) >= 100:
                        break
                    last_height = new_height               

            # 이미지 다운로드
            images = driver.find_elements(By.CSS_SELECTOR, '.wzrbN img')
            for idx2, img in enumerate(images):
                img_url = img.get_attribute('src')
                if img_url:
                    img_path = os.path.join(download_folder, f'image_{idx}_{idx2}.jpg')
                    urllib.request.urlretrieve(img_url, img_path)

            # 다시 목록으로 돌아가기
            driver.switch_to.default_content()
            wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, 'searchIframe')))

        except Exception as e:
            print(f"Error: {e}")
            continue

driver.quit()
