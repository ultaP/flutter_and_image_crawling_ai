from selenium import webdriver
import time
import os
import base64
import requests
from selenium.webdriver.common.by import By

# 검색어 리스트
search_queries = ["피자", "햄버거", "파스타"]
save_directories = ["downloaded_images/pizza", "downloaded_images/burger", "downloaded_images/pasta"]

# 크롬 드라이버 실행
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # 창 숨기기
driver = webdriver.Chrome(options=options)

# 각 검색어와 저장 디렉토리를 함께 반복
for search_query, save_directory in zip(search_queries, save_directories):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 구글 이미지 검색 URL 설정
    # url = f"https://www.google.com/search?q={search_query}&tbm=isch"
    # url = f"https://search.naver.com/search.naver?where=image&query={search_query}"
    url = f"https://pixabay.com/ko/images/search/{search_query}/"
    # 구글 이미지 검색 페이지 열기
    driver.get(url)

    # 페이지 스크롤 다운 (더 많은 이미지 로딩)
    scrolls = 5
    for _ in range(scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # 이미지 다운로드
    for i, h3_tag in enumerate(driver.find_elements(By.XPATH, 
                                                     # '//h3[@class="ob5Hkd"]') #google
                                                    #  '//div[@class="thumb"]') #naver
                                                     '//div[@class="cell--B7yKd') #pixabay
                                                [:150]
                                ):
        try:
            # h3 태그의 자식인 img 태그의 src 속성 값을 가져옴
            img_tag = h3_tag.find_element(By.XPATH, './/img')
            image_url = img_tag.get_attribute("src")

            # 이미지 형식이 base64 코드인지 확인
            if image_url.startswith("data:image"):
                # base64로 인코딩된 이미지를 디코딩하여 이미지 데이터로 변환
                image_data = base64.b64decode(image_url.split(",")[1])
                # 이미지를 파일로 저장
                with open(os.path.join(save_directory, f"image_{i}_base64.jpg"), "wb") as f:
                    f.write(image_data)
                print(f"Base64 image {i} downloaded and saved successfully for '{search_query}'")
            else:
                # 이미지가 URL 형식인 경우
                image_data = requests.get(image_url).content
                # 이미지를 파일로 저장
                with open(os.path.join(save_directory, f"image_{i}.jpg"), "wb") as f:
                    f.write(image_data)
                print(f"Image {i} downloaded and saved successfully for '{search_query}'")
        except Exception as e:
            print(f"Failed to download image {i} for '{search_query}': {e}")

# 드라이버 종료
driver.quit()
