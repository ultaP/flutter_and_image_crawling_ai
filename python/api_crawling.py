import requests
import os

# Pixabay API 키
API_KEY = '43531811-44979105e0ef34d6db3e529d1'

# 이미지를 검색할 키워드
query = '스테이크'

# 가져올 이미지 수
total_images = 500

# 한 페이지당 가져올 이미지 수
per_page = 100

# 이미지 저장 디렉토리 경로
save_directory = "downloaded_images/stake"

# 이미지 저장 디렉토리 생성
os.makedirs(save_directory, exist_ok=True)

# 총 페이지 수 계산
total_pages = -(-total_images // per_page)  # 올림 연산

# Pixabay API 엔드포인트 URL
for page in range(1, total_pages + 1):
    url = f'https://pixabay.com/api/?key={API_KEY}&q={query}&image_type=photo&page={page}&per_page={per_page}'
    
    # API 요청을 보냄
    response = requests.get(url)
    
    # 요청이 성공적으로 수행되었는지 확인
    if response.status_code == 200:
        # JSON 응답 데이터를 딕셔너리로 파싱
        data = response.json()

        # 이미지 URL을 추출하여 다운로드
        for i, item in enumerate(data['hits']):
            image_url = item['largeImageURL']
            # 이미지를 다운로드하거나 필요한 작업을 수행할 수 있음
            print('Downloading:', image_url)
            # 이미지 다운로드
            image_data = requests.get(image_url).content
            # 이미지를 파일로 저장
            with open(os.path.join(save_directory, f"image_{(page-1)*per_page+i}.jpg"), "wb") as f:
                f.write(image_data)
            print(f"Image {(page-1)*per_page+i} downloaded and saved successfully for '{image_url}'")
    else:
        print(f'Failed to fetch images from Pixabay API on page {page}.')
