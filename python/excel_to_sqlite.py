import pandas as pd
import sqlite3

# 엑셀 파일 로드
file_path = 'for_DB.csv'  # 이 부분을 실제 엑셀 파일 경로로 변경

# 엑셀 파일 읽기
df = pd.read_csv(file_path, encoding='cp949')

# 칼럼명을 영어로 변환
column_mappings = {
    '식품명': 'food_name',
    '식품대분류명': 'food_category',
    '대표식품명': 'representative_food',
    '영양성분함량기준량': 'nutrition_standard',
    '에너지(kcal)': 'calories',
    '수분(g)': 'moisture',
    '단백질(g)': 'protein',
    '지방(g)': 'fat',
    '회분(g)': 'ash',
    '탄수화물(g)': 'carbohydrates',
    '당류(g)': 'sugars',
    '식이섬유(g)': 'dietary_fiber',
    '칼슘(mg)': 'calcium',
    '철(mg)': 'iron',
    '인(mg)': 'phosphorus',
    '칼륨(mg)': 'potassium',
    '나트륨(mg)': 'sodium',
    '비타민 A(μg RAE)': 'vitamin_A',
    '레티놀(μg)': 'retinol',
    '베타카로틴(μg)': 'beta_carotene',
    '티아민(mg)': 'thiamine',
    '리보플라빈(mg)': 'riboflavin',
    '니아신(mg)': 'niacin',
    '비타민 C(mg)': 'vitamin_C',
    '비타민 D(μg)': 'vitamin_D',
    '콜레스테롤(mg)': 'cholesterol',
    '포화지방산(g)': 'saturated_fat',
    '트랜스지방산(g)': 'trans_fat',
    '자당(g)': 'sucrose',
    '포도당(g)': 'glucose',
    '과당(g)': 'fructose',
    '유당(g)': 'lactose',
    '맥아당(g)': 'maltose',
    '마그네슘(㎎)': 'magnesium',
    '아연(㎎)': 'zinc',
    '구리(㎎)': 'copper',
    '망간(㎎)': 'manganese',
    '셀레늄(㎍)': 'selenium',
    '토코페롤(㎎)': 'tocopherol',
    '토코트리에놀(㎎)': 'tocotrienol',
    '엽산(DFE)(㎍)': 'folate',
    '비타민 B12(㎍)': 'vitamin_B12',
    '총 아미노산(㎎)': 'total_amino_acids',
    '이소류신(㎎)': 'isoleucine',
    '류신(㎎)': 'leucine',
    '라이신(㎎)': 'lysine',
    '메티오닌(㎎)': 'methionine',
    '페닐알라닌(㎎)': 'phenylalanine',
    '트레오닌(㎎)': 'threonine',
    '발린(㎎)': 'valine',
    '히스티딘(㎎)': 'histidine',
    '아르기닌(㎎)': 'arginine',
    '티로신(㎎)': 'tyrosine',
    '시스테인(㎎)': 'cysteine',
    '알라닌(㎎)': 'alanine',
    '아스파르트산(㎎)': 'aspartic_acid',
    '글루탐산(㎎)': 'glutamic_acid',
    '글리신(㎎)': 'glycine',
    '프롤린(㎎)': 'proline',
    '세린(㎎)': 'serine',
    '부티르산(4:0)(g)': 'butyric_acid',
    '카프로산(6:0)(g)': 'caproic_acid',
    '카프릴산(8:0)(g)': 'caprylic_acid',
    '카프르산(10:0)(g)': 'capric_acid',
    '라우르산(12:0)(g)': 'lauric_acid',
    '미리스트산(14:0)(g)': 'myristic_acid',
    '팔미트산(16:0)(g)': 'palmitic_acid',
    '스테아르산(18:0)(g)': 'stearic_acid',
    '아라키드산(20:0)(g)': 'arachidic_acid',
    '미리스톨레산(14:1)(g)': 'myristoleic_acid',
    '팔미톨레산(16:1)(g)': 'palmitoleic_acid',
    '올레산(18:1(n-9))(g)': 'oleic_acid',
    '박센산(18:1(n-7))(g)': 'vaccenic_acid',
    '가돌레산(20:1)(g)': 'gadoleic_acid',
    '리놀레산(18:2(n-6)c)(g)': 'linoleic_acid',
    '알파 리놀렌산(18:3(n-3))(g)': 'alpha_linolenic_acid',
    '감마 리놀렌산(18:3(n-6))(g)': 'gamma_linolenic_acid',
    '에이코사디에노산(20:2(n-6))(g)': 'eicosadienoic_acid',
    '아라키돈산(20:4(n-6))(g)': 'arachidonic_acid',
    '에이코사트리에노산(20:3(n-6))(g)': 'eicosatrienoic_acid',
    '에이코사펜타에노산(20:5(n-3))(g)': 'eicosapentaenoic_acid',
    '도코사펜타에노산(22:5(n-3))(g)': 'docosapentaenoic_acid',
    '도코사헥사에노산(22:6(n-3))(g)': 'docosahexaenoic_acid',
    '트랜스 올레산(18:1(n-9)t)(g)': 'trans_oleic_acid',
    '트랜스 리놀레산(18:2t)(g)': 'trans_linoleic_acid',
    '트랜스 리놀렌산(18:3t)(g)': 'trans_linolenic_acid',
    '카페인(㎎)': 'caffeine',
    '출처코드': 'source_code',
    '출처명': 'source_name',
    '식품중량': 'food_weight',
    '업체명': 'company_name',
    '총칼로리': 'total_calories'
}

# 영어 칼럼명으로 변경
df.rename(columns=column_mappings, inplace=True)

# SQLite 데이터베이스에 연결
conn = sqlite3.connect('food_nutrition.db')
cursor = conn.cursor()

# food_nutrition 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS food_nutrition (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    food_name TEXT,
    food_category TEXT,
    representative_food TEXT,
    nutrition_standard TEXT,
    calories REAL,
    moisture REAL,
    protein REAL,
    fat REAL,
    ash REAL,
    carbohydrates REAL,
    sugars REAL,
    dietary_fiber REAL,
    calcium REAL,
    iron REAL,
    phosphorus REAL,
    potassium REAL,
    sodium REAL,
    vitamin_A REAL,
    retinol REAL,
    beta_carotene REAL,
    thiamine REAL,
    riboflavin REAL,
    niacin REAL,
    vitamin_C REAL,
    vitamin_D REAL,
    cholesterol REAL,
    saturated_fat REAL,
    trans_fat REAL,
    sucrose REAL,
    glucose REAL,
    fructose REAL,
    lactose REAL,
    maltose REAL,
    magnesium REAL,
    zinc REAL,
    copper REAL,
    manganese REAL,
    selenium REAL,
    tocopherol REAL,
    tocotrienol REAL,
    folate REAL,
    vitamin_B12 REAL,
    total_amino_acids REAL,
    isoleucine REAL,
    leucine REAL,
    lysine REAL,
    methionine REAL,
    phenylalanine REAL,
    threonine REAL,
    valine REAL,
    histidine REAL,
    arginine REAL,
    tyrosine REAL,
    cysteine REAL,
    alanine REAL,
    aspartic_acid REAL,
    glutamic_acid REAL,
    glycine REAL,
    proline REAL,
    serine REAL,
    butyric_acid REAL,
    caproic_acid REAL,
    caprylic_acid REAL,
    capric_acid REAL,
    lauric_acid REAL,
    myristic_acid REAL,
    palmitic_acid REAL,
    stearic_acid REAL,
    arachidic_acid REAL,
    myristoleic_acid REAL,
    palmitoleic_acid REAL,
    oleic_acid REAL,
    vaccenic_acid REAL,
    gadoleic_acid REAL,
    linoleic_acid REAL,
    alpha_linolenic_acid REAL,
    gamma_linolenic_acid REAL,
    eicosadienoic_acid REAL,
    arachidonic_acid REAL,
    eicosatrienoic_acid REAL,
    eicosapentaenoic_acid REAL,
    docosapentaenoic_acid REAL,
    docosahexaenoic_acid REAL,
    trans_oleic_acid REAL,
    trans_linoleic_acid REAL,
    trans_linolenic_acid REAL,
    caffeine REAL,
    source_code TEXT,
    source_name TEXT,
    food_weight REAL,
    company_name TEXT,
    total_calories REAL
)
''')

# column_mappings 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS column_mappings (
    korean_name TEXT PRIMARY KEY,
    english_name TEXT
)
''')

# column_mappings 데이터 삽입
for korean_name, english_name in column_mappings.items():
    cursor.execute('INSERT INTO column_mappings (korean_name, english_name) VALUES (?, ?)', (korean_name, english_name))

# food_nutrition 데이터 삽입
df.to_sql('food_nutrition', conn, if_exists='append', index=False)

# 변경사항 저장 및 연결 종료
conn.commit()
conn.close()
