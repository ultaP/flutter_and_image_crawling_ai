import fitz  # PyMuPDF
import pandas as pd
import tabula
import openpyxl

# PDF 파일 경로 및 특정 페이지 번호 설정
pdf_path = '2020ko.pdf'
page_number = 19  # 추출하려는 페이지 번호
excel_path = 'output{}.xlsx'.format(page_number)
# PDF 파일 열기
pdf_document = fitz.open(pdf_path)

# 특정 페이지 추출
page = pdf_document.load_page(page_number - 1)  # 페이지 번호는 0부터 시작
page_text = page.get_text("text")

# 페이지의 텍스트를 확인하여 표가 있는 부분을 찾기
print(page_text)

# Tabula를 사용하여 특정 페이지에서 표 추출
# tabula.read_pdf는 페이지 번호가 1부터 시작
tables = tabula.read_pdf(pdf_path, pages=page_number, multiple_tables=True)

# 추출된 표를 데이터프레임으로 변환 및 확인
if tables:
    df = tables[0]  # 첫 번째 표를 선택
    print(df)
    
    # 엑셀 파일로 저장
    
    df.to_excel(excel_path, index=False)
    print(f'Table extracted and saved to {excel_path}')
else:
    print('No tables found on the specified page.')

# PDF 파일 닫기
pdf_document.close()
