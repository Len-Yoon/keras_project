import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- 1단계: 데이터 로드 및 전처리 ---

# 캡션 CSV 파일 경로 설정
caption_csv_path = './sample_data/compressed_captions.csv'

# 파일 존재 여부 확인
if not os.path.exists(caption_csv_path):
    raise FileNotFoundError(f"캡션 파일이 존재하지 않습니다: {caption_csv_path}")

# CSV 파일에서 캡션 불러오기 (utf-8-sig 인코딩)
print("데이터를 불러오는 중입니다...")
df = pd.read_csv(caption_csv_path, encoding='utf-8-sig')

# 유효한 캡션만 필터링 (null 값 및 공백 제거)
df.dropna(subset=['caption'], inplace=True)
df = df[df['caption'].str.strip() != '']
# 인덱스 재설정
df.reset_index(drop=True, inplace=True)

print(f"전처리 후 총 {len(df)}개의 캡션을 사용합니다.")


# --- 2단계: TF-IDF 벡터화 ---

# TF-IDF Vectorizer 객체 생성
# TfidfVectorizer는 단어의 중요도를 계산하여 텍스트를 벡터로 변환합니다.
print("텍스트 데이터를 벡터로 변환하는 중입니다...")
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))

# 데이터셋의 모든 캡션을 TF-IDF 행렬로 변환
tfidf_matrix = vectorizer.fit_transform(df['caption'])
print("벡터 변환 완료.")


# --- 3단계: 텍스트 검색 함수 정의 ---

def find_similar_captions(query, vectorizer, tfidf_matrix, dataframe, top_n=5):
    """
    주어진 쿼리와 가장 유사한 캡션을 top_n개 만큼 찾아 반환합니다.
    """
    if not query.strip():
        print("검색어가 비어있습니다.")
        return

    print(f"\n--- 검색어 '{query}'에 대한 유사도 검색 결과 (Top {top_n}) ---")

    # 1. 검색어를 TF-IDF 벡터로 변환
    query_vector = vectorizer.transform([query])

    # 2. 코사인 유사도 계산
    # 검색어 벡터와 전체 캡션 벡터 간의 유사도를 계산합니다.
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # 3. 유사도가 높은 순서대로 인덱스 정렬
    # argsort()는 값을 정렬했을 때의 인덱스를 반환합니다.
    # [::-1]로 뒤집어 내림차순으로 만듭니다.
    similar_indices = cosine_similarities.argsort()[::-1]

    # 4. 상위 N개 결과 출력
    for i in range(top_n):
        idx = similar_indices[i]
        score = cosine_similarities[idx]
        caption = dataframe['caption'][idx]

        # 유사도가 0인 경우는 의미 없는 결과이므로 제외
        if score == 0:
            continue

        print(f"[{i+1}위] 유사도: {score:.4f} | 내용: {caption}")


# --- 4단계: 테스트 실행 ---

# !!! 아래 검색어를 원하는 텍스트로 바꾸어 테스트해 보세요 !!!
search_query_1 = "a pink pokémon"
find_similar_captions(search_query_1, vectorizer, tfidf_matrix, df, top_n=5)

search_query_2 = "legendary bird of fire"
find_similar_captions(search_query_2, vectorizer, tfidf_matrix, df, top_n=5)

search_query_3 = "turtle with cannons on its back"
find_similar_captions(search_query_3, vectorizer, tfidf_matrix, df, top_n=5)