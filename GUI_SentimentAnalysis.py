import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from underthesea import word_tokenize
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import pickle

from opencage.geocoder import OpenCageGeocode

key = '71b566a96e254c919a6632917d6b837f'
geocoder = OpenCageGeocode(key)

def geocode_address(address):
    results = geocoder.geocode(address)
    if results and len(results):
        return results[0]['geometry']['lat'], results[0]['geometry']['lng']
    else:
        return None, None


# Chuyển đổi ngày thành mùa
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    
# Làm sạch và chuyển đổi 'Review Date' để phân tích xu hướng
def clean_review_date(date):
    cleaned_date = re.sub(r'Đã nhận xét vào ', '', date)
    return cleaned_date

# Tính điểm trung bình theo thời gian lưu trú
def extract_stay_duration(stay_details):
    match = re.search(r'(\d+)', stay_details)
    return int(match.group(1)) if match else 0

# Hàm xử lý văn bản
def preprocess_text(text):
    # Chuyển đổi thành chữ thường
    text = text.lower()
    # Loại bỏ dấu câu và ký tự đặc biệt
    text = re.sub(f"[{string.punctuation}]", "", text)
    # Loại bỏ các khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    # Tách từ
    text = word_tokenize(text, format="text")
    return text

# load dữ liệu
comments_data = pd.read_csv('data_hotel_comments_clean.csv')
profiles_data = pd.read_csv('data_hotel_profiles_cleaned.csv')

# Distinct hotel ID
df_hotel_id = profiles_data['Hotel ID'] + '\t' + profiles_data['Hotel Name']

# # GUI
# st.title("ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE")
# st.header('Sentiment Analysis', )
# st.subheader('HV1: _Thang Tuấn Văn_')
# st.subheader('HV2: _Nguyễn Xuân Trường_')
# st.divider()
# st.write("""### Giới thiệu về project
# - Hỗ trợ phân loại các phản hồi của khách hàng thành các nhóm: tích cực, tiêu cực, trung tính dựa trên dữ liệu dạng văn bản.
# - Hỗ trợ thống kê, cung cấp insight cụ thể, chính xác cho cho chủ khách sạn/resort khi họ đăng nhập vào hệ thống, giúp họ thể biết được những phản hồi nhanh chóng của khách hàng về dịch vụ của họ để cải thiện hơn trong dịch vụ. 
# """)
# st.divider()

# Thêm tiêu đề vào sidebar
st.sidebar.header("ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE")
st.sidebar.markdown("#### Sentiment Analysis")
st.sidebar.write('HV1: _Thang Tuấn Văn_')
st.sidebar.write('HV2: _Nguyễn Xuân Trường_')
st.sidebar.divider()

menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)

st.sidebar.divider()

st.sidebar.write("""#### Giới thiệu về project
- Hỗ trợ phân loại các phản hồi của khách hàng thành các nhóm: tích cực, tiêu cực, trung tính dựa trên dữ liệu dạng văn bản.
- Hỗ trợ thống kê, cung cấp insight cụ thể, chính xác cho cho chủ khách sạn/resort khi họ đăng nhập vào hệ thống, giúp họ thể biết được những phản hồi nhanh chóng của khách hàng về dịch vụ của họ để cải thiện hơn trong dịch vụ. 
""")

if choice == 'Business Objective':  
    st.subheader("Business Objective")

    # Chọn khách sạn - Multiselect
    option = st.selectbox(
        "Nhập ID khách sạn",
        df_hotel_id,
        index=None,
        placeholder="Ví dụ: 1_1",
    )

    # Kết hợp dữ liệu từ hai bảng dựa trên Hotel ID
    merged_data = comments_data.merge(profiles_data, on='Hotel ID', how='left')


    # Thêm cột mùa vào dữ liệu
    merged_data['Review Month'] = merged_data['Review Date'].apply(lambda x: int(re.search(r'\d+', x).group()))
    merged_data['Season'] = merged_data['Review Month'].apply(get_season)

    hotel_data = pd.DataFrame()
    # Lọc lại thông tin khách sạn và comment 
    if option != None:
        hotel_id = option.split('\t')[0]
        
        # Lấy profiles và comments với ID khách sạn đã nhập
        hotel_data = merged_data[merged_data['Hotel ID'] == hotel_id]

    tab1, tab2, tab3 = st.tabs(["Thông tin khách sạn", "Phân loại phản hồi", "Thống kê"])

    if hotel_data.empty != True:
        with tab1:  # Thông tin khách sạn
            # Lấy thông tin tổng quan của khách sạn
            hotel_info = hotel_data.iloc[0]
            hotel_name = hotel_info['Hotel Name']
            hotel_address = hotel_info['Hotel Address']
            hotel_rank = hotel_info['Hotel Rank']
            total_score = hotel_info['Total Score']
            location_score = hotel_info['Vị trí']
            cleanliness_score = hotel_info['Độ sạch sẽ']
            service_score = hotel_info['Dịch vụ']
            facilities_score = hotel_info['Tiện nghi']
            value_for_money_score = hotel_info['Đáng giá tiền']
            comfort_score = hotel_info['Sự thoải mái và chất lượng phòng']  

            # In thông tin tổng quan của khách sạn
            st.write(f"\nThông tin tổng quan về khách sạn: {hotel_name} (ID: {hotel_id})")
            st.write(f"Địa chỉ: {hotel_address}")
            st.write(f"Xếp hạng: {hotel_rank}")
            st.write(f"Điểm tổng thể: {total_score}")
            st.write(f"Điểm vị trí: {location_score}")
            st.write(f"Điểm độ sạch sẽ: {cleanliness_score}")
            st.write(f"Điểm dịch vụ: {service_score}")
            st.write(f"Điểm tiện nghi: {facilities_score}")
            st.write(f"Điểm đáng giá tiền: {value_for_money_score}")
            st.write(f"Điểm sự thoải mái và chất lượng phòng: {comfort_score}")

            # Plot Map
            # Chuyển đổi địa chỉ thành tọa độ
            latitude, longitude = geocode_address(hotel_address)
            # Nếu tìm thấy tọa độ, hiển thị bản đồ
            if latitude and longitude:
                st.write(f"Coordinates: Latitude = {latitude}, Longitude = {longitude}")
                
                # Tạo DataFrame với tọa độ
                df = pd.DataFrame({
                    'lat': [latitude],
                    'lon': [longitude]
                })

                # Vẽ bản đồ
                st.map(df)
            else:
                st.write("Could not find the location. Please check the address.")

        with tab2: # Phân loại phản hồi
            # Tạo WordCloud cho các phản hồi tích cực
            positive_texts = " ".join(str(review) for review in hotel_data[hotel_data['Sentiment'] == 'positive']['Body_new'])
            
            # Hiển thị WordClouds
            if positive_texts.strip:
                st.subheader("Phản hồi tích cực")
      
                positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)

                plt.figure(figsize=(10, 5))
                fig = plt.imshow(positive_wordcloud, interpolation='bilinear')
                plt.title('Positive Reviews')
                plt.axis('off')
                # plt.show()
                st.pyplot(fig.figure)

            # Tạo WordCloud cho các phản hồi trung tính
            neutral_texts = " ".join(str(review) for review in hotel_data[hotel_data['Sentiment'] == 'neutral']['Body_new'])

            # Hiển thị WordClouds
            if neutral_texts.strip:
                st.subheader("Phản hồi trung tính")

                neutral_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(neutral_texts)

                plt.figure(figsize=(10, 5))
                fig = plt.imshow(neutral_wordcloud, interpolation='bilinear')
                plt.title('Neutral Reviews')
                plt.axis('off')
                # plt.show()
                st.pyplot(fig.figure)


            # Tạo WordCloud cho các phản hồi tiêu cực
            negative_texts = " ".join(str(review) for review in hotel_data[hotel_data['Sentiment'] == 'negative']['Body_new'])

            # Hiển thị WordClouds
            if negative_texts.strip:
                st.subheader("Phản hồi tiêu cực")

                negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_texts)

                plt.figure(figsize=(10, 5))
                fig = plt.imshow(negative_wordcloud, interpolation='bilinear')
                plt.title('Negative Reviews')
                plt.axis('off')
                # plt.show()
                st.pyplot(fig.figure)


        with tab3:  # Thống kê
            ##############################################################
            st.subheader("Phân tích điểm số trung bình theo loại phòng")
            
            # Phân tích điểm số trung bình theo loại phòng
            room_type_scores = hotel_data.groupby('Room Type')['Score'].mean().sort_values()
            st.write("\nĐiểm số trung bình theo loại phòng:")
            st.write(room_type_scores)

            # Hiển thị biểu đồ điểm số theo loại phòng
            plt.figure(figsize=(10, 5))
            fig = room_type_scores.plot(kind='bar', color='skyblue')
            plt.title('Điểm số trung bình theo loại phòng')
            plt.xlabel('Loại phòng')
            plt.ylabel('Điểm số trung bình')
            plt.xticks(rotation=45, ha='right')
            # plt.show()
            # st.pyplot(fig.figure)
            st.pyplot(plt.gcf())
            plt.clf()

            ##############################################################
            st.subheader("Phân tích điểm số trung bình theo nhóm khách")
            
            # Phân tích điểm số trung bình theo nhóm khách
            group_scores = hotel_data.groupby('Group Name')['Score'].mean().sort_values()
            st.write("\nĐiểm số trung bình theo nhóm khách:")
            st.write(group_scores) 
            
            # Hiển thị biểu đồ điểm số theo nhóm khách
            plt.figure(figsize=(10, 5))
            fig = group_scores.plot(kind='bar', color='lightgreen')
            plt.title('Điểm số trung bình theo nhóm khách')
            plt.xlabel('Nhóm khách')
            plt.ylabel('Điểm số trung bình')
            plt.xticks(rotation=45, ha='right')
            # plt.show()
            # st.pyplot(fig.figure)
            st.pyplot(plt.gcf())
            plt.clf()

            ##############################################################
            st.subheader("Phân bố nhóm khách")
            
            # Phân bố nhóm khách
            group_distribution = hotel_data['Group Name'].value_counts(normalize=True)
            st.write("\nTỷ lệ nhóm khách:")
            st.write(group_distribution)        

            # Hiển thị biểu đồ phân bố nhóm khách
            plt.figure(figsize=(10, 5))
            fig = group_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set3'))
            plt.title('Phân bố nhóm khách')
            plt.ylabel('')
            # plt.show()
            # st.pyplot(fig.figure)
            st.pyplot(plt.gcf())
            plt.clf()

            ##############################################################
            st.subheader("Phân bố theo thời gian lưu trú")
            
            # Phân bố theo thời gian lưu trú
            hotel_data['Stay Duration'] = hotel_data['Stay Details'].apply(extract_stay_duration)
            average_score_by_stay_duration = hotel_data.groupby('Stay Duration')['Score'].mean()
            st.write("\nĐiểm số trung bình theo thời gian lưu trú:")
            st.write(average_score_by_stay_duration)

            fig = sns.barplot(x=average_score_by_stay_duration.index, y=average_score_by_stay_duration.values)
            plt.title('Average Score by Stay Duration')
            plt.xlabel('Stay Duration')
            plt.ylabel('Average Score')
            # plt.show()
            # st.pyplot(fig.figure)
            st.pyplot(plt.gcf())
            plt.clf()

            ##############################################################
            st.subheader("Phân bố theo quốc tịch")

            nationality_distribution = hotel_data['Nationality'].value_counts(normalize=True)
            st.write("\nTỷ lệ theo quốc tịch:")
            st.write(nationality_distribution)

            plt.figure(figsize=(10, 6))
            fig = sns.barplot(x=nationality_distribution.index, y=nationality_distribution.values, palette='coolwarm')
            plt.title('Phân bố quốc tịch tại khách sạn')
            plt.xlabel('Quốc tịch')
            plt.ylabel('Tỷ lệ')

            for p in fig.patches:
                height = p.get_height()
                fig.annotate(f"{height:.1%}", (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                            textcoords='offset points')
            plt.ylim(0, nationality_distribution.max() + 0.1)
            plt.xticks(rotation=45, ha='right')     

            # st.pyplot(fig.figure)
            st.pyplot(plt.gcf())
            plt.clf()

            ##############################################################
            st.subheader("Phân tích điểm số trung bình theo quốc tịch")
            
            nationality_scores = hotel_data.groupby('Nationality')['Score'].mean().sort_values()
            st.write("\nĐiểm số trung bình theo quốc tịch:")
            st.write(nationality_scores)

            # Hiển thị biểu đồ điểm số theo quốc tịch
            plt.figure(figsize=(12, 6))
            fig = nationality_scores.plot(kind='bar', color='salmon')
            plt.title('Điểm số trung bình theo quốc tịch')
            plt.xlabel('Quốc tịch')
            plt.ylabel('Điểm số trung bình')
            plt.xticks(rotation=45, ha='right')
            # plt.show()
            # st.pyplot(fig.figure)
            st.pyplot(plt.gcf())
            plt.clf()

            ##############################################################
            st.subheader("Phân tích xu hướng điểm số theo thời gian")
                
            hotel_data['Review Year'] = hotel_data['Review Date'].apply(lambda x: int(re.search(r'\d{4}', x).group()))
            time_trend_scores = hotel_data.groupby(['Review Year', 'Review Month'])['Score'].mean().sort_index()
            st.write("\nXu hướng điểm số theo thời gian:")
            st.write(time_trend_scores)

            # Hiển thị biểu đồ xu hướng điểm số theo thời gian
            plt.figure(figsize=(14, 7))
            fig = time_trend_scores.plot(kind='line', marker='o', color='purple')
            plt.title('Xu hướng điểm số theo thời gian')
            plt.xlabel('Thời gian (Năm, Tháng)')
            plt.ylabel('Điểm số trung bình')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            # plt.show()
            # st.pyplot(fig.figure)
            st.pyplot(plt.gcf())
            plt.clf()

            ##############################################################
            st.subheader("Phân tích điểm số theo mùa")
            
            # Phân tích điểm số theo mùa
            season_scores = hotel_data.groupby('Season')['Score'].mean().sort_values()
            st.write("\nĐiểm số trung bình theo mùa:")
            st.write(season_scores)

            # Hiển thị biểu đồ điểm số theo mùa
            plt.figure(figsize=(8, 5))
            fig = season_scores.plot(kind='bar', color='orange')
            plt.title('Điểm số trung bình theo mùa')
            plt.xlabel('Mùa')
            plt.ylabel('Điểm số trung bình')
            plt.xticks(rotation=45, ha='right')
            # plt.show()
            # st.pyplot(fig.figure)
            st.pyplot(plt.gcf())
            plt.clf()

            # ##############################################################
            # st.subheader("Phân tích phản hồi từ nhóm khách")
            # st.write("\nPhản hồi từ nhóm khách:")
            # for group in hotel_data['Group Name'].unique():
            #     st.write(f"\nNhóm khách: {group}")
            #     st.write(hotel_data[hotel_data['Group Name'] == group]['Body'].sample(3, random_state=42).tolist())

elif choice == 'Build Project':
    # st.subheader("Build Project")

    # Chuẩn hóa văn bản và mã hóa nhãn sentiment
    comments_data['Body_new'] = comments_data['Body_new'].str.lower()

    # Xử lý giá trị thiếu
    comments_data = comments_data.dropna(subset=['Body_new'])

    # Trực quan hóa dữ liệu
    st.subheader('Phân bố cảm xúc')
    sns.set(style="whitegrid")
    sentiment_counts = comments_data['Sentiment'].value_counts(normalize=True)
    plt.figure(figsize=(8, 5))
    fig = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
    plt.title('Phân bố cảm xúc')
    plt.xlabel('Cảm xúc')
    plt.ylabel('Tỷ lệ')
    plt.ylim(0, 1)
    # plt.show()
    st.pyplot(fig.figure)
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X = comments_data['Body_new']
    y = comments_data['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

    # Vector hóa dữ liệu văn bản bằng TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)        

    # Huấn luyện mô hình Naïve Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_predictions = nb_model.predict(X_test_tfidf)
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    nb_report = classification_report(y_test, nb_predictions)
    nb_confusion_matrix = confusion_matrix(y_test, nb_predictions) 

    # Huấn luyện mô hình Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_predictions = lr_model.predict(X_test_tfidf)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    lr_report = classification_report(y_test, lr_predictions)
    lr_confusion_matrix = confusion_matrix(y_test, lr_predictions)

    # Đánh giá mô hình
    st.write(f"Độ chính xác Naïve Bayes: {nb_accuracy}")
    st.write("Báo cáo phân loại Naïve Bayes:\n", nb_report)
    st.write("Naïve Bayes Confusion Matrix:\n", nb_confusion_matrix)

    st.write(f"Độ chính xác Logistic Regression: {lr_accuracy}")
    st.write("Báo cáo phân loại Logistic Regression:\n", lr_report)
    st.write("Logistic Regression Confusion Matrix:\n", lr_confusion_matrix)

    # Vẽ ma trận nhầm lẫn cho Naïve Bayes
    nb_cm = confusion_matrix(y_test, nb_predictions)
    plt.figure(figsize=(10, 4))
    fig = sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Ma trận nhầm lẫn Naïve Bayes')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    # plt.show()
    st.pyplot(fig.figure)

    # Vẽ ma trận nhầm lẫn cho Logistic Regression
    lr_cm = confusion_matrix(y_test, lr_predictions)
    plt.figure(figsize=(10, 4))
    fig = sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Ma trận nhầm lẫn Logistic Regression')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    # plt.show()
    st.pyplot(fig.figure)

    st.write("""
             #### Nhận xét
1. Naive Bayes
- Độ chính xác tổng thể: 0.839 (83.9%)
- Cảm xúc tiêu cực (negative):
    + Precision cao (0.95) nhưng recall thấp (0.24), điều này cho thấy mô hình dự đoán chính xác các mẫu tiêu cực nhưng bỏ sót nhiều mẫu tiêu cực thực tế.
- Cảm xúc trung tính (neutral) và tích cực (positive):
    + Có tỷ lệ recall cao cho tích cực, nhưng đối với neutral thì kết quả cũng tương đối tốt so với tiêu cực.
2. Logistic Regression
- Độ chính xác tổng thể: 0.978 (97.8%)
- Cảm xúc tiêu cực (negative):
    + Cải thiện đáng kể với precision 0.91 và recall 0.71 so với Naive Bayes, cho thấy mô hình này phát hiện tốt hơn các đánh giá tiêu cực.
- Cảm xúc trung tính (neutral) và tích cực (positive):
    + Rất hiệu quả với precision và recall đều cao cho cả hai loại, đặc biệt là tích cực với recall lên đến 0.99.

=> Dựa trên kết quả, Logistic Regression là lựa chọn tốt hơn để triển khai trong thực tế do có độ chính xác cao và đánh giá tốt cho tất cả các loại cảm xúc.
- Tuy nhiên, cả hai mô hình đều cho thấy độ nhạy (recall) thấp đối với cảm xúc tiêu cực. Có thể cải thiện bằng cách điều chỉnh ngưỡng phân loại để cân bằng dữ liệu.           
             """) 
    

    # luu model linear regression
    pkl_filename = "sentiment_model.pkl"  
    with open(pkl_filename, 'wb') as file:  
        pickle.dump(lr_model, file)

    # luu model TFIDF
    pkl_tfidf = "tfidf_model.pkl"  
    with open(pkl_tfidf, 'wb') as file:  
        pickle.dump(tfidf_vectorizer, file)


elif choice == 'New Prediction':
    st.subheader("Select data")
    
    # Load các mô hình đã lưu
    pkl_filename = "sentiment_model.pkl"
    with open(pkl_filename, 'rb') as file:  
        lr_sentiment_model = pickle.load(file)

    pkl_tfidf = "tfidf_model.pkl"
    with open(pkl_tfidf, 'rb') as file:  
        tfidf_model = pickle.load(file)

    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type == "Upload":
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            if lines.shape[1] == 1:  # Nếu chỉ có một cột
                lines.columns = ['Input Data']  # Đặt tên cột cho dữ liệu
            else:
                # Gộp các cột thành một, bỏ qua các giá trị NaN
                lines[0] = lines.apply(lambda row: ' '.join(row.dropna().values.astype(str)), axis=1)
                lines = lines[[0]]  # Chỉ giữ lại cột đã gộp
                lines.columns = ['Input Data']  # Đặt tên cột là 'Input Data'
            lines = lines.dropna()  # Loại bỏ các giá trị NaN nếu có
            st.write(lines)  # Hiển thị bảng dữ liệu đã upload
            flag = True                          
    if type == "Input":        
        content = st.text_area(label="Input your content:")
        if content != "":
            lines = pd.DataFrame([content], columns=['Input Data'])
            flag = True

    if flag:
        st.write("Content:")
        if len(lines) > 0:
            # Xử lý văn bản đầu vào
            processed_lines = lines['Input Data'].apply(preprocess_text)

            # Chuyển đổi văn bản đã xử lý thành TF-IDF
            x_new = tfidf_model.transform(processed_lines)

            # Dự đoán
            y_pred_new = lr_sentiment_model.predict(x_new)
            
            # Thêm cột kết quả dự đoán vào dataframe
            lines['Prediction'] = y_pred_new

            # Hiển thị bảng dữ liệu kèm kết quả dự đoán
            st.write(lines)




