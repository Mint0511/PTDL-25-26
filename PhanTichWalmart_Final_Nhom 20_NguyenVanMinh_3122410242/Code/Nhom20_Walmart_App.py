import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Walmart Analytics Dashboard", 
    page_icon="🏪", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Không custom CSS - dùng giao diện mặc định Streamlit

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

DATA_DIR = "data"
OUTPUTS_DIR = "outputs"
RANDOM_STATE = 42

# ==================== HELPER FUNCTIONS ====================

def format_currency(value):
    """Format số tiền với dấu $ và phân cách hàng nghìn"""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.1f}K"
    return f"${value:,.0f}"

def format_percent(value):
    """Format phần trăm"""
    return f"{value:.1f}%"

def show_info_box(text, box_type="info"):
    """Hiển thị info box đơn giản"""
    if box_type == "info":
        st.info(text)
    elif box_type == "warning":
        st.warning(text)
    elif box_type == "success":
        st.success(text)
    else:
        st.info(text)

def explain_term(term, explanation):
    """Hiển thị thuật ngữ với explanation"""
    with st.expander(f"❓ {term} là gì?"):
        st.write(explanation)

def format_dept_label(dept_num, df=None, show_stats=True):
    """Format Department label với thông tin doanh số
    
    Args:
        dept_num: Số department
        df: DataFrame để tính stats (optional)
        show_stats: Có hiển thị stats không
    
    Returns:
        Formatted string
    """
    if dept_num == "Tất cả":
        return "Tất cả phòng ban"
    
    dept_num = int(dept_num)
    
    if not show_stats or df is None:
        return f"Dept #{dept_num:02d}"
    
    # Tính stats cho department này
    dept_data = df[df["Dept"] == dept_num]
    if len(dept_data) == 0:
        return f"Dept #{dept_num:02d}"
    
    avg_sales = dept_data["Weekly_Sales"].mean()
    
    # Phân loại theo doanh số (giống Store Type)
    if avg_sales >= 20000:
        tier = "A"  # Top tier
    elif avg_sales >= 15000:
        tier = "B"  # High tier
    elif avg_sales >= 10000:
        tier = "C"  # Mid tier
    else:
        tier = "D"  # Low tier
    
    return f"[{tier}] Dept #{dept_num:02d} (TB: {format_currency(avg_sales)})"

def get_store_ranking_info(df):
    """Tạo thông tin ranking cho tất cả stores"""
    store_stats = df.groupby("Store").agg({
        "Store_Total_Sales": ["mean", "median", "count"]
    }).round(0)
    store_stats.columns = ["Avg", "Median", "Count"]
    store_stats = store_stats.sort_values("Avg", ascending=False)
    store_stats["Rank"] = range(1, len(store_stats) + 1)
    return store_stats

# ==================== DATA LOADING ====================

@st.cache_data(show_spinner=False)
def load_data():
    """Load và chuẩn bị dữ liệu theo đúng notebook"""
    
    # Load datasets
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
    
    # Merge datasets (giống notebook)
    df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = df.merge(stores, on='Store', how='left')
    
    # Chuẩn hóa ngày tháng
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    
    # Xử lý missing values cho MarkDown columns
    markdown_cols = [col for col in df.columns if 'MarkDown' in col]
    for col in markdown_cols:
        df[col].fillna(0, inplace=True)
    
    # Xóa dữ liệu trùng lặp
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        print(f"Đã xóa {duplicates_count} dòng trùng lặp")
        df = df.drop_duplicates()
    
    # Xóa doanh số âm (giống notebook)
    negative_count = (df['Weekly_Sales'] < 0).sum()
    if negative_count > 0:
        print(f"Đã xóa {negative_count} dòng có doanh số âm")
        df = df[df['Weekly_Sales'] >= 0].copy()
    
    # Tạo dataset tổng hợp theo cửa hàng (giống notebook)
    df_store = df.groupby(['Store', 'Date']).agg({
        'Weekly_Sales': 'sum',           # Tổng doanh số tất cả departments
        'IsHoliday': 'first',            # Giữ nguyên (giống nhau cho tất cả dept)
        'Temperature': 'first',
        'Fuel_Price': 'first',
        'CPI': 'first',
        'Unemployment': 'first',
        'Type': 'first',
        'Size': 'first',
        'Month': 'first',
        'Quarter': 'first',
        'Year': 'first'
    }).reset_index()
    
    # Đổi tên cột Weekly_Sales thành Store_Total_Sales để rõ ràng hơn
    df_store.rename(columns={'Weekly_Sales': 'Store_Total_Sales'}, inplace=True)
    
    print(f"Dataset gốc: {df.shape[0]:,} records (chi tiết theo department)")
    print(f"Dataset tổng hợp: {df_store.shape[0]:,} records (tổng hợp theo cửa hàng)")
    
    return df_store, df, train, features, stores

# ==================== SIDEBAR ====================

def sidebar(df):
    """Tạo sidebar với filters và navigation"""
    st.sidebar.title("🏪 Walmart Analytics")
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("ℹ️ Về App Này", expanded=False):
        st.markdown("""
        ### Dashboard Phân Tích Walmart
        
        **Dữ liệu:** 6,435 records từ 45 cửa hàng (2010-2012)
        
        **Chức năng chính:**
        - 📊 Phân tích xu hướng & patterns
        - 🎯 Phân nhóm cửa hàng thông minh
        - 💡 Insights & khuyến nghị thực tế
        
        **Cách dùng:**
        1. Lọc dữ liệu ở sidebar này
        2. Chọn trang phân tích bên dưới
        3. Tương tác với biểu đồ & thông số
        4. Đọc insights để ra quyết định
        
        💼 *Công cụ hỗ trợ ra quyết định kinh doanh*
        """)
    
    st.sidebar.markdown("### 🔍 Bộ Lọc Dữ Liệu")
    
    # Filter năm
    year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
    y1, y2 = st.sidebar.slider(
        "📅 Khoảng thời gian", 
        min_value=year_min, 
        max_value=year_max, 
        value=(year_min, year_max),
        help="Chọn khoảng năm để phân tích. Dữ liệu từ 2010-2012."
    )
    
    # Filter loại cửa hàng
    types = sorted(df["Type"].dropna().unique())
    type_sel = st.sidebar.multiselect(
        "🏪 Loại cửa hàng", 
        options=types, 
        default=types,
        help="A=Super Center (lớn), B=Discount Store (trung), C=Neighborhood Market (nhỏ)"
    )
    
    # Filter cửa hàng với thông tin ranking
    store_list = ["Tất cả"] + sorted([int(x) for x in df["Store"].dropna().unique()])
    
    # Tạo mapping store -> label
    store_labels = {}
    store_stats = get_store_ranking_info(df)
    
    for store in store_list:
        if store == "Tất cả":
            store_labels[store] = "🏪 Tất cả cửa hàng"
        else:
            avg_sales = store_stats.loc[store, "Avg"]
            rank = int(store_stats.loc[store, "Rank"])
            store_type = df[df["Store"] == store]["Type"].iloc[0]
            
            # Tiếp đầu ngữ theo ranking
            if rank <= 15:
                tier = "A"
            elif rank <= 30:
                tier = "B"
            else:
                tier = "C"
            
            store_labels[store] = f"[{tier}] Store #{store:02d} ({store_type}, #{rank}, TB: {format_currency(avg_sales)})"
    
    store_option = st.sidebar.selectbox(
        "🏪 Cửa hàng (Store)", 
        options=store_list,
        format_func=lambda x: store_labels[x],
        help="Chọn cửa hàng - [A/B/C] theo doanh số: A=Cao nhất, C=Thấp nhất"
    )
    
    # Áp dụng filters
    df_view = df[(df["Year"].between(y1, y2)) & (df["Type"].isin(type_sel))].copy()
    if store_option != "Tất cả":
        df_view = df_view[df_view["Store"] == store_option]
    
    # Thống kê filter
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Dữ liệu đã lọc:**")
    st.sidebar.info(f"""  
    - {df_view['Store'].nunique()} cửa hàng
    - {len(df_view):,} records
    - {format_currency(df_view['Store_Total_Sales'].sum())} tổng doanh số
    """)
    
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📑 Chọn Trang Phân Tích",
        (
            "🏠 Tổng quan",
            "📊 So sánh cửa hàng",
            "📈 Xu hướng thời gian",
            "🎉 Phân tích ngày lễ",
            "🔍 Phân nhóm thông minh",
            "🌳 Dự đoán Decision Tree",
            "💡 Dự đoán doanh số"
        ),
        help="Chọn trang để xem phân tích chi tiết"
    )
    
    return page, df_view

# ==================== PAGE: TỔNG QUAN ====================

def page_overview(train, features, stores, df):
    """Trang tổng quan với insights thực tế"""
    st.title("🏠 Tổng Quan Phân Tích Walmart")
    
    st.markdown("""
    ### Chào mừng đến với Dashboard Phân Tích Walmart! 👋
    
    App này giúp bạn khám phá **dữ liệu doanh số thực tế** từ 45 cửa hàng Walmart trong giai đoạn 2010-2012.
    Không chỉ đơn thuần là số liệu, chúng ta sẽ tìm ra **insights có giá trị** để ra quyết định kinh doanh!
    """)
    
    st.markdown("---")
    
    # Key Metrics
    st.subheader("📊 Chỉ Số Quan Trọng")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = df['Store_Total_Sales'].sum()
        st.metric(
            "💰 Tổng Doanh Số", 
            format_currency(total_sales),
            help="Tổng doanh số của tất cả cửa hàng trong giai đoạn phân tích"
        )
        
    with col2:
        avg_sales = df['Store_Total_Sales'].mean()
        st.metric(
            "📊 Doanh Số TB/Tuần", 
            format_currency(avg_sales),
            help="Doanh số trung bình mỗi tuần, mỗi phòng ban"
        )
    
    with col3:
        cv = (df['Store_Total_Sales'].std() / df['Store_Total_Sales'].mean()) * 100
        st.metric(
            "📈 Độ Biến Động", 
            f"{cv:.1f}%",
            help="Coefficient of Variation - đo mức độ biến động doanh số"
        )
    
    with col4:
        st.metric(
            "🏪 Số Cửa Hàng", 
            f"{df['Store'].nunique()}",
            help="Tổng số cửa hàng Walmart trong phân tích"
        )
    
        explain_term(
            "Độ Biến Động (CV) có ý nghĩa gì?",
            f"""
**Coefficient of Variation (CV) = {cv:.1f}%** cho thấy:

- **CV < 15%**: Doanh số rất ổn định (dễ dự đoán)
- **CV 15-30%**: Biến động trung bình (có thể quản lý)
- **CV > 30%**: Biến động cao (khó dự đoán) ← **Walmart đang ở đây!**

**Ý nghĩa:** Có những tuần bán rất tốt (ngày lễ) và tuần bán yếu (sau lễ).
Cần chiến lược linh hoạt để tối ưu hàng tồn kho và nhân sự.
"""
        )
    
    # Phân bố doanh số
    st.subheader("📊 Phân Bố Doanh Số - Insight Quan Trọng")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df["Store_Total_Sales"], bins=50, alpha=0.7, color="skyblue", edgecolor='black')
        ax.axvline(df["Store_Total_Sales"].mean(), color='red', linestyle='--', linewidth=2, label=f'Trung bình: {format_currency(avg_sales)}')
        ax.axvline(df["Store_Total_Sales"].median(), color='green', linestyle='--', linewidth=2, label=f'Trung vị: {format_currency(df["Store_Total_Sales"].median())}')
        ax.set_xlabel("Doanh Số Hàng Tuần ($)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Số Lượng Tuần", fontsize=11, fontweight='bold')
        ax.set_title("Histogram Phân Bố Doanh Số\n(Hình dạng phân bố tiết lộ nhiều thông tin!)", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### 🔍 Phát Hiện Quan Trọng")
        
        mean_val = df["Store_Total_Sales"].mean()
        median_val = df["Store_Total_Sales"].median()
        
        show_info_box(f"""
        **📌 Phân bố lệch phải!**
        
        - **Trung bình**: {format_currency(mean_val)}
        - **Trung vị**: {format_currency(median_val)}
        - Có nhiều tuần bán thấp, ít tuần bán rất cao
        - Những tuần cao thường là ngày lễ hoặc khuyến mãi lớn
        
        **💡 Insight:**
        Walmart không đều đặn - cần:
        - Dự báo chính xác tuần nào "hot"
        - Chuẩn bị hàng hóa linh hoạt
        - Tối ưu nhân sự theo mùa
        """, "info")
    
    st.markdown("---")
    
    # So sánh Type
    st.subheader("🏪 So Sánh Theo Loại Cửa Hàng")
    
    type_comparison = df.groupby('Type')['Store_Total_Sales'].agg(['count', 'mean', 'sum']).round(0)
    type_comparison.columns = ['Số Tuần', 'TB Doanh Số', 'Tổng DS']
    type_comparison['% Contribution'] = (type_comparison['Tổng DS'] / type_comparison['Tổng DS'].sum() * 100).round(1)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot([df[df['Type']==t]['Store_Total_Sales'] for t in ['A', 'B', 'C']], 
                       patch_artist=True, widths=0.5,
                       boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2.5))
        ax.set_xticklabels(['A', 'B', 'C'])
        ax.set_ylabel("Doanh Số TB ($)", fontsize=11, fontweight='bold')
        ax.set_title("Boxplot Doanh Số Theo Loại Cửa Hàng", fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### 📊 Bảng So Sánh Chi Tiết")
        display_df = type_comparison.copy()
        display_df['TB Doanh Số'] = display_df['TB Doanh Số'].apply(lambda x: format_currency(x))
        display_df['Tổng DS'] = display_df['Tổng DS'].apply(lambda x: format_currency(x))
        display_df['% Contribution'] = display_df['% Contribution'].apply(lambda x: f"{x}%")
        st.dataframe(display_df, use_container_width=True)
        
        show_info_box("""
        **🎯 Kết luận rõ ràng:**
        
        - **Type A** Super Center: Chiếm ưu thế tuyệt đối
        - **Type B** Discount Store: Trung bình khá
        - **Type C** Neighborhood Market: Yếu nhất
        
        **💡 Khuyến nghị:**
        - Ưu tiên mở rộng Type A
        - Nâng cấp Type B lên Type A nếu có thể
        - Cân nhắc đóng/chuyển đổi Type C kém hiệu quả
        """, "info")
    
    st.markdown("---")
    
    # Lời khuyên tổng hợp
    st.subheader("💡 Khuyến Nghị Hành Động")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_info_box("""
        ### 📅 Theo Mùa Vụ
        
        1. **Quý 4** Oct-Dec: Chuẩn bị 150-200% hàng tồn kho
        2. **Quý 1** Jan-Mar: Giảm giá mạnh để thanh lý
        3. **Ngày lễ**: Tăng nhân sự 30-50%
        """, "success")
    
    with col2:
        show_info_box("""
        ### 🏪 Theo Cửa Hàng
        
        1. **Top performers**: Nhân rộng mô hình
        2. **Average**: Cải thiện marketing địa phương
        3. **Bottom**: Đánh giá lại hoặc đóng cửa
        """, "warning")
    
    with col3:
        show_info_box("""
        ### 🎯 Chiến Lược Chung
        
        1. Tập trung Type A
        2. Dự báo doanh số chính xác
        3. Linh hoạt với biến động cao
        4. Tối ưu theo từng cụm cửa hàng
        """, "info")

# ==================== PAGE: SO SÁNH CỬA HÀNG ====================

def page_compare_stores(df):
    """Trang so sánh cửa hàng với phân tích sâu"""
    st.title("📊 So Sánh Cửa Hàng Chi Tiết")
    
    st.markdown("""
    ### Mục đích: Tìm cửa hàng nào hoạt động tốt nhất và tại sao? 🎯
    
    Chọn tối đa 5 cửa hàng để so sánh xu hướng, hiệu suất và đặc điểm.
    **Mẹo:** Chọn cửa hàng cùng loại (A/B/C) để so sánh công bằng!
    """)
    
    if df.empty:
        st.warning("⚠️ Không có dữ liệu theo bộ lọc hiện tại.")
        return
    
    st.markdown("---")
    
    # Store selection
    stores = sorted(df["Store"].unique())
    store_info = df.groupby('Store')[['Type', 'Size']].first()
    
    def format_store(store):
        info = store_info.loc[store]
        size_category = "Nhỏ" if info['Size'] < 100000 else "Trung bình" if info['Size'] < 150000 else "Lớn"
        return f"Store {store} | Loại {info['Type']} | {size_category} ({info['Size']:,} sq ft)"
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_stores = st.multiselect(
            "🏪 Chọn cửa hàng để so sánh (tối đa 5)", 
            options=stores, 
            default=stores[:3], 
            max_selections=5,
            format_func=format_store,
            help="Chọn các cửa hàng bạn muốn so sánh. Nên chọn cùng loại để dễ phân tích."
        )
    
    with col2:
        comparison_metric = st.selectbox(
            "📊 Chỉ số so sánh",
            ["Doanh số", "Độ ổn định", "Xu hướng"],
            help="Chọn góc nhìn để so sánh cửa hàng"
        )
    
    if not selected_stores:
        st.info("👆 Vui lòng chọn ít nhất một cửa hàng để bắt đầu so sánh.")
        return
    
    st.markdown("---")
    
    # Aggregate data (đã tổng hợp theo store rồi)
    df_filtered = df[df["Store"].isin(selected_stores)]
    df_agg = df_filtered.copy()  # Đã là level store
    
    # Visualization based on comparison metric
    st.subheader(f"📊 Biểu Đồ So Sánh: {comparison_metric}")
    
    if comparison_metric == "Doanh số":
        # Bar chart comparing average sales
        fig, ax = plt.subplots(figsize=(12, 6))
        sales_data = df_agg.groupby("Store")["Store_Total_Sales"].mean().sort_values(ascending=False)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sales_data)))
        bars = ax.bar(range(len(sales_data)), sales_data.values, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xlabel("Cửa Hàng", fontsize=12, fontweight='bold')
        ax.set_ylabel("Doanh Số TB ($)", fontsize=12, fontweight='bold')
        ax.set_title("So Sánh Doanh Số Trung Bình Giữa Các Cửa Hàng", fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(sales_data)))
        ax.set_xticklabels([f'Store {store}' for store in sales_data.index], rotation=45)
        
        for i, v in enumerate(sales_data.values):
            ax.text(i, v, format_currency(v), ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
        # Insights
        best_store = sales_data.idxmax()
        worst_store = sales_data.idxmin()
        best_sales = sales_data.max()
        worst_sales = sales_data.min()
        ratio = best_sales / worst_sales
        
        st.markdown("#### 🏆 Phân Tích Nhanh")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Tốt nhất", f"Store {best_store}", format_currency(best_sales))
        with col2:
            st.metric("📉 Yếu nhất", f"Store {worst_store}", format_currency(worst_sales))
        with col3:
            st.metric("📊 Chênh lệch", f"{ratio:.2f}x", "Tốt nhất gấp X lần")
        
    elif comparison_metric == "Độ ổn định":
        # CV comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        cv_data = df_agg.groupby("Store")["Store_Total_Sales"].std() / df_agg.groupby("Store")["Store_Total_Sales"].mean() * 100
        cv_data = cv_data.sort_values()
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cv_data)))
        bars = ax.bar(range(len(cv_data)), cv_data.values, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xlabel("Cửa Hàng", fontsize=12, fontweight='bold')
        ax.set_ylabel("Hệ Số Biến Động (%)", fontsize=12, fontweight='bold')
        ax.set_title("So Sánh Độ Ổn Định (CV %)\n(Thấp = Ổn định hơn)", fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(cv_data)))
        ax.set_xticklabels([f'Store {store}' for store in cv_data.index], rotation=45)
        
        for i, v in enumerate(cv_data.values):
            ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
        # Insights
        most_stable = cv_data.idxmin()
        least_stable = cv_data.idxmax()
        
        show_info_box(f"""
        **✅ Ổn định nhất:** Store {most_stable} CV: {cv_data.min():.1f}%
        **⚠️ Biến động nhất:** Store {least_stable} CV: {cv_data.max():.1f}%
        
        **💡 Ý nghĩa:** CV dưới 20% = Rất ổn định, dễ dự đoán
        """, "warning")
        
    else:  # Xu hướng
        # Trend comparison over time
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for store in selected_stores:
            store_data = df_agg[df_agg["Store"] == store]
            monthly_avg = store_data.groupby("Month")["Store_Total_Sales"].mean()
            ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, 
                   label=f'Store {store}', markersize=6)
        
        ax.set_xlabel("Tháng", fontsize=12, fontweight='bold')
        ax.set_ylabel("Doanh Số TB ($)", fontsize=12, fontweight='bold')
        ax.set_title("Xu Hướng Doanh Số Theo Tháng Của Các Cửa Hàng", fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        show_info_box("""
        **📈 Cách đọc biểu đồ:**
        - Đường thẳng = Ổn định
        - Đường lên/xuống = Có xu hướng
        - Khoảng cách lớn = Khác biệt rõ rệt
        
        **💡 Ứng dụng:** Xem cửa hàng nào bị ảnh hưởng nhiều bởi mùa vụ
        """, "info")
    
    st.markdown("---")
    
    # Statistics comparison
    st.subheader("📊 Bảng So Sánh Chi Tiết")
    
    stats_df = df_agg.groupby("Store")["Store_Total_Sales"].agg([
        ('Doanh Số TB', 'mean'),
        ('Trung Vị', 'median'),
        ('Độ Lệch Chuẩn', 'std'),
        ('Thấp Nhất', 'min'),
        ('Cao Nhất', 'max')
    ]).round(0)
    
    # Tính thêm CV và trend
    stats_df['CV (%)'] = (stats_df['Độ Lệch Chuẩn'] / stats_df['Doanh Số TB'] * 100).round(1)
    
    # Highlight best performers
    best_avg = stats_df['Doanh Số TB'].idxmax()
    most_stable = stats_df['CV (%)'].idxmin()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_df = stats_df.copy()
        for col in ['Doanh Số TB', 'Trung Vị', 'Độ Lệch Chuẩn', 'Thấp Nhất', 'Cao Nhất']:
            display_df[col] = display_df[col].apply(lambda x: format_currency(x))
        display_df['CV (%)'] = display_df['CV (%)'].apply(lambda x: f"{x}%")
        st.dataframe(display_df, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏆 Phân Tích Nhanh")
        
        show_info_box(f"""
        **Cửa hàng tốt nhất (Store {best_avg}):**
        - Doanh số TB cao nhất
        - Có thể là mô hình chuẩn để học hỏi
        
        **Cửa hàng ổn định nhất (Store {most_stable}):**
        - CV thấp nhất = Dễ dự đoán
        - Quản lý tốt, ít biến động
        """, "success")
        
        explain_term(
            "CV (%) - Coefficient of Variation",
            """
            **CV đo độ biến động tương đối:**
            
            - **CV dưới 20%**: Rất ổn định tuyệt vời!
            - **CV 20-40%**: Biến động trung bình chấp nhận được
            - **CV trên 40%**: Biến động cao cần cải thiện
            
            **Ví dụ:**
            - Store A: TB $100K, Độ lệch $20K → CV = 20%
            - Store B: TB $100K, Độ lệch $50K → CV = 50%
            
            → Store A ổn định hơn dù cùng TB!
            """
        )

# ==================== PAGE: XU HƯỚNG THỜI GIAN ====================

def page_time_trends(df):
    """Trang phân tích xu hướng thời gian với seasonality"""
    st.title("📈 Xu Hướng Doanh Số Theo Thời Gian")
    
    if df.empty:
        st.warning("⚠️ Không có dữ liệu theo bộ lọc hiện tại.")
        return
    
    st.markdown("""
    ### Khám phá mùa vụ và xu hướng! 📅
    
    Hiểu rõ **khi nào** doanh số cao/thấp giúp:
    - Lập kế hoạch hàng tồn kho chính xác
    - Phân bổ nhân sự hợp lý
    - Tối ưu ngân sách marketing
    """)
    
    st.markdown("---")
    
    # Time aggregation options
    col1, col2, col3 = st.columns(3)
    with col1:
        view_type = st.selectbox(
            "📊 Xem theo",
            ["Tháng", "Quý", "Năm"],
            help="Chọn cách nhóm thời gian để phân tích"
        )
    with col2:
        show_holiday = st.checkbox("🎉 Hiển thị ngày lễ", value=True, help="Đánh dấu tuần có ngày lễ")
    with col3:
        show_trend = st.checkbox("📈 Hiển thị đường xu hướng", value=True, help="Thêm đường xu hướng tuyến tính")
    
    st.markdown("---")
    
    # Monthly trend
    if view_type == "Tháng":
        st.subheader("📅 Phân Tích Theo Tháng")
        
        monthly = df.groupby("Month").agg({
            'Store_Total_Sales': ['mean', 'sum', 'count', 'std']
        }).round(0)
        monthly.columns = ['TB Doanh Số', 'Tổng DS', 'Số Tuần', 'Độ Lệch']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = monthly.index
            y = monthly['TB Doanh Số'].values
            
            # Bar chart với gradient color
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(x)))
            bars = ax.bar(x, y, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Highlight Q4
            for i, (month, val) in enumerate(zip(x, y)):
                if month in [10, 11, 12]:
                    bars[i].set_color('#FFD700')
                    bars[i].set_alpha(0.9)
            
            # Trend line
            if show_trend:
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", linewidth=2, label='Xu hướng')
            
            ax.set_xlabel("Tháng", fontsize=12, fontweight='bold')
            ax.set_ylabel("Doanh Số TB ($)", fontsize=12, fontweight='bold')
            ax.set_title("Doanh Số Trung Bình Theo Tháng\n(Vàng = Tháng cao điểm Q4)", 
                         fontsize=13, fontweight='bold')
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'])
            
            # Add value labels
            for i, v in enumerate(y):
                ax.text(x[i], v, format_currency(v), ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            if show_trend:
                ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📊 Thống Kê Tháng")
            
            # Find peak and low months
            peak_month = monthly['TB Doanh Số'].idxmax()
            low_month = monthly['TB Doanh Số'].idxmin()
            
            show_info_box(f"""
            **🔥 Tháng cao điểm:** Tháng {peak_month}
            - Doanh số: {format_currency(monthly.loc[peak_month, 'TB Doanh Số'])}
            - Cao hơn TB: {((monthly.loc[peak_month, 'TB Doanh Số'] / monthly['TB Doanh Số'].mean() - 1) * 100):.1f}%
            
            **❄️ Tháng thấp điểm:** Tháng {low_month}
            - Doanh số: {format_currency(monthly.loc[low_month, 'TB Doanh Số'])}
            - Thấp hơn TB: {((1 - monthly.loc[low_month, 'TB Doanh Số'] / monthly['TB Doanh Số'].mean()) * 100):.1f}%
            
            **📈 Biến động:** Cao nhất/Thấp nhất = {(monthly.loc[peak_month, 'TB Doanh Số'] / monthly.loc[low_month, 'TB Doanh Số']):.2f}x
            """, "info")
            
            month_names = {
                1: "Tháng 1 Sau Tết", 2: "Tháng 2", 3: "Tháng 3", 
                4: "Tháng 4", 5: "Tháng 5", 6: "Tháng 6",
                7: "Tháng 7", 8: "Tháng 8", 9: "Tháng 9",
                10: "Tháng 10", 11: "Tháng 11 Black Friday", 12: "Tháng 12 Christmas"
            }
            
            display_monthly = monthly.copy()
            display_monthly.index = display_monthly.index.map(month_names)
            display_monthly['TB Doanh Số'] = display_monthly['TB Doanh Số'].apply(lambda x: format_currency(x))
            display_monthly['Tổng DS'] = display_monthly['Tổng DS'].apply(lambda x: format_currency(x))
            st.dataframe(display_monthly[['TB Doanh Số', 'Số Tuần']], use_container_width=True)
    
    # Quarterly trend
    elif view_type == "Quý":
        st.subheader("📅 Phân Tích Theo Quý")
        
        quarterly = df.groupby("Quarter").agg({
            'Store_Total_Sales': ['mean', 'sum', 'count']
        }).round(0)
        quarterly.columns = ['TB Doanh Số', 'Tổng DS', 'Số Tuần']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#3498db', '#f39c12', '#e74c3c', '#2ecc71']
            bars = ax.bar(quarterly.index, quarterly['TB Doanh Số'], 
                          color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            
            # Highlight Q4
            bars[3].set_color('#FFD700')
            bars[3].set_linewidth(3)
            
            ax.set_xlabel("Quý", fontsize=12, fontweight='bold')
            ax.set_ylabel("Doanh Số TB ($)", fontsize=12, fontweight='bold')
            ax.set_title("Doanh Số Trung Bình Theo Quý\n(Q4 = Vàng = Mùa Vàng!)", 
                         fontsize=13, fontweight='bold')
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(['Q1\n(Jan-Mar)', 'Q2\n(Apr-Jun)', 'Q3\n(Jul-Sep)', 'Q4\n(Oct-Dec)'])
            
            for i, v in enumerate(quarterly['TB Doanh Số']):
                ax.text(i+1, v, format_currency(v), ha='center', va='bottom', 
                        fontweight='bold', fontsize=11)
            
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📊 Insights Theo Quý")
            
            q4_boost = (quarterly.loc[4, 'TB Doanh Số'] / quarterly['TB Doanh Số'].mean() - 1) * 100
            q1_drop = (1 - quarterly.loc[1, 'TB Doanh Số'] / quarterly['TB Doanh Số'].mean()) * 100
            
            show_info_box(f"""
            **🎄 Quý 4 - Mùa Vàng:**
            - Cao hơn TB: **+{q4_boost:.1f}%**
            - Nguyên nhân: Black Friday, Thanksgiving, Christmas
            - Hành động: Tăng 150% hàng tồn kho
            
            **❄️ Quý 1 - Mùa Khó:**
            - Thấp hơn TB: **-{q1_drop:.1f}%**
            - Nguyên nhân: Sau lễ, khách hết tiền
            - Hành động: Khuyến mãi mạnh, thanh lý tồn kho
            """, "warning")
            
            st.markdown("---")
            
            display_q = quarterly.copy()
            display_q['TB Doanh Số'] = display_q['TB Doanh Số'].apply(lambda x: format_currency(x))
            display_q['Tổng DS'] = display_q['Tổng DS'].apply(lambda x: format_currency(x))
            st.dataframe(display_q, use_container_width=True)
    
    # Yearly trend
    else:  # view_type == "Năm"
        st.subheader("📅 Phân Tích Theo Năm")
        
        yearly = df.groupby("Year").agg({
            'Store_Total_Sales': ['mean', 'sum', 'count']
        }).round(0)
        yearly.columns = ['TB Doanh Số', 'Tổng DS', 'Số Tuần']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = yearly.index
            y = yearly['TB Doanh Số'].values
            
            ax.plot(x, y, marker='o', linewidth=3, markersize=12, color='#2ecc71', markeredgecolor='black', markeredgewidth=2)
            ax.fill_between(x, y, alpha=0.3, color='#2ecc71')
            
            # Trend line
            if show_trend and len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", linewidth=2, label=f'Xu hướng: {z[0]:+.0f}$/năm')
                ax.legend()
            
            ax.set_xlabel("Năm", fontsize=12, fontweight='bold')
            ax.set_ylabel("Doanh Số TB ($)", fontsize=12, fontweight='bold')
            ax.set_title("Doanh Số Trung Bình Theo Năm\n(Xu hướng tăng hay giảm?)", 
                         fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            
            for i, v in enumerate(y):
                ax.text(x[i], v, format_currency(v), ha='center', va='bottom', 
                        fontweight='bold', fontsize=11)
            
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📊 Phân Tích Năm")
            
            if len(yearly) > 1:
                first_year = yearly.index[0]
                last_year = yearly.index[-1]
                growth = (yearly.loc[last_year, 'TB Doanh Số'] / yearly.loc[first_year, 'TB Doanh Số'] - 1) * 100
                
                if growth > 0:
                    show_info_box(f"""
                    **📈 Xu hướng tăng trưởng:**
                    - Tăng {growth:.1f}% từ {first_year} đến {last_year}
                    - Trung bình: {growth/(last_year-first_year):.1f}%/năm
                    
                    **💡 Ý nghĩa:**
                    - Walmart đang phát triển tốt
                    - Chiến lược hiệu quả
                    - Nên tiếp tục mở rộng
                    """, "success")
                else:
                    show_info_box(f"""
                    **📉 Xu hướng giảm:**
                    - Giảm {abs(growth):.1f}% từ {first_year} đến {last_year}
                    
                    **⚠️ Cần hành động:**
                    - Điều tra nguyên nhân
                    - Cải thiện sản phẩm/dịch vụ
                    - Tăng cường marketing
                    """, "warning")
            
            display_y = yearly.copy()
            display_y['TB Doanh Số'] = display_y['TB Doanh Số'].apply(lambda x: format_currency(x))
            display_y['Tổng DS'] = display_y['Tổng DS'].apply(lambda x: format_currency(x))
            st.dataframe(display_y, use_container_width=True)
    
    st.markdown("---")
    
    # Heatmap by year and month
    st.subheader("🔥 Bản Đồ Nhiệt Doanh Số Theo Năm và Tháng")
    
    # Create pivot table for year vs month
    pivot_data = df.pivot_table(
        values='Store_Total_Sales',
        index='Year',  # Rows: Years
        columns='Month',  # Columns: Months
        aggfunc='mean'  # Average sales
    ).round(0)  # Round to whole numbers for cleaner display
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn', 
                linewidths=1, cbar_kws={'label': 'Doanh Số TB ($)'}, ax=ax)
    ax.set_xlabel("Tháng", fontsize=11, fontweight='bold')
    ax.set_ylabel("Năm", fontsize=11, fontweight='bold')
    ax.set_title("Bản Đồ Nhiệt: Doanh Số TB Theo Năm & Tháng\n(Đỏ = Cao, Xanh = Thấp)", 
                 fontsize=12, fontweight='bold')
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'], rotation=0)
    st.pyplot(fig)
    
    explain_term(
        "Cách đọc Bản Đồ Nhiệt",
        """
        **Bản đồ nhiệt (Heatmap) giúp:**
        
        1. **Tìm mùa vụ**: Tháng nào đỏ nhiều = Cao điểm
        2. **So sánh năm**: Năm nào đỏ hơn = Phát triển tốt
        3. **Phát hiện xu hướng**: Có tăng dần theo năm?
        
        **Insights chính:**
        - **Tháng 11-12**: Đỏ nhất (Black Friday, Christmas)
        - **Tháng 1-2**: Xanh nhất (Sau lễ, khách hết tiền)
        - **Q4 > Q1**: Mùa vụ rõ ràng
        """
    )


# ==================== PAGE: PHÂN TÍCH NGÀY LỄ ====================

def page_holiday(df):
    """Trang phân tích ngày lễ chi tiết - Q5 từ notebook"""
    st.title("🎉 Phân Tích Ngày Lễ Chi Tiết")
    
    if df.empty:
        st.info("Không có dữ liệu theo bộ lọc.")
        return

    st.markdown("""
    ### Khám phá tác động của từng ngày lễ cụ thể! 🎯
    
    Phân tích chi tiết **Super Bowl, Labor Day, Thanksgiving, Christmas** so với tuần thường.
    """)

    # Options for analysis
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.selectbox(
            "🎯 Chọn kiểu phân tích",
            ["Tất cả ngày lễ", "So sánh với tuần thường", "So sánh giữa các lễ"],
            help="Chọn cách phân tích tác động của ngày lễ"
        )
    
    with col2:
        if analysis_type == "Tất cả ngày lễ":
            selected_holidays = st.multiselect(
                "Chọn ngày lễ để phân tích",
                ["Super Bowl", "Labor Day", "Thanksgiving", "Christmas"],
                default=["Super Bowl", "Labor Day", "Thanksgiving", "Christmas"],
                help="Chọn các ngày lễ muốn phân tích"
            )
        else:
            selected_holidays = ["Super Bowl", "Labor Day", "Thanksgiving", "Christmas"]

    # Phân loại ngày lễ
    def classify_holiday(month):
        if month == 2:
            return 'Super Bowl'
        elif month == 9:
            return 'Labor Day'
        elif month == 11:
            return 'Thanksgiving'
        elif month == 12:
            return 'Christmas'
        else:
            return 'Other Holiday'

    df_holiday = df.copy()
    df_holiday['Holiday_Name'] = df_holiday.apply(
        lambda row: classify_holiday(row['Month']) if row['IsHoliday'] == 1 else 'Non-Holiday',
        axis=1
    )

    if analysis_type == "Tất cả ngày lễ":
        # Tính doanh số trung bình
        holiday_sales = df_holiday.groupby('Holiday_Name')['Store_Total_Sales'].mean().sort_values(ascending=False)
        baseline_sales = holiday_sales['Non-Holiday']

        st.subheader("📊 Tác Động Của Từng Ngày Lễ")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            holiday_plot = holiday_sales.drop('Non-Holiday').sort_values()
            
            if selected_holidays:
                holiday_plot = holiday_plot[holiday_plot.index.isin(selected_holidays)]
            
            bars = ax.barh(holiday_plot.index, holiday_plot.values, 
                           color=['#e74c3c', '#3498db', '#f1c40f', '#2ecc71'], 
                           alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.axvline(baseline_sales, color='black', linestyle='--', linewidth=2, 
                       label=f'Tuần thường: ${baseline_sales:,.0f}')

            for bar in bars:
                width = bar.get_width()
                growth = (width / baseline_sales - 1) * 100
                ax.text(width + 10000, bar.get_y() + bar.get_height()/2,
                        f'${width:,.0f}\n({growth:+.1f}%)',
                        va='center', ha='left', fontsize=9, fontweight='bold')

            ax.set_xlabel('Doanh số trung bình ($)', fontsize=11, fontweight='bold')
            ax.set_title('So Sánh Tác Động Của Từng Ngày Lễ', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.markdown("#### 📋 Chi Tiết")
            for holiday in selected_holidays:
                if holiday in holiday_sales.index:
                    sales = holiday_sales[holiday]
                    growth = (sales / baseline_sales - 1) * 100
                    emoji = '🟢' if growth > 0 else '🔴'
                    st.write(f"{emoji} **{holiday}**: ${sales:,.0f} ({growth:+.1f}%)")
            
            st.markdown("---")
            st.metric("Tuần thường (Baseline)", f"${baseline_sales:,.0f}")

    elif analysis_type == "So sánh với tuần thường":
        st.subheader("⚖️ So Sánh Ngày Lễ vs Tuần Thường")
        
        # Chọn ngày lễ cụ thể
        selected_holiday = st.selectbox(
            "Chọn ngày lễ để so sánh",
            ["Super Bowl", "Labor Day", "Thanksgiving", "Christmas"],
            help="Chọn ngày lễ muốn so sánh chi tiết với tuần thường"
        )
        
        if selected_holiday in df_holiday['Holiday_Name'].values:
            holiday_data = df_holiday[df_holiday['Holiday_Name'] == selected_holiday]['Store_Total_Sales']
            normal_data = df_holiday[df_holiday['Holiday_Name'] == 'Non-Holiday']['Store_Total_Sales']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Box plot comparison
                data = [normal_data, holiday_data]
                labels = ['Tuần Thường', selected_holiday]
                
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                colors = ['lightblue', 'lightcoral']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_ylabel('Doanh Số ($)', fontsize=11, fontweight='bold')
                ax.set_title(f'Phân Phối Doanh Số: {selected_holiday} vs Tuần Thường', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Statistics
                holiday_mean = holiday_data.mean()
                normal_mean = normal_data.mean()
                diff = holiday_mean - normal_mean
                pct_diff = (diff / normal_mean) * 100
                
                st.markdown(f"#### 📊 Thống Kê {selected_holiday}")
                st.metric("Doanh số TB ngày lễ", f"${holiday_mean:,.0f}")
                st.metric("Doanh số TB thường", f"${normal_mean:,.0f}")
                st.metric("Chênh lệch", f"${diff:,.0f} ({pct_diff:+.1f}%)")
                
                # T-test
                if len(holiday_data) > 0 and len(normal_data) > 0:
                    t_stat, p_value = stats.ttest_ind(holiday_data, normal_data)
                    significance = "Có ý nghĩa" if p_value < 0.05 else "Không ý nghĩa"
                    st.metric("P-value", f"{p_value:.4f} ({significance})")
        
        # Nhận xét tùy chỉnh
        st.markdown("---")
        st.subheader("💭 Nhận Xét & Khuyến Nghị")
        
        holiday_insights = {
            "Super Bowl": """
            **Super Bowl (Tháng 2):**
            - Thường tăng nhẹ do người xem TV nhiều
            - Tập trung quảng cáo trong giờ nghỉ giải lao
            - Chuẩn bị đồ ăn nhẹ, bia rượu
            """,
            "Labor Day": """
            **Labor Day (Tháng 9):**
            - Cuối hè, người dân mua sắm cuối tuần
            - Khuyến mãi đồ gia dụng, đồ outdoor
            - Thời điểm tốt để thanh lý hàng tồn kho hè
            """,
            "Thanksgiving": """
            **Thanksgiving (Tháng 11):**
            - Tăng mạnh do mua sắm trước Giáng sinh
            - Tập trung thực phẩm, quà tặng
            - Chuẩn bị nhân sự tăng 50%
            """,
            "Christmas": """
            **Christmas (Tháng 12):**
            - Cao điểm nhất trong năm
            - Tăng 150-200% hàng tồn kho
            - Quảng cáo mạnh, chương trình khuyến mãi
            """
        }
        
        if selected_holiday in holiday_insights:
            show_info_box(holiday_insights[selected_holiday], "info")

    else:  # So sánh giữa các lễ
        st.subheader("🔄 So Sánh Giữa Các Ngày Lễ")
        
        # Tính doanh số cho từng lễ
        holiday_comparison = df_holiday[df_holiday['Holiday_Name'] != 'Non-Holiday'].groupby('Holiday_Name')['Store_Total_Sales'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).round(0)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))
            means = holiday_comparison['mean'].sort_values(ascending=False)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(means)))
            bars = ax.bar(range(len(means)), means.values, color=colors, alpha=0.8, edgecolor='black')
            
            ax.set_xlabel("Ngày Lễ", fontsize=11, fontweight='bold')
            ax.set_ylabel("Doanh Số TB ($)", fontsize=11, fontweight='bold')
            ax.set_title("So Sánh Doanh Số Giữa Các Ngày Lễ", fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(means.index, rotation=45)
            
            for i, v in enumerate(means.values):
                ax.text(i, v, format_currency(v), ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📊 Xếp Hạng")
            
            ranking = holiday_comparison['mean'].sort_values(ascending=False)
            for i, (holiday, sales) in enumerate(ranking.items(), 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
                st.write(f"{emoji} **#{i} {holiday}**: ${sales:,.0f}")
            
            st.markdown("---")
            st.markdown("#### 💡 Insights")
            show_info_box("""
            **Mùa lễ mạnh nhất:** Christmas > Thanksgiving
            **Mùa lễ yếu nhất:** Super Bowl, Labor Day
            **Chiến lược:** Tập trung nguồn lực vào Q4
            """, "success")

    st.markdown("---")
    
    # T-test cho từng lễ (chỉ hiển thị khi cần)
    if analysis_type == "Tất cả ngày lễ":
        st.subheader("📈 Phân Tích Thống Kê")
        
        for holiday in selected_holidays:
            if holiday in df_holiday['Holiday_Name'].values:
                holiday_data = df_holiday[df_holiday['Holiday_Name'] == holiday]['Store_Total_Sales']
                normal_data = df_holiday[df_holiday['Holiday_Name'] == 'Non-Holiday']['Store_Total_Sales']
                
                if len(holiday_data) > 0 and len(normal_data) > 0:
                    t_stat, p_value = stats.ttest_ind(holiday_data, normal_data)
                    mean_diff = holiday_data.mean() - normal_data.mean()
                    pct_diff = (mean_diff / normal_data.mean()) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{holiday} - Chênh lệch", f"{format_currency(mean_diff)} ({pct_diff:.1f}%)")
                    with col2:
                        st.metric(f"{holiday} - T-stat", f"{t_stat:.2f}")
                    with col3:
                        significance = "Có ý nghĩa" if p_value < 0.05 else "Không ý nghĩa"
                        st.metric(f"{holiday} - P-value", f"{p_value:.4f} ({significance})")

# ==================== PAGE: PHÂN NHÓM THÔNG MINH ====================

def page_clustering(df):
    """Trang phân nhóm K-Means"""
    st.title("🔍 Phân Nhóm Cửa Hàng (K-Means)")
    
    if df.empty:
        st.info("Không có dữ liệu theo bộ lọc.")
        return

    st.markdown("""
    **Phần này làm gì?** Nhóm 45 cửa hàng thành các cụm tương tự dựa trên đặc tính (doanh số, quy mô, kinh tế).

    **Chọn gì để làm gì?** 
    - Chọn đặc tính để phân cụm (càng nhiều càng chính xác, nhưng phức tạp hơn).
    - Chọn số cụm K (dùng Elbow để chọn K tốt).

    **Ý nghĩa kết quả:** Biết cửa hàng nào giống nhau, tối ưu chiến lược cho từng nhóm.

    **Nên làm gì tiếp theo?** Áp dụng chiến lược khác nhau cho từng cụm (ví dụ: quảng bá cho cụm yếu).
    """)

    # Chọn features cho clustering
    available_features = ["Store_Total_Sales", "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    feature_labels = {
        "Store_Total_Sales": "Doanh số trung bình",
        "Size": "Quy mô cửa hàng",
        "Temperature": "Nhiệt độ trung bình",
        "Fuel_Price": "Giá xăng trung bình",
        "CPI": "Chỉ số CPI",
        "Unemployment": "Tỷ lệ thất nghiệp"
    }
    
    selected_features = st.multiselect(
        "Chọn đặc tính để phân cụm", 
        options=available_features, 
        default=["Store_Total_Sales", "Size", "CPI", "Unemployment"],
        format_func=lambda x: feature_labels[x],
        help="Chọn các yếu tố bạn muốn dùng để phân nhóm cửa hàng"
    )

    if len(selected_features) < 2:
        st.warning("⚠️ Vui lòng chọn ít nhất 2 đặc tính để phân cụm.")
        return

    # Aggregate per store
    store_features = df.groupby("Store").agg({
        feat: "mean" if feat != "Size" else "first" for feat in selected_features
    }).dropna()

    if len(store_features) < 3:
        st.warning("⚠️ Không đủ dữ liệu để phân cụm.")
        return

    # Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(store_features)

    # Elbow Method
    st.subheader("📊 Chọn số cụm K tối ưu (Elbow Method)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        wcss = []
        max_k = min(10, len(store_features)-1)
        for i in range(1, max_k+1):
            kmeans_temp = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans_temp.fit(X_scaled)
            wcss.append(kmeans_temp.inertia_)
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(range(1, max_k+1), wcss, marker='o', linewidth=2, markersize=8, color='#2ecc71')
        ax.set_xlabel("Số cụm K", fontsize=12)
        ax.set_ylabel("WCSS", fontsize=12)
        ax.set_title("Elbow Method để chọn K tối ưu", fontsize=13)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.info("""
Cách đọc:

- Tìm điểm "khuỷu tay" nơi đường cong bắt đầu phẳng
- Đó là K tối ưu
- Thường K=3-5 là tốt

Ý nghĩa:
- K nhỏ: Đơn giản nhưng ít chi tiết
- K lớn: Chi tiết nhưng phức tạp
        """)

    # Slider K
    k = st.slider("Chọn số cụm K", min_value=2, max_value=max_k, value=min(3, max_k),
                  help="Số nhóm bạn muốn chia cửa hàng")

    # Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Silhouette Score
    sil_score = silhouette_score(X_scaled, clusters)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số cụm", k)
    with col2:
        st.metric("Silhouette Score", f"{sil_score:.3f}")
    with col3:
        if sil_score > 0.5:
            st.metric("Đánh giá", "✅ Tốt")
        elif sil_score > 0.25:
            st.metric("Đánh giá", "⚠️ Khá")
        else:
            st.metric("Đánh giá", "❌ Kém")

    with st.expander("❓ Silhouette Score là gì?"):
        st.markdown(f"""
Score = {sil_score:.3f} đo chất lượng phân cụm:

- > 0.7: Xuất sắc (cụm tách biệt rất rõ)
- 0.5-0.7: Tốt
- 0.2-0.5: Chấp nhận được ← Bạn ở đây
- < 0.2: Kém (cụm chồng lấn nhiều)

💡 Với dữ liệu thực tế, 0.2-0.4 là bình thường!
        """)

    # Thêm cluster vào dataframe
    store_features["Cluster"] = clusters

    # Visualization
    st.subheader("📈 Visualize Các Cụm")
    
    if "Size" in selected_features and "Store_Total_Sales" in selected_features:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10,7))
            scatter = ax.scatter(
                store_features["Size"], 
                store_features["Store_Total_Sales"], 
                c=clusters, 
                cmap="viridis", 
                s=150, 
                alpha=0.7,
                edgecolor='black'
            )
            
            # Add labels
            for idx, row in store_features.iterrows():
                ax.annotate(f"S{idx}", (row["Size"], row["Store_Total_Sales"]), 
                           fontsize=8, ha='center')
            
            ax.set_xlabel("Quy mô cửa hàng (Size)", fontsize=12)
            ax.set_ylabel("Doanh số TB (Store_Total_Sales)", fontsize=12)
            ax.set_title(f"Phân Cụm {k} Nhóm Cửa Hàng", fontsize=13)
            ax.legend(*scatter.legend_elements(), title="Cụm")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        

    else:
        st.info("💡 Chọn cả 'Doanh số TB' và 'Quy mô' để xem biểu đồ scatter!")

    # Cluster summary
    st.subheader("📊 Đặc Điểm Từng Cụm")
    
    cluster_summary = store_features.groupby("Cluster")[selected_features].mean().round(2)
    
    # Display table
    display_summary = cluster_summary.copy()
    if "Store_Total_Sales" in selected_features:
        display_summary["Store_Total_Sales"] = display_summary["Store_Total_Sales"].apply(lambda x: format_currency(x))
    
    st.dataframe(display_summary, use_container_width=True)

    # Heatmap
    if len(selected_features) > 2:
        fig, ax = plt.subplots(figsize=(10,6))
        
        # Normalize for better visualization
        cluster_norm = (cluster_summary - cluster_summary.min()) / (cluster_summary.max() - cluster_summary.min())
        
        sns.heatmap(cluster_norm.T, annot=cluster_summary.T.values, fmt='.0f',
                    cmap='RdYlGn', linewidths=2, ax=ax)
        ax.set_xlabel("Cụm", fontsize=12)
        ax.set_ylabel("Đặc tính", fontsize=12)
        ax.set_title("Heatmap Đặc Điểm Cụm (Đỏ=Cao, Xanh=Thấp)", fontsize=13)
        ax.set_yticklabels([feature_labels[f] for f in selected_features], rotation=0)
        st.pyplot(fig)
        plt.close()

    # Lời khuyên chiến lược
    st.subheader("💡 Lời Khuyên Chiến Lược")
    
    if "Store_Total_Sales" in selected_features:
        best_cluster = cluster_summary["Store_Total_Sales"].idxmax()
        worst_cluster = cluster_summary["Store_Total_Sales"].idxmin()
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_info_box(f"""
🌟 Cụm mạnh nhất {best_cluster}:

- Doanh số cao: {format_currency(cluster_summary.loc[best_cluster, 'Store_Total_Sales'])}
- Học hỏi mô hình quản lý
- Nhân rộng thành công
- Tăng đầu tư cho cụm này
            """, "success")
        
        with col2:
            show_info_box(f"""
⚠️ Cụm yếu nhất {worst_cluster}:

- Doanh số thấp: {format_currency(cluster_summary.loc[worst_cluster, 'Store_Total_Sales'])}
- Cần cải thiện: marketing, đào tạo
- Kiểm tra vị trí, cạnh tranh
- Cân nhắc đóng cửa nếu không cải thiện
            """, "warning")
    
    st.info("""
Chiến lược tổng thể:
- Phân bổ nguồn lực dựa trên cụm
- Cá nhân hóa chiến lược marketing
- Tập trung hỗ trợ cụm yếu
- Nhân rộng mô hình cụm mạnh
    """)





# ==================== PAGE: DỰ ĐOÁN DECISION TREE ====================

def page_decision_tree(df):
    """Trang dự đoán Decision Tree - Q8 từ notebook"""
    st.title("🌳 Dự Đoán Doanh Số (Decision Tree)")
    
    st.markdown("""
    **Phần này làm gì?** Sử dụng Decision Tree để dự đoán tuần có doanh số cao hay thấp.

    **Cách hoạt động:** 
    - Chia doanh số thành 2 nhóm: Cao (> trung vị) và Thấp (<= trung vị)
    - Dùng các yếu tố kinh tế để dự đoán nhóm nào

    **Ý nghĩa:** Kiểm tra xem có thể dự đoán được tuần cao điểm không?
    """)
    
    if df.empty:
        st.warning("⚠️ Không có dữ liệu theo bộ lọc hiện tại.")
        return
    
    # Chuẩn bị dữ liệu
    st.subheader("📊 Chuẩn Bị Dữ Liệu")
    
    # Tạo target: High_Sales
    median_sales = df["Store_Total_Sales"].median()
    df["High_Sales"] = (df["Store_Total_Sales"] > median_sales).astype(int)
    
    # Features
    features_list = ["Size", "Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"]
    X = df[features_list].copy()
    y = df["High_Sales"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    st.info(f"""
    **Dữ liệu huấn luyện:**
    - Tổng mẫu: {len(df):,}
    - Trung vị doanh số: {format_currency(median_sales)}
    - Tuần cao điểm: {y.sum():,} ({y.mean()*100:.1f}%)
    - Tuần thấp điểm: {len(y) - y.sum():,} ({(1-y.mean())*100:.1f}%)
    """)
    
    # Huấn luyện Decision Tree
    st.subheader("🌳 Huấn Luyện Decision Tree")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_depth = st.slider("Độ Sâu Tối Đa", 3, 10, 5, help="Độ sâu tối đa của cây quyết định")
        min_samples_split = st.slider("Số Mẫu Tối Thiểu Để Chia", 2, 20, 10, help="Số mẫu tối thiểu để chia nhánh")
    
    with col2:
        criterion = st.selectbox("Tiêu Chí Chia Nhánh", ["gini", "entropy"], index=0, help="Tiêu chí để chọn cách chia dữ liệu tại mỗi nút")
        random_state = RANDOM_STATE
        
        # Giải thích tiêu chí
        if criterion == "gini":
            st.info("**Gini Impurity:** Đo độ hỗn loạn của dữ liệu. Giá trị thấp = dữ liệu đồng nhất hơn.")
        else:
            st.info("**Entropy:** Đo độ bất định của dữ liệu. Giá trị thấp = dự đoán chính xác hơn.")
    
    # Fit model
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state
    )
    dt_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt_model.predict(X_test)
    
    # Accuracy
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)
    
    st.success(f"**Độ chính xác trên test set:** {accuracy:.1f}%")
    
    # Feature Importance
    st.subheader("📊 Tầm Quan Trọng Của Các Yếu Tố")
    
    feature_importance = pd.DataFrame({
        'Feature': features_list,
        'Importance': dt_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='#2ecc71', alpha=0.8)
        ax.set_xlabel("Tầm Quan Trọng", fontsize=12)
        ax.set_title("Tầm Quan Trọng Yếu Tố - Decision Tree", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### 🔍 Insights")
        top_feature = feature_importance.iloc[0]['Feature']
        top_imp = feature_importance.iloc[0]['Importance'] * 100
        
        st.info(f"""
        **Yếu tố quan trọng nhất:** {top_feature}
        - Tầm quan trọng: {top_imp:.1f}%
        - Ảnh hưởng mạnh nhất đến dự đoán
        """)
        
        explain_term(
            "Feature Importance là gì?",
            """
            **Feature Importance** đo mức độ đóng góp của từng yếu tố trong việc dự đoán.
            
            - Giá trị cao = Yếu tố quan trọng
            - Giá trị thấp = Yếu tố ít ảnh hưởng
            
            **Ví dụ:** Nếu Size = 80%, nghĩa là quy mô cửa hàng quyết định 80% dự đoán!
            """
        )
    
    # Confusion Matrix
    st.subheader("📈 Ma Trận Nhầm Lẫn")
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Thấp', 'Cao'], yticklabels=['Thấp', 'Cao'], ax=ax)
    ax.set_xlabel("Dự Đoán", fontsize=12)
    ax.set_ylabel("Thực Tế", fontsize=12)
    ax.set_title("Ma Trận Nhầm Lẫn", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()
    
    # Decision Tree Plot
    st.subheader("🌳 Cấu Trúc Cây Quyết Định")
    
    fig, ax = plt.subplots(figsize=(20, 10))
    from sklearn.tree import plot_tree
    plot_tree(dt_model, feature_names=features_list, class_names=['Thấp', 'Cao'], 
              filled=True, rounded=True, fontsize=10, ax=ax)
    ax.set_title("Cấu Trúc Cây Quyết Định", fontsize=16, fontweight='bold')
    st.pyplot(fig)
    plt.close()
    
    # Classification Report
    st.subheader("📋 Báo Cáo Phân Loại")
    
    report = classification_report(y_test, y_pred, target_names=['Thấp', 'Cao'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    st.dataframe(report_df.style.format({
        'precision': '{:.2f}',
        'recall': '{:.2f}', 
        'f1-score': '{:.2f}',
        'support': '{:.0f}'
    }))
    
    # Kết luận
    st.subheader("💡 Kết Luận")
    
    if accuracy > 0.7:
        show_info_box("""
        ✅ **Mô hình khá tốt!**
        
        - Độ chính xác > 70%
        - Có thể dự đoán được tuần cao điểm
        - Dùng để lập kế hoạch kinh doanh
        """, "success")
    elif accuracy > 0.6:
        show_info_box("""
        ⚠️ **Mô hình chấp nhận được**
        
        - Độ chính xác 60-70%
        - Dự đoán có ích nhưng không hoàn hảo
        - Kết hợp với kinh nghiệm thực tế
        """, "info")
    else:
        show_info_box("""
        ❌ **Mô hình chưa tốt**
        
        - Độ chính xác < 60%
        - Khó dự đoán tuần cao điểm
        - Cần cải thiện hoặc dùng phương pháp khác
        """, "warning")
    
    st.info("""
    **Lưu ý:** Đây là thử nghiệm trên dữ liệu lịch sử. Trong thực tế, cần validation kỹ hơn và cập nhật mô hình định kỳ.
    """)

# ==================== PAGE: DỰ TOÁN TƯƠNG TÁC ====================

def page_forecast(df):
    """Trang dự toán doanh số tương tác"""
    st.title("💡 Dự Toán Doanh Số Tương Tác")
    
    st.markdown("""
    **Phần này làm gì?** Công cụ mô phỏng doanh số dựa trên dữ liệu thực tế - Không phải AI dự đoán, mà là "What-if Analysis".

    **Chọn gì để làm gì?** Điều chỉnh các thông số (nhiệt độ, giá xăng, ngày lễ...) để xem ảnh hưởng đến doanh số.

    **Ý nghĩa kết quả:** Hiểu được yếu tố nào ảnh hưởng bao nhiêu %, lập kế hoạch kinh doanh.

    **Nên làm gì tiếp theo?** Dùng insights để điều chỉnh chiến lược theo điều kiện thị trường.
    """)
    
    if df.empty:
        st.info("Không có dữ liệu theo bộ lọc.")
        return

    st.info("⚠️ **Lưu ý:** Đây là mô phỏng dựa trên dữ liệu lịch sử, không phải dự đoán AI chính xác tuyệt đối.")
    
    # Tính baseline từ dữ liệu thực
    st.subheader("📊 Bước 1: Chọn Điểm Xuất Phát")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Chọn cửa hàng
        store_options = ["Trung bình toàn hệ thống"] + [f"Store {s}" for s in sorted(df["Store"].unique())]
        store_choice = st.selectbox("🏪 Cửa hàng", store_options)
        
        if store_choice == "Trung bình toàn hệ thống":
            baseline_df = df.copy()
            store_id = None
        else:
            store_id = int(store_choice.split()[1])
            baseline_df = df[df["Store"] == store_id].copy()
    
    with col2:
        # Chọn khoảng thời gian làm baseline
        time_options = ["Toàn bộ lịch sử", "6 tháng gần nhất", "1 năm gần nhất"]
        time_choice = st.selectbox("📅 Khoảng thời gian tham khảo", time_options)
        
        if time_choice == "6 tháng gần nhất":
            cutoff = baseline_df["Date"].max() - pd.Timedelta(days=180)
            baseline_df = baseline_df[baseline_df["Date"] >= cutoff]
        elif time_choice == "1 năm gần nhất":
            cutoff = baseline_df["Date"].max() - pd.Timedelta(days=365)
            baseline_df = baseline_df[baseline_df["Date"] >= cutoff]
    
    # Tính baseline metrics
    baseline_sales = baseline_df["Store_Total_Sales"].mean()
    baseline_temp = baseline_df["Temperature"].mean()
    baseline_fuel = baseline_df["Fuel_Price"].mean()
    baseline_cpi = baseline_df["CPI"].mean()
    baseline_unemp = baseline_df["Unemployment"].mean()
    
    st.success(f"""
    ✅ **Baseline được chọn:**
    - Doanh số TB: {format_currency(baseline_sales)}/tuần
    - Nhiệt độ TB: {baseline_temp:.1f}°F
    - Giá xăng TB: ${baseline_fuel:.2f}
    - CPI TB: {baseline_cpi:.2f}
    - Thất nghiệp TB: {baseline_unemp:.2f}%
    """)
    
    st.markdown("---")
    st.subheader("🎮 Bước 2: Điều Chỉnh Các Yếu Tố")
    
    st.markdown("""
    **Cách dùng:** Kéo slider để thay đổi các yếu tố và xem tác động lên doanh số.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Nhiệt độ
        temp_change = st.slider(
            "🌡️ Nhiệt độ (°F)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Thay đổi nhiệt độ so với baseline"
        )
        new_temp = baseline_temp + temp_change
        
        # Ngày lễ
        is_holiday = st.checkbox(
            "🎉 Tuần lễ",
            value=False,
            help="Có phải tuần có ngày lễ không?"
        )
    
    with col2:
        # Giá xăng
        fuel_change = st.slider(
            "⛽ Giá xăng ($)",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Thay đổi giá xăng so với baseline"
        )
        new_fuel = baseline_fuel + fuel_change
        
        # CPI
        cpi_change = st.slider(
            "📊 CPI",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Thay đổi chỉ số giá tiêu dùng"
        )
        new_cpi = baseline_cpi + cpi_change
    
    with col3:
        # Thất nghiệp
        unemp_change = st.slider(
            "💼 Thất nghiệp (%)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            help="Thay đổi tỷ lệ thất nghiệp"
        )
        new_unemp = baseline_unemp + unemp_change
        
        # Khuyến mãi
        has_promo = st.checkbox(
            "🎁 Có khuyến mãi",
            value=False,
            help="Có chương trình giảm giá không?"
        )
    
    st.markdown("---")
    st.subheader("📈 Bước 3: Kết Quả Dự Toán")
    
    # Tính toán ảnh hưởng dựa trên correlation từ dữ liệu thực
    # Phân tích correlation
    temp_corr = df[["Temperature", "Store_Total_Sales"]].corr().iloc[0, 1]
    fuel_corr = df[["Fuel_Price", "Store_Total_Sales"]].corr().iloc[0, 1]
    cpi_corr = df[["CPI", "Store_Total_Sales"]].corr().iloc[0, 1]
    unemp_corr = df[["Unemployment", "Store_Total_Sales"]].corr().iloc[0, 1]
    
    # Tính % thay đổi
    temp_impact = (temp_change / baseline_temp) * temp_corr * 100 if baseline_temp != 0 else 0
    fuel_impact = (fuel_change / baseline_fuel) * fuel_corr * 100 if baseline_fuel != 0 else 0
    cpi_impact = (cpi_change / baseline_cpi) * cpi_corr * 100 if baseline_cpi != 0 else 0
    unemp_impact = (unemp_change / baseline_unemp) * unemp_corr * 100 if baseline_unemp != 0 else 0
    
    # Ảnh hưởng ngày lễ (từ dữ liệu thực)
    holiday_sales = df[df["IsHoliday"] == 1]["Store_Total_Sales"].mean()
    normal_sales = df[df["IsHoliday"] == 0]["Store_Total_Sales"].mean()
    holiday_impact = ((holiday_sales - normal_sales) / normal_sales * 100) if is_holiday else 0
    
    # Ảnh hưởng khuyến mãi (estimate)
    promo_impact = 8.0 if has_promo else 0  # Giả định +8% dựa trên industry standard
    
    # Tổng hợp
    total_impact = temp_impact + fuel_impact + cpi_impact + unemp_impact + holiday_impact + promo_impact
    estimated_sales = baseline_sales * (1 + total_impact / 100)
    
    # Hiển thị kết quả
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Doanh số Baseline",
            format_currency(baseline_sales),
            help="Doanh số trung bình từ dữ liệu lịch sử"
        )
    
    with col2:
        delta_color = "normal" if total_impact >= 0 else "inverse"
        st.metric(
            "Tổng Tác Động",
            f"{total_impact:+.2f}%",
            help="Tổng % thay đổi từ tất cả yếu tố"
        )
    
    with col3:
        st.metric(
            "Doanh Số Dự Toán",
            format_currency(estimated_sales),
            delta=f"{total_impact:+.2f}%",
            help="Doanh số ước tính sau khi điều chỉnh"
        )
    
    # Breakdown chi tiết
    st.subheader("🔍 Phân Tích Chi Tiết Từng Yếu Tố")
    
    breakdown_data = {
        "Yếu tố": [
            "🌡️ Nhiệt độ",
            "⛽ Giá xăng",
            "📊 CPI",
            "💼 Thất nghiệp",
            "🎉 Ngày lễ",
            "🎁 Khuyến mãi"
        ],
        "Thay đổi": [
            f"{temp_change:+.1f}°F",
            f"${fuel_change:+.2f}",
            f"{cpi_change:+.1f}",
            f"{unemp_change:+.1f}%",
            "Có" if is_holiday else "Không",
            "Có" if has_promo else "Không"
        ],
        "Ảnh hưởng": [
            f"{temp_impact:+.2f}%",
            f"{fuel_impact:+.2f}%",
            f"{cpi_impact:+.2f}%",
            f"{unemp_impact:+.2f}%",
            f"{holiday_impact:+.2f}%",
            f"{promo_impact:+.2f}%"
        ],
        "Tác động ($)": [
            format_currency(baseline_sales * temp_impact / 100),
            format_currency(baseline_sales * fuel_impact / 100),
            format_currency(baseline_sales * cpi_impact / 100),
            format_currency(baseline_sales * unemp_impact / 100),
            format_currency(baseline_sales * holiday_impact / 100),
            format_currency(baseline_sales * promo_impact / 100)
        ]
    }
    
    breakdown_df = pd.DataFrame(breakdown_data)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    impacts = [temp_impact, fuel_impact, cpi_impact, unemp_impact, holiday_impact, promo_impact]
    labels = ["Nhiệt độ", "Giá xăng", "CPI", "Thất nghiệp", "Ngày lễ", "Khuyến mãi"]
    colors = ['#3498db' if x >= 0 else '#e74c3c' for x in impacts]
    
    ax.barh(labels, impacts, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel("Ảnh hưởng (%)", fontsize=12)
    ax.set_title("Phân Tích Tác Động Từng Yếu Tố", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (label, value) in enumerate(zip(labels, impacts)):
        x_pos = value + (1 if value >= 0 else -1)
        ax.text(x_pos, i, f"{value:+.2f}%", va='center', fontweight='bold')
    
    st.pyplot(fig)
    plt.close()
    
    # Lời khuyên chiến lược
    st.subheader("💡 Lời Khuyên Chiến Lược")
    
    if total_impact > 10:
        show_info_box(f"""
✅ **Điều kiện rất thuận lợi** {total_impact:.1f}%

- Tăng hàng tồn kho 150-200%
- Tăng nhân viên ca làm việc
- Đẩy mạnh marketing
- Chuẩn bị logistics tốt
- Tối đa hóa doanh thu
        """, "success")
    elif total_impact > 0:
        show_info_box(f"""
📈 **Điều kiện tích cực** {total_impact:.1f}%

- Duy trì mức tồn kho cao hơn bình thường
- Marketing vừa phải
- Theo dõi sát tình hình
- Sẵn sàng điều chỉnh
        """, "info")
    elif total_impact > -10:
        show_info_box(f"""
⚠️ **Điều kiện khó khăn** {total_impact:.1f}%

- Giảm tồn kho, tránh ứ đọng
- Tập trung giảm chi phí
- Khuyến mãi để kích cầu
- Tối ưu hiệu quả vận hành
        """, "warning")
    else:
        show_info_box(f"""
❌ **Điều kiện rất khó** {total_impact:.1f}%

- Tối thiểu hóa tồn kho
- Cắt giảm chi phí mạnh
- Khuyến mãi sâu nếu cần
- Cân nhắc đóng cửa tạm thời một số cửa hàng
        """, "warning")
    
    # Case studies
    with st.expander("📚 Ví Dụ Thực Tế"):
        st.markdown("""
**Case 1: Mùa Hè Nóng Bức**
- Nhiệt độ: +15°F
- Tác động: +2-3% (nước giải khát, kem tăng)

**Case 2: Tăng Giá Xăng**
- Giá xăng: +$0.50
- Tác động: -3-5% (giảm chi tiêu tùy ý)

**Case 3: Black Friday**
- Ngày lễ: Có
- Khuyến mãi: Có
- Tác động: +15-25% (tăng mạnh)

**Case 4: Suy Thoái**
- Thất nghiệp: +3%
- CPI: +10
- Tác động: -8-12% (giảm sức mua)
        """)
    
    st.info("""
💡 **Lưu ý quan trọng:**
- Đây là mô phỏng dựa trên dữ liệu lịch sử 2010-2012
- Kết quả mang tính tham khảo, không phải dự đoán chính xác
- Nên kết hợp với kinh nghiệm thực tế và phân tích thị trường hiện tại
- Các yếu tố khác (cạnh tranh, xu hướng...) cũng ảnh hưởng lớn
    """)

# ==================== MAIN APP ====================

def main():
    """Main application entry point"""
    # Load data
    with st.spinner("🔄 Đang tải dữ liệu..."):
        df_store, df, train, features, stores = load_data()
    
    # Sidebar
    page, df_view = sidebar(df_store)
    
    # Route to pages
    if page == "🏠 Tổng quan":
        page_overview(train, features, stores, df_view)
    elif page == "📊 So sánh cửa hàng":
        page_compare_stores(df_view)
    elif page == "📈 Xu hướng thời gian":
        page_time_trends(df_view)
    elif page == "🎉 Phân tích ngày lễ":
        page_holiday(df_view)
    elif page == "🔍 Phân nhóm thông minh":
        page_clustering(df_view)
    elif page == "🌳 Dự đoán Decision Tree":
        page_decision_tree(df_view)
    elif page == "💡 Dự đoán doanh số":
        page_forecast(df_view)
    else:
        st.error("❌ Trang không tồn tại!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>📊 Walmart Analytics Dashboard | 💼 Phân tích dữ liệu 2010-2012</p>
        <p><small>Được xây dựng với ❤️ bởi Nguyễn Văn Minh (3122410242)</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
