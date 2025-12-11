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

def get_dept_ranking_info(df):
    """Tạo thông tin ranking cho tất cả departments"""
    dept_stats = df.groupby("Dept").agg({
        "Weekly_Sales": ["mean", "median", "count"]
    }).round(0)
    dept_stats.columns = ["Avg", "Median", "Count"]
    dept_stats = dept_stats.sort_values("Avg", ascending=False)
    dept_stats["Rank"] = range(1, len(dept_stats) + 1)
    return dept_stats

# ==================== DATA LOADING ====================

@st.cache_data(show_spinner=False)
def load_data():
    """Load và xử lý dữ liệu"""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))

    # Xử lý datetime
    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])
    
    # Convert IsHoliday to int
    train["IsHoliday"] = train["IsHoliday"].astype(int)
    features["IsHoliday"] = features["IsHoliday"].astype(int)

    # Merge datasets
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    # Tạo features thời gian
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Fill missing MarkDown
    markdown_cols = [col for col in df.columns if "MarkDown" in col]
    for col in markdown_cols:
        df[col] = df[col].fillna(0)

    return train, features, stores, df

# ==================== SIDEBAR ====================

def sidebar(df):
    """Tạo sidebar với filters và navigation"""
    st.sidebar.title("🏪 Walmart Analytics")
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("ℹ️ Về App Này", expanded=False):
        st.markdown("""
        ### Dashboard Phân Tích Walmart
        
        **Dữ liệu:** 421,570 records từ 45 cửa hàng (2010-2012)
        
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
    
    # Filter phòng ban với thông tin ranking
    dept_list = ["Tất cả"] + sorted([int(x) for x in df["Dept"].dropna().unique()])
    
    # Tạo mapping dept -> label
    dept_labels = {}
    dept_stats = get_dept_ranking_info(df)
    
    for dept in dept_list:
        if dept == "Tất cả":
            dept_labels[dept] = "📋 Tất cả phòng ban"
        else:
            avg_sales = dept_stats.loc[dept, "Avg"]
            rank = int(dept_stats.loc[dept, "Rank"])
            
            # Tiếp đầu ngữ (A/B/C/D) theo ranking
            if rank <= 20:
                tier = "A"
            elif rank <= 40:
                tier = "B"
            elif rank <= 60:
                tier = "C"
            else:
                tier = "D"
            
            dept_labels[dept] = f"[{tier}] Dept #{dept:02d} (#{rank}, TB: {format_currency(avg_sales)})"
    
    dept_option = st.sidebar.selectbox(
        "🏬 Phòng ban (Department)", 
        options=dept_list,
        format_func=lambda x: dept_labels[x],
        help="Chọn phòng ban - [A/B/C/D] theo doanh số: A=Cao nhất, D=Thấp nhất"
    )
    
    # Áp dụng filters
    df_view = df[(df["Year"].between(y1, y2)) & (df["Type"].isin(type_sel))].copy()
    if dept_option != "Tất cả":
        df_view = df_view[df_view["Dept"] == dept_option]
    
    # Thống kê filter
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Dữ liệu đã lọc:**")
    st.sidebar.info(f"""  
    - {df_view['Store'].nunique()} cửa hàng
    - {df_view['Dept'].nunique()} phòng ban
    - {len(df_view):,} records
    - {format_currency(df_view['Weekly_Sales'].sum())} tổng doanh số
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
            "📉 Hiệu quả khuyến mãi",
            "💡 Dự toán doanh số"
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
        total_sales = df['Weekly_Sales'].sum()
        st.metric(
            "💰 Tổng Doanh Số", 
            format_currency(total_sales),
            help="Tổng doanh số của tất cả cửa hàng trong giai đoạn phân tích"
        )
        
    with col2:
        avg_sales = df['Weekly_Sales'].mean()
        st.metric(
            "📊 Doanh Số TB/Tuần", 
            format_currency(avg_sales),
            help="Doanh số trung bình mỗi tuần, mỗi phòng ban"
        )
    
    with col3:
        cv = (df['Weekly_Sales'].std() / df['Weekly_Sales'].mean()) * 100
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
    
    # Giải thích CV
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
    
    st.markdown("---")
    
    # Phân bố doanh số
    st.subheader("📊 Phân Bố Doanh Số - Insight Quan Trọng")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df["Weekly_Sales"], bins=50, alpha=0.7, color="skyblue", edgecolor='black')
        ax.axvline(df["Weekly_Sales"].mean(), color='red', linestyle='--', linewidth=2, label=f'Trung bình: {format_currency(avg_sales)}')
        ax.axvline(df["Weekly_Sales"].median(), color='green', linestyle='--', linewidth=2, label=f'Trung vị: {format_currency(df["Weekly_Sales"].median())}')
        ax.set_xlabel("Doanh Số Hàng Tuần ($)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Số Lượng Tuần", fontsize=11, fontweight='bold')
        ax.set_title("Histogram Phân Bố Doanh Số\n(Hình dạng phân bố tiết lộ nhiều thông tin!)", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### 🔍 Phát Hiện Quan Trọng")
        
        mean_val = df["Weekly_Sales"].mean()
        median_val = df["Weekly_Sales"].median()
        
        show_info_box(f"""
        **📌 Phân bố lệch phải!**
        
        - **Trung bình** ({format_currency(mean_val)}) > **Trung vị** ({format_currency(median_val)})
        - Có nhiều tuần bán thấp, ít tuần bán rất cao
        - Những tuần cao thường là ngày lễ hoặc khuyến mãi lớn
        
        **💡 Insight:**
        Walmart không đều đặn - cần:
        - Dự báo chính xác tuần nào "hot"
        - Chuẩn bị hàng hóa linh hoạt
        - Tối ưu nhân sự theo mùa
        """, "info")
    
    st.markdown("---")
    
    # Top/Bottom Performance
    st.subheader("🏆 Phân Tích Hiệu Suất Cửa Hàng")
    
    store_performance = df.groupby('Store').agg({
        'Weekly_Sales': ['sum', 'mean', 'std'],
        'Type': 'first',
        'Size': 'first'
    }).round(0)
    store_performance.columns = ['Tổng DS', 'TB DS', 'Độ Lệch', 'Loại', 'Quy Mô']
    store_performance = store_performance.sort_values('Tổng DS', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🥇 Top 5 Cửa Hàng Tốt Nhất")
        top5 = store_performance.head(5).copy()
        top5['Tổng DS'] = top5['Tổng DS'].apply(lambda x: format_currency(x))
        top5['TB DS'] = top5['TB DS'].apply(lambda x: format_currency(x))
        st.dataframe(top5, use_container_width=True)
        
        top_store = store_performance.index[0]
        top_type = store_performance.iloc[0]['Loại']
        show_info_box(f"""
        **Cửa hàng số {top_store}** (Loại {top_type}) là nhà vô địch!
        
        🎯 **Chiến lược:** Nghiên cứu mô hình cửa hàng này để nhân rộng.
        """, "success")
    
    with col2:
        st.markdown("#### ⚠️ Bottom 5 Cửa Hàng Cần Cải Thiện")
        bottom5 = store_performance.tail(5).copy()
        bottom5['Tổng DS'] = bottom5['Tổng DS'].apply(lambda x: format_currency(x))
        bottom5['TB DS'] = bottom5['TB DS'].apply(lambda x: format_currency(x))
        st.dataframe(bottom5, use_container_width=True)
        
        weak_store = store_performance.index[-1]
        weak_type = store_performance.iloc[-1]['Loại']
        show_info_box(f"""
        **Cửa hàng số {weak_store}** (Loại {weak_type}) cần hỗ trợ.
        
        ⚠️ **Hành động:** Phân tích nguyên nhân (vị trí, cạnh tranh, quản lý).
        """, "warning")
    
    st.markdown("---")
    
    # So sánh Type
    st.subheader("🏪 So Sánh Theo Loại Cửa Hàng")
    
    type_comparison = df.groupby('Type')['Weekly_Sales'].agg(['count', 'mean', 'sum']).round(0)
    type_comparison.columns = ['Số Tuần', 'TB Doanh Số', 'Tổng DS']
    type_comparison['% Contribution'] = (type_comparison['Tổng DS'] / type_comparison['Tổng DS'].sum() * 100).round(1)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        type_comparison['TB Doanh Số'].plot(kind='bar', ax=ax, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel("Loại Cửa Hàng", fontsize=11, fontweight='bold')
        ax.set_ylabel("Doanh Số TB ($)", fontsize=11, fontweight='bold')
        ax.set_title("Doanh Số TB Theo Loại Cửa Hàng", fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        for i, v in enumerate(type_comparison['TB Doanh Số']):
            ax.text(i, v, format_currency(v), ha='center', va='bottom', fontweight='bold')
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
        
        - **Type A** (Super Center): Chiếm ưu thế tuyệt đối
        - **Type B** (Discount Store): Trung bình khá
        - **Type C** (Neighborhood Market): Yếu nhất
        
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
        
        1. **Quý 4** (Oct-Dec): Chuẩn bị 150-200% hàng tồn kho
        2. **Quý 1** (Jan-Mar): Giảm giá mạnh để thanh lý
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
        return f"Store {store} | Loại {info['Type']} | Quy mô {info['Size']:,} sq ft"
    
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
    
    # Aggregate data
    df_filtered = df[df["Store"].isin(selected_stores)]
    df_agg = df_filtered.groupby(["Date", "Store"])["Weekly_Sales"].sum().reset_index()
    
    # Time series comparison
    st.subheader("📈 Xu Hướng Doanh Số Theo Thời Gian")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for store in selected_stores:
        store_data = df_agg[df_agg["Store"] == store]
        ax.plot(store_data["Date"], store_data["Weekly_Sales"], 
                marker='o', markersize=3, label=f"Store {store}", linewidth=2, alpha=0.8)
    
    ax.set_xlabel("Thời Gian", fontsize=12, fontweight='bold')
    ax.set_ylabel("Doanh Số Hàng Tuần ($)", fontsize=12, fontweight='bold')
    ax.set_title("So Sánh Xu Hướng Doanh Số\n(Đường nào ổn định hơn? Đường nào tăng trưởng?)", 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    explain_term(
        "Cách đọc biểu đồ xu hướng",
        """
        **Những điều cần chú ý:**
        
        1. **Đường có nhiều "răng cưa"** = Doanh số không ổn định → Cần điều tra nguyên nhân
        2. **Đường có xu hướng đi lên** = Đang tăng trưởng → Mô hình tốt!
        3. **Đường đi ngang** = Ổn định nhưng không tăng → Cần chiến lược mới
        4. **Đường có đỉnh cao vào Q4** = Tận dụng tốt mùa lễ → Đúng hướng!
        
        **So sánh:**
        - Cửa hàng nào có đường cao hơn = Doanh số tốt hơn
        - Cửa hàng nào ít biến động = Dễ dự đoán và quản lý hơn
        """
    )
    
    st.markdown("---")
    
    # Statistics comparison
    st.subheader("📊 Bảng So Sánh Chi Tiết")
    
    stats_df = df_agg.groupby("Store")["Weekly_Sales"].agg([
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
    
    col1, col2 = st.columns(2)
    
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
            
            - **CV < 20%**: Rất ổn định (tuyệt vời!)
            - **CV 20-40%**: Biến động trung bình (chấp nhận được)
            - **CV > 40%**: Biến động cao (cần cải thiện)
            
            **Ví dụ:**
            - Store A: TB $100K, Độ lệch $20K → CV = 20%
            - Store B: TB $100K, Độ lệch $50K → CV = 50%
            
            → Store A ổn định hơn dù cùng TB!
            """
        )
    
    st.markdown("---")
    
    # Performance ranking
    st.subheader("🥇 Xếp Hạng Hiệu Suất")
    
    ranking = stats_df.copy()
    ranking['Score'] = (
        ranking['Doanh Số TB'] / ranking['Doanh Số TB'].max() * 50 +  # 50% for sales
        (1 - ranking['CV (%)'] / ranking['CV (%)'].max()) * 30 +  # 30% for stability
        (ranking['Cao Nhất'] / ranking['Cao Nhất'].max()) * 20  # 20% for peak potential
    ).round(1)
    ranking = ranking.sort_values('Score', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FFD700', '#C0C0C0', '#CD7F32'] + ['#4CAF50'] * (len(ranking) - 3)
        ranking['Score'].plot(kind='barh', ax=ax, color=colors[:len(ranking)], alpha=0.8, edgecolor='black')
        ax.set_xlabel("Điểm Tổng Hợp", fontsize=11, fontweight='bold')
        ax.set_ylabel("Store", fontsize=11, fontweight='bold')
        ax.set_title("Xếp Hạng Tổng Hợp\n(Kết hợp: Doanh số 50% + Ổn định 30% + Tiềm năng 20%)", 
                     fontsize=12, fontweight='bold')
        for i, v in enumerate(ranking['Score']):
            ax.text(v, i, f' {v:.1f}', va='center', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### 🎯 Giải Thích Điểm Số")
        st.markdown("""
        **Công thức tính điểm:**
        
        1. **50%** từ Doanh Số TB
           - Cửa hàng bán nhiều = Điểm cao
        
        2. **30%** từ Độ Ổn Định
           - CV thấp = Điểm cao
        
        3. **20%** từ Tiềm Năng
           - Đỉnh cao = Điểm cao
        
        **Ý nghĩa:**
        - Điểm > 80: Xuất sắc 🏆
        - Điểm 60-80: Tốt ⭐
        - Điểm < 60: Cần cải thiện ⚠️
        """)
    
    st.markdown("---")
    
    # Action recommendations
    st.subheader("💡 Khuyến Nghị Hành Động")
    
    top_store = ranking.index[0]
    bottom_store = ranking.index[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_info_box(f"""
        ### 🌟 Học Hỏi Từ Store {top_store}
        
        **Điểm mạnh:**
        - Doanh số cao & ổn định
        - Mô hình đáng học hỏi
        
        **Hành động:**
        1. Phỏng vấn quản lý: Bí quyết là gì?
        2. Phân tích: Vị trí, marketing, dịch vụ
        3. Nhân rộng: Áp dụng cho cửa hàng khác
        4. Đầu tư: Mở rộng nếu có thể
        """, "success")
    
    with col2:
        show_info_box(f"""
        ### ⚠️ Cải Thiện Store {bottom_store}
        
        **Vấn đề có thể:**
        - Doanh số thấp hoặc không ổn định
        - Quản lý chưa tối ưu
        
        **Hành động:**
        1. Điều tra: Nguyên nhân gốc rễ?
        2. So sánh: Với store tốt cùng vùng
        3. Thử nghiệm: Marketing mới, layout mới
        4. Quyết định: Cải thiện hoặc đóng cửa
        """, "warning")

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
            'Weekly_Sales': ['mean', 'sum', 'count', 'std']
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
            **🔥 Tháng cao điểm:**
            - **Tháng {peak_month}**: {format_currency(monthly.loc[peak_month, 'TB Doanh Số'])}
            - Cao hơn TB {((monthly.loc[peak_month, 'TB Doanh Số'] / monthly['TB Doanh Số'].mean() - 1) * 100):.1f}%
            
            **❄️ Tháng thấp điểm:**
            - **Tháng {low_month}**: {format_currency(monthly.loc[low_month, 'TB Doanh Số'])}
            - Thấp hơn TB {((1 - monthly.loc[low_month, 'TB Doanh Số'] / monthly['TB Doanh Số'].mean()) * 100):.1f}%
            
            **📈 Biến động:**
            - Cao nhất / Thấp nhất = {(monthly.loc[peak_month, 'TB Doanh Số'] / monthly.loc[low_month, 'TB Doanh Số']):.2f}x
            """, "info")
            
            month_names = {
                1: "Tháng 1 (Sau Tết)", 2: "Tháng 2", 3: "Tháng 3", 
                4: "Tháng 4", 5: "Tháng 5", 6: "Tháng 6",
                7: "Tháng 7", 8: "Tháng 8", 9: "Tháng 9",
                10: "Tháng 10", 11: "Tháng 11 (Black Friday)", 12: "Tháng 12 (Christmas)"
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
            'Weekly_Sales': ['mean', 'sum', 'count']
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
            - Nguyên nhân: Sau lễ, khách "hết tiền"
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
            'Weekly_Sales': ['mean', 'sum', 'count']
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
    
    # Heatmap by month and day of week (if data available)
    if 'DayOfYear' in df.columns and view_type == "Tháng":
        st.subheader("🔥 Bản Đồ Nhiệt Doanh Số")
        
        # Create pivot table
        df_temp = df.copy()
        df_temp['Week_of_Month'] = (df_temp['Date'].dt.day - 1) // 7 + 1
        pivot_data = df_temp.pivot_table(
            values='Weekly_Sales',
            index='Month',
            columns='Week_of_Month',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn', 
                    linewidths=1, cbar_kws={'label': 'Doanh Số ($)'}, ax=ax)
        ax.set_xlabel("Tuần trong Tháng", fontsize=11, fontweight='bold')
        ax.set_ylabel("Tháng", fontsize=11, fontweight='bold')
        ax.set_title("Bản Đồ Nhiệt: Doanh Số TB Theo Tháng & Tuần\n(Đỏ = Cao, Xanh = Thấp)", 
                     fontsize=12, fontweight='bold')
        ax.set_yticklabels(['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'], rotation=0)
        st.pyplot(fig)
        
        explain_term(
            "Cách đọc Bản Đồ Nhiệt",
            """
            **Bản đồ nhiệt (Heatmap) giúp:**
            
            1. **Tìm "điểm nóng"**: Ô màu đỏ = Doanh số cao
            2. **Phát hiện patterns**: Cột/hàng nào đỏ nhiều?
            3. **Lập kế hoạch**: Chuẩn bị cho điểm nóng
            
            **Ví dụ:**
            - Tháng 11-12 đỏ nhiều → Mùa lễ
            - Tuần 4 đỏ hơn → Cuối tháng lương về?
            - Tháng 1-2 xanh nhiều → Mùa thấp điểm
            """
        )
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Khuyến Nghị Chiến Lược")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_info_box("""
        ### 🎄 Mùa Cao Điểm (Q4)
        
        **Chuẩn bị:**
        1. Tăng 150-200% hàng tồn kho
        2. Thuê thêm nhân sự part-time
        3. Tăng ngân sách marketing 2x
        
        **Thời điểm:**
        - Black Friday (T11)
        - Thanksgiving (T11)
        - Christmas (T12)
        """, "success")
    
    with col2:
        show_info_box("""
        ### ❄️ Mùa Thấp Điểm (Q1)
        
        **Phục hồi:**
        1. Khuyến mãi mạnh (20-30%)
        2. Thanh lý hàng tồn kho
        3. Giảm chi phí vận hành
        
        **Mục tiêu:**
        - Duy trì cash flow
        - Giữ chân khách hàng cũ
        """, "warning")
    
    with col3:
        show_info_box("""
        ### 📊 Quản Lý Linh Hoạt
        
        **Chiến thuật:**
        1. Dự báo hàng tuần/tháng
        2. Điều chỉnh nhân sự linh hoạt
        3. Marketing theo mùa
        
        **Công cụ:**
        - Dashboard real-time
        - Alert doanh số bất thường
        """, "info")

# ==================== PAGE: PHÂN TÍCH NGÀY LỄ ====================

def page_holiday(df):
    """Trang phân tích ngày lễ"""
    st.title("🎉 Ngày Lễ vs Tuần Thường")
    
    if df.empty:
        st.info("Không có dữ liệu theo bộ lọc.")
        return

    st.markdown("""
    **Phần này làm gì?** Kiểm tra xem ngày lễ có làm tăng doanh số không.

    **Chọn gì để làm gì?** Chọn "Tổng thể" hoặc "Theo loại cửa hàng" để so sánh chi tiết hơn.

    **Ý nghĩa kết quả:** Biết được hiệu quả của ngày lễ, lập kế hoạch kinh doanh.

    **Nên làm gì tiếp theo?** Nếu lễ tăng mạnh, tăng hàng tồn kho; nếu không, tập trung khuyến mãi khác.
    """)

    # Option để so sánh theo nhóm
    group_by = st.selectbox("So sánh theo", ["Tổng thể", "Theo loại cửa hàng (A/B/C)"])

    if group_by == "Tổng thể":
        holiday_sales = df[df["IsHoliday"] == 1]["Weekly_Sales"]
        normal_sales = df[df["IsHoliday"] == 0]["Weekly_Sales"]

        if len(holiday_sales) == 0 or len(normal_sales) == 0:
            st.warning("Không đủ dữ liệu để so sánh.")
            return

        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Doanh số tuần thường", f"{format_currency(normal_sales.mean())} ± {format_currency(normal_sales.std())}")
            st.metric("Doanh số tuần lễ", f"{format_currency(holiday_sales.mean())} ± {format_currency(holiday_sales.std())}")
            
            diff = holiday_sales.mean() - normal_sales.mean()
            pct = (diff / normal_sales.mean()) * 100
            st.metric("Chênh lệch", f"{format_currency(diff)} ({pct:.1f}%)")

        with col2:
            # Barplot
            fig, ax = plt.subplots(figsize=(6,5))
            means = [normal_sales.mean(), holiday_sales.mean()]
            ax.bar(["Tuần thường", "Tuần lễ"], means, color=["#3498db", "#e74c3c"], alpha=0.8)
            ax.set_ylabel("Doanh số trung bình ($)")
            ax.set_title("Doanh số TB: Tuần thường vs Tuần lễ")
            for i, v in enumerate(means):
                ax.text(i, v, format_currency(v), ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        # T-test
        t_stat, p_value = stats.ttest_ind(holiday_sales, normal_sales)
        st.write(f"**T-test:** T-statistic = {t_stat:.2f}, P-value = {p_value:.6f}")
        
        if p_value < 0.05:
            st.success("✅ **Kết luận:** Sự khác biệt có ý nghĩa thống kê (p < 0.05). Ngày lễ THỰC SỰ làm tăng doanh số!")
        else:
            st.warning("⚠️ **Kết luận:** Sự khác biệt không rõ ràng (p >= 0.05). Có thể do ngẫu nhiên.")

    else:
        # So sánh theo Type
        types = sorted(df["Type"].unique())
        for typ in types:
            st.subheader(f"📊 Loại cửa hàng {typ}")
            df_typ = df[df["Type"] == typ]
            holiday_sales = df_typ[df_typ["IsHoliday"] == 1]["Weekly_Sales"]
            normal_sales = df_typ[df_typ["IsHoliday"] == 0]["Weekly_Sales"]
            
            if len(holiday_sales) > 0 and len(normal_sales) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tuần thường", format_currency(normal_sales.mean()))
                with col2:
                    st.metric("Tuần lễ", format_currency(holiday_sales.mean()))
                with col3:
                    diff = holiday_sales.mean() - normal_sales.mean()
                    st.metric("Chênh lệch", format_currency(diff))
            else:
                st.write("Không đủ dữ liệu để so sánh.")
            
            st.markdown("---")

    # Lời khuyên chiến lược
    st.subheader("💡 Lời Khuyên Chiến Lược")
    
    if group_by == "Tổng thể" and len(holiday_sales) > 0 and len(normal_sales) > 0:
        diff_pct = (holiday_sales.mean() - normal_sales.mean()) / normal_sales.mean() * 100
        
        if diff_pct > 10:
            show_info_box(f"""
Hiệu quả lễ cao ({diff_pct:.1f}%):

- Tăng đầu tư quảng bá mùa lễ 50-100%
- Chuẩn bị hàng tồn kho 150-200%
- Thuê thêm nhân viên part-time
- Marketing sớm 2-3 tuần trước lễ
            """, "success")
        elif diff_pct > 0:
            show_info_box(f"""
Hiệu quả lễ nhẹ ({diff_pct:.1f}%):

- Kết hợp với khuyến mãi khác để tăng tác động
- A/B testing các chiến lược marketing
- Tập trung vào trải nghiệm khách hàng
            """, "info")
        else:
            show_info_box(f"""
Không hiệu quả ({diff_pct:.1f}%):

- Tập trung vào tuần thường thay vì lễ
- Cải thiện trải nghiệm khách hàng
- Phân tích nguyên nhân sâu xa
            """, "warning")
    else:
        st.info("Theo loại cửa hàng: Loại A thường hiệu quả nhất. Ưu tiên đầu tư cho loại A trong mùa lễ.")

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
    available_features = ["Weekly_Sales", "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    feature_labels = {
        "Weekly_Sales": "Doanh số trung bình",
        "Size": "Quy mô cửa hàng",
        "Temperature": "Nhiệt độ trung bình",
        "Fuel_Price": "Giá xăng trung bình",
        "CPI": "Chỉ số CPI",
        "Unemployment": "Tỷ lệ thất nghiệp"
    }
    
    selected_features = st.multiselect(
        "Chọn đặc tính để phân cụm", 
        options=available_features, 
        default=["Weekly_Sales", "Size", "CPI", "Unemployment"],
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
    
    if "Size" in selected_features and "Weekly_Sales" in selected_features:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10,7))
            scatter = ax.scatter(
                store_features["Size"], 
                store_features["Weekly_Sales"], 
                c=clusters, 
                cmap="viridis", 
                s=150, 
                alpha=0.7,
                edgecolor='black'
            )
            
            # Add labels
            for idx, row in store_features.iterrows():
                ax.annotate(f"S{idx}", (row["Size"], row["Weekly_Sales"]), 
                           fontsize=8, ha='center')
            
            ax.set_xlabel("Quy mô cửa hàng (Size)", fontsize=12)
            ax.set_ylabel("Doanh số TB (Weekly_Sales)", fontsize=12)
            ax.set_title(f"Phân Cụm {k} Nhóm Cửa Hàng", fontsize=13)
            ax.legend(*scatter.legend_elements(), title="Cụm")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Pie chart
            cluster_counts = store_features["Cluster"].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(cluster_counts.values, labels=[f"Cụm {i}" for i in cluster_counts.index],
                   autopct='%1.1f%%', startangle=90)
            ax.set_title(f"Phân bố {len(store_features)} cửa hàng")
            st.pyplot(fig)
            plt.close()
            
            for cluster_id, count in cluster_counts.items():
                st.write(f"**Cụm {cluster_id}:** {count} cửa hàng")
    else:
        st.info("💡 Chọn cả 'Doanh số TB' và 'Quy mô' để xem biểu đồ scatter!")

    # Cluster summary
    st.subheader("📊 Đặc Điểm Từng Cụm")
    
    cluster_summary = store_features.groupby("Cluster")[selected_features].mean().round(2)
    
    # Display table
    display_summary = cluster_summary.copy()
    if "Weekly_Sales" in selected_features:
        display_summary["Weekly_Sales"] = display_summary["Weekly_Sales"].apply(lambda x: format_currency(x))
    
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
    
    if "Weekly_Sales" in selected_features:
        best_cluster = cluster_summary["Weekly_Sales"].idxmax()
        worst_cluster = cluster_summary["Weekly_Sales"].idxmin()
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_info_box(f"""
🌟 Cụm mạnh nhất ({best_cluster}):

- Doanh số cao: {format_currency(cluster_summary.loc[best_cluster, 'Weekly_Sales'])}
- Học hỏi mô hình quản lý
- Nhân rộng thành công
- Tăng đầu tư cho cụm này
            """, "success")
        
        with col2:
            show_info_box(f"""
⚠️ Cụm yếu nhất ({worst_cluster}):

- Doanh số thấp: {format_currency(cluster_summary.loc[worst_cluster, 'Weekly_Sales'])}
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

# ==================== PAGE: HIỆU QUẢ KHUYẾN MÃI ====================

def page_promo(df):
    """Trang phân tích khuyến mãi"""
    st.title("📉 Ảnh Hưởng Khuyến Mãi")
    
    st.markdown("""
    **Phần này làm gì?** Phân tích xem chương trình khuyến mãi (giảm giá) có tăng doanh số không.

    **Chọn gì để làm gì?** 
    - Chọn phạm vi (toàn bộ hoặc cửa hàng cụ thể).
    - Chọn ngưỡng giảm giá để định nghĩa "có khuyến mãi".
    - Lọc ngày lễ để xem hiệu quả trong bối cảnh khác nhau.

    **Ý nghĩa kết quả:** Biết được khuyến mãi có hiệu quả, lập kế hoạch marketing.

    **Nên làm gì tiếp theo?** Nếu hiệu quả, tăng đầu tư khuyến mãi; nếu không, thử chiến lược khác.
    """)
    
    st.info("**Giảm giá là gì?** Đây là số tiền giảm giá ($) áp dụng cho tuần đó tại cửa hàng. Giá trị >0 = có khuyến mãi, =0 = không có.")
    
    if df.empty:
        st.info("Không có dữ liệu theo bộ lọc hiện tại.")
        return

    # Options
    st.subheader("⚙️ Tùy Chọn Phân Tích")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scope = st.radio("Phạm vi", ["Toàn bộ hệ thống", "Theo cửa hàng"], horizontal=True)
        if scope == "Theo cửa hàng":
            store_ids = sorted(df["Store"].unique())
            store_id = st.selectbox("Chọn Store", store_ids)
            dff = df[df["Store"] == store_id].copy()
        else:
            dff = df.copy()
    
    with col2:
        include_holiday = st.selectbox(
            "Bộ lọc ngày lễ", 
            ["Tất cả", "Chỉ tuần lễ", "Chỉ tuần thường"], 
            index=0,
            help="Lọc theo tuần lễ để xem hiệu ứng khác nhau"
        )
        
        if include_holiday == "Chỉ tuần lễ":
            dff = dff[dff["IsHoliday"] == 1]
        elif include_holiday == "Chỉ tuần thường":
            dff = dff[dff["IsHoliday"] == 0]

    if "MarkDown1" not in dff.columns:
        st.warning("⚠️ Thiếu cột MarkDown1 trong dữ liệu hiện tại.")
        return
    
    threshold = st.number_input(
        "Ngưỡng Giảm giá (>= là có khuyến mãi)", 
        min_value=0.0, 
        value=0.0, 
        step=10.0,
        help="Chọn mức giảm giá từ bao nhiêu trở lên được coi là 'Có khuyến mãi'"
    )
    
    dff["Promo"] = (dff["MarkDown1"].fillna(0) >= threshold).astype(int)

    # Comparison
    st.subheader("📊 So Sánh: Có Khuyến Mãi vs Không")
    
    grp = dff.groupby("Promo")["Weekly_Sales"].agg(["mean", "median", "count", "std"]).rename(
        index={0: "Không KM", 1: "Có KM"}
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(grp.style.format({
            "mean": "${:,.0f}",
            "median": "${:,.0f}",
            "std": "${:,.0f}",
            "count": "{:.0f}"
        }))
        
        st.caption("**mean**: trung bình | **median**: trung vị | **count**: số tuần | **std**: độ lệch chuẩn")
    
    with col2:
        # Bar chart
        if len(grp) >= 2:
            fig, ax = plt.subplots(figsize=(7,5))
            means = grp["mean"].values
            ax.bar(["Không KM", "Có KM"], means, color=["#3498db", "#e74c3c"], alpha=0.8)
            ax.set_ylabel("Doanh số trung bình ($)")
            ax.set_title("So sánh doanh số")
            for i, v in enumerate(means):
                ax.text(i, v, format_currency(v), ha='center', va='bottom', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()

    # Statistical test
    a = dff.loc[dff["Promo"] == 1, "Weekly_Sales"].astype(float)
    b = dff.loc[dff["Promo"] == 0, "Weekly_Sales"].astype(float)
    
    if len(a) > 2 and len(b) > 2:
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        
        # Cohen's d
        n1, n0 = len(a), len(b)
        m1, m0 = a.mean(), b.mean()
        s1, s0 = a.std(ddof=1), b.std(ddof=1)
        pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / max(n1 + n0 - 2, 1))
        cohens_d = (m1 - m0) / pooled if pooled > 0 else np.nan
        
        st.subheader("📈 Kiểm Định Thống Kê")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P-value", f"{p_val:.4f}")
        with col2:
            st.metric("Chênh lệch TB", format_currency(m1 - m0))
        with col3:
            st.metric("Cohen's d", f"{cohens_d:.3f}")
        
        with st.expander("❓ Giải thích các chỉ số"):
            st.markdown("""
**P-value**: Xác suất kết quả do ngẫu nhiên
- ≤ 0.05: Khác biệt THỰC SỰ, không phải tình cờ ✅
- > 0.05: Chưa đủ bằng chứng ⚠️

**Cohen's d**: Đo "độ mạnh" khác biệt
- < 0.2: Rất nhỏ (không đáng kể)
- 0.2-0.5: Nhỏ đến vừa
- 0.5-0.8: Vừa đến lớn ✅
- > 0.8: Rất lớn! 🎉
            """)
        
        # Conclusion
        significant = p_val <= 0.05
        delta = m1 - m0
        
        if significant and delta > 0:
            if abs(cohens_d) >= 0.5:
                show_info_box("""
✅ Khuyến mãi hiệu quả cao!

- Có ý nghĩa thống kê
- Effect size lớn
- Nên tăng đầu tư khuyến mãi
                """, "success")
            else:
                show_info_box("""
⚠️ Khuyến mãi có hiệu quả nhẹ

- Có ý nghĩa thống kê
- Nhưng effect size nhỏ
- Cân nhắc ROI trước khi đầu tư
                """, "info")
        else:
            show_info_box("""
❌ Chưa rõ hiệu quả

- Không có ý nghĩa thống kê
- Không nên đầu tư nhiều vào khuyến mãi
- Thử chiến lược khác
            """, "warning")
    
    # Lời khuyên chiến lược
    st.subheader("💡 Lời Khuyên Chiến Lược")
    
    if len(a) > 2 and len(b) > 2:
        delta = m1 - m0
        significant = p_val <= 0.05
        
        if significant and delta > 0:
            st.success("""
Khuyến mãi hiệu quả - Hành động:
1. Tăng tần suất khuyến mãi
2. Tăng mức giảm giá hợp lý
3. Mở rộng sản phẩm khuyến mãi
4. Marketing mạnh mẽ hơn
5. Chuẩn bị hàng tồn kho đầy đủ
            """)
        elif significant and delta < 0:
            st.warning("""
Khuyến mãi không hiệu quả - Hành động:
1. Giảm ngân sách khuyến mãi
2. Tập trung vào chất lượng sản phẩm
3. Cải thiện dịch vụ khách hàng
4. Xây dựng loyalty program
5. Phân tích nguyên nhân sâu xa
            """)
        else:
            st.info("""
Chưa rõ hiệu quả - Hành động:
1. Thu thập thêm dữ liệu
2. A/B testing với mức giảm giá khác nhau
3. Phân tích theo từng phân khúc khách hàng
4. Kết hợp nhiều chiến lược marketing
            """)
    else:
        st.warning("⚠️ Dữ liệu không đủ để phân tích chi tiết.")

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
    baseline_sales = baseline_df["Weekly_Sales"].mean()
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
    temp_corr = df[["Temperature", "Weekly_Sales"]].corr().iloc[0, 1]
    fuel_corr = df[["Fuel_Price", "Weekly_Sales"]].corr().iloc[0, 1]
    cpi_corr = df[["CPI", "Weekly_Sales"]].corr().iloc[0, 1]
    unemp_corr = df[["Unemployment", "Weekly_Sales"]].corr().iloc[0, 1]
    
    # Tính % thay đổi
    temp_impact = (temp_change / baseline_temp) * temp_corr * 100 if baseline_temp != 0 else 0
    fuel_impact = (fuel_change / baseline_fuel) * fuel_corr * 100 if baseline_fuel != 0 else 0
    cpi_impact = (cpi_change / baseline_cpi) * cpi_corr * 100 if baseline_cpi != 0 else 0
    unemp_impact = (unemp_change / baseline_unemp) * unemp_corr * 100 if baseline_unemp != 0 else 0
    
    # Ảnh hưởng ngày lễ (từ dữ liệu thực)
    holiday_sales = df[df["IsHoliday"] == 1]["Weekly_Sales"].mean()
    normal_sales = df[df["IsHoliday"] == 0]["Weekly_Sales"].mean()
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
✅ **Điều kiện rất thuận lợi (+{total_impact:.1f}%)**

- Tăng hàng tồn kho 150-200%
- Tăng nhân viên ca làm việc
- Đẩy mạnh marketing
- Chuẩn bị logistics tốt
- Tối đa hóa doanh thu
        """, "success")
    elif total_impact > 0:
        show_info_box(f"""
📈 **Điều kiện tích cực (+{total_impact:.1f}%)**

- Duy trì mức tồn kho cao hơn bình thường
- Marketing vừa phải
- Theo dõi sát tình hình
- Sẵn sàng điều chỉnh
        """, "info")
    elif total_impact > -10:
        show_info_box(f"""
⚠️ **Điều kiện khó khăn ({total_impact:.1f}%)**

- Giảm tồn kho, tránh ứ đọng
- Tập trung giảm chi phí
- Khuyến mãi để kích cầu
- Tối ưu hiệu quả vận hành
        """, "warning")
    else:
        show_info_box(f"""
❌ **Điều kiện rất khó ({total_impact:.1f}%)**

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
        train, features, stores, df = load_data()
    
    # Sidebar
    page, df_view = sidebar(df)
    
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
    elif page == "📉 Hiệu quả khuyến mãi":
        page_promo(df_view)
    elif page == "💡 Dự toán doanh số":
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
