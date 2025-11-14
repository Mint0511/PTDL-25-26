import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
#import pyfolio as pf

# ==============================================================================
# DANH SÁCH MÃ CỔ PHIẾU VN30
# ==============================================================================
VN30_TICKERS = [
    'ACB.VN', 'BCM.VN', 'BID.VN', 'BWE.VN', 'CTG.VN', 'FPT.VN', 'GAS.VN', 
    'GVR.VN', 'HDB.VN', 'HPG.VN', 'MBB.VN', 'MSN.VN', 'MWG.VN', 'PLX.VN', 
    'POW.VN', 'SAB.VN', 'SHB.VN', 'SSB.VN', 'SSI.VN', 'STB.VN', 'TCB.VN', 
    'TPB.VN', 'VCB.VN', 'VHM.VN', 'VIB.VN', 'VIC.VN', 'VJC.VN', 'VNM.VN', 
    'VPB.VN', 'VRE.VN'
]

# Đặt cấu hình trang Streamlit
st.set_page_config(page_title="Dashboard Tài chính VN30", layout="wide")

#==============================================================================
# Tab 1 Tóm tắt
#==============================================================================

@st.cache_data(show_spinner="Đang tải dữ liệu tóm tắt...")
def get_summary_data(ticker):
    """
    Lấy dữ liệu tóm tắt cơ bản từ yfinance.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    summary_data = {
        "Giá đóng cửa phiên trước": info.get('previousClose'),
        "Giá mở cửa": info.get('open'),
        "Biên độ ngày": f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
        "Biên độ 52 tuần": f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
        "Khối lượng GD": info.get('volume'),
        "KL trung bình (3T)": info.get('averageVolume'),
        "Vốn hóa": info.get('marketCap'),
        "Beta (5Y)": info.get('beta'),
        "P/E (TTM)": info.get('trailingPE'),
        "EPS (TTM)": info.get('trailingEps'),
        "Cổ tức & Tỷ suất": f"{info.get('dividendRate', 'N/A')} ({info.get('dividendYield', 0) * 100:.2f}%)"
    }
    
    df = pd.DataFrame.from_dict(summary_data, orient='index', columns=['Giá trị'])
    df.index.name = 'Chỉ số'

    # Normalize all values to strings to avoid mixed-type column issues
    def _format_val(v):
        if v is None:
            return 'N/A'
        # numpy nan
        try:
            if isinstance(v, float) and np.isnan(v):
                return 'N/A'
        except Exception:
            pass

        # Numbers -> nicely formatted string
        try:
            if isinstance(v, (int, float, np.integer, np.floating)):
                if float(v).is_integer():
                    return f"{int(v):,}"
                return f"{v:,.2f}"
        except Exception:
            pass

        # Otherwise, fallback to string
        return str(v)

    df['Giá trị'] = df['Giá trị'].apply(_format_val)
    # Return a copy to avoid fragmented memory view warnings in pandas
    return df.copy()

@st.cache_data(show_spinner="Đang tải dữ liệu biểu đồ...")
def getstockdata(ticker):
    """
    Lấy dữ liệu giá lịch sử tối đa.
    """
    stockdata = yf.download(ticker, period='max', auto_adjust=False, progress=False)
    
    # Flatten multi-index columns if present (happens with some tickers)
    if isinstance(stockdata.columns, pd.MultiIndex):
        stockdata.columns = stockdata.columns.get_level_values(0)
    
    return stockdata
    
def tab1():
    st.title("Tổng quan")
    st.write(f"### {ticker}")
    
    if ticker != '-':
        # --- Block 1: Bảng Tóm tắt ---
        try:
            summary_df = get_summary_data(ticker)
            st.dataframe(summary_df, width='stretch')
        except Exception as e:
            st.error(f"Không thể tải dữ liệu thông tin cơ bản cho mã {ticker}.")
            st.warning("Dữ liệu có thể không có sẵn hoặc API gặp lỗi.")
            
        # --- Block 2: Biểu đồ Area ---
        try:
            chartdata = getstockdata(ticker) 
            if chartdata.empty:
                st.warning(f"Không có dữ liệu lịch sử giá cho mã {ticker}.")
            else:
                fig = px.area(chartdata, x=chartdata.index, y=chartdata['Close'], title=f"Biểu đồ giá {ticker} (Toàn bộ)")
                fig.update_xaxes(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1T", step="month", stepmode="backward"),
                            dict(count=3, label="3T", step="month", stepmode="backward"),
                            dict(count=6, label="6T", step="month", stepmode="backward"),
                            dict(count=1, label="Đầu năm", step="year", stepmode="todate"),
                            dict(count=1, label="1N", step="year", stepmode="backward"),
                            dict(count=3, label="3N", step="year", stepmode="backward"),
                            dict(count=5, label="5N", step="year", stepmode="backward"),
                            dict(label = "Tất cả", step="all")
                        ])
                    )
                )
                st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.error(f"Không thể tải dữ liệu biểu đồ cho mã {ticker}.")
            st.warning("Dữ liệu lịch sử giá có thể không có sẵn.")
            

#==============================================================================
# Tab 2 Chart (Biểu đồ Kỹ thuật)
#==============================================================================

@st.cache_data(show_spinner="Đang tải dữ liệu biểu đồ chi tiết...")
def getchartdata(ticker, duration, inter, start_date, end_date):
    """
    Lấy dữ liệu SMA và dữ liệu biểu đồ chính, sau đó gộp lại.
    """
    try:
        # 1. Lấy dữ liệu MAX để tính SMA
        SMA_data = yf.download(ticker, period='max', auto_adjust=False, progress=False)
        if SMA_data.empty:
            return pd.DataFrame()

        # Flatten multi-index columns if present
        if isinstance(SMA_data.columns, pd.MultiIndex):
            SMA_data.columns = SMA_data.columns.get_level_values(0)
        
        SMA_data['SMA'] = SMA_data['Close'].rolling(50).mean()
        SMA_data = SMA_data.reset_index()
        SMA_data['Date'] = pd.to_datetime(SMA_data['Date']).dt.tz_localize(None)
        SMA = SMA_data[['Date', 'SMA']].copy()

        # 2. Lấy dữ liệu biểu đồ chính
        if duration != '-':
            chartdata = yf.download(ticker, period=duration, interval=inter, auto_adjust=False, progress=False)
        else:
            chartdata = yf.download(ticker, start=start_date, end=end_date, interval=inter, auto_adjust=False, progress=False)

        if chartdata.empty:
            return pd.DataFrame()

        # Flatten multi-index columns if present
        if isinstance(chartdata.columns, pd.MultiIndex):
            chartdata.columns = chartdata.columns.get_level_values(0)

        chartdata = chartdata.reset_index()
        chartdata['Date'] = pd.to_datetime(chartdata['Date']).dt.tz_localize(None)

        # Sort by Date
        SMA = SMA.sort_values('Date').reset_index(drop=True)
        chartdata = chartdata.sort_values('Date').reset_index(drop=True)

        # Merge SMA using merge_asof for time-series alignment
        chartdata_merged = pd.merge_asof(chartdata, SMA, on='Date', direction='backward')
        
        # Return copy to avoid fragmentation warnings
        return chartdata_merged.copy()
        
    except Exception as ex:
        print(f"getchartdata error for {ticker}: {ex}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def tab2():
    st.title("Biểu đồ Kỹ thuật")
    st.write(f"### {ticker}")
    
    st.info("💡 Chọn 'Khoảng thời gian' = '-' nếu muốn tùy chỉnh khoảng ngày cụ thể")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        start_date = st.date_input("Từ ngày", datetime.today().date() - timedelta(days=365))
    with c2:
        end_date = st.date_input("Đến ngày", datetime.today().date())        
    with c3:
        duration = st.selectbox("Khoảng thời gian", ['-', '1mo', '3mo', '6mo', 'ytd','1y', '3y','5y', 'max'], key='duration_tab2')          
    with c4: 
        inter = st.selectbox("Khung thời gian", ['1d', '1wk', '1mo'], key='interval_tab2')
    with c5:
        plot = st.selectbox("Kiểu biểu đồ", ['Đường', 'Nến'], key='plot_tab2')
        
    if ticker != '-':
        try:
            # Validate dates before calling the cached data loader
            if duration == '-' and start_date >= end_date:
                st.error("Lỗi: Ngày bắt đầu phải trước ngày kết thúc.")
                return

            chartdata = getchartdata(ticker, duration, inter, start_date, end_date)

            if not chartdata.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                if plot == 'Đường':
                    fig.add_trace(go.Scatter(x=chartdata['Date'], y=chartdata['Close'], mode='lines', 
                                             name = 'Giá đóng cửa'), secondary_y = False)
                else:
                    fig.add_trace(go.Candlestick(x = chartdata['Date'], open = chartdata['Open'], 
                                                 high = chartdata['High'], low = chartdata['Low'], close = chartdata['Close'], name = 'Nến Nhật'))
                  
                fig.add_trace(go.Scatter(x=chartdata['Date'], y=chartdata['SMA'], mode='lines', name = 'SMA 50'), secondary_y = False)
                fig.add_trace(go.Bar(x = chartdata['Date'], y = chartdata['Volume'], name = 'Khối lượng GD'), secondary_y = True)
                fig.update_yaxes(range=[0, chartdata['Volume'].max()*3], showticklabels=False, secondary_y=True)
                
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Không có dữ liệu với tham số đã chọn.")
                st.write("Nguyên nhân: mã không hợp lệ, không có dữ liệu, hoặc lỗi kết nối.")
                st.write("Kiểm tra bằng cách chạy:")
                st.code("import yfinance as yf\nprint(yf.download('ACB.VN', period='1mo'))", language='python')
        
        except Exception as e:
            st.error(f"Lỗi khi tạo biểu đồ cho mã {ticker}: {e}")

#==============================================================================
# Tab 3 Dividends & Splits (TAB MỚI thay thế Statistics)
#==============================================================================

@st.cache_data(show_spinner="Đang tải lịch sử giao dịch...")
def get_actions(ticker):
    """
    Lấy lịch sử Cổ tức và Chia tách.
    """
    stock = yf.Ticker(ticker)
    dividends = stock.dividends
    splits = stock.splits
    
    dividends = dividends.sort_index(ascending=False)
    splits = splits.sort_index(ascending=False)
    
    return dividends, splits

def tab3():
    st.title("Cổ tức & Chia tách")
    st.write(f"### {ticker}")
    
    if ticker != '-':
        try:
            dividends, splits = get_actions(ticker)
            
            st.subheader("Lịch sử Chi trả Cổ tức")
            if not dividends.empty:
                st.dataframe(dividends, width='stretch')
            else:
                st.info(f"Chưa có dữ liệu chi trả cổ tức cho mã {ticker}.")
            
            st.subheader("Lịch sử Chia tách Cổ phiếu")
            if not splits.empty:
                st.dataframe(splits, width='stretch')
            else:
                st.info(f"Chưa có dữ liệu chia tách cổ phiếu cho mã {ticker}.")
        
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu cho mã {ticker}: {e}")
            
#==============================================================================
# Tab 4 Financials (Báo cáo Tài chính)
#==============================================================================

@st.cache_data(show_spinner="Đang tải báo cáo tài chính...")
def get_financials(ticker, period='yearly'):
    stock = yf.Ticker(ticker)
    if period == 'yearly':
        return stock.financials
    else:
        return stock.quarterly_financials

@st.cache_data(show_spinner="Đang tải bảng cân đối kế toán...")
def get_balance_sheet(ticker, period='yearly'):
    stock = yf.Ticker(ticker)
    if period == 'yearly':
        return stock.balance_sheet
    else:
        return stock.quarterly_balance_sheet

@st.cache_data(show_spinner="Đang tải báo cáo lưu chuyển tiền tệ...")
def get_cash_flow(ticker, period='yearly'):
    stock = yf.Ticker(ticker)
    if period == 'yearly':
        return stock.cashflow
    else:
        return stock.quarterly_cashflow

def tab4():
    st.title("Báo cáo Tài chính")
    st.write(f"### {ticker}")
      
    statement = st.selectbox("Loại báo cáo", ['Báo cáo Thu nhập', 'Bảng Cân đối Kế toán', 'Báo cáo Lưu chuyển Tiền tệ'])
    period = st.selectbox("Chu kỳ", ['Năm', 'Quý'])
      
    if ticker != '-':
        try:
            data = pd.DataFrame()
            period_eng = 'yearly' if period == 'Năm' else 'quarterly'
            
            if statement == 'Báo cáo Thu nhập':
                data = get_financials(ticker, period_eng)
            elif statement == 'Bảng Cân đối Kế toán':
                data = get_balance_sheet(ticker, period_eng)
            elif statement == 'Báo cáo Lưu chuyển Tiền tệ':
                data = get_cash_flow(ticker, period_eng)
            
            if data.empty:
                st.warning(f"Không tìm thấy dữ liệu '{statement}' cho mã {ticker}.")
            else:
                st.dataframe(data, width='stretch')
            
        except Exception as e:
            st.error(f"Không thể tải dữ liệu tài chính cho {ticker}.")
            st.warning("Yahoo Finance có thể không cung cấp báo cáo chi tiết cho mã này.")

#==============================================================================
# Tab 5 Holders & Recommendations (TAB MỚI thay thế Analysis)
#==============================================================================

@st.cache_data(show_spinner="Đang tải dữ liệu cổ đông...")
def get_analysis_data(ticker):
    """
    Lấy dữ liệu Khuyến nghị và Cổ đông.
    """
    stock = yf.Ticker(ticker)
    recs = stock.recommendations
    inst_holders = stock.institutional_holders
    mf_holders = stock.mutualfund_holders
    
    return recs, inst_holders, mf_holders

def tab5():
    st.title("Phân tích & Cổ đông")
    st.write(f"### {ticker}")
    
    if ticker != '-':
        try:
            recs, inst_holders, mf_holders = get_analysis_data(ticker)
            
            st.subheader("Khuyến nghị của Nhà phân tích")
            if recs is not None and not recs.empty:
                st.dataframe(recs.tail(10).sort_index(ascending=False), width='stretch')
            else:
                st.info(f"Chưa có khuyến nghị phân tích cho mã {ticker}.")

            st.subheader("Cổ đông Tổ chức Lớn")
            if inst_holders is not None and not inst_holders.empty:
                st.dataframe(inst_holders, width='stretch')
            else:
                st.info(f"Chưa có dữ liệu cổ đông tổ chức cho mã {ticker}.")
            
            st.subheader("Cổ đông Quỹ Đầu tư Lớn")
            if mf_holders is not None and not mf_holders.empty:
                st.dataframe(mf_holders, width='stretch')
            else:
                st.info(f"Chưa có dữ liệu cổ đông quỹ cho mã {ticker}.")
        
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu phân tích cho mã {ticker}: {e}")
            
#==============================================================================
# Tab 6 Monte Carlo Simulation
#==============================================================================

@st.cache_data(show_spinner="Đang chạy mô phỏng Monte Carlo...")
def montecarlo(ticker, time_horizon, simulations):
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    stock_price = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    
    if stock_price.empty:
        return pd.DataFrame(), None
    
    # Flatten multi-index columns if present
    if isinstance(stock_price.columns, pd.MultiIndex):
        stock_price.columns = stock_price.columns.get_level_values(0)
        
    close_price = stock_price['Close']
    daily_return = close_price.pct_change()
    daily_volatility = np.std(daily_return)
    
    last_price = close_price.iloc[-1]

    # Build simulation matrix in-memory to avoid DataFrame fragmentation
    sim_matrix = np.empty((time_horizon, simulations), dtype=float)

    for i in range(simulations):
        current_sim_price = last_price
        col = np.empty(time_horizon, dtype=float)
        for x in range(time_horizon):
            future_return = np.random.normal(0, daily_volatility)
            future_price = current_sim_price * (1 + future_return)
            col[x] = future_price
            current_sim_price = future_price
        sim_matrix[:, i] = col

    simulation_df = pd.DataFrame(sim_matrix)
    return simulation_df, last_price

def tab6():
    st.title("Mô phỏng Monte Carlo")
    st.write(f"### {ticker}")
     
    simulations = st.selectbox("Số kịch bản mô phỏng", [200, 500, 1000], key='sim_count')
    time_horizon = st.selectbox("Số ngày dự báo", [30, 60, 90], key='sim_horizon')
     
    if ticker != '-':
        try:
            mc_df, last_price = montecarlo(ticker, time_horizon, simulations)
            
            if last_price is None:
                st.warning(f"Không đủ dữ liệu lịch sử cho mã {ticker} để chạy mô phỏng.")
                return

            fig, ax = plt.subplots(figsize=(15, 10))
            ax.plot(mc_df)
            plt.title(f"Mô phỏng Monte Carlo - {ticker} ({time_horizon} phiên giao dịch)")
            plt.xlabel('Phiên')
            plt.ylabel('Giá (VND)')
            
            plt.axhline(y=last_price, color='red', linestyle='--', label=f'Giá hiện tại: {np.round(last_price, 2):,.0f} VND')
            plt.legend()
            st.pyplot(fig, width='stretch')
            
            st.subheader('Giá trị Rủi ro (VaR - Value at Risk)')
            ending_price = mc_df.iloc[-1:, :].values[0, ]
            fig1, ax = plt.subplots(figsize=(15, 10))
            ax.hist(ending_price, bins=50)
            percentile_5 = np.percentile(ending_price, 5)
            plt.axvline(percentile_5, color='red', linestyle='--', linewidth=1)
            plt.legend([f'Ngưỡng 5%: {np.round(percentile_5, 2):,.0f} VND'])
            plt.title('Phân phối giá dự báo cuối kỳ')
            plt.xlabel('Giá (VND)')
            plt.ylabel('Số lần xuất hiện')
            st.pyplot(fig1, width='stretch')
            
            VaR = last_price - percentile_5
            st.write(f'**VaR (95% tin cậy):** {np.round(VaR, 2):,.0f} VND - Mức lỗ tối đa có thể xảy ra với xác suất 5%')
        
        except Exception as e:
            st.error(f"Lỗi khi chạy mô phỏng Monte Carlo: {e}")

#==============================================================================
# Tab 7 Your Portfolio's Trend
#==============================================================================

@st.cache_data(show_spinner="Đang tải dữ liệu danh mục...")
def get_portfolio_data(tickers):
    """
    Tải dữ liệu đóng cửa cho nhiều mã.
    """
    all_data = yf.download(tickers, period='5y', auto_adjust=False, progress=False)
    
    # Handle both single ticker and multiple tickers
    if isinstance(all_data.columns, pd.MultiIndex):
        # Multiple tickers: extract 'Close' level
        if 'Close' in all_data.columns.get_level_values(0):
            all_data = all_data['Close']
    else:
        # Single ticker: already flat, just select Close if it exists
        if 'Close' in all_data.columns:
            all_data = all_data['Close']
    
    return all_data

def tab7():
    st.title("Danh mục Đầu tư")
    
    alltickers = VN30_TICKERS
    selected_tickers = st.multiselect("Chọn các mã cổ phiếu trong danh mục", options=alltickers, default=['FPT.VN', 'VCB.VN', 'HPG.VN'])
      
    if selected_tickers: 
        try:
            all_data = get_portfolio_data(selected_tickers)
            
            if len(selected_tickers) == 1:
                df = all_data.to_frame(name=selected_tickers[0])
            else:
                df = all_data
            
            if df.empty:
                st.warning("Không thể tải dữ liệu cho các mã đã chọn.")
            else:
                # Chuẩn hóa (Normalize) dữ liệu để so sánh
                normalized_df = (df / df.iloc[0])
                
                st.subheader("So sánh Hiệu suất Đầu tư")
                st.write("Biểu đồ cho thấy tăng trưởng của 1 đồng đầu tư vào mỗi mã (chuẩn hóa về 1.0) trong 5 năm qua.")
                fig = px.line(normalized_df, title="So sánh Tăng trưởng Danh mục (5 năm)")
                st.plotly_chart(fig, width='stretch')
                
                st.subheader("Bảng Giá Lịch sử (VND)")
                st.dataframe(df, width='stretch')

        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu danh mục: {e}")
    else:
        st.info("Vui lòng chọn ít nhất một mã để phân tích.")
    
#==============================================================================
# Main body (Chương trình chính)
#==============================================================================

def run():
    
    st.sidebar.title("Dashboard Phân tích VN30")
    
    ticker_list = ['-'] + VN30_TICKERS
    
    global ticker
    ticker = st.sidebar.selectbox("Chọn mã cổ phiếu", ticker_list)
    
    # Các tab được việt hóa
    tab_options = ['Tổng quan', 'Biểu đồ Kỹ thuật', 'Báo cáo Tài chính', 'Cổ tức & Chia tách', 
                   'Phân tích & Cổ đông', 'Mô phỏng Monte Carlo', "Danh mục Đầu tư"]
    select_tab = st.sidebar.radio("Chọn mục xem", tab_options)
    
    # Logic điều hướng
    if select_tab == 'Tổng quan':
        tab1()
    elif select_tab == 'Biểu đồ Kỹ thuật':
        tab2()
    elif select_tab == 'Báo cáo Tài chính':
        tab4()
    elif select_tab == 'Cổ tức & Chia tách':
        tab3() 
    elif select_tab == 'Phân tích & Cổ đông':
        tab5()
    elif select_tab == 'Mô phỏng Monte Carlo':
        tab6()
    elif select_tab == "Danh mục Đầu tư":
        tab7()
       
    
if __name__ == "__main__":
    run()