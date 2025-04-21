import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

class CryptoVisualizer:
    def __init__(self):
        """
        암호화폐 데이터 시각화 도구
        """
        # 저장 디렉토리 생성
        os.makedirs('reports/figures', exist_ok=True)
    
    def plot_price_history(self, df, price_col='close', date_col=None, title='가격 추이'):
        """
        가격 추이 시각화
        
        Args:
            df (pd.DataFrame): 데이터프레임
            price_col (str): 가격 열 이름
            date_col (str): 날짜 열 이름 (None이면 인덱스 사용)
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        if date_col is None:
            # 인덱스가 날짜인지 확인
            if isinstance(df.index, pd.DatetimeIndex):
                x = df.index
            else:
                x = np.arange(len(df))
        else:
            x = df[date_col]
        
        ax.plot(x, df[price_col], color='blue')
        
        ax.set_title(title)
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격')
        ax.grid(True)
        
        # 그래프 저장
        plt.savefig(f'reports/figures/price_history.png')
        
        return fig
    
    def plot_candlestick(self, df, date_col=None, title='캔들스틱 차트'):
        """
        캔들스틱 차트 시각화 (Plotly 사용)
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임
            date_col (str): 날짜 열 이름 (None이면 인덱스 사용)
            title (str): 그래프 제목
            
        Returns:
            plotly.graph_objects.Figure: 그래프 객체
        """
        if date_col is None:
            # 인덱스가 날짜인지 확인
            if isinstance(df.index, pd.DatetimeIndex):
                x = df.index
            else:
                x = np.arange(len(df))
        else:
            x = df[date_col]
        
        fig = go.Figure(data=[go.Candlestick(
            x=x,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='캔들스틱'
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title='날짜',
            yaxis_title='가격',
            xaxis_rangeslider_visible=False
        )
        
        # HTML 파일로 저장
        os.makedirs('reports/html', exist_ok=True)
        fig.write_html('reports/html/candlestick_chart.html')
        
        return fig
    
    def plot_technical_indicators(self, df, date_col=None, title='기술적 지표'):
        """
        기술적 지표 시각화
        
        Args:
            df (pd.DataFrame): 기술적 지표가 포함된 데이터프레임
            date_col (str): 날짜 열 이름 (None이면 인덱스 사용)
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        if date_col is None:
            # 인덱스가 날짜인지 확인
            if isinstance(df.index, pd.DatetimeIndex):
                x = df.index
            else:
                x = np.arange(len(df))
        else:
            x = df[date_col]
        
        # 필요한 기술적 지표 확인
        indicators = []
        if 'MA7' in df.columns: indicators.append('MA7')
        if 'MA14' in df.columns: indicators.append('MA14')
        if 'MA30' in df.columns: indicators.append('MA30')
        if 'RSI' in df.columns: indicators.append('RSI')
        if 'MACD' in df.columns: indicators.append('MACD')
        
        # 서브플롯 개수 결정
        n_plots = 2  # 가격 + RSI
        if 'MACD' in indicators:
            n_plots += 1
        
        # 그래프 생성
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
        
        # 가격 및 이동평균선 플롯
        axes[0].plot(x, df['close'], label='가격', color='black')
        
        for indicator in indicators:
            if indicator.startswith('MA'):
                axes[0].plot(x, df[indicator], label=indicator)
        
        if 'BB_middle' in df.columns:
            axes[0].plot(x, df['BB_middle'], label='BB Middle', color='blue', linestyle='--')
            axes[0].plot(x, df['BB_upper'], label='BB Upper', color='red', linestyle='--')
            axes[0].plot(x, df['BB_lower'], label='BB Lower', color='green', linestyle='--')
            axes[0].fill_between(x, df['BB_upper'], df['BB_lower'], alpha=0.1, color='gray')
        
        axes[0].set_title('가격 및 이동평균선')
        axes[0].set_ylabel('가격')
        axes[0].legend()
        axes[0].grid(True)
        
        # RSI 플롯
        if 'RSI' in indicators:
            axes[1].plot(x, df['RSI'], label='RSI', color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--')
            axes[1].axhline(y=30, color='g', linestyle='--')
            axes[1].set_title('RSI (Relative Strength Index)')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True)
        
        # MACD 플롯
        if 'MACD' in indicators:
            idx = 2
            axes[idx].plot(x, df['MACD'], label='MACD', color='blue')
            axes[idx].plot(x, df['MACD_signal'], label='Signal', color='red')
            axes[idx].bar(x, df['MACD_hist'], label='Histogram', color='green', alpha=0.5)
            axes[idx].set_title('MACD (Moving Average Convergence Divergence)')
            axes[idx].set_ylabel('MACD')
            axes[idx].legend()
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        # 그래프 저장
        plt.savefig(f'reports/figures/technical_indicators.png')
        
        return fig
    
    def plot_correlation_matrix(self, df, title='특성 상관관계'):
        """
        특성 간 상관관계 시각화
        
        Args:
            df (pd.DataFrame): 데이터프레임
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        # 수치형 열만 선택
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 상관관계 계산
        corr = numeric_df.corr()
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 히트맵 플롯
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        
        ax.set_title(title)
        
        # 그래프 저장
        plt.savefig(f'reports/figures/correlation_matrix.png')
        
        return fig
    
    def plot_volume_analysis(self, df, title='거래량 분석'):
        """
        거래량 분석 시각화
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 인덱스가 날짜인지 확인
        if isinstance(df.index, pd.DatetimeIndex):
            x = df.index
        else:
            x = np.arange(len(df))
        
        # 가격 플롯
        axes[0].plot(x, df['close'], label='가격', color='blue')
        axes[0].set_title('가격 추이')
        axes[0].set_ylabel('가격')
        axes[0].legend()
        axes[0].grid(True)
        
        # 거래량 플롯
        axes[1].bar(x, df['volume'], label='거래량', color='green', alpha=0.7)
        
        # 이동평균 거래량 추가
        if len(df) > 20:
            volume_ma = df['volume'].rolling(window=20).mean()
            axes[1].plot(x, volume_ma, label='20일 이동평균 거래량', color='red')
        
        axes[1].set_title('거래량 추이')
        axes[1].set_ylabel('거래량')
        axes[1].set_xlabel('날짜')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        # 그래프 저장
        plt.savefig(f'reports/figures/volume_analysis.png')
        
        return fig
    
    def plot_returns_distribution(self, df, title='수익률 분포'):
        """
        수익률 분포 시각화
        
        Args:
            df (pd.DataFrame): 가격 데이터프레임
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        # 일일 수익률 계산
        returns = df['close'].pct_change().dropna()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 수익률 시계열 플롯
        axes[0].plot(returns.index if isinstance(returns.index, pd.DatetimeIndex) else np.arange(len(returns)), 
                    returns, color='blue')
        axes[0].set_title('일일 수익률 추이')
        axes[0].set_ylabel('수익률')
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].grid(True)
        
        # 수익률 분포 히스토그램
        sns.histplot(returns, kde=True, ax=axes[1], color='green')
        
        # 정규분포 추가
        from scipy import stats
        x = np.linspace(returns.min(), returns.max(), 100)
        axes[1].plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 
                    'r-', label='정규분포')
        
        axes[1].set_title('일일 수익률 분포')
        axes[1].set_xlabel('수익률')
        axes[1].set_ylabel('빈도')
        axes[1].legend()
        axes[1].grid(True)
        
        # 통계 정보 추가
        stats_text = (f"평균: {returns.mean():.4f}\n"
                    f"표준편차: {returns.std():.4f}\n"
                    f"왜도: {returns.skew():.4f}\n"
                    f"첨도: {returns.kurtosis():.4f}")
        
        axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        # 그래프 저장
        plt.savefig(f'reports/figures/returns_distribution.png')
        
        return fig
    
    def plot_volatility_analysis(self, df, window=20, title='변동성 분석'):
        """
        변동성 분석 시각화
        
        Args:
            df (pd.DataFrame): 가격 데이터프레임
            window (int): 이동 표준편차 윈도우 크기
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        # 일일 수익률 계산
        returns = df['close'].pct_change().dropna()
        
        # 변동성 계산 (이동 표준편차)
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # 연간화
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 가격 플롯
        axes[0].plot(df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df)), 
                    df['close'], label='가격', color='blue')
        axes[0].set_title('가격 추이')
        axes[0].set_ylabel('가격')
        axes[0].legend()
        axes[0].grid(True)
        
        # 변동성 플롯
        axes[1].plot(volatility.index if isinstance(volatility.index, pd.DatetimeIndex) else np.arange(len(volatility)), 
                    volatility, label=f'{window}일 이동 변동성', color='red')
        axes[1].set_title('변동성 추이')
        axes[1].set_ylabel('연간화 변동성')
        axes[1].set_xlabel('날짜')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        # 그래프 저장
        plt.savefig(f'reports/figures/volatility_analysis.png')
        
        return fig
    
    def plot_interactive_dashboard(self, df, output_file='crypto_dashboard.html'):
        """
        인터랙티브 대시보드 생성 (Plotly 사용)
        
        Args:
            df (pd.DataFrame): OHLCV 데이터프레임
            output_file (str): 출력 HTML 파일 이름
            
        Returns:
            plotly.graph_objects.Figure: 대시보드 객체
        """
        # 서브플롯 생성 (3행 1열)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('가격 차트', '거래량', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # 캔들스틱 차트 추가
        fig.add_trace(
            go.Candlestick(
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df)),
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='가격'
            ),
            row=1, col=1
        )
        
        # 이동평균선 추가
        if 'MA7' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df)),
                    y=df['MA7'],
                    name='MA7',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        if 'MA30' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df)),
                    y=df['MA30'],
                    name='MA30',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
        
        # 볼린저 밴드 추가
        if 'BB_upper' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df)),
                    y=df['BB_upper'],
                    name='BB Upper',
                    line=dict(color='rgba(255, 0, 0, 0.5)', dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df)),
                    y=df['BB_middle'],
                    name='BB Middle',
                    line=dict(color='rgba(0, 0, 255, 0.5)', dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df)),
                    y=df['BB_lower'],
                    name='BB Lower',
                    line=dict(color='rgba(0, 255, 0, 0.5)', dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.2)'
                ),
                row=1, col=1
            )
        
        # 거래량 추가
        fig.add_trace(
            go.Bar(
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df)),
                y=df['volume'],
                name='거래량',
                marker=dict(color='green')
            ),
            row=2, col=1
        )
        
        # RSI 추가
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df)),
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
            
            # RSI 기준선 추가
            fig.add_shape(
                type="line",
                x0=df.index[0] if isinstance(df.index, pd.DatetimeIndex) else 0,
                y0=70,
                x1=df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else len(df)-1,
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
                row=3, col=1
            )
            
            fig.add_shape(
                type="line",
                x0=df.index[0] if isinstance(df.index, pd.DatetimeIndex) else 0,
                y0=30,
                x1=df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else len(df)-1,
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
                row=3, col=1
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title='암호화폐 기술적 분석 대시보드',
            xaxis_title='날짜',
            yaxis_title='가격',
            xaxis_rangeslider_visible=False,
            height=900,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Y축 범위 설정
        if 'RSI' in df.columns:
            fig.update_yaxes(range=[0, 100], row=3, col=1)
        
        # HTML 파일로 저장
        os.makedirs('reports/html', exist_ok=True)
        fig.write_html(f'reports/html/{output_file}')
        
        return fig
    
    def plot_seasonal_analysis(self, df, title='계절성 분석'):
        """
        계절성 분석 시각화
        
        Args:
            df (pd.DataFrame): 가격 데이터프레임 (인덱스가 날짜여야 함)
            title (str): 그래프 제목
            
        Returns:
            matplotlib.figure.Figure: 그래프 객체
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("데이터프레임의 인덱스가 날짜형식이어야 합니다.")
        
        # 연도, 월, 요일 추출
        df_copy = df.copy()
        df_copy['year'] = df_copy.index.year
        df_copy['month'] = df_copy.index.month
        df_copy['day_of_week'] = df_copy.index.dayofweek
        
        # 일일 수익률 계산
        df_copy['returns'] = df_copy['close'].pct_change()
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 월별 평균 수익률
        monthly_returns = df_copy.groupby('month')['returns'].mean() * 100  # 퍼센트로 변환
        axes[0, 0].bar(monthly_returns.index, monthly_returns.values, color='blue')
        axes[0, 0].set_title('월별 평균 수익률')
        axes[0, 0].set_xlabel('월')
        axes[0, 0].set_ylabel('평균 일일 수익률 (%)')
        axes[0, 0].set_xticks(range(1, 13))
        axes[0, 0].grid(True, axis='y')
        
        # 요일별 평균 수익률
        day_names = ['월', '화', '수', '목', '금', '토', '일']
        daily_returns = df_copy.groupby('day_of_week')['returns'].mean() * 100  # 퍼센트로 변환
        axes[0, 1].bar(daily_returns.index, daily_returns.values, color='green')
        axes[0, 1].set_title('요일별 평균 수익률')
        axes[0, 1].set_xlabel('요일')
        axes[0, 1].set_ylabel('평균 일일 수익률 (%)')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(day_names)
        axes[0, 1].grid(True, axis='y')
        
        # 연도별 누적 수익률
        yearly_returns = df_copy.groupby('year')['returns'].apply(lambda x: (1 + x).cumprod().iloc[-1] - 1) * 100
        axes[1, 0].bar(yearly_returns.index, yearly_returns.values, color='purple')
        axes[1, 0].set_title('연도별 누적 수익률')
        axes[1, 0].set_xlabel('연도')
        axes[1, 0].set_ylabel('누적 수익률 (%)')
        axes[1, 0].grid(True, axis='y')
        
        # 월별 거래량
        monthly_volume = df_copy.groupby('month')['volume'].mean()
        axes[1, 1].bar(monthly_volume.index, monthly_volume.values, color='orange')
        axes[1, 1].set_title('월별 평균 거래량')
        axes[1, 1].set_xlabel('월')
        axes[1, 1].set_ylabel('평균 거래량')
        axes[1, 1].set_xticks(range(1, 13))
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        # 그래프 저장
        plt.savefig(f'reports/figures/seasonal_analysis.png')
        
        return fig

# 사용 예시
if __name__ == "__main__":
    # 가상의 데이터 생성 (실제로는 전처리된 데이터를 로드해야 함)
    dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'open': np.random.normal(30000, 1000, 365),
        'high': np.random.normal(30500, 1000, 365),
        'low': np.random.normal(29500, 1000, 365),
        'close': np.random.normal(30000, 1000, 365),
        'volume': np.random.normal(1000000, 200000, 365)
    }, index=dates)
    
    # 데이터 정리 (high는 항상 open, close, low보다 크게, low는 항상 작게)
    for i in range(len(df)):
        values = [df.iloc[i]['open'], df.iloc[i]['close']]
        df.iloc[i, df.columns.get_loc('high')] = max(values) + abs(np.random.normal(500, 100))
        df.iloc[i, df.columns.get_loc('low')] = min(values) - abs(np.random.normal(500, 100))
        df.iloc[i, df.columns.get_loc('volume')] = abs(df.iloc[i]['volume'])
    
    # 기술적 지표 추가
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA14'] = df['close'].rolling(window=14).mean()
    df['MA30'] = df['close'].rolling(window=30).mean()
    
    # RSI 계산
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    # 결측치 제거
    df.dropna(inplace=True)
    
    # 시각화 객체 생성
    visualizer = CryptoVisualizer()
    
    # 다양한 시각화 수행
    visualizer.plot_price_history(df)
    visualizer.plot_candlestick(df)
    visualizer.plot_technical_indicators(df)
    visualizer.plot_correlation_matrix(df)
    visualizer.plot_volume_analysis(df)
    visualizer.plot_returns_distribution(df)
    visualizer.plot_volatility_analysis(df)
    visualizer.plot_interactive_dashboard(df)
    visualizer.plot_seasonal_analysis(df)
    
    print("모든 시각화가 완료되었습니다.") 