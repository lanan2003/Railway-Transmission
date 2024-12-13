import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, stats
from datetime import datetime
import json
import pandas as pd
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import pyqtgraph as pg

# 配置日志记录
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 时域特征提取类
class TimeFeatureExtractor:
    """时域特征提取器"""

    def __init__(self):
        self.window_size = 1000  # 默认窗口大小

    def extract_basic_features(self, data):
        """提取基本统计特征"""
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'max': np.max(data),
            'min': np.min(data),
            'peak_to_peak': np.max(data) - np.min(data),
            'rms': np.sqrt(np.mean(data ** 2)),
            'zero_crossings': len(np.where(np.diff(np.signbit(data)))[0]),
            'dominant_freq': 0.0,  # 默认值
            'power_total': 0.0  # 默认值
        }
        return features

    def extract_shape_features(self, data):
        """提取波形特征"""
        rms = np.sqrt(np.mean(np.square(data)))
        peak = np.max(np.abs(data))
        features = {
            'crest_factor': peak / rms if rms != 0 else 0,
            'form_factor': rms / (np.mean(np.abs(data)) if np.mean(np.abs(data)) != 0 else 1),
            'impulse_factor': peak / (np.mean(np.abs(data)) if np.mean(np.abs(data)) != 0 else 1),
            'kurtosis': stats.kurtosis(data),
            'skewness': stats.skew(data)
        }
        return features

    def extract_correlation_features(self, data):
        """提取相关性特征"""
        n = len(data)
        if n < 2:
            return {}

        features = {
            'zero_crossing_rate': len(np.where(np.diff(np.signbit(data)))[0]) / n,
            'mean_crossing_rate': len(np.where(np.diff(np.signbit(data - np.mean(data))))[0]) / n
        }

        if n > 1 and np.var(data) != 0:
            correlation = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')[n - 1:] / (
                        n * np.var(data))
            features['first_correlation'] = correlation[1] if len(correlation) > 1 else 0

        return features

    def extract_features(self, data):
        """提取所有时域特征"""
        all_features = {}
        all_features.update(self.extract_basic_features(data))
        all_features.update(self.extract_shape_features(data))
        all_features.update(self.extract_correlation_features(data))
        return all_features

# 频域特征提取类
class FrequencyFeatureExtractor:
    """频域特征提取器"""

    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        self.window = signal.windows.hann(1024)  # 汉宁窗

    def compute_fft(self, data):
        """计算FFT"""
        if len(data) <= 1024:
            windowed_data = data * self.window[:len(data)]
        else:
            windowed_data = data[:1024] * self.window  # 若数据长度大于1024，则截断

        fft_result = np.fft.fft(windowed_data)
        freqs = np.fft.fftfreq(len(windowed_data), 1 / self.sampling_rate)

        positive_mask = freqs >= 0
        freqs = freqs[positive_mask]
        fft_result = np.abs(fft_result[positive_mask])

        return freqs, fft_result

    def compute_psd(self, data):
        """计算功率谱密度"""
        freqs, psd = signal.welch(
            data,
            self.sampling_rate,
            window='hann',
            nperseg=min(1024, len(data)),
            scaling='density'
        )
        return freqs, psd

    def extract_spectral_features(self, data):
        """提取频谱特征"""
        freqs, fft_result = self.compute_fft(data)
        psd_freqs, psd = self.compute_psd(data)

        features = {}

        # 主频特征
        if len(fft_result) > 0:
            max_freq_idx = np.argmax(fft_result)
            features['dominant_frequency'] = freqs[max_freq_idx]
            features['dominant_amplitude'] = fft_result[max_freq_idx]
        else:
            features['dominant_frequency'] = 0
            features['dominant_amplitude'] = 0

        # 频谱统计特征
        total_power = np.sum(psd) if len(psd) > 0 else 0
        weighted_freq_sum = np.sum(psd_freqs * psd) if len(psd) > 0 else 0
        features['spectral_centroid'] = weighted_freq_sum / total_power if total_power != 0 else 0

        # 定义频带范围并计算频带能量特征
        bands = [(0, 100), (100, 200), (200, 300)]
        for i, (low, high) in enumerate(bands):
            mask = (psd_freqs >= low) & (psd_freqs < high)
            band_power = np.sum(psd[mask]) if len(psd) > 0 else 0
            features[f'band_{i}_power'] = band_power / total_power if total_power != 0 else 0

        # 频谱形状特征
        if total_power != 0:
            spectral_spread = np.sqrt(np.sum(((psd_freqs - features['spectral_centroid']) ** 2) * psd) / total_power)
            features['spectral_spread'] = spectral_spread
        else:
            features['spectral_spread'] = 0

        # 几何均值计算需要psd中非零项
        nonzero_psd = psd[psd > 0]
        if len(nonzero_psd) > 0:
            spectral_flatness = stats.mstats.gmean(nonzero_psd) / (np.mean(psd) if np.mean(psd) != 0 else 1)
            features['spectral_flatness'] = spectral_flatness
        else:
            features['spectral_flatness'] = 0

        return features

    def extract_features(self, data):
        """提取所有频域特征"""
        return self.extract_spectral_features(data)

# 信号生成器类
class SignalGenerator:
    """信号生成器类，用于生成测试数据"""

    def __init__(self, sampling_rate=1000, duration=1.0):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.time_points = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    def generate_sine_wave(self, frequency, amplitude=1.0, phase=0):
        """生成正弦波信号"""
        return amplitude * np.sin(2 * np.pi * frequency * self.time_points + phase)

    def generate_noise(self, std=0.1):
        """生成高斯噪声"""
        return np.random.normal(0, std, len(self.time_points))

    def generate_test_signal(self, signal_type='good'):
        """生成测试信号"""
        if signal_type == 'good':
            main_signal = self.generate_sine_wave(frequency=100, amplitude=1.0)
            noise = self.generate_noise(std=0.05)
        else:
            main_signal = self.generate_sine_wave(frequency=100, amplitude=0.7)
            noise = self.generate_noise(std=0.3)

        return main_signal + noise

# 数据采集线程类
class DataAcquisitionThread(QThread):
    """数据采集线程类"""
    data_acquired = pyqtSignal(dict)
    acquisition_finished = pyqtSignal()

    def __init__(self, sampling_rate=1000, sampling_duration=1.0, signal_type='good'):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.sampling_duration = sampling_duration
        self.signal_type = signal_type
        self.signal_generator = SignalGenerator(sampling_rate, sampling_duration)
        self.time_extractor = TimeFeatureExtractor()
        self.freq_extractor = FrequencyFeatureExtractor(sampling_rate)
        self.is_running = False
        self.buffer = []

    def remove_noise(self, data, cutoff=100.0):
        """使用低通滤波器去噪"""
        filter_order = 4
        nyq = 0.5 * self.sampling_rate
        cutoff_normalized = cutoff / nyq
        b, a = signal.butter(filter_order, cutoff_normalized, 'low')
        return signal.filtfilt(b, a, data)

    def extract_features(self, data):
        """提取综合特征（包含时域与频域特征）"""
        time_features = self.time_extractor.extract_features(data)
        freq_features = self.freq_extractor.extract_features(data)
        features = {**time_features, **freq_features}
        return features

    def run(self):
        """线程运行方法"""
        self.is_running = True
        try:
            while self.is_running:
                # 生成测试信号
                raw_signal = self.signal_generator.generate_test_signal(self.signal_type)

                # 数据预处理
                preprocessed_signal = self.remove_noise(raw_signal, cutoff=100.0)

                # 特征提取
                features = self.extract_features(preprocessed_signal)

                # 获取当前时间戳
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

                # 准备数据字典
                data_dict = {
                    'timestamp': timestamp,
                    'signal_type': self.signal_type,
                    'raw_data': raw_signal.tolist(),
                    'processed_features': features
                }

                # 存入缓存
                self.buffer.append(data_dict)

                # 发送信号以更新GUI
                self.data_acquired.emit(data_dict)

                # 等待下一个采样周期
                self.sleep_duration = self.sampling_duration
                self.msleep(int(self.sleep_duration * 1000))  # QThread sleep 接受毫秒
        except Exception as e:
            logging.error(f"数据采集过程中出现错误：{e}")
        finally:
            self.is_running = False
            self.acquisition_finished.emit()

    def stop(self):
        """停止数据采集"""
        self.is_running = False
        self.wait()

    def get_buffer(self):
        """获取缓存中的数据"""
        data_to_save = self.buffer.copy()
        self.buffer.clear()
        return data_to_save

# 主窗口类
class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('铁路信号质量监测系统 - 数据采集')
        self.resize(1200, 800)

        # 初始化组件
        self.signal_generator = SignalGenerator()
        self.time_extractor = TimeFeatureExtractor()
        self.freq_extractor = FrequencyFeatureExtractor()
        self.acquisition_thread = None
        self.buffer = []

        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 标题标签
        title_label = QLabel('铁路信号质量监测系统 - 数据采集')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet('font-size: 18pt; font-weight: bold; margin: 10px;')
        main_layout.addWidget(title_label)

        # 配置布局
        config_layout = QHBoxLayout()
        main_layout.addLayout(config_layout)

        # 采样率配置
        config_layout.addWidget(QLabel('采样率 (Hz):'))
        self.input_sampling_rate = QLineEdit('1000')
        config_layout.addWidget(self.input_sampling_rate)

        # 采样时长配置
        config_layout.addWidget(QLabel('采样时长 (秒):'))
        self.input_sampling_duration = QLineEdit('1.0')
        config_layout.addWidget(self.input_sampling_duration)

        # 信号类型选择
        config_layout.addWidget(QLabel('信号类型:'))
        self.combo_signal_type = QComboBox()
        self.combo_signal_type.addItems(['good', 'bad'])
        config_layout.addWidget(self.combo_signal_type)

        # 输出文件路径配置
        config_layout.addWidget(QLabel('输出文件:'))
        self.input_output_file = QLineEdit('collected_data.csv')
        config_layout.addWidget(self.input_output_file)
        btn_browse = QPushButton('浏览')
        btn_browse.clicked.connect(self.browse_file)
        config_layout.addWidget(btn_browse)

        # 按钮布局
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        # 开始采集按钮
        self.btn_start = QPushButton('开始采集')
        self.btn_start.clicked.connect(self.start_acquisition)
        button_layout.addWidget(self.btn_start)

        # 停止采集按钮
        self.btn_stop = QPushButton('停止采集')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_acquisition)
        button_layout.addWidget(self.btn_stop)

        # 导出数据按钮
        self.btn_export = QPushButton('导出数据')
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setEnabled(False)
        button_layout.addWidget(self.btn_export)

        # 添加结果显示标签
        self.result_label = QLabel('结果将在这里显示')
        self.result_label.setStyleSheet('font-size: 11pt; margin: 10px;')
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        # 添加状态显示标签
        self.status_label = QLabel('状态: 未采集')
        self.status_label.setStyleSheet('font-size: 11pt; margin: 10px;')
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # 添加表格显示加载的数据
        self.table = QTableWidget()
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        main_layout.addWidget(self.table)

        # 添加数据可视化按钮
        self.btn_visualize = QPushButton('可视化特征分布')
        self.btn_visualize.clicked.connect(self.visualize_features)
        self.btn_visualize.setStyleSheet('font-size: 12pt; padding: 5px;')
        self.btn_visualize.setEnabled(False)
        main_layout.addWidget(self.btn_visualize)

        # 实时绘图
        plot_layout = QHBoxLayout()
        main_layout.addLayout(plot_layout)

        # 原始信号绘图
        self.plot_raw = pg.PlotWidget(title="原始信号")
        self.plot_raw.setYRange(-2, 2)
        self.plot_raw.setLabel('left', '幅度')
        self.plot_raw.setLabel('bottom', '时间', '秒')
        self.curve_raw = self.plot_raw.plot(pen='b')
        plot_layout.addWidget(self.plot_raw)

        # 处理后信号绘图
        self.plot_processed = pg.PlotWidget(title="处理后信号")
        self.plot_processed.setYRange(-2, 2)
        self.plot_processed.setLabel('left', '幅度')
        self.plot_processed.setLabel('bottom', '时间', '秒')
        self.curve_processed = self.plot_processed.plot(pen='r')
        plot_layout.addWidget(self.plot_processed)

    def browse_file(self):
        """浏览选择输出CSV文件"""
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "选择输出CSV文件",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if filename:
            self.input_output_file.setText(filename)

    def start_acquisition(self):
        """开始数据采集"""
        try:
            sampling_rate = float(self.input_sampling_rate.text())
            sampling_duration = float(self.input_sampling_duration.text())
            signal_type = self.combo_signal_type.currentText()
            output_file = self.input_output_file.text()

            if sampling_rate <= 0 or sampling_duration <= 0:
                raise ValueError("采样率和采样时长必须为正数。")

            # 禁用输入控件和开始按钮
            self.input_sampling_rate.setEnabled(False)
            self.input_sampling_duration.setEnabled(False)
            self.combo_signal_type.setEnabled(False)
            self.input_output_file.setEnabled(False)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_export.setEnabled(False)

            # 更新状态
            self.status_label.setText('状态: 采集中...')

            # 初始化数据采集线程
            self.acquisition_thread = DataAcquisitionThread(
                sampling_rate=sampling_rate,
                sampling_duration=sampling_duration,
                signal_type=signal_type
            )
            self.acquisition_thread.data_acquired.connect(self.update_data)
            self.acquisition_thread.acquisition_finished.connect(self.on_acquisition_finished)
            self.acquisition_thread.start()

        except ValueError as ve:
            QMessageBox.warning(self, "输入错误", str(ve))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"发生错误: {str(e)}")

    def stop_acquisition(self):
        """停止数据采集"""
        if self.acquisition_thread and self.acquisition_thread.isRunning():
            self.acquisition_thread.stop()
            self.btn_stop.setEnabled(False)

    def on_acquisition_finished(self):
        """处理采集结束事件"""
        self.status_label.setText('状态: 采集结束')
        self.btn_export.setEnabled(True)
        # 重新启用输入控件和开始按钮
        self.input_sampling_rate.setEnabled(True)
        self.input_sampling_duration.setEnabled(True)
        self.combo_signal_type.setEnabled(True)
        self.input_output_file.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.information(self, "采集结束", "数据采集已完成。")

    def update_data(self, data):
        """处理采集到的数据并更新GUI"""
        # 存储数据
        self.buffer.append(data)

        # 更新状态标签
        self.status_label.setText(f"最新采集时间: {data['timestamp']}\n信号类型: {data['signal_type']}")

        # 更新绘图
        time_points = np.linspace(0, self.acquisition_thread.sampling_duration, len(data['raw_data']), endpoint=False)
        self.curve_raw.setData(time_points, data['raw_data'])
        self.curve_processed.setData(time_points, self.acquisition_thread.remove_noise(np.array(data['raw_data'])))

    def export_data(self):
        """将缓存中的数据保存到CSV文件"""
        if not self.buffer:
            QMessageBox.information(self, "无数据", "没有数据需要导出。")
            return

        # 检查每个数据字典是否包含 'signal_type'
        for idx, data in enumerate(self.buffer, start=1):
            if 'signal_type' not in data:
                logging.error(f"Data at index {idx} missing 'signal_type': {data}")
                QMessageBox.warning(self, "数据错误", f"数据条目 {idx} 缺少 'signal_type' 字段。")
                return

        output_file = self.input_output_file.text()
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.buffer)

            # 添加日志以确认 'signal_type' 列存在
            logging.info(f"Exporting DataFrame columns before reordering: {df.columns.tolist()}")
            logging.debug(f"Exporting DataFrame head before reordering:\n{df.head()}")

            # 确保 'processed_features' 是字符串格式
            df['processed_features'] = df['processed_features'].apply(json.dumps)

            # 重新排列列顺序，确保只包含需要的列
            columns_order = ['signal_type', 'raw_data', 'processed_features']
            df = df[columns_order]

            # 添加日志以确认列顺序
            logging.info(f"Exporting DataFrame columns after reordering: {df.columns.tolist()}")
            logging.debug(f"Exporting DataFrame head after reordering:\n{df.head()}")

            # 检查文件是否存在
            file_exists = os.path.isfile(output_file)

            # 如果文件存在，追加数据，否则写入新文件
            if file_exists:
                df.to_csv(output_file, mode='a', index=False, header=False)
            else:
                df.to_csv(output_file, mode='w', index=False, header=True)

            # 添加日志以确认导出成功
            logging.info(f"已将 {len(df)} 条数据保存到 {output_file}")

            QMessageBox.information(self, "导出成功", f"已将 {len(df)} 条数据保存到 {output_file}")
            self.buffer.clear()
            self.btn_export.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"保存数据到CSV时出错: {str(e)}")
            logging.error(f"导出数据时出错: {e}")

    def visualize_features(self):
        """可视化特征分布图"""
        if not self.buffer:
            self.result_label.setText("没有采集的数据可供可视化。")
            return

        try:
            # 提取特征
            features = [data['processed_features'] for data in self.buffer]
            features_df = pd.DataFrame(features)

            # 添加信号类型
            signal_types = [data['signal_type'] for data in self.buffer]
            features_df['signal_type'] = signal_types

            # 特征列表（排除 'signal_type'）
            feature_names = list(features_df.columns)
            feature_names.remove('signal_type')

            # 绘制每个特征的分布直方图
            num_features = len(feature_names)
            cols = 3
            rows = num_features // cols + (num_features % cols > 0)

            plt.figure(figsize=(15, 5 * rows))

            for i, feature in enumerate(feature_names, 1):
                plt.subplot(rows, cols, i)
                for label, color in zip(['good', 'bad'], ['blue', 'red']):
                    subset = features_df[features_df['signal_type'] == label]
                    plt.hist(subset[feature], bins=20, alpha=0.5, label=f'{label}', color=color)
                plt.title(f'{feature} 分布')
                plt.xlabel(feature)
                plt.ylabel('频率')
                plt.legend()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.error(f"绘制特征对比图时出错: {e}")
            QMessageBox.warning(self, '可视化失败', f'绘制特征对比图时出错：{str(e)}')

    def closeEvent(self, event):
        """处理窗口关闭事件，确保数据采集线程正确关闭并保存数据"""
        if self.acquisition_thread and self.acquisition_thread.isRunning():
            self.acquisition_thread.stop()

        # 保存数据
        if self.buffer:
            output_file = self.input_output_file.text()
            try:
                # Convert buffer to DataFrame
                df = pd.DataFrame(self.buffer)
                df['id'] = range(1, len(df) + 1)

                # 确保 'processed_features' 是字符串格式
                df['processed_features'] = df['processed_features'].apply(json.dumps)

                # 检查文件是否存在
                file_exists = os.path.isfile(output_file)

                # 如果文件存在，追加数据，否则写入新文件
                if file_exists:
                    df.to_csv(output_file, mode='a', index=False, header=False)
                else:
                    df.to_csv(output_file, mode='w', index=False, header=True)

                print(f"已将 {len(df)} 条数据保存到 {output_file}")
            except Exception as e:
                print(f"保存数据到CSV时出错: {e}")
        event.accept()

# 主函数
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
