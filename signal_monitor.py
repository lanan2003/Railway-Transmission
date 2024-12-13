import sys
import json
import logging
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QComboBox)
import sqlite3
import pywt
import zlib
import base64

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DatabaseManager:
    """数据库管理类，用于处理SQLite数据库的操作"""

    def __init__(self, db_path='signals.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()

    def create_tables(self):
        """创建必要的数据库表"""
        cursor = self.conn.cursor()
        # 创建signals表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type TEXT NOT NULL,
                raw_data TEXT NOT NULL,
                compressed_data BLOB,
                encoded_data TEXT
            )
        ''')
        # 创建features表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                mean REAL,
                std REAL,
                max REAL,
                min REAL,
                peak_to_peak REAL,
                rms REAL,
                zero_crossings INTEGER,
                dominant_freq REAL,
                power_total REAL,
                FOREIGN KEY(signal_id) REFERENCES signals(id)
            )
        ''')
        self.conn.commit()

    def insert_signal(self, signal_type, raw_data, compressed_data=None, encoded_data=None):
        """插入信号数据，并返回signal_id"""
        cursor = self.conn.cursor()
        
        # 确保 raw_data 是列表而不是 numpy 数组
        if isinstance(raw_data, np.ndarray):
            raw_data = raw_data.tolist()
        
        cursor.execute('''
            INSERT INTO signals (signal_type, raw_data, compressed_data, encoded_data)
            VALUES (?, ?, ?, ?)
        ''', (
            signal_type,
            json.dumps(raw_data),  # raw_data 现在确保是列表
            compressed_data,
            encoded_data
        ))
        self.conn.commit()
        return cursor.lastrowid

    def insert_feature(self, signal_id, features):
        """插入特征数据"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO features (
                signal_id, mean, std, max, min, peak_to_peak,
                rms, zero_crossings, dominant_freq, power_total
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_id,
            features['mean'],
            features['std'],
            features['max'],
            features['min'],
            features['peak_to_peak'],
            features['rms'],
            features['zero_crossings'],
            features['dominant_freq'],
            features['power_total']
        ))
        self.conn.commit()

    def fetch_all_signals_and_features(self):
        """获取所有信号及其对应的特征"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT s.id, s.signal_type, s.raw_data, s.compressed_data, s.encoded_data,
                   f.mean, f.std, f.max, f.min, f.peak_to_peak,
                   f.rms, f.zero_crossings, f.dominant_freq, f.power_total
            FROM signals s
            JOIN features f ON s.id = f.signal_id
        ''')
        rows = cursor.fetchall()
        data = []
        for row in rows:
            (signal_id, signal_type, raw_data, compressed_data, encoded_data,
             mean, std, max_, min_, peak_to_peak, rms, zero_crossings, dominant_freq, power_total) = row
            data.append({
                'signal_id': signal_id,
                'signal_type': signal_type,
                'raw_data': json.loads(raw_data),
                'compressed_data': compressed_data,
                'encoded_data': encoded_data,
                'processed_features': {
                    'mean': mean,
                    'std': std,
                    'max': max_,
                    'min': min_,
                    'peak_to_peak': peak_to_peak,
                    'rms': rms,
                    'zero_crossings': zero_crossings,
                    'dominant_freq': dominant_freq,
                    'power_total': power_total
                }
            })
        return pd.DataFrame(data)

    def update_signal_compression(self, signal_id, compressed_data, encoded_data):
        """更新指定信号的压缩和编码数据"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE signals
            SET compressed_data = ?, encoded_data = ?
            WHERE id = ?
        ''', (
            compressed_data,
            encoded_data,
            signal_id
        ))
        self.conn.commit()

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

    def clear_database(self):
        """清空数据库中的所有数据"""
        cursor = self.conn.cursor()
        try:
            # 先删除features表中的所有数据（因为有外键约束）
            cursor.execute('DELETE FROM features')
            # 再删除signals表中的所有数据
            cursor.execute('DELETE FROM signals')
            # 重置自增ID
            cursor.execute('DELETE FROM sqlite_sequence WHERE name="signals" OR name="features"')
            self.conn.commit()
            logging.info("数据库已清空")
        except Exception as e:
            logging.error(f"清空数据库时出错: {e}")
            raise


class SignalOptimizer:
    """信号优化器类，用于压缩和编码信号数据"""

    def __init__(self, compression_level=6):
        """
        参数：
            compression_level: 压缩级别（0-9），越高压缩率越高，但压缩和解压缩时间越长
        """
        self.compression_level = compression_level
        self.signal_generator = None  # 在 MainWindow 中设置

    def compress_signal(self, data, threshold_ratio=0.1):
        """
        使用离散小波变换（DWT）进行信号压缩，并使用zlib进行压缩
        参数：
            data: 原始信号数据（numpy数组）
            threshold_ratio: 保留系数的比例（0-1），较低的值表示更多系数被设为零
        返回：
            压缩后的信号数据（bytes）
        """
        try:
            # 进行小波变换，选择适当的小波函数和分解层数
            coeffs = pywt.wavedec(data, 'db4', level=4)

            # 计算阈值
            all_coeffs = np.concatenate([c.flatten() for c in coeffs])
            threshold = np.percentile(np.abs(all_coeffs), 100 * (1 - threshold_ratio))

            # 应用阈值，将小于阈值的系数设为零
            coeffs_thresholded = [coeffs[0]]  # 近似系数通常保留
            for detail in coeffs[1:]:
                detail_thresh = np.where(np.abs(detail) >= threshold, detail, 0)
                coeffs_thresholded.append(detail_thresh.tolist())

            # 将阈值后的系数序列化为字节
            coeffs_bytes = json.dumps(coeffs_thresholded).encode('utf-8')

            # 使用zlib进行压缩
            compressed_data = zlib.compress(coeffs_bytes, level=self.compression_level)
            return compressed_data
        except Exception as e:
            logging.error(f"压缩信号时出错: {e}")
            raise

    def decompress_signal(self, compressed_data):
        """
        解压缩信号数据
        参数：
            compressed_data: 压缩后的信号数据（bytes）
        返回：
            解压后的信号数据（numpy数组）
        """
        try:
            # 解压缩
            coeffs_reduced = json.loads(zlib.decompress(compressed_data).decode('utf-8'))

            # 将每个系数从列表转换为numpy数组
            coeffs_reduced = [np.array(c) for c in coeffs_reduced]

            # 进行小波重构，使用合适的mode参数
            data_reconstructed = pywt.waverec(coeffs_reduced, 'db4', mode='periodization')

            # 确认重构后的信号长度
            if self.signal_generator:
                original_length = len(self.signal_generator.time_points)
                reconstructed_length = len(data_reconstructed)
                logging.debug(f"原始信号长度: {original_length}, 重构信号长度: {reconstructed_length}")

                # 如果长度不一致，进行调整
                if reconstructed_length > original_length:
                    data_reconstructed = data_reconstructed[:original_length]
                elif reconstructed_length < original_length:
                    data_reconstructed = np.pad(data_reconstructed, (0, original_length - reconstructed_length),
                                                'constant')
            return data_reconstructed
        except Exception as e:
            logging.error(f"解压缩信号时出错: {e}")
            raise

    def encode_signal(self, compressed_data):
        """
        对压缩后的信号数据进行编码（Base64编码）
        参数：
            compressed_data: 压缩后的信号数据（bytes）
        返回：
            编码后的信号数据（字符串）
        """
        try:
            encoded_data = base64.b64encode(compressed_data).decode('utf-8')
            return encoded_data
        except Exception as e:
            logging.error(f"编码信号时出错: {e}")
            raise

    def decode_signal(self, encoded_data):
        """
        解码信号数据（Base64解码）
        参数：
            encoded_data: 编码后的信号数据（字符串）
        返回：
            压缩后的信号数据（bytes）
        """
        try:
            compressed_data = base64.b64decode(encoded_data.encode('utf-8'))
            return compressed_data
        except Exception as e:
            logging.error(f"解码信号时出错: {e}")
            raise


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
        """生成测试信号

        参数：
            signal_type: 'good' 或 'bad'，表示信号质量
        """
        if signal_type == 'good':
            # 生成高质量信号
            main_signal = self.generate_sine_wave(frequency=100, amplitude=1.0)
            noise = self.generate_noise(std=0.05)
        else:
            # 生成低质量信号
            main_signal = self.generate_sine_wave(frequency=100, amplitude=0.7)
            noise = self.generate_noise(std=0.3)

        return main_signal + noise


class SignalProcessor:
    """信号处理器类"""

    def __init__(self):
        self.filter_order = 4
        self.cutoff_freq = 100.0  # 低通滤波器截止频率（Hz）

    def remove_noise(self, data, fs=1000.0, cutoff=100.0):
        """使用低通滤波器去噪"""
        nyq = 0.5 * fs
        cutoff_normalized = cutoff / nyq
        b, a = signal.butter(self.filter_order, cutoff_normalized, 'low')
        return signal.filtfilt(b, a, data)

    def extract_features(self, data):
        """提取信号特征"""
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'max': np.max(data),
            'min': np.min(data),
            'peak_to_peak': np.max(data) - np.min(data),
            'rms': np.sqrt(np.mean(data ** 2)),
            'zero_crossings': len(np.where(np.diff(np.signbit(data)))[0]),
            'dominant_freq': 0.0,  # 默认值
            'power_total': 0.0  # 默��值
        }

        try:
            # 添加频域特征
            freqs, psd = signal.welch(data, fs=1000, nperseg=len(data) // 2)
            features['dominant_freq'] = freqs[np.argmax(psd)]
            features['power_total'] = np.sum(psd)
        except Exception as e:
            logging.warning(f"提取频域特征时出错: {e}")

        return features


class SignalClassifier:
    """信号分类器类"""

    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['mean', 'std', 'max', 'min', 'peak_to_peak', 'rms', 'zero_crossings', 'dominant_freq',
                              'power_total']

    def prepare_features(self, features_dict_list):
        """准备特征数据"""
        try:
            X = np.array([[d[name] for name in self.feature_names] for d in features_dict_list])
            return X
        except KeyError as e:
            missing_feature = e.args[0]
            logging.error(f"缺少特征: {missing_feature}")
            raise ValueError(f"缺少特征: {missing_feature}")

    def train(self, X, y):
        """训练模型"""
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logging.info(f"模型训练完成。准确率: {accuracy:.2f}")
        logging.info(f"分类报告:\n{report}")

        return accuracy, report

    def predict(self, X):
        """预测信号质量"""
        if not self.is_trained:
            raise ValueError("模型尚未训练。请先生成并分析信号。")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('铁路信号质量监测系统')
        self.resize(1200, 800)

        # 初始化组件
        self.signal_generator = SignalGenerator()
        self.signal_processor = SignalProcessor()
        self.classifier = SignalClassifier()
        self.db_manager = DatabaseManager()  # 初始化数据库管理器
        self.optimizer = SignalOptimizer(compression_level=6)  # 初始化信号优化器
        self.optimizer.signal_generator = self.signal_generator  # 设置 signal_generator

        self.loaded_df = None  # 用于存储加载的数据

        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 添加标题标签
        title_label = QLabel('铁路信号质量监测系统')
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

        # 生成并分析信号按钮
        self.btn_generate = QPushButton('生成并分析信号')
        self.btn_generate.clicked.connect(self.generate_and_analyze)
        self.btn_generate.setStyleSheet('font-size: 12pt; padding: 5px;')
        button_layout.addWidget(self.btn_generate)

        # 加载并优化信号按钮
        self.btn_load_optimize = QPushButton('加载并优化信号')
        self.btn_load_optimize.clicked.connect(self.load_and_optimize_signals)
        self.btn_load_optimize.setStyleSheet('font-size: 12pt; padding: 5px;')
        button_layout.addWidget(self.btn_load_optimize)

        # 导出分析结果按钮
        self.btn_export = QPushButton('导出分析结果')
        self.btn_export.clicked.connect(self.export_analysis)
        self.btn_export.setStyleSheet('font-size: 12pt; padding: 5px;')
        self.btn_export.setEnabled(False)
        button_layout.addWidget(self.btn_export)

        # 添加结果显示标签
        self.result_label = QLabel('结果将在这里显示')
        self.result_label.setStyleSheet('font-size: 11pt; margin: 10px;')
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

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

        # 添加按钮以从数据库加载数据
        db_button_layout = QHBoxLayout()
        main_layout.addLayout(db_button_layout)

        self.btn_load_db = QPushButton('从数据库加载数据')
        self.btn_load_db.clicked.connect(self.load_data_from_db)
        self.btn_load_db.setStyleSheet('font-size: 12pt; padding: 5px;')
        db_button_layout.addWidget(self.btn_load_db)

        self.btn_clear_db = QPushButton('清空数据库')
        self.btn_clear_db.clicked.connect(self.clear_data)
        self.btn_clear_db.setStyleSheet('font-size: 12pt; padding: 5px;')
        db_button_layout.addWidget(self.btn_clear_db)

        self.btn_clear_table = QPushButton('清空列表')
        self.btn_clear_table.clicked.connect(self.clear_table)
        self.btn_clear_table.setStyleSheet('font-size: 12pt; padding: 5px;')
        db_button_layout.addWidget(self.btn_clear_table)

    def browse_file(self):
        """浏览选择输出CSV文件（用于导出分析结果）"""
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

    def generate_and_analyze(self):
        """生成并分析信号"""
        try:
            # 获取用户输入的采样参数
            sampling_rate_text = self.input_sampling_rate.text()
            sampling_duration_text = self.input_sampling_duration.text()
            signal_type = self.combo_signal_type.currentText()
            output_file = self.input_output_file.text()

            # 验证输入
            try:
                sampling_rate = float(sampling_rate_text)
                sampling_duration = float(sampling_duration_text)
            except ValueError:
                raise ValueError("采样率和采样时长必须为数字。")

            if sampling_rate <= 0 or sampling_duration <= 0:
                raise ValueError("采样率和采样时长必须为正数。")

            # 更新信号生成器的采样参数
            self.signal_generator = SignalGenerator(sampling_rate=sampling_rate, duration=sampling_duration)
            self.signal_processor = SignalProcessor()

            # 生成测试数据
            num_signals = 50  # 生成的信号数量
            logging.info("开始生成good信号")
            good_signals = [self.signal_generator.generate_test_signal('good') for _ in range(num_signals)]
            logging.info("完成生成good信号")

            logging.info("开始生成bad信号")
            bad_signals = [self.signal_generator.generate_test_signal('bad') for _ in range(num_signals)]
            logging.info("完成生成bad信号")

            # 提取特征
            logging.info("开始提取good信号特征")
            good_features = [self.signal_processor.extract_features(sig) for sig in good_signals]
            logging.info("完成提取good信号特征")

            logging.info("开始提取bad信号特征")
            bad_features = [self.signal_processor.extract_features(sig) for sig in bad_signals]
            logging.info("完成提取bad信号特征")

            # 保存信号和特征到数据库（仅保存原始信号）
            logging.info("开始保存信号和特征到数据库")
            for i in range(num_signals):
                # 保存good信号
                signal_id = self.db_manager.insert_signal(
                    'good',
                    good_signals[i].tolist()  # 转换为列表
                )
                self.db_manager.insert_feature(signal_id, good_features[i])

                # 保存bad信号
                signal_id = self.db_manager.insert_signal(
                    'bad',
                    bad_signals[i].tolist()  # 转换为列表
                )
                self.db_manager.insert_feature(signal_id, bad_features[i])
            logging.info("完成保存信号和特征到数据库")

            # 准备训练数据
            combined_features = good_features + bad_features
            X = self.classifier.prepare_features(combined_features)
            y = ['good'] * num_signals + ['bad'] * num_signals

            # 训练模型
            logging.info("开始训练分类器")
            accuracy, report = self.classifier.train(X, y)
            logging.info("完成训练分类器")

            # 显示结果
            self.result_label.setText(f'模型准确率: {accuracy:.2f}\n\n{report}')

            # 保存数据到CSV（可选）
            self.save_to_csv(output_file, good_signals, bad_signals, good_features, bad_features)

            # 绘制原始信号
            self.plot_signals(good_signals[0], bad_signals[0])

            QMessageBox.information(self, '成功', '信号生成、分析和模型训练完成！')

        except ValueError as ve:
            QMessageBox.warning(self, '输入错误', str(ve))
        except Exception as e:
            logging.error(f"处理过程中出现错误: {e}")
            QMessageBox.warning(self, '错误', f'处理过程中出现错误：{str(e)}')

    def save_to_csv(self, filename, good_signals, bad_signals, good_features, bad_features):
        """将生成的信号和特征保存到CSV文件"""
        try:
            data = []
            for sig, feat in zip(good_signals, good_features):
                data.append({
                    'signal_type': 'good',
                    'raw_data': json.dumps(sig.tolist()),
                    'processed_features': json.dumps(feat),
                    'compressed_data': None,  # 优化后数据不保存
                    'encoded_data': None  # 优化后数据不保存
                })
            for sig, feat in zip(bad_signals, bad_features):
                data.append({
                    'signal_type': 'bad',
                    'raw_data': json.dumps(sig.tolist()),
                    'processed_features': json.dumps(feat),
                    'compressed_data': None,  # 优化后数据不保存
                    'encoded_data': None  # 优化后数据不保存
                })

            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"已将 {len(df)} 条数据保存到 {filename}")
        except Exception as e:
            logging.error(f"保存数据到CSV时出错: {e}")
            QMessageBox.warning(self, '保存失败', f'保存数据到CSV时出错：{str(e)}')

    def plot_signals(self, good_signal, bad_signal):
        """绘制原始信号对比图"""
        try:
            plt.figure(figsize=(14, 8))

            # 绘制高质量信号
            plt.subplot(2, 2, 1)
            plt.plot(self.signal_generator.time_points, good_signal, 'b-')
            plt.title('原始高质量信号')
            plt.xlabel('时间 (s)')
            plt.ylabel('幅度')
            plt.grid(True)

            # 绘制低质量信号
            plt.subplot(2, 2, 3)
            plt.plot(self.signal_generator.time_points, bad_signal, 'r-')
            plt.title('原始低质量信号')
            plt.xlabel('时间 (s)')
            plt.ylabel('幅度')
            plt.grid(True)

            plt.tight_layout()
            plt.show()
        except ValueError as ve:
            logging.error(f"绘制原始信号时出错: {ve}")
            QMessageBox.warning(self, '绘图失败', f'绘制原始信号时出错：{str(ve)}')
        except Exception as e:
            logging.error(f"绘制原始信号时出错: {e}")
            QMessageBox.warning(self, '绘图失败', f'绘制原始信号时出错：{str(e)}')

    def load_and_optimize_signals(self):
        """加载信号数据并进行优化"""
        try:
            # 选择数据来源
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(
                self,
                '选择数据文件',
                '',
                'CSV files (*.csv);;All Files (*)',
                options=options
            )

            if filename:
                df = pd.read_csv(filename)

                # 检查必要的列是否存在
                required_columns = ['signal_type', 'raw_data', 'processed_features']
                for col in required_columns:
                    if col not in df.columns:
                        raise ValueError(f"缺少必要的列：{col}")

                # 解析JSON字段
                try:
                    # 移除raw_data中可能的额外引号
                    df['raw_data'] = df['raw_data'].apply(lambda x: json.loads(x.replace("'", '"') if isinstance(x, str) else x))
                    df['processed_features'] = df['processed_features'].apply(lambda x: json.loads(x.replace("'", '"') if isinstance(x, str) else x))
                except json.JSONDecodeError as e:
                    logging.error(f"JSON解析错误: {e}")
                    raise ValueError(f"数据格式错误，无法解析JSON: {e}")

                # 优化信号并更新数据库
                logging.info("开始优化信号")
                for index, row in df.iterrows():
                    try:
                        signal_type = row['signal_type']
                        
                        # 确保raw_data是数值列表
                        if isinstance(row['raw_data'], str):
                            raw_data = json.loads(row['raw_data'])
                        else:
                            raw_data = row['raw_data']
                        
                        # 转换为numpy数组
                        raw_signal = np.array(raw_data, dtype=float)

                        # 压缩信号
                        compressed_signal = self.optimizer.compress_signal(raw_signal)

                        # 编码信号
                        encoded_signal = self.optimizer.encode_signal(compressed_signal)

                        # 插入优化后的信号到数据库
                        signal_id = self.db_manager.insert_signal(
                            signal_type,
                            raw_signal,  # insert_signal方法会处理转换
                            compressed_data=compressed_signal,
                            encoded_data=encoded_signal
                        )

                        # 处理特征数据
                        processed_features = {}
                        for k, v in row['processed_features'].items():
                            if isinstance(v, (np.number, np.ndarray)):
                                processed_features[k] = float(v)
                            else:
                                processed_features[k] = v

                        # 插入特征到数据库
                        self.db_manager.insert_feature(signal_id, processed_features)
                    except Exception as e:
                        logging.error(f"处理第{index}行数据时出错: {e}")
                        continue

                logging.info("完成优化信号并保存到数据库")

                # 加载优化后的信号进行展示
                optimized_df = self.db_manager.fetch_all_signals_and_features()
                self.loaded_df = optimized_df

                # 显示数据到表格
                self.display_table(optimized_df)

                # 启用导出和可视化按钮
                self.btn_export.setEnabled(True)
                self.btn_visualize.setEnabled(True)

                QMessageBox.information(self, '成功', '信号加载、优化和展示完成！')

        except ValueError as ve:
            QMessageBox.warning(self, '输入错误', str(ve))
        except Exception as e:
            logging.error(f"加载并优化信号过程中出现错误: {e}")
            QMessageBox.warning(self, '错误', f'加载并优化信号过程中出错：{str(e)}')

    def plot_optimization_comparison(self, raw_signal, optimized_signal):
        """绘制优化前后的信号对比图"""
        try:
            plt.figure(figsize=(14, 8))

            # 绘制优化前信号
            plt.subplot(2, 1, 1)
            plt.plot(self.signal_generator.time_points, raw_signal, 'b-')
            plt.title('优化前信号')
            plt.xlabel('时间 (s)')
            plt.ylabel('幅度')
            plt.grid(True)

            # 绘制优化后信号
            plt.subplot(2, 1, 2)
            plt.plot(self.signal_generator.time_points, optimized_signal, 'g-')
            plt.title('优化后信号')
            plt.xlabel('时间 (s)')
            plt.ylabel('幅度')
            plt.grid(True)

            plt.tight_layout()
            plt.show()
        except ValueError as ve:
            logging.error(f"绘制优化信号对比时出错: {ve}")
            QMessageBox.warning(self, '绘图失败', f'绘制优化信号对比时出错：{str(ve)}')
        except Exception as e:
            logging.error(f"绘制优化信号对比时出错: {e}")
            QMessageBox.warning(self, '绘图失败', f'绘制优化信号对比时出错：{str(e)}')

    def load_data_from_db(self):
        """从数据库加载数据并进行分析"""
        try:
            df = self.db_manager.fetch_all_signals_and_features()
            if df.empty:
                QMessageBox.information(self, '信息', '数据库中没有可用的数据。请先生成并分析信号。')
                return

            # 准备特征和标签
            features = df['processed_features'].tolist()
            X = self.classifier.prepare_features(features)
            y_true = df['signal_type'].tolist()

            # 检查分类器是否已训练
            if not self.classifier.is_trained:
                raise ValueError("分类器尚未训练。请先生成并分析信号。")

            # 预测
            y_pred = self.classifier.predict(X)

            # 添加预测结果到DataFrame
            df['predicted_label'] = y_pred

            # 计算准确率和分类报告
            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred)

            # 显示结果
            self.result_label.setText(f'数据库数据模型准确率: {accuracy:.2f}\n\n{report}')

            # 显示数据到表格
            self.display_table(df)

            # 存储加载的数据以便可视化
            self.loaded_df = df

            # 启用导出和可视化按钮
            self.btn_export.setEnabled(True)
            self.btn_visualize.setEnabled(True)

            QMessageBox.information(self, '成功', '数据库中的信号数据加载和分析完成！')
        except ValueError as ve:
            QMessageBox.warning(self, '输入错误', str(ve))
        except Exception as e:
            logging.error(f"从数据库加载数据过程中出现错误: {e}")
            QMessageBox.warning(self, '错误', f'从数据库加载数据过程中出错：{str(e)}')

    def display_table(self, df):
        """在表格中显示加载的数据"""
        # 设置表格列数和列名
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())

        # 填充表格内容
        for i in range(len(df)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                self.table.setItem(i, j, item)

    def visualize_features(self):
        """可视化特征分布图"""
        if self.loaded_df is None:
            self.result_label.setText("没有加载的数据可供可视化。")
            return

        try:
            # 提取特征
            features = self.loaded_df['processed_features'].tolist()
            features_df = pd.DataFrame(features)

            # 特征列表
            feature_names = self.classifier.feature_names

            plt.figure(figsize=(16, 12))

            for i, feature in enumerate(feature_names, 1):
                plt.subplot(3, 3, i)
                for label, color in zip(['good', 'bad'], ['blue', 'red']):
                    subset = features_df[self.loaded_df['signal_type'] == label]
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

    def export_analysis(self):
        """将分析结果导出到CSV文件"""
        try:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "保存分析结果",
                "",
                "CSV Files (*.csv);;All Files (*)",
                options=options
            )
            if filename:
                # 获取当前表格内容
                row_count = self.table.rowCount()
                column_count = self.table.columnCount()
                headers = [self.table.horizontalHeaderItem(i).text() for i in range(column_count)]
                data = []
                for row in range(row_count):
                    row_data = {}
                    for col in range(column_count):
                        item = self.table.item(row, col)
                        row_data[headers[col]] = item.text() if item else ''
                    data.append(row_data)
                export_df = pd.DataFrame(data)
                export_df.to_csv(filename, index=False)
                QMessageBox.information(self, '成功', f'分析结果已保存到 {filename}')
        except Exception as e:
            logging.error(f"导出分析结果时出错: {e}")
            QMessageBox.warning(self, '导出失败', f'导出分析结果时出错：{str(e)}')

    def closeEvent(self, event):
        """关闭窗口时关闭数据库连接"""
        self.db_manager.close()
        event.accept()

    def clear_data(self):
        """清空数据库和表格"""
        reply = QMessageBox.question(
            self, 
            '确认清空', 
            '确定要清空数据库吗？此操作不可恢复！',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # 清空数据库
                self.db_manager.clear_database()
                
                # 清空表格
                self.clear_table()
                
                # 清空加载的数据
                self.loaded_df = None
                
                # 禁用相关按钮
                self.btn_export.setEnabled(False)
                self.btn_visualize.setEnabled(False)
                
                # 更新结果标签
                self.result_label.setText('数据已清空')
                
                QMessageBox.information(self, '成功', '数据库和列表已清空！')
            except Exception as e:
                logging.error(f"清空数据时出错: {e}")
                QMessageBox.warning(self, '错误', f'清空数据时出错：{str(e)}')

    def clear_table(self):
        """只清空表格显示"""
        try:
            # 清空表格内容
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            
            # 更新表格显示
            self.table.update()
        except Exception as e:
            logging.error(f"清空表格时出错: {e}")
            QMessageBox.warning(self, '错误', f'清空表格时出错：{str(e)}')


def main():
    """主函数"""
    # 创建应用
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle('Fusion')

    # 创建并显示主窗口
    window = MainWindow()
    window.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
