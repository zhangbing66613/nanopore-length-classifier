"""
基于神经网络的纳米孔Length分类系统
使用 Dwell Time(s), Amplitude(pA), ECD(pC), size 四个参数进行Length分类
版本：2.2 (已集成孔内特征对比分析)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============ 检查并安装必要的库 ============
REQUIRED_LIBRARIES = [
    ('tensorflow', 'tensorflow'),
    ('scipy', 'scipy'),
    ('sklearn', 'scikit-learn'),
    ('seaborn', 'seaborn'),
    ('joblib', 'joblib'),
    ('openpyxl', 'openpyxl')
]

print("=" * 80)
print("检查依赖库...")
print("=" * 80)

missing_libs = []
for import_name, pip_name in REQUIRED_LIBRARIES:
    try:
        __import__(import_name)
        print(f"✅ {import_name}")
    except ImportError:
        print(f"❌ {import_name}")
        missing_libs.append(pip_name)

if missing_libs:
    print(f"\n缺少以下库: {', '.join(missing_libs)}")
    install = input("是否自动安装? (y/n) [默认: y]: ").strip().lower()
    if install != 'n':
        import subprocess

        for lib in missing_libs:
            print(f"正在安装 {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print("✅ 所有依赖库安装完成!")
    else:
        print("请手动安装缺少的库后重新运行程序。")
        sys.exit(1)

# ============ 导入库 ============
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
# ============ 中文显示设置 ============
# 设置中文字体（解决中文显示为方框问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ============ 中文设置结束 ============
# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)


# ============ 主类定义 ============
class NanoPoreLengthClassifier:
    """纳米孔Length分类器"""

    def __init__(self):
        """初始化分类器"""
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = ['Dwell Time (s)', 'Amplitude (pA)', 'ECD (pC)', 'size']
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        self.config = {}

    # ============ 核心方法 ============

    def run_interactive_setup(self):
        """交互式参数设置"""
        print("\n" + "=" * 80)
        print("神经网络Length分类系统 - 交互式设置")
        print("=" * 80)

        # 1. 数据路径
        print("\n[1/7] 数据文件夹设置")
        default_folder = r"D:\ecd\analysis\纳米孔数据"
        while True:
            folder = input(f"数据文件夹路径 [默认: {default_folder}]: ").strip()
            folder = folder if folder else default_folder

            if os.path.exists(folder):
                self.config['data_folder'] = folder
                print(f"✅ 数据文件夹: {folder}")
                break
            else:
                print(f"❌ 路径不存在: {folder}")
                if input("重试? (y/n) [y]: ").strip().lower() == 'n':
                    sys.exit(0)

        # 2. 列名配置
        print("\n[2/7] 列名配置")
        self.config['pore_column'] = input("Pore Number列名 [默认: 'Pore Number']: ").strip() or "Pore Number"
        print(f"✅ Pore Number列: {self.config['pore_column']}")

        # 3. Length选择
        print("\n[3/7] Length选择")
        print("1. 自动选择最常见Length")
        print("2. 手动指定Length")
        choice = input("选择方式 (1/2) [默认: 1]: ").strip() or "1"

        if choice == "1":
            self.config['target_lengths'] = None
            min_samples = input("每个类别最少样本数 [默认: 100]: ").strip()
            self.config['min_samples'] = int(min_samples) if min_samples else 100
            print(f"✅ 自动选择 (最少{self.config['min_samples']}样本)")
        else:
            lengths = input("输入Length值，用逗号分隔 (如: 100,200,300,400,500): ").strip()
            if lengths:
                self.config['target_lengths'] = [int(x.strip()) for x in lengths.split(',')]
                print(f"✅ 手动指定: {self.config['target_lengths']}")
            else:
                self.config['target_lengths'] = None
                print("⚠ 使用自动选择")

        # 4. 数据分割
        print("\n[4/7] 数据分割")
        test_size = input("测试集比例 (0.1-0.4) [默认: 0.2]: ").strip()
        self.config['test_size'] = float(test_size) if test_size else 0.2
        print(f"✅ 测试集比例: {self.config['test_size']}")

        # 5. 神经网络结构
        print("\n[5/7] 神经网络结构")
        print("1. 简单 (32-16)")
        print("2. 中等 (64-32-16) [推荐]")
        print("3. 复杂 (128-64-32-16)")
        print("4. 自定义")

        nn_choice = input("选择结构 (1/2/3/4) [默认: 2]: ").strip() or "2"

        if nn_choice == "1":
            self.config['hidden_layers'] = [32, 16]
        elif nn_choice == "2":
            self.config['hidden_layers'] = [64, 32, 16]
        elif nn_choice == "3":
            self.config['hidden_layers'] = [128, 64, 32, 16]
        else:
            custom = input("输入隐藏层神经元数，逗号分隔 (如: 128,64,32): ").strip()
            self.config['hidden_layers'] = [int(x) for x in custom.split(',')] if custom else [64, 32, 16]

        print(f"✅ 网络结构: {self.config['hidden_layers']}")

        # 6. 训练参数
        print("\n[6/7] 训练参数")
        self.config['epochs'] = int(input("训练轮数 [默认: 100]: ").strip() or 100)
        self.config['batch_size'] = int(input("批次大小 [默认: 32]: ").strip() or 32)
        self.config['learning_rate'] = float(input("学习率 [默认: 0.001]: ").strip() or 0.001)
        self.config['dropout_rate'] = float(input("Dropout率 [默认: 0.3]: ").strip() or 0.3)

        print(f"✅ 训练轮数: {self.config['epochs']}, 批次: {self.config['batch_size']}")
        print(f"✅ 学习率: {self.config['learning_rate']}, Dropout: {self.config['dropout_rate']}")

        # 7. 输出设置
        print("\n[7/7] 输出设置")
        default_output = os.path.join(os.path.dirname(self.config['data_folder']), "length_classification_results")
        output = input(f"输出文件夹 [默认: {default_output}]: ").strip() or default_output
        self.config['output_folder'] = output
        os.makedirs(output, exist_ok=True)
        print(f"✅ 输出文件夹: {output}")

        # 显示配置摘要
        self._show_config_summary()

        return self.config

    def _show_config_summary(self):
        """显示配置摘要"""
        print("\n" + "=" * 80)
        print("配置摘要")
        print("=" * 80)

        summary = f"""
        数据文件夹: {self.config['data_folder']}
        Pore Number列: {self.config['pore_column']}
        Length选择: {'自动选择' if self.config['target_lengths'] is None else f'手动指定: {self.config["target_lengths"]}'}
        测试集比例: {self.config['test_size']}
        网络结构: {self.config['hidden_layers']}
        训练参数: {self.config['epochs']}轮, 批次{self.config['batch_size']}
        学习率: {self.config['learning_rate']}, Dropout: {self.config['dropout_rate']}
        输出文件夹: {self.config['output_folder']}
        """
        print(summary)

        confirm = input("\n配置是否正确? 开始执行? (y/n) [y]: ").strip().lower()
        if confirm == 'n':
            print("重新配置...")
            self.run_interactive_setup()

    def load_and_preprocess_data(self):
        """加载并预处理数据（支持列名大小写模糊匹配）"""
        print("\n" + "=" * 80)
        print("数据加载与预处理")
        print("=" * 80)

        data_folder = self.config['data_folder']
        excel_exts = ('.xlsx', '.xls', '.xlsm', '.xlsb')
        all_data = []
        problematic_files = []

        print(f"扫描文件夹: {data_folder}")

        # 定义必需列（标准名称）
        required_columns = self.feature_names + ['Length', self.config['pore_column']]

        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.lower().endswith(excel_exts):
                    try:
                        file_path = os.path.join(root, file)
                        df = pd.read_excel(file_path)

                        # 列名标准化：不区分大小写匹配
                        column_mapping = {}
                        missing_cols = []

                        for std_col in required_columns:
                            # 查找匹配的列（不区分大小写）
                            matches = [col for col in df.columns
                                       if str(col).strip().lower() == std_col.lower()]

                            if matches:
                                actual_col = matches[0]
                                if actual_col != std_col:
                                    column_mapping[actual_col] = std_col
                            else:
                                missing_cols.append(std_col)

                        if missing_cols:
                            problematic_files.append((file, f"缺少列: {missing_cols}"))
                            continue

                        # 重命名列
                        if column_mapping:
                            df = df.rename(columns=column_mapping)

                        # 添加文件信息
                        df['Source_File'] = file

                        # 尝试查找EventID列（不区分大小写）
                        event_id_candidates = ['EventID', 'eventid', 'Event_ID', 'event_id', 'Id', 'ID', 'Index']
                        for candidate in event_id_candidates:
                            matches = [col for col in df.columns
                                       if str(col).strip().lower() == candidate.lower()]
                            if matches:
                                df['EventID'] = df[matches[0]]
                                break

                        all_data.append(df)

                    except Exception as e:
                        problematic_files.append((file, f"读取错误: {str(e)[:100]}"))

        # 合并数据
        if not all_data:
            raise ValueError("❌ 未找到任何有效数据文件！")

        self.data = pd.concat(all_data, ignore_index=True)

        print(f"\n✅ 数据加载完成")
        print(f"   成功加载文件: {len(all_data)}个")
        print(f"   总数据行数: {len(self.data):,}")

        if problematic_files:
            print(f"\n⚠ 有问题的文件 ({len(problematic_files)}个):")
            for file, reason in problematic_files[:10]:  # 只显示前10个
                print(f"   - {file}: {reason}")
            if len(problematic_files) > 10:
                print(f"   ... 还有{len(problematic_files) - 10}个文件")

        # 显示基本信息
        print(f"\n📊 数据基本信息:")
        print(f"   列: {list(self.data.columns)}")
        print(f"   唯一Length值: {self.data['Length'].nunique()}个")

        # Length分布
        length_counts = self.data['Length'].value_counts()
        print(f"\n📈 Length分布 (前10):")
        for length, count in length_counts.head(10).items():
            print(f"   Length {length}: {count:,}行 ({count / len(self.data) * 100:.1f}%)")

        return self.data

    def select_target_lengths(self):
        """选择目标Length"""
        print("\n" + "=" * 80)
        print("Length选择")
        print("=" * 80)

        length_counts = self.data['Length'].value_counts()

        if self.config['target_lengths'] is None:
            # 自动选择
            min_samples = self.config.get('min_samples', 100)
            common_lengths = length_counts[length_counts >= min_samples].index.tolist()

            if len(common_lengths) < 2:
                print(f"⚠ 数据量不足，降低要求...")
                common_lengths = length_counts.head(5).index.tolist()

            self.config['target_lengths'] = common_lengths
            print(f"✅ 自动选择 {len(common_lengths)} 个Length: {common_lengths}")
        else:
            # 检查手动指定的Length是否存在
            available = set(self.data['Length'].unique())
            specified = set(self.config['target_lengths'])
            missing = specified - available

            if missing:
                print(f"⚠ 以下Length不存在: {missing}")
                self.config['target_lengths'] = list(specified & available)
                print(f"✅ 使用存在的Length: {self.config['target_lengths']}")

        # 筛选数据
        before = len(self.data)
        self.data = self.data[self.data['Length'].isin(self.config['target_lengths'])].copy()
        after = len(self.data)

        print(f"\n✅ 数据筛选完成")
        print(f"   筛选前: {before:,}行")
        print(f"   筛选后: {after:,}行 (保留{after / before * 100:.1f}%)")

        # 显示最终分布
        print(f"\n📊 最终Length分布:")
        final_counts = self.data['Length'].value_counts()
        for length, count in final_counts.items():
            print(f"   Length {length}: {count:,}行 ({count / after * 100:.1f}%)")

        return self.data

    def explore_data(self):
        """探索性数据分析"""
        print("\n" + "=" * 80)
        print("探索性数据分析")
        print("=" * 80)

        output_dir = os.path.join(self.config['output_folder'], "exploration")
        os.makedirs(output_dir, exist_ok=True)

        # 1. 基本统计
        print("\n📋 特征统计:")
        stats_df = self.data[self.feature_names].describe().round(4)
        print(stats_df)
        stats_df.to_csv(os.path.join(output_dir, "basic_statistics.csv"))

        # 2. 相关性分析
        print("\n🔗 特征相关性:")
        corr_matrix = self.data[self.feature_names].corr().round(3)
        print(corr_matrix)
        corr_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

        # 可视化相关性
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('特征相关性热图')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        # 3. Length分布图
        plt.figure(figsize=(10, 6))
        length_counts = self.data['Length'].value_counts().sort_index()
        plt.bar(range(len(length_counts)), length_counts.values)
        plt.xticks(range(len(length_counts)), [str(x) for x in length_counts.index], rotation=45)
        plt.xlabel('Length')
        plt.ylabel('样本数量')
        plt.title('Length分布')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'length_distribution.png'), dpi=300)
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        print(f"\n✅ 探索性分析完成")
        print(f"   结果保存到: {output_dir}")

        return True

    def analyze_ecd_outliers(self):
        """分析ECD异常值（包含详细来源）"""
        print("\n" + "=" * 80)
        print("ECD异常值分析")
        print("=" * 80)

        ecd_series = self.data['ECD (pC)']

        print("📊 ECD统计摘要:")
        print(f"   中位数: {ecd_series.median():.2f} pC")
        print(f"   均值:   {ecd_series.mean():.2f} pC")
        print(f"   标准差: {ecd_series.std():.2f} pC")
        print(f"   最小值: {ecd_series.min():.2f} pC")
        print(f"   最大值: {ecd_series.max():.2f} pC")
        print(f"   99%分位数: {ecd_series.quantile(0.99):.2f} pC")

        # 使用IQR方法识别异常值
        Q1 = ecd_series.quantile(0.25)
        Q3 = ecd_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (ecd_series < lower_bound) | (ecd_series > upper_bound)
        outliers_count = outliers_mask.sum()

        print(f"\n🔍 异常值检测 (IQR方法):")
        print(f"   Q1 (25%): {Q1:.2f} pC")
        print(f"   Q3 (75%): {Q3:.2f} pC")
        print(f"   IQR: {IQR:.2f} pC")
        print(f"   正常范围: [{lower_bound:.2f}, {upper_bound:.2f}] pC")
        print(f"   异常值数量: {outliers_count:,} ({outliers_count / len(ecd_series) * 100:.2f}%)")

        if outliers_count > 0:
            # 创建详细报告
            outliers_data = self.data[outliers_mask].copy()

            # 确保有EventID列
            if 'EventID' not in outliers_data.columns:
                outliers_data['EventID'] = outliers_data.index

            # 选择要输出的列
            output_cols = ['Source_File', 'EventID', 'ECD (pC)', 'Length']
            for col in self.feature_names:
                if col != 'ECD (pC)' and col in outliers_data.columns:
                    output_cols.append(col)

            outliers_report = outliers_data[output_cols].copy()
            outliers_report = outliers_report.sort_values('ECD (pC)', ascending=False)

            # 保存报告
            output_dir = os.path.join(self.config['output_folder'], "outlier_analysis")
            os.makedirs(output_dir, exist_ok=True)

            report_path = os.path.join(output_dir, "ecd_outliers_detailed.csv")
            outliers_report.to_csv(report_path, index=False, encoding='utf-8-sig')

            print(f"\n📋 异常值详细报告:")
            print(f"   异常值总数: {outliers_count}")
            print(f"   按文件分布:")
            file_dist = outliers_data['Source_File'].value_counts()
            for file, count in file_dist.head(10).items():
                print(f"     - {file}: {count}个")

            print(f"\n📄 前10个最严重的异常值:")
            print(outliers_report.head(10).to_string(index=False))

            print(f"\n💾 完整报告已保存: {report_path}")

            # 询问如何处理
            print("\n" + "-" * 40)
            print("异常值处理选项:")
            print("1. 移除所有异常值")
            print("2. 仅移除ECD > 99%分位数的极端值")
            print("3. 保留所有数据（不处理）")
            print("4. 查看详细报告后手动处理")

            choice = input("\n请选择处理方式 (1/2/3/4) [默认: 2]: ").strip() or "2"

            if choice == "1":
                # 移除所有IQR异常值
                clean_data = self.data[~outliers_mask].copy()
                removed = len(self.data) - len(clean_data)
                self.data = clean_data
                print(f"✅ 已移除所有异常值: {removed:,}行")

            elif choice == "2":
                # 移除99%分位数以上的极端值
                threshold = ecd_series.quantile(0.99)
                extreme_mask = ecd_series > threshold
                clean_data = self.data[~extreme_mask].copy()
                removed = extreme_mask.sum()
                self.data = clean_data
                print(f"✅ 已移除ECD > {threshold:.2f} pC的极端值: {removed:,}行")

            elif choice == "4":
                print(f"\n📋 请查看详细报告: {report_path}")
                print("您可以在Excel中打开CSV文件查看所有异常值")
                input("按Enter键继续...")

        return outliers_count if 'outliers_count' in locals() else 0

    def prepare_training_data(self):
        """准备训练数据（已加入数据清洗）"""
        print("\n" + "=" * 80)
        print("准备训练数据")
        print("=" * 80)
        print("\n🔧 基于孔内分析创建新的相对特征...")

        # 确保有Pore Number列
        pore_column = self.config.get('pore_column', 'Pore Number')

        if pore_column in self.data.columns:
            # 为每个孔计算400bp的平均特征值作为参考
            pore_400_means = self.data[self.data['Length'] == 400].groupby(pore_column)[self.feature_names].mean()
            pore_400_means = pore_400_means.rename(columns={col: f'{col}_400_ref' for col in self.feature_names})

            # 将参考值合并到原始数据
            self.data = self.data.merge(pore_400_means, how='left', left_on=pore_column, right_index=True)

            # 创建相对特征（当前事件 / 同孔400bp平均值）
            for feature in self.feature_names:
                ref_col = f'{feature}_400_ref'
                if ref_col in self.data.columns:
                    # 避免除零
                    valid_mask = self.data[ref_col] != 0
                    self.data.loc[valid_mask, f'{feature}_ratio_to_400'] = self.data.loc[valid_mask, feature] / \
                                                                           self.data.loc[valid_mask, ref_col]

            # 更新特征名称列表
            new_features = [f'{feature}_ratio_to_400' for feature in self.feature_names]
            self.feature_names.extend(new_features)

            print(f"✅ 已创建 {len(new_features)} 个新的相对特征:")
            # 【在第507行之后添加以下代码】
            # 3. 创建物理意义的复合特征
            print(f"\n🔧 创建物理意义的复合特征...")

            # Dwell Time和ECD都是长度敏感的，它们的比值可能更稳定
            if 'ECD (pC)' in self.data.columns and 'Dwell Time (s)' in self.data.columns:
                valid_mask = self.data['Dwell Time (s)'] != 0
                self.data.loc[valid_mask, 'ECD_per_Dwell'] = self.data.loc[valid_mask, 'ECD (pC)'] / self.data.loc[
                    valid_mask, 'Dwell Time (s)']
                self.feature_names.append('ECD_per_Dwell')
                print(f"   - ECD_per_Dwell: 电荷转移速率(ECD/Dwell Time)")

            # Amplitude与size的比值可能反映孔径大小的影响
            if 'Amplitude (pA)' in self.data.columns and 'size' in self.data.columns:
                valid_mask = self.data['size'] != 0
                self.data.loc[valid_mask, 'Amp_per_size'] = self.data.loc[valid_mask, 'Amplitude (pA)'] / self.data.loc[
                    valid_mask, 'size']
                self.feature_names.append('Amp_per_size')
                print(f"   - Amp_per_size: 单位size的电流幅度")

            # 创建对数变换特征（对于偏态分布可能更有效）
            for col in ['Dwell Time (s)', 'ECD (pC)', 'Amplitude (pA)']:
                if col in self.data.columns:
                    # 确保所有值为正
                    positive_mask = self.data[col] > 0
                    if positive_mask.any():
                        log_col = f'log_{col.replace(" (s)", "").replace(" (pA)", "").replace(" (pC)", "")}'
                        self.data.loc[positive_mask, log_col] = np.log(self.data.loc[positive_mask, col])
                        # 对负值或零用最小值填充
                        if not positive_mask.all():
                            min_log = self.data.loc[positive_mask, log_col].min()
                            self.data.loc[~positive_mask, log_col] = min_log
                        self.feature_names.append(log_col)
                        print(f"   - {log_col}: {col}的对数变换")

        else:
            print("⚠ 未找到Pore Number列，跳过相对特征创建")
        # ============ 新增特征结束 ============

        # 提取特征和标签（现在包含更多特征）
        X = self.data[self.feature_names].values
        y = self.data['Length'].values

        # 提取特征和标签
        X = self.data[self.feature_names].values
        y = self.data['Length'].values

        # ============ 数据清洗：处理NaN和无穷大值 ============
        print("\n🔧 数据清洗：处理缺失值和异常值...")
        X = pd.DataFrame(X, columns=self.feature_names)  # 先转为DataFrame便于处理

        for col in self.feature_names:
            # 处理无穷大值
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            # 统计并处理NaN值
            nan_count_before = X[col].isna().sum()
            if nan_count_before > 0:
                # 用中位数填充NaN（保留所有样本）
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"   特征 '{col}': 填充了 {nan_count_before} 个NaN值 (使用中位数: {median_val:.4f})")

        X = X.values  # 转换回numpy数组
        print("✅ 数据清洗完成")
        # ============ 数据清洗结束 ============

        # 编码标签
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)

        print(f"\n📊 数据信息:")
        print(f"   特征形状: {X.shape}")
        print(f"   类别数: {len(self.label_encoder.classes_)}")
        print(f"   类别编码:")
        for i, cls in enumerate(self.label_encoder.classes_):
            print(f"     {i} -> Length {cls}")

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded,
            test_size=self.config['test_size'],
            random_state=42,
            stratify=y_encoded
        )

        print(f"\n📈 数据分割:")
        print(f"   训练集: {X_train.shape[0]:,} 样本")
        print(f"   测试集: {X_test.shape[0]:,} 样本")

        print(f"\n🎯 训练集类别分布:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = self.label_encoder.inverse_transform([label])[0]
            percentage = count / len(y_train) * 100
            print(f"   Length {label_name}: {count:,}样本 ({percentage:.1f}%)")

        # 检查类别不平衡
        max_count = counts.max()
        min_count = counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        if imbalance_ratio > 3:  # 如果最大类比最小类多3倍以上
            print(f"⚠️  检测到类别不平衡: 最大/最小样本比 = {imbalance_ratio:.1f}:1")
            print("   将应用类别权重进行平衡...")

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.X = X_scaled
        self.y = y_encoded

        return X_train, X_test, y_train, y_test

    def select_important_features(self, importance_threshold=0.05):
        """基于特征重要性选择重要特征"""
        print("\n🔍 基于重要性筛选特征...")

        # 根据重要性报告，手动选择重要特征
        important_features = [
            'ECD_per_Dwell',
            'Dwell Time (s)_ratio_to_400',
            'Amplitude (pA)_ratio_to_400',
            'ECD (pC)_ratio_to_400',
            'size',
            'Amp_per_size',
            'ECD (pC)',
            'Dwell Time (s)',
            'Amplitude (pA)',
            'log_ECD',
            'log_Dwell Time'
        ]

        # 检查这些特征是否存在于当前特征列表中
        current_features = set(self.feature_names)
        selected = [f for f in important_features if f in current_features]

        # 添加可能遗漏的重要原始特征
        for feature in ['Dwell Time (s)', 'Amplitude (pA)', 'ECD (pC)', 'size']:
            if feature not in selected and feature in current_features:
                selected.append(feature)

        removed = len(self.feature_names) - len(selected)
        self.feature_names = selected

        print(f"✅ 特征选择完成: 从{len(self.feature_names) + removed}个特征中选择{len(self.feature_names)}个")
        print(f"   移除了 {removed} 个不重要的特征")
        print(f"   保留的特征: {self.feature_names}")

        return self.feature_names

    def build_model(self):
        """构建优化的神经网络模型"""
        print("\n" + "=" * 80)
        print("构建优化神经网络模型")
        print("=" * 80)

        input_dim = len(self.feature_names)
        n_classes = len(self.label_encoder.classes_)

        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        # 针对更多特征增加网络容量
        model.add(layers.Dense(128, activation='relu',
                               kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Dense(64, activation='relu',
                               kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.2))

        # 输出层
        model.add(layers.Dense(n_classes, activation='softmax'))

        # 使用更小的初始学习率
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)

        # 添加更多评估指标
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy',
                               keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_accuracy'),
                               keras.metrics.SparseCategoricalCrossentropy(name='xentropy')])

        print("🧠 优化模型架构:")
        model.summary()
        self.model = model
        return model

    def train_ensemble_model(self):
        """训练简单的集成学习模型（快速实现）"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        print("\n" + "=" * 80)
        print("🌲 训练随机森林集成模型")
        print("=" * 80)

        # 使用相同的训练数据
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        # 训练随机森林
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        print("正在训练随机森林...")
        rf_model.fit(X_train, y_train)

        # 评估
        y_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred)

        print(f"📊 随机森林性能:")
        print(f"   测试准确率: {rf_accuracy:.4f}")

        # 如果已经训练过神经网络，则比较
        if hasattr(self, 'history') and self.history:
            nn_val_accuracy = max(self.history.history['val_accuracy'])
            print(f"   vs 神经网络最佳验证准确率: {nn_val_accuracy:.4f}")

        # 保存模型
        output_dir = os.path.join(self.config['output_folder'], "ensemble")
        os.makedirs(output_dir, exist_ok=True)

        joblib.dump(rf_model, os.path.join(output_dir, "random_forest_model.pkl"))

        # 特征重要性
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🌳 随机森林特征重要性:")
        print(rf_importance.head(10).to_string(index=False))

        # 保存特征重要性
        rf_importance.to_csv(os.path.join(output_dir, "rf_feature_importance.csv"), index=False)

        # 与神经网络比较
        if hasattr(self, 'history') and self.history:
            nn_val_accuracy = max(self.history.history['val_accuracy'])
            if rf_accuracy > nn_val_accuracy:
                improvement = (rf_accuracy - nn_val_accuracy) * 100
                print(f"\n🎯 随机森林比神经网络最佳验证准确率高 {improvement:.1f}%")
                print("   建议使用随机森林作为主要模型")
            else:
                print(f"\n🎯 神经网络仍然是最佳模型")

        self.ensemble_model = rf_model
        self.rf_accuracy = rf_accuracy

        return rf_accuracy
    def train_model(self):
        """训练模型（已加入类别平衡）"""
        print("\n" + "=" * 80)
        print("训练神经网络")
        print("=" * 80)

        output_dir = os.path.join(self.config['output_folder'], "training")
        os.makedirs(output_dir, exist_ok=True)

        # ============ 类别平衡：计算类别权重 ============
        print("\n⚖️  计算类别权重以平衡数据...")
        from sklearn.utils import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        print("类别权重:")
        for class_idx, weight in class_weight_dict.items():
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            print(f"   Length {class_name}: 权重 = {weight:.3f}")
        # ============ 类别平衡结束 ============

        # 回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        print(f"\n⚙️ 训练参数:")
        print(f"   轮数: {self.config['epochs']}")
        print(f"   批次: {self.config['batch_size']}")
        print(f"   验证比例: 10%")

        # 训练（应用类别权重）
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=0.1,
            callbacks=callbacks,
            class_weight=class_weight_dict,  # 应用类别权重
            verbose=1
        )

        # 绘制训练历史
        self._plot_training_history(output_dir)

        print(f"\n✅ 训练完成!")
        print(f"   最佳模型已保存: {os.path.join(output_dir, 'best_model.h5')}")

        return self.history

    def _plot_training_history(self, output_dir):
        """绘制训练历史"""
        history = self.history.history

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(history['loss'], label='训练损失', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(history['accuracy'], label='训练准确率', linewidth=2)
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='验证准确率', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('训练和验证准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        # 保存历史数据
        pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

        best_val_acc = max(history['val_accuracy']) if 'val_accuracy' in history else None
        if best_val_acc:
            print(f"   最佳验证准确率: {best_val_acc:.4f}")

    def evaluate_model(self):
        """评估模型"""
        print("\n" + "=" * 80)
        print("模型评估")
        print("=" * 80)

        output_dir = os.path.join(self.config['output_folder'], "evaluation")
        os.makedirs(output_dir, exist_ok=True)

        # 评估
        # 修改第812行
        evaluation_results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        test_loss, test_accuracy = evaluation_results[0], evaluation_results[1]
        print(f"📊 测试集性能:")
        print(f"   损失: {test_loss:.4f}")
        print(f"   准确率: {test_accuracy:.4f}")

        # 预测
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1) if len(self.label_encoder.classes_) > 2 else (
                y_pred_proba > 0.5).astype(int).flatten()

        # 解码
        y_test_decoded = self.label_encoder.inverse_transform(self.y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)

        # 分类报告
        print(f"\n📋 分类报告:")
        report = classification_report(y_test_decoded, y_pred_decoded,
                                       target_names=[str(c) for c in self.label_encoder.classes_],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        print(report_df.round(4))
        report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))

        # 混淆矩阵
        cm = confusion_matrix(y_test_decoded, y_pred_decoded,
                              labels=self.label_encoder.classes_)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        # 保存混淆矩阵
        cm_df = pd.DataFrame(cm,
                             index=self.label_encoder.classes_,
                             columns=self.label_encoder.classes_)
        cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))

        # 特征重要性
        self._analyze_feature_importance(output_dir)

        # 保存模型和工具
        self._save_model_and_tools(output_dir)

        return test_accuracy

    def _analyze_feature_importance(self, output_dir):
        """分析特征重要性"""
        print(f"\n🔍 特征重要性分析:")

        if len(self.model.layers) > 0:
            # 基于第一层权重
            weights = self.model.layers[0].get_weights()[0]
            importance = np.mean(np.abs(weights), axis=1)

            importance_df = pd.DataFrame({
                '特征': self.feature_names,
                '重要性': importance
            }).sort_values('重要性', ascending=False)

            print(importance_df.to_string(index=False))

            # 可视化
            plt.figure(figsize=(10, 5))
            bars = plt.barh(range(len(importance_df)), importance_df['重要性'])
            plt.yticks(range(len(importance_df)), importance_df['特征'])
            plt.xlabel('重要性得分')
            plt.title('特征重要性')
            plt.gca().invert_yaxis()

            for bar, score in zip(bars, importance_df['重要性']):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                         f'{score:.3f}', va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
            plt.show(block=False)
            plt.pause(2)
            plt.close()

            importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    def _save_model_and_tools(self, output_dir):
        """保存模型和工具"""
        # 保存完整模型
        model_path = os.path.join(output_dir, 'final_model.h5')
        self.model.save(model_path)

        # 保存预处理工具
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))

        # 保存配置
        pd.DataFrame([self.config]).to_csv(os.path.join(output_dir, 'training_config.csv'), index=False)

        print(f"\n💾 模型和工具已保存:")
        print(f"   模型: {model_path}")
        print(f"   标准化器: {os.path.join(output_dir, 'scaler.pkl')}")
        print(f"   标签编码器: {os.path.join(output_dir, 'label_encoder.pkl')}")

    def analyze_intra_pore_features(self):
        """分析同一孔内400bp与其他长度的特征关系"""
        print("\n" + "=" * 80)
        print("同一孔内特征对比分析: 400bp vs 其他长度")
        print("=" * 80)

        if self.data is None:
            print("错误: 请先加载数据 (调用 load_and_preprocess_data)")
            return

        # 检查必要的列
        pore_column = self.config.get('pore_column', 'Pore Number')
        required_cols = [pore_column, 'Length'] + self.feature_names
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            print(f"错误: 数据缺少以下列: {missing}")
            return
        output_dir = os.path.join(self.config['output_folder'], "intra_pore_analysis")

        # 确保目录存在（更稳健的创建方式）
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ 创建输出目录: {output_dir}")
        except Exception as e:
            print(f"❌ 创建目录失败: {e}")
            # 尝试创建父目录
            parent_dir = os.path.dirname(output_dir)
            try:
                os.makedirs(parent_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                print(f"✅ 重新创建输出目录: {output_dir}")
            except Exception as e2:
                print(f"❌ 最终创建目录失败，使用临时目录")
                output_dir = os.path.join(os.getcwd(), "temp_intra_pore_analysis")
                os.makedirs(output_dir, exist_ok=True)

        print(f"📁 输出目录确认: {output_dir}")
        # ============ 目录创建结束 ============

        # 2. 筛选出包含400bp事件的孔
        pores_with_400 = self.data[self.data['Length'] == 400][pore_column].unique()
        analysis_data = self.data[self.data[pore_column].isin(pores_with_400)].copy()
        # 筛选出包含400bp事件的孔
        pores_with_400 = self.data[self.data['Length'] == 400][pore_column].unique()
        analysis_data = self.data[self.data[pore_column].isin(pores_with_400)].copy()

        print(f"📈 数据概览:")
        print(f"   总孔数: {self.data[pore_column].nunique()}")
        print(f"   包含400bp事件的孔数: {len(pores_with_400)}")
        print(f"   用于分析的事件数: {len(analysis_data):,}")

        # 按孔统计事件数，识别有效孔
        pore_stats = analysis_data.groupby(pore_column).agg({
            'Length': ['count', 'nunique']
        }).round(2)
        pore_stats.columns = ['总事件数', '不同Length数']
        pore_stats = pore_stats.sort_values('总事件数', ascending=False)

        # 定义有效孔的标准 (可调整)
        MIN_EVENTS_PER_PORE = 10  # 一个孔至少要有10个事件
        MIN_LENGTHS_PER_PORE = 2  # 一个孔至少要有2种不同的Length

        valid_pores = pore_stats[
            (pore_stats['总事件数'] >= MIN_EVENTS_PER_PORE) &
            (pore_stats['不同Length数'] >= MIN_LENGTHS_PER_PORE)
            ].index

        print(f"\n🔍 孔过滤标准: 事件数≥{MIN_EVENTS_PER_PORE}, 不同Length数≥{MIN_LENGTHS_PER_PORE}")
        print(f"   过滤前孔数: {len(pore_stats)}")
        print(f"   有效孔数: {len(valid_pores)}")

        if len(valid_pores) == 0:
            print("警告: 没有找到符合条件的孔，正在降低标准...")
            valid_pores = pore_stats[pore_stats['总事件数'] >= 5].index
            print(f"   使用宽松标准后的有效孔数: {len(valid_pores)}")

        if len(valid_pores) == 0:
            print("错误: 没有足够的数据进行孔内分析")
            return

        # 使用有效孔的数据
        valid_data = analysis_data[analysis_data[pore_column].isin(valid_pores)].copy()

        # 创建输出目录
        output_dir = os.path.join(self.config['output_folder'], "intra_pore_analysis")
        os.makedirs(output_dir, exist_ok=True)

        # 分析每个有效孔的特征分布
        print(f"\n📊 正在分析 {len(valid_pores)} 个有效孔的特征分布...")

        all_pore_results = []

        for i, pore in enumerate(valid_pores[:20]):  # 先分析前20个孔作为示例
            pore_events = valid_data[valid_data[pore_column] == pore]

            # 检查该孔是否有400bp和其他长度
            lengths_in_pore = pore_events['Length'].unique()
            if 400 not in lengths_in_pore or len(lengths_in_pore) < 2:
                continue

            # 按长度分组计算统计量
            length_groups = pore_events.groupby('Length')

            pore_result = {pore_column: pore, 'Total Events': len(pore_events)}

            for length, group in length_groups:
                for feature in self.feature_names:
                    if feature in group.columns:
                        prefix = f"L{length}_{feature[:3]}"  # 例如: L400_Dwe, L500_Dwe
                        pore_result[f"{prefix}_mean"] = group[feature].mean()
                        pore_result[f"{prefix}_std"] = group[feature].std()
                        pore_result[f"{prefix}_median"] = group[feature].median()

            all_pore_results.append(pore_result)

            # 为每个孔绘制特征对比图 (可选，前5个孔)
            if i < 5 and len(lengths_in_pore) >= 2:
                self._plot_pore_features(pore_events, pore, pore_column, output_dir)

        # 汇总分析结果
        if all_pore_results:
            results_df = pd.DataFrame(all_pore_results)
            results_path = os.path.join(output_dir, "intra_pore_feature_summary.csv")
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')

            print(f"\n✅ 孔内分析完成!")
            print(f"   分析了 {len(results_df)} 个孔的数据")
            print(f"   详细结果保存至: {results_path}")

            # 重点: 计算400bp相对于其他长度的特征差异
            self._calculate_400bp_relative_features(valid_data, pore_column, output_dir)

        return valid_data

    def _plot_pore_features(self, pore_data, pore_id, pore_column, output_dir):
        """为单个孔绘制特征对比图"""
        # 设置图形
        features_to_plot = ['Dwell Time (s)', 'Amplitude (pA)', 'ECD (pC)']
        n_features = len(features_to_plot)

        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 6))
        if n_features == 1:
            axes = [axes]

        pore_data = pore_data.copy()

        # 为不同长度创建颜色映射
        unique_lengths = sorted(pore_data['Length'].unique())
        palette = sns.color_palette("husl", len(unique_lengths))
        length_to_color = dict(zip(unique_lengths, palette))

        for idx, feature in enumerate(features_to_plot):
            ax = axes[idx]

            # 创建箱线图
            box_data = []
            box_labels = []
            for length in unique_lengths:
                length_data = pore_data[pore_data['Length'] == length][feature].dropna()
                if len(length_data) > 0:
                    box_data.append(length_data)
                    box_labels.append(str(length))

            if box_data:
                box_plot = ax.boxplot(box_data, labels=box_labels, patch_artist=True)

                # 为每个箱体上色
                for i, (patch, length) in enumerate(zip(box_plot['boxes'], unique_lengths)):
                    if length in length_to_color:
                        patch.set_facecolor(length_to_color[length])
                        patch.set_alpha(0.7)

                # 突出显示400bp
                if 400 in unique_lengths:
                    idx_400 = list(unique_lengths).index(400)
                    if idx_400 < len(box_plot['boxes']):
                        box_plot['boxes'][idx_400].set_edgecolor('red')
                        box_plot['boxes'][idx_400].set_linewidth(2)

            ax.set_title(f'Pore {pore_id}: {feature}')
            ax.set_xlabel('Length (bp)')
            ax.set_ylabel(feature)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Pore {pore_id} - 不同Length的特征分布对比', fontsize=14)
        plt.tight_layout()

        # 保存图像
        plot_path = os.path.join(output_dir, f"pore_{pore_id}_feature_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 额外: 绘制散点矩阵
        if len(pore_data) >= 10:
            fig = sns.pairplot(pore_data,
                               hue='Length',
                               vars=features_to_plot,
                               palette=length_to_color,
                               plot_kws={'alpha': 0.6, 's': 30})
            fig_path = os.path.join(output_dir, f"pore_{pore_id}_pairplot.png")
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()

    def _calculate_400bp_relative_features(self, data, pore_column, output_dir):
        """计算400bp相对于其他长度的特征差异"""
        print(f"\n🔬 计算400bp的相对特征...")

        results = []

        # 按孔分组处理
        for pore, pore_group in data.groupby(pore_column):
            # 检查该孔是否有400bp事件
            events_400 = pore_group[pore_group['Length'] == 400]
            other_events = pore_group[pore_group['Length'] != 400]

            if len(events_400) == 0 or len(other_events) == 0:
                continue

            # 计算400bp的特征平均值
            mean_400 = events_400[self.feature_names].mean()

            # 计算其他长度的平均值
            mean_other = other_events[self.feature_names].mean()

            # 计算比值 (400bp / 其他)
            ratio = mean_400 / mean_other.replace(0, np.nan)  # 避免除零

            # 计算差值
            diff = mean_400 - mean_other

            # 计算标准化差值 (Z-score)
            std_other = other_events[self.feature_names].std()
            z_diff = diff / std_other.replace(0, np.nan)

            # 保存结果
            pore_result = {
                'Pore_Number': pore,
                'Events_400bp': len(events_400),
                'Events_Other': len(other_events),
                'Other_Lengths': ','.join(map(str, other_events['Length'].unique()))
            }

            for feature in self.feature_names:
                pore_result[f'{feature}_400_mean'] = mean_400[feature]
                pore_result[f'{feature}_other_mean'] = mean_other[feature]
                pore_result[f'{feature}_ratio'] = ratio[feature]
                pore_result[f'{feature}_diff'] = diff[feature]
                pore_result[f'{feature}_z_diff'] = z_diff[feature]

            results.append(pore_result)

        if results:
            rel_df = pd.DataFrame(results)

            # 保存详细结果
            detail_path = os.path.join(output_dir, "400bp_relative_features_detailed.csv")
            rel_df.to_csv(detail_path, index=False, encoding='utf-8-sig')

            # 生成汇总统计
            print(f"\n📈 400bp相对特征汇总 (基于 {len(rel_df)} 个孔):")

            # 对每个特征，统计比值和差异
            for feature in self.feature_names:
                ratio_col = f'{feature}_ratio'
                diff_col = f'{feature}_diff'

                if ratio_col in rel_df.columns:
                    ratios = rel_df[ratio_col].dropna()
                    if len(ratios) > 0:
                        print(f"\n  {feature}:")
                        print(f"    比值(400/其他) - 中位数: {ratios.median():.3f}, 均值: {ratios.mean():.3f}")
                        print(f"    范围: [{ratios.min():.3f}, {ratios.max():.3f}]")

            # 保存汇总统计
            summary_cols = [col for col in rel_df.columns if any(x in col for x in ['_ratio', '_diff', '_z_diff'])]
            if summary_cols:
                summary = rel_df[summary_cols].describe().round(4)
                summary_path = os.path.join(output_dir, "400bp_relative_features_summary.csv")
                summary.to_csv(summary_path, encoding='utf-8-sig')

                print(f"\n📋 汇总统计已保存: {summary_path}")
                print(summary)

            print(f"\n💡 关键洞察建议:")
            print("  1. 如果某个特征的比值稳定远离1.0，它是好的区分指标")
            print("  2. Z-score差异越大，该特征在孔内的区分度越好")
            print("  3. 可以将这些比值/差异作为新特征加入分类模型")

            # 可视化比值分布
            self._plot_relative_feature_distributions(rel_df, output_dir)

        return rel_df if 'rel_df' in locals() else None

    def _plot_relative_feature_distributions(self, rel_df, output_dir):
        """绘制相对特征的分布图"""
        # 筛选比值特征
        ratio_cols = [col for col in rel_df.columns if '_ratio' in col]

        if not ratio_cols:
            return

        n_features = len(ratio_cols)
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))

        if n_features == 1:
            axes = [axes]

        for idx, col in enumerate(ratio_cols):
            ax = axes[idx]
            feature_name = col.replace('_ratio', '')

            # 绘制分布直方图
            ratios = rel_df[col].dropna()
            ax.hist(ratios, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

            # 添加参考线 (比值=1表示无差异)
            ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='无差异线(比值=1)')

            # 添加中位数线
            median_val = ratios.median()
            ax.axvline(x=median_val, color='green', linestyle='-', linewidth=2,
                       label=f'中位数: {median_val:.2f}')

            ax.set_xlabel(f'{feature_name} 比值 (400bp/其他)')
            ax.set_ylabel('孔的数量')
            ax.set_title(f'{feature_name}: 400bp相对比值分布')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('400bp相对其他长度的特征比值分布', fontsize=14)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "400bp_relative_ratios_distribution.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 相对特征分布图已保存: {plot_path}")

    def run_pipeline(self):
        """运行完整流程"""
        try:
            print("\n" + "=" * 80)
            print("🚀 开始纳米孔Length分类流程")
            print("=" * 80)

            # 1. 交互式设置
            self.run_interactive_setup()

            # 2. 加载数据
            self.load_and_preprocess_data()

            # 3. 选择Length
            self.select_target_lengths()

            # 4. 探索性分析（可选）
            if input("\n进行探索性数据分析? (y/n) [y]: ").strip().lower() != 'n':
                self.explore_data()

            # 5. ECD异常值分析（关键步骤）
            print("\n" + "=" * 80)
            print("⚠️  ECD异常值处理（强烈推荐）")
            print("=" * 80)
            print("您的数据显示ECD存在极端异常值，可能严重影响模型性能。")

            if input("进行ECD异常值分析? (y/n) [y]: ").strip().lower() != 'n':
                self.analyze_ecd_outliers()
            else:
                print("已跳过异常值分析。")

            # 6. 同一孔内特征对比分析（新增步骤）
            print("\n" + "=" * 80)
            print("🔬 同一孔内特征对比分析")
            print("=" * 80)
            print("分析同一纳米孔内400bp与其他长度的特征关系，用于发现更好的区分特征。")

            if input("进行同一孔内特征对比分析? (y/n) [y]: ").strip().lower() != 'n':
                self.analyze_intra_pore_features()
                input("\n按Enter键继续训练流程...")

            # 7. 准备训练数据
            self.prepare_training_data()

            # 8. 构建模型
            self.build_model()

            # 9. 训练模型
            self.train_model()

            # 10. 评估模型
            accuracy = self.evaluate_model()

            # 11. 交叉验证（可选）
            if input("\n进行交叉验证? (y/n) [n]: ").strip().lower() == 'y':
                self._run_cross_validation()

            # 12. 预测新数据（可选）
            if input("\n预测新数据? (y/n) [n]: ").strip().lower() == 'y':
                self._predict_new_data()

            print("\n" + "=" * 80)
            print("🎉 流程完成!")
            print("=" * 80)
            print(f"最终测试准确率: {accuracy:.4f}")
            print(f"\n所有结果已保存到: {self.config['output_folder']}")
            print("\n📁 输出文件夹结构:")
            print("   exploration/        - 探索性分析结果")
            print("   outlier_analysis/   - 异常值详细报告")
            print("   intra_pore_analysis/ - 同一孔内特征对比分析")
            print("   training/          - 训练过程和最佳模型")
            print("   evaluation/        - 模型评估结果")
            print("\n感谢使用纳米孔Length分类系统!")

        except Exception as e:
            print(f"\n❌ 程序执行出错: {str(e)}")
            import traceback
            traceback.print_exc()

            if input("\n是否重新运行? (y/n) [n]: ").strip().lower() == 'y':
                self.__init__()
                self.run_pipeline()

    def _run_cross_validation(self):
        """交叉验证（简化版）"""
        print("\n进行交叉验证...")
        # 这里可以添加交叉验证代码
        print("交叉验证功能待实现")

    def _predict_new_data(self):
        """预测新数据（简化版）"""
        print("\n预测新数据...")
        # 这里可以添加预测新数据的代码
        print("新数据预测功能待实现")


# ============ 主程序 ============
def main():
    """主函数"""
    print("=" * 80)
    print("🧬 纳米孔Length分类系统 v2.2")
    print("=" * 80)
    print("基于四个参数区分Length:")
    print("  1. Dwell Time (s)")
    print("  2. Amplitude (pA)")
    print("  3. ECD (pC)")
    print("  4. size")
    print("\n✅ 已集成以下改进:")
    print("  • 自动数据清洗（处理NaN/Inf值）")
    print("  • 类别平衡（自动计算类别权重）")
    print("  • 同一孔内特征对比分析（新增）")
    print("=" * 80)

    # 创建分类器并运行
    classifier = NanoPoreLengthClassifier()
    classifier.run_pipeline()


if __name__ == "__main__":
    main()