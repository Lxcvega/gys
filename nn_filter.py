import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from data_model import (
    SupplierCapability, WastewaterType, ChemicalType,
    GovernmentEmissionConstraint
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Any, cast
@dataclass
class DataAuthenticityResult:
    supplier_id: str
    authenticity_score: float
    is_authentic: bool
    suspicious_fields: List[str]
    confidence_level: float

@dataclass
class SupplierFilterResult:
    supplier_id: str
    supplier_name: str
    passed_initial_filter: bool
    nn_prediction_score: float
    authenticity_score: float
    emission_compliance: bool
    overall_score: float
    filter_reason: str

class NeuralNetworkFilterModule:
    def __init__(self):
        self.nn_model: Any = None
        self.reg_model: Any = None
        self.scaler = MinMaxScaler()
        self.reg_scaler = MinMaxScaler()
        self.is_trained = False
        self.reg_is_trained = False
        self.historical_data: List[List[float]] = []
        self.historical_labels: List[int] = []
        self.reg_data: List[List[float]] = []
        self.reg_labels: List[float] = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.r2_train: Optional[float] = None
        self.r2_test: Optional[float] = None
        self.class_weights = None
        
        self.authenticity_threshold = 0.7
        self.emission_compliance_weight = 0.3
        self.nn_prediction_weight = 0.4
        self.authenticity_weight = 0.3
        
        self.feature_names = [
            'material_unit_price', 'transport_cost', 'payment_cycle',
            'effective_ingredient_content', 'delivery_timeliness',
            'inventory_capacity', 'delivery_capability',
            'emergency_supply_capability', 'safety_management_level',
            'sludge_treatment_rate'
        ]
        
        self.emission_constraints = {
            WastewaterType.INDUSTRIAL: GovernmentEmissionConstraint(
                max_daily_emission=5000.0,
                max_monthly_emission=150000.0,
                max_annual_emission=1800000.0,
                pollutant_type="重金属",
                penalty_rate=0.5
            ),
            WastewaterType.DOMESTIC: GovernmentEmissionConstraint(
                max_daily_emission=10000.0,
                max_monthly_emission=300000.0,
                max_annual_emission=3600000.0,
                pollutant_type="有机物",
                penalty_rate=0.3
            ),
            WastewaterType.MEDICAL: GovernmentEmissionConstraint(
                max_daily_emission=2000.0,
                max_monthly_emission=60000.0,
                max_annual_emission=720000.0,
                pollutant_type="病原体",
                penalty_rate=0.6
            ),
            WastewaterType.AGRICULTURAL: GovernmentEmissionConstraint(
                max_daily_emission=8000.0,
                max_monthly_emission=240000.0,
                max_annual_emission=2880000.0,
                pollutant_type="氮磷",
                penalty_rate=0.4
            ),
            WastewaterType.CHEMICAL: GovernmentEmissionConstraint(
                max_daily_emission=3000.0,
                max_monthly_emission=90000.0,
                max_annual_emission=1080000.0,
                pollutant_type="化学污染物",
                penalty_rate=0.7
            )
        }

    def add_training_data(self, supplier_capability: SupplierCapability, is_reliable: bool):
        features = self._extract_features(supplier_capability)
        self.historical_data.append(features)
        self.historical_labels.append(1 if is_reliable else 0)

    def add_regression_training_data(self, supplier_capability: SupplierCapability, actual_score: float):
        features = self._extract_features(supplier_capability)
        self.reg_data.append(features)
        self.reg_labels.append(max(0, min(100, actual_score)))

    def clear_training_data(self):
        self.historical_data = []
        self.historical_labels = []
        self.reg_data = []
        self.reg_labels = []
        self.nn_model = None
        self.reg_model = None
        self.is_trained = False
        self.reg_is_trained = False

    def train_regression_model(self):
        if len(self.reg_data) < 10:
            print(f"回归训练数据不足（{len(self.reg_data)}条），跳过R^2评估")
            return False
        
        X = np.array(self.reg_data)
        y = np.array(self.reg_labels)
        
        _model_dir = os.path.join(os.path.dirname(__file__), 'data', 'models')
        reg_path = os.path.join(_model_dir, 'nn_best_reg.pkl')
        if os.path.exists(reg_path):
            try:
                with open(reg_path, 'rb') as f:
                    loaded = pickle.load(f)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.375, random_state=42
                )

                if isinstance(loaded, dict) and 'model' in loaded:
                    self.reg_model = loaded.get('model')
                    saved_reg_scaler = loaded.get('scaler')
                    if saved_reg_scaler is not None:
                        try:
                            X_train_scaled = saved_reg_scaler.transform(X_train)
                            X_test_scaled = saved_reg_scaler.transform(X_test)
                            self.reg_scaler = saved_reg_scaler
                        except Exception:
                            self.reg_scaler = MinMaxScaler()
                            X_train_scaled = self.reg_scaler.fit_transform(X_train)
                            X_test_scaled = self.reg_scaler.transform(X_test)
                    else:
                        self.reg_scaler = MinMaxScaler()
                        X_train_scaled = self.reg_scaler.fit_transform(X_train)
                        X_test_scaled = self.reg_scaler.transform(X_test)
                else:
                    self.reg_model = loaded
                    self.reg_scaler = MinMaxScaler()
                    X_train_scaled = self.reg_scaler.fit_transform(X_train)
                    X_test_scaled = self.reg_scaler.transform(X_test)

                reg_model = cast(MLPRegressor, self.reg_model)
                train_pred = reg_model.predict(X_train_scaled)
                test_pred = reg_model.predict(X_test_scaled)
                self.r2_train = r2_score(y_train, train_pred)
                self.r2_test = r2_score(y_test, test_pred)
                from sklearn.metrics import mean_squared_error
                self.rmse_train_ = np.sqrt(mean_squared_error(y_train, train_pred))
                self.rmse_test_ = np.sqrt(mean_squared_error(y_test, test_pred))
                rmse_train = self.rmse_train_
                rmse_test = self.rmse_test_

                print(f"{'─'*55}")
                print(f"  R^2  训练集: {self.r2_train:.4f}")
                print(f"  R^2  测试集: {self.r2_test:.4f}")
                print(f"  RMSE训练集: {rmse_train:.2f} 分")
                print(f"  RMSE测试集: {rmse_test:.2f} 分")
                print(f"{'─'*55}")

                if self.r2_test >= 0.80:
                    print(f"  BEST: R^2>=0.80 -> 评分模型高度可靠，能准确预测供应商评分")
                    self.reg_is_trained = True
                elif self.r2_test >= 0.60:
                    print(f"  GOOD: R^2>=0.60 -> 评分模型较为可靠，有一定预测能力")
                    self.reg_is_trained = True
                elif self.r2_test >= 0.30:
                    print(f"  WARN: R^2>=0.30 -> 评分模型预测能力一般，建议优化")
                    self.reg_is_trained = True
                elif self.r2_test >= 0:
                    print(f"  WARN: R^2<0.30 -> 评分模型预测能力弱，需大幅改进")
                    self.reg_is_trained = False
                else:
                    print(f"  FAIL: R^2<0 -> 比猜平均值还差，模型有问题")
                    self.reg_is_trained = False

                print(f"\n  真实评分 vs 预测评分 (前10个测试样本):")
                print(f"  {'真实值':>8} {'预测值':>8} {'差值':>8}")
                print(f"  {'─'*26}")
                for i in range(min(10, len(y_test))):
                    diff = y_test[i] - test_pred[i]
                    print(f"  {y_test[i]:>8.1f} {test_pred[i]:>8.2f} {diff:>+8.2f}")
                print(f"{'─'*55}")

                return True
            except Exception as e:
                print(f"  警告：加载已保存回归模型失败({e})，将重新训练")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.375, random_state=42
        )
        self.reg_scaler = MinMaxScaler()
        X_train_scaled = self.reg_scaler.fit_transform(X_train)
        X_test_scaled = self.reg_scaler.transform(X_test)
        
        self.reg_model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        print("\n=== 训练回归模型（用于R^2真实性验证） ===")
        self.reg_model.fit(X_train_scaled, y_train)
        
        train_pred = self.reg_model.predict(X_train_scaled)
        test_pred = self.reg_model.predict(X_test_scaled)
        
        self.r2_train = r2_score(y_train, train_pred)
        self.r2_test = r2_score(y_test, test_pred)
        
        from sklearn.metrics import mean_squared_error
        self.rmse_train_ = np.sqrt(mean_squared_error(y_train, train_pred))
        self.rmse_test_ = np.sqrt(mean_squared_error(y_test, test_pred))
        rmse_train = self.rmse_train_
        rmse_test = self.rmse_test_
        
        print(f"{'─'*55}")
        print(f"  R^2  训练集: {self.r2_train:.4f}")
        print(f"  R^2  测试集: {self.r2_test:.4f}")
        print(f"  RMSE训练集: {rmse_train:.2f} 分")
        print(f"  RMSE测试集: {rmse_test:.2f} 分")
        print(f"{'─'*55}")
        
        if self.r2_test >= 0.80:
            print(f"  BEST: R^2>=0.80 -> 评分模型高度可靠，能准确预测供应商评分")
            self.reg_is_trained = True
        elif self.r2_test >= 0.60:
            print(f"  GOOD: R^2>=0.60 -> 评分模型较为可靠，有一定预测能力")
            self.reg_is_trained = True
        elif self.r2_test >= 0.30:
            print(f"  WARN: R^2>=0.30 -> 评分模型预测能力一般，建议优化")
            self.reg_is_trained = True
        elif self.r2_test >= 0:
            print(f"  WARN: R^2<0.30 -> 评分模型预测能力弱，需大幅改进")
            self.reg_is_trained = False
        else:
            print(f"  FAIL: R^2<0 -> 比猜平均值还差，模型有问题")
            self.reg_is_trained = False
        
        print(f"\n  真实评分 vs 预测评分 (前10个测试样本):")
        print(f"  {'真实值':>8} {'预测值':>8} {'差值':>8}")
        print(f"  {'─'*26}")
        for i in range(min(10, len(y_test))):
            diff = y_test[i] - test_pred[i]
            print(f"  {y_test[i]:>8.1f} {test_pred[i]:>8.2f} {diff:>+8.2f}")
        print(f"{'─'*55}")
        
        return True

    def train_neural_network(self):
        if len(self.historical_data) < 10:
            print(f"训练数据不足，当前有 {len(self.historical_data)} 条，至少需要10条")
            print("将使用供应商评分作为可靠性依据（兜底方案）")
            return False
        
        X = np.array(self.historical_data)
        y = np.array(self.historical_labels)
        
        _model_dir = os.path.join(os.path.dirname(__file__), 'data', 'models')
        clf_path = os.path.join(_model_dir, 'nn_best_clf.pkl')
        if os.path.exists(clf_path):
            try:
                with open(clf_path, 'rb') as f:
                    loaded = pickle.load(f)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.375, random_state=42, stratify=y
                )

                if isinstance(loaded, dict) and 'model' in loaded:
                    self.nn_model = loaded.get('model')
                    saved_scaler = loaded.get('scaler')
                    if saved_scaler is not None:
                        try:
                            X_train_scaled = saved_scaler.transform(X_train)
                            X_test_scaled = saved_scaler.transform(X_test)
                            self.scaler = saved_scaler
                        except Exception:
                            self.scaler = MinMaxScaler()
                            X_train_scaled = self.scaler.fit_transform(X_train)
                            X_test_scaled = self.scaler.transform(X_test)
                    else:
                        self.scaler = MinMaxScaler()
                        X_train_scaled = self.scaler.fit_transform(X_train)
                        X_test_scaled = self.scaler.transform(X_test)
                else:
                    self.nn_model = loaded
                    self.scaler = MinMaxScaler()
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)

                self.X_train = X_train_scaled
                self.X_test = X_test_scaled
                self.y_train = y_train
                self.y_test = y_test
                self.is_trained = True

                nn_model = cast(MLPClassifier, self.nn_model)
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                train_pred = nn_model.predict(self.X_train)
                test_pred = nn_model.predict(self.X_test)
                train_acc = accuracy_score(self.y_train, train_pred)
                test_acc = accuracy_score(self.y_test, test_pred)

                print("\n=== 已加载分类模型评估报告 ===")
                print(f"训练集样本: {len(self.y_train)}, 测试集样本: {len(self.y_test)}")
                print(f"训练集准确率: {train_acc:.4f}")
                print(f"测试集准确率: {test_acc:.4f}")

                if test_acc >= 0.0:
                    train_precision = precision_score(self.y_train, train_pred, zero_division=0)
                    train_recall = recall_score(self.y_train, train_pred, zero_division=0)
                    train_f1 = f1_score(self.y_train, train_pred, zero_division=0)
                    test_precision = precision_score(self.y_test, test_pred, zero_division=0)
                    test_recall = recall_score(self.y_test, test_pred, zero_division=0)
                    test_f1 = f1_score(self.y_test, test_pred, zero_division=0)

                    print(f"训练集精确率: {train_precision:.4f}, 召回率: {train_recall:.4f}, F1: {train_f1:.4f}")
                    print(f"测试集精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}, F1: {test_f1:.4f}")

                    train_cm = confusion_matrix(self.y_train, train_pred)
                    test_cm = confusion_matrix(self.y_test, test_pred)

                    print(f"训练集混淆矩阵:")
                    if train_cm.shape == (2, 2):
                        print(f"  TN={train_cm[0,0]}, FP={train_cm[0,1]}")
                        print(f"  FN={train_cm[1,0]}, TP={train_cm[1,1]}")
                    else:
                        print(f"  [{train_cm[0,0]}]")

                    print(f"测试集混淆矩阵:")
                    if test_cm.shape == (2, 2):
                        print(f"  TN={test_cm[0,0]}, FP={test_cm[0,1]}")
                        print(f"  FN={test_cm[1,0]}, TP={test_cm[1,1]}")
                    else:
                        print(f"  [{test_cm[0,0]}]")

                    if test_acc >= 0.75 and abs(train_acc - test_acc) < 0.25:
                        print(f"\nGOOD: 分类模型质量良好，可用于供应商可靠性预测")
                    elif test_acc >= 0.6:
                        print(f"\nWARN: 分类模型质量一般，建议补充更多训练数据")
                    else:
                        print(f"\nWARN: 分类模型质量较差，将使用评分兜底方案")
                        self.is_trained = False

                if len(self.reg_data) >= 10:
                    self.train_regression_model()

                return True
            except Exception as e:
                print(f"  警告：加载已保存分类模型失败({e})，将重新训练")

        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        print(f"训练数据标签分布: 可靠={n_pos}({n_pos/len(y)*100:.1f}%), 不可靠={n_neg}({n_neg/len(y)*100:.1f}%)")
        
        if n_pos == 0 or n_neg == 0:
            print("警告：训练数据只有一个类别，无法训练有效分类器")
            print("将使用供应商评分作为可靠性依据（兜底方案）")
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.375, random_state=42, stratify=y
        )
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        if abs(n_pos - n_neg) / len(y) > 0.2:
            classes = np.unique(y)
            self.class_weights = dict(zip(classes, compute_class_weight(class_weight='balanced', classes=classes, y=y)))
            class_weight = self.class_weights
        else:
            class_weight = None
        
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        print("开始训练神经网络（基于真实历史标签）...")
        
        if class_weight is not None:
            sample_weight_arr = np.array([class_weight[label] for label in y_train])
            self.nn_model.fit(X_train_scaled, y_train, sample_weight=sample_weight_arr)  # type: ignore[call-arg]
        else:
            self.nn_model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        train_pred = self.nn_model.predict(self.X_train)
        test_pred = self.nn_model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        print(f"\n=== 神经网络训练报告 ===")
        print(f"训练集样本: {len(self.y_train)}, 测试集样本: {len(self.y_test)}")
        print(f"训练集准确率: {train_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        
        if test_acc >= 0.0:
            train_precision = precision_score(self.y_train, train_pred, zero_division=0)
            train_recall = recall_score(self.y_train, train_pred, zero_division=0)
            train_f1 = f1_score(self.y_train, train_pred, zero_division=0)
            test_precision = precision_score(self.y_test, test_pred, zero_division=0)
            test_recall = recall_score(self.y_test, test_pred, zero_division=0)
            test_f1 = f1_score(self.y_test, test_pred, zero_division=0)
            
            print(f"训练集精确率: {train_precision:.4f}, 召回率: {train_recall:.4f}, F1: {train_f1:.4f}")
            print(f"测试集精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}, F1: {test_f1:.4f}")
            
            train_cm = confusion_matrix(self.y_train, train_pred)
            test_cm = confusion_matrix(self.y_test, test_pred)
            
            print(f"训练集混淆矩阵:")
            if train_cm.shape == (2, 2):
                print(f"  TN={train_cm[0,0]}, FP={train_cm[0,1]}")
                print(f"  FN={train_cm[1,0]}, TP={train_cm[1,1]}")
            else:
                print(f"  [{train_cm[0,0]}]")
            
            print(f"测试集混淆矩阵:")
            if test_cm.shape == (2, 2):
                print(f"  TN={test_cm[0,0]}, FP={test_cm[0,1]}")
                print(f"  FN={test_cm[1,0]}, TP={test_cm[1,1]}")
            else:
                print(f"  [{test_cm[0,0]}]")
            
            if test_acc >= 0.75 and abs(train_acc - test_acc) < 0.25:
                print(f"\nGOOD: 分类模型质量良好，可用于供应商可靠性预测")
            elif test_acc >= 0.6:
                print(f"\nWARN: 分类模型质量一般，建议补充更多训练数据")
            else:
                print(f"\nWARN: 分类模型质量较差，将使用评分兜底方案")
                self.is_trained = False
        
        if len(self.reg_data) >= 10:
            self.train_regression_model()

        return True

    def tune_models(self, cv: int = 3, n_iter: int = 8, random_state: int = 42):
        try:
            from sklearn.model_selection import RandomizedSearchCV
        except Exception:
            print("警告：当前环境缺少 RandomizedSearchCV，无法调参")
            return False
        
        results = {}
        if len(self.historical_data) >= 10:
            X = np.array(self.historical_data)
            y = np.array(self.historical_labels)
            X_scaled = self.scaler.fit_transform(X)
            param_dist_clf = {
                'hidden_layer_sizes': [(32,), (64,32), (128,64)],
                'alpha': [1e-4, 1e-3, 1e-2],
                'learning_rate_init': [1e-4, 5e-4, 1e-3],
                'activation': ['relu', 'tanh']
            }
            base_clf = MLPClassifier(solver='adam', max_iter=2000, early_stopping=True,
                                     validation_fraction=0.1, n_iter_no_change=20, random_state=random_state)
            print("开始对分类模型进行随机搜索调参（快速模式）...")
            rs_clf = RandomizedSearchCV(base_clf, param_distributions=param_dist_clf,
                                        n_iter=n_iter, cv=cv, scoring='f1', random_state=random_state, n_jobs=-1)
            try:
                rs_clf.fit(X_scaled, y)
                best_clf = rs_clf.best_estimator_
                self.nn_model = best_clf
                self.is_trained = True
                results['clf_best_params'] = rs_clf.best_params_
                results['clf_best_score'] = rs_clf.best_score_
                print(f"分类调参完成，最佳参数: {rs_clf.best_params_}, CV得分: {rs_clf.best_score_:.4f}")
            except Exception as e:
                print(f"分类调参失败: {e}")
        else:
            print("分类调参跳过：分类训练数据不足（<10）")
        
        if len(self.reg_data) >= 10:
            Xr = np.array(self.reg_data)
            yr = np.array(self.reg_labels)
            Xr_scaled = self.reg_scaler.fit_transform(Xr)
            param_dist_reg = {
                'hidden_layer_sizes': [(32,), (64,32), (128,64)],
                'alpha': [1e-4, 1e-3, 1e-2],
                'learning_rate_init': [1e-4, 5e-4, 1e-3],
                'activation': ['relu', 'tanh']
            }
            base_reg = MLPRegressor(solver='adam', max_iter=2000, early_stopping=True,
                                    validation_fraction=0.1, n_iter_no_change=20, random_state=random_state)
            print("开始对回归模型进行随机搜索调参（快速模式）...")
            rs_reg = RandomizedSearchCV(base_reg, param_distributions=param_dist_reg,
                                        n_iter=n_iter, cv=cv, scoring='r2', random_state=random_state, n_jobs=-1)
            try:
                rs_reg.fit(Xr_scaled, yr)
                best_reg = rs_reg.best_estimator_
                self.reg_model = best_reg
                self.reg_is_trained = True
                results['reg_best_params'] = rs_reg.best_params_
                results['reg_best_score'] = rs_reg.best_score_
                print(f"回归调参完成，最佳参数: {rs_reg.best_params_}, CV得分: {rs_reg.best_score_:.4f}")
            except Exception as e:
                print(f"回归调参失败: {e}")
        else:
            print("回归调参跳过：回归训练数据不足（<10）")
        
        return results

    def load_best_models(self) -> bool:
        loaded = False
        _model_dir = os.path.join(os.path.dirname(__file__), 'data', 'models')
        clf_path = os.path.join(_model_dir, 'nn_best_clf.pkl')
        reg_path = os.path.join(_model_dir, 'nn_best_reg.pkl')
        try:
            if os.path.exists(clf_path):
                with open(clf_path, 'rb') as f:
                    clf_loaded = pickle.load(f)
                if isinstance(clf_loaded, dict) and 'model' in clf_loaded:
                    self.nn_model = clf_loaded['model']
                    saved_scaler = clf_loaded.get('scaler')
                    if saved_scaler is not None and len(self.historical_data) > 0:
                        try:
                            X = np.array(self.historical_data)
                            saved_scaler.transform(X)
                            self.scaler = saved_scaler
                        except Exception:
                            pass
                else:
                    self.nn_model = clf_loaded
                scaler_needs_fit = (
                    not hasattr(self, 'scaler') or
                    (hasattr(self, 'scaler') and not hasattr(self.scaler, 'scale_'))
                )
                if len(self.historical_data) > 0 and scaler_needs_fit:
                    X = np.array(self.historical_data)
                    self.scaler = MinMaxScaler().fit(X)
                if len(self.historical_data) > 0:
                    X = np.array(self.historical_data)
                    if hasattr(self, 'scaler') and hasattr(self.scaler, 'scale_'):
                        X_scaled = self.scaler.transform(X)
                    else:
                        self.scaler = MinMaxScaler().fit(X)
                        X_scaled = self.scaler.transform(X)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, np.array(self.historical_labels), test_size=0.375,
                        random_state=42, stratify=np.array(self.historical_labels)
                    )
                    self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
                self.is_trained = True
                loaded = True
                print('已加载最佳分类模型：', clf_path)
            if os.path.exists(reg_path):
                with open(reg_path, 'rb') as f:
                    reg_loaded = pickle.load(f)
                if isinstance(reg_loaded, dict) and 'model' in reg_loaded:
                    self.reg_model = reg_loaded['model']
                    saved_reg_scaler = reg_loaded.get('scaler')
                    if saved_reg_scaler is not None and len(self.reg_data) > 0:
                        try:
                            Xr = np.array(self.reg_data)
                            saved_reg_scaler.transform(Xr)
                            self.reg_scaler = saved_reg_scaler
                        except Exception:
                            pass
                else:
                    self.reg_model = reg_loaded
                if len(self.reg_data) > 0 and self.reg_model is not None:
                    Xr = np.array(self.reg_data)
                    yr = np.array(self.reg_labels)
                    if not hasattr(self, 'reg_scaler') or not hasattr(self.reg_scaler, 'scale_'):
                        self.reg_scaler = MinMaxScaler().fit(Xr)
                    Xr_scaled = self.reg_scaler.transform(Xr)
                    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
                        Xr_scaled, yr, test_size=0.375, random_state=42
                    )
                    try:
                        r2_train_pred = self.reg_model.predict(Xr_train)
                        r2_test_pred = self.reg_model.predict(Xr_test)
                        self.r2_train = r2_score(yr_train, r2_train_pred)
                        self.r2_test = r2_score(yr_test, r2_test_pred)
                        self.reg_is_trained = self.r2_test >= 0.30
                        if self.r2_test >= 0.30:
                            print(f'已加载最佳回归模型：{reg_path} (训练R²={self.r2_train:.4f}, 测试R²={self.r2_test:.4f})')
                        else:
                            print(f'警告：已加载回归模型但R²={self.r2_test:.4f}<0.30，将不使用回归预测')
                    except Exception as e:
                        print(f'警告：加载的回归模型评估失败({e})，将不使用')
                        self.reg_is_trained = False
                else:
                    self.reg_is_trained = False
                loaded = True
        except Exception as e:
            print(f'加载预训练模型失败: {e}')
        return loaded

    def _extract_features(self, capability: SupplierCapability) -> List[float]:
        return [
            capability.material_unit_price,
            capability.transport_cost,
            capability.payment_cycle,
            capability.effective_ingredient_content,
            capability.delivery_timeliness,
            capability.inventory_capacity,
            capability.delivery_capability,
            capability.emergency_supply_capability,
            capability.safety_management_level,
            capability.sludge_treatment_rate
        ]

    def _validate_feature_dimension(self, n_features: int, model_type: str) -> bool:
        if model_type == 'classifier' and self.nn_model is not None:
            expected = self.nn_model.coefs_[0].shape[0] if hasattr(self.nn_model, 'coefs_') else None
        elif model_type == 'regressor' and self.reg_model is not None:
            expected = self.reg_model.coefs_[0].shape[0] if hasattr(self.reg_model, 'coefs_') else None
        else:
            return True

        if expected is not None and n_features != expected:
            print(f"  ⚠️  特征维度不匹配: 当前 {n_features} 维，模型期望 {expected} 维。"
                  f"请重新训练模型或检查 data_processor 配置")
            return False
        return True

    def predict_supplier_reliability(self, capability: SupplierCapability,
                                     internal_score: Optional[float] = None) -> float:
        features = self._extract_features(capability)
        
        if self.reg_is_trained and self.reg_model is not None:
            if self._validate_feature_dimension(len(features), 'regressor'):
                try:
                    features_scaled = self.reg_scaler.transform(np.array([features]))
                    predicted_score = float(self.reg_model.predict(features_scaled)[0])
                    return min(1.0, max(0.0, predicted_score / 100.0))
                except Exception as e:
                    print(f"  警告：回归预测失败({e})")

        if self.is_trained and self.nn_model is not None:
            if self._validate_feature_dimension(len(features), 'classifier'):
                try:
                    features_scaled = self.scaler.transform(np.array([features]))
                    if hasattr(self.nn_model, 'predict_proba'):
                        prediction_proba = self.nn_model.predict_proba(features_scaled)[0]
                        proba = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
                        return float(proba)
                    else:
                        pred = self.nn_model.predict(features_scaled)[0]
                        return float(pred)
                except Exception as e:
                    print(f"  警告：分类预测失败({e})，使用评分兜底")
        
        if internal_score is not None:
            return min(1.0, max(0.0, internal_score / 100.0))
        
        return 0.5

    def get_training_predictions(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.is_trained or self.X_train is None or self.nn_model is None:
            return None, None
        
        y_pred = self.nn_model.predict(self.X_train)
        return self.y_train, y_pred
    
    def get_test_predictions(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.is_trained or self.X_test is None or self.nn_model is None:
            return None, None
        
        y_pred = self.nn_model.predict(self.X_test)
        return self.y_test, y_pred

    def check_data_authenticity(self, capability: SupplierCapability, 
                               historical_records: Dict[str, List[float]]) -> DataAuthenticityResult:
        suspicious_fields = []
        authenticity_scores = []
        
        current_features = self._extract_features(capability)
        
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in historical_records:
                historical_values = historical_records[feature_name]
                if len(historical_values) > 0:
                    mean_val = np.mean(historical_values)
                    std_val = np.std(historical_values)
                    
                    if std_val > 0:
                        z_score = abs(current_features[i] - mean_val) / std_val
                        if z_score > 4.0:
                            suspicious_fields.append(feature_name)
                            authenticity_scores.append(max(0, 1.0 - z_score / 6.0))
                        else:
                            authenticity_scores.append(1.0)
                    else:
                        authenticity_scores.append(0.9)
                else:
                    authenticity_scores.append(0.8)
            else:
                authenticity_scores.append(0.9)
        
        if authenticity_scores:
            overall_authenticity = float(np.mean(authenticity_scores))
        else:
            overall_authenticity = 0.7
        
        confidence_level = min(1.0, len(historical_records) / 10.0) if historical_records else 0.5
        
        return DataAuthenticityResult(
            supplier_id=capability.supplier_id,
            authenticity_score=overall_authenticity,
            is_authentic=bool(overall_authenticity >= self.authenticity_threshold),
            suspicious_fields=suspicious_fields,
            confidence_level=confidence_level
        )

    def check_emission_compliance(self, capability: SupplierCapability, 
                                 wastewater_type: WastewaterType,
                                 required_quantity: float) -> Tuple[bool, float]:
        constraint = self.emission_constraints.get(wastewater_type)
        if not constraint:
            return True, 1.0
        
        estimated_emission = required_quantity * (1.0 - (capability.safety_management_level / 5.0) * 0.5)
        
        daily_compliance = estimated_emission <= constraint.max_daily_emission
        monthly_compliance = estimated_emission * 30 <= constraint.max_monthly_emission
        annual_compliance = estimated_emission * 365 <= constraint.max_annual_emission
        
        is_compliant = daily_compliance and monthly_compliance and annual_compliance
        
        compliance_score = 1.0
        if not is_compliant:
            excess_ratio = max(
                estimated_emission / constraint.max_daily_emission,
                (estimated_emission * 30) / constraint.max_monthly_emission,
                (estimated_emission * 365) / constraint.max_annual_emission
            )
            compliance_score = max(0.0, 1.0 - (excess_ratio - 1.0) * constraint.penalty_rate)
        
        return is_compliant, compliance_score

    def initial_supplier_filter(self, suppliers: List[SupplierCapability],
                               wastewater_type: WastewaterType,
                               required_quantity: float,
                               historical_records: Optional[Dict[str, List[float]]] = None,
                               internal_scores: Optional[Dict[str, float]] = None) -> List[SupplierFilterResult]:
        if historical_records is None:
            historical_records = {}
        if internal_scores is None:
            internal_scores = {}
        
        filter_results = []
        
        for supplier in suppliers:
            score = internal_scores.get(supplier.supplier_id)
            nn_score = self.predict_supplier_reliability(supplier, internal_score=score)
            
            authenticity_result = self.check_data_authenticity(supplier, historical_records)
            authenticity_score = authenticity_result.authenticity_score
            
            emission_compliant, emission_score = self.check_emission_compliance(
                supplier, wastewater_type, required_quantity
            )
            
            overall_score = (
                nn_score * self.nn_prediction_weight +
                authenticity_score * self.authenticity_weight +
                emission_score * self.emission_compliance_weight
            )
            
            passed_filter = (
                nn_score >= 0.5 and
                authenticity_result.is_authentic and
                emission_compliant
            )
            
            filter_reason = []
            if nn_score < 0.5:
                filter_reason.append(f"可靠性评分过低({nn_score:.2f})")
            if not authenticity_result.is_authentic:
                filter_reason.append(f"数据真实性不足({authenticity_score:.2f})")
            if not emission_compliant:
                filter_reason.append(f"排放量不合规")
            
            filter_results.append(SupplierFilterResult(
                supplier_id=supplier.supplier_id,
                supplier_name=supplier.supplier_name,
                passed_initial_filter=passed_filter,
                nn_prediction_score=nn_score,
                authenticity_score=authenticity_score,
                emission_compliance=emission_compliant,
                overall_score=overall_score,
                filter_reason="; ".join(filter_reason) if filter_reason else "通过初筛"
            ))
        
        return filter_results

    def get_filtered_suppliers(self, filter_results: List[SupplierFilterResult],
                              original_suppliers: List[SupplierCapability]) -> List[SupplierCapability]:
        filtered_ids = {result.supplier_id for result in filter_results if result.passed_initial_filter}
        return [supplier for supplier in original_suppliers if supplier.supplier_id in filtered_ids]

    def generate_filter_report(self, filter_results: List[SupplierFilterResult]) -> str:
        total_suppliers = len(filter_results)
        passed_suppliers = sum(1 for result in filter_results if result.passed_initial_filter)
        failed_suppliers = total_suppliers - passed_suppliers
        
        report = f"供应商初筛报告\n"
        report += f"{'='*60}\n"
        report += f"总供应商数量: {total_suppliers}\n"
        report += f"通过初筛: {passed_suppliers}\n"
        report += f"未通过初筛: {failed_suppliers}\n"
        report += f"通过率: {passed_suppliers/total_suppliers*100:.1f}%\n\n"
        
        report += f"通过初筛的供应商:\n"
        report += f"{'-'*60}\n"
        for result in filter_results:
            if result.passed_initial_filter:
                report += f"• {result.supplier_name} (ID: {result.supplier_id})\n"
                report += f"  综合评分: {result.overall_score:.3f}\n"
                report += f"  神经网络评分: {result.nn_prediction_score:.3f}\n"
                report += f"  真实性评分: {result.authenticity_score:.3f}\n\n"
        
        report += f"未通过初筛的供应商:\n"
        report += f"{'-'*60}\n"
        for result in filter_results:
            if not result.passed_initial_filter:
                report += f"• {result.supplier_name} (ID: {result.supplier_id})\n"
                report += f"  原因: {result.filter_reason}\n\n"
        
        return report